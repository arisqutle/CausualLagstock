from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:  # pragma: no cover - optional dependency path
    torchdiffeq_odeint = None

ODEIntFn = Callable[..., torch.Tensor] | None
GraphMode = Literal["full", "no_graph", "a_only", "t_only", "random"]


@dataclass(frozen=True)
class NeuralODEConfig:
    text_emb_dim: int = 768
    event_type_emb_dim: int = 32
    event_profile_dim: int = 4
    impact_dim: int = 64
    hidden_dim: int = 128
    price_hidden_dim: int = 64
    stock_hidden_dim: int = 64
    ode_steps: int = 8
    ode_method: str = "dopri5"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    gate_scale: float = 5.0
    dropout: float = 0.1


class ImpactInitializer(nn.Module):
    def __init__(self, n_event_types: int, config: NeuralODEConfig):
        super().__init__()
        self.type_embedding = nn.Embedding(n_event_types, config.event_type_emb_dim)
        in_dim = config.text_emb_dim + config.event_type_emb_dim + 1 + config.event_profile_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.impact_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        event_types: torch.Tensor,
        magnitudes: torch.Tensor,
        event_profiles: torch.Tensor,
    ) -> torch.Tensor:
        type_emb = self.type_embedding(event_types)
        signed_mag = magnitudes.unsqueeze(-1)
        fused = torch.cat([text_embeddings, type_emb, signed_mag, event_profiles], dim=-1)
        return self.net(fused)


class ODEEventPropagator(nn.Module):
    def __init__(
        self,
        n_event_types: int,
        causal_matrix: torch.Tensor | None,
        lag_matrix: torch.Tensor | None,
        graph_mode: GraphMode,
        config: NeuralODEConfig,
    ):
        super().__init__()
        if graph_mode == "no_graph":
            causal_matrix = None
            lag_matrix = None
        elif graph_mode == "a_only":
            lag_matrix = None
        elif graph_mode == "t_only":
            causal_matrix = None
        elif graph_mode == "random":
            if causal_matrix is None or lag_matrix is None:
                raise ValueError("random graph mode requires causal_matrix and lag_matrix")
        elif graph_mode != "full":
            raise ValueError(f"unsupported graph_mode: {graph_mode}")

        if causal_matrix is not None and int(causal_matrix.shape[0]) != int(n_event_types):
            raise ValueError("causal_matrix must match n_event_types")
        if lag_matrix is not None and int(lag_matrix.shape[0]) != int(n_event_types):
            raise ValueError("lag_matrix must match n_event_types")
        if causal_matrix is not None and lag_matrix is not None and causal_matrix.shape != lag_matrix.shape:
            raise ValueError("causal_matrix and lag_matrix must have the same shape")

        self.config = config
        self.graph_mode = graph_mode
        self.n_event_types = int(n_event_types)
        self.impact_dim = int(config.impact_dim)
        self.register_buffer("offdiag_mask", (1.0 - torch.eye(self.n_event_types)).float())
        if causal_matrix is not None:
            self.register_buffer("causal_matrix", causal_matrix.float())
        if lag_matrix is not None:
            self.register_buffer("lag_matrix", lag_matrix.float())
        self.decay_raw = nn.Parameter(torch.zeros(self.n_event_types, 1))
        self.func = nn.Sequential(
            nn.Linear(self.impact_dim * 2 + 1, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, self.impact_dim),
        )

    def ode_func(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.graph_mode == "no_graph":
            agg = torch.zeros_like(state)
        elif self.graph_mode == "a_only":
            causal_matrix = cast(torch.Tensor, self.causal_matrix)
            time_adj = causal_matrix
            agg = torch.einsum("ji,bjd->bid", time_adj, state)
        elif self.graph_mode == "t_only":
            lag_matrix = cast(torch.Tensor, self.lag_matrix)
            offdiag_mask = cast(torch.Tensor, self.offdiag_mask)
            time_adj = torch.sigmoid(self.config.gate_scale * (t - lag_matrix)) * offdiag_mask
            agg = torch.einsum("ji,bjd->bid", time_adj, state)
        else:
            lag_matrix = cast(torch.Tensor, self.lag_matrix)
            causal_matrix = cast(torch.Tensor, self.causal_matrix)
            gate = torch.sigmoid(self.config.gate_scale * (t - lag_matrix))
            time_adj = causal_matrix * gate
            agg = torch.einsum("ji,bjd->bid", time_adj, state)
        t_feat = torch.full(
            (state.shape[0], state.shape[1], 1),
            float(t.item()),
            dtype=state.dtype,
            device=state.device,
        )
        fused = torch.cat([state, agg, t_feat], dim=-1)
        drift = self.func(fused)
        decay = F.softplus(self.decay_raw).unsqueeze(0)
        return drift - decay * state

    def _euler_fallback(self, initial_state: torch.Tensor, time_points: torch.Tensor) -> torch.Tensor:
        states = [initial_state]
        current = initial_state
        for idx in range(1, time_points.shape[0]):
            prev_t = time_points[idx - 1]
            next_t = time_points[idx]
            dt = next_t - prev_t
            current = current + dt * self.ode_func(prev_t, current)
            states.append(current)
        return torch.stack(states, dim=0)

    @staticmethod
    def _is_underflow_error(exc: BaseException) -> bool:
        return "underflow in dt" in str(exc).lower()

    def forward(self, initial_state: torch.Tensor, time_points: torch.Tensor) -> torch.Tensor:
        odeint_fn = cast(ODEIntFn, torchdiffeq_odeint)
        if odeint_fn is not None:
            try:
                return odeint_fn(
                    self.ode_func,
                    initial_state,
                    time_points,
                    method=self.config.ode_method,
                    rtol=self.config.ode_rtol,
                    atol=self.config.ode_atol,
                )
            except (AssertionError, RuntimeError) as exc:
                if not self._is_underflow_error(exc):
                    raise

                step_size = None
                if time_points.shape[0] >= 2:
                    step_size = float((time_points[1] - time_points[0]).item())
                if step_size is not None and step_size > 0.0:
                    return odeint_fn(
                        self.ode_func,
                        initial_state,
                        time_points,
                        method="rk4",
                        options={"step_size": step_size},
                    )
        return self._euler_fallback(initial_state, time_points)


class EventToStockMapper(nn.Module):
    def __init__(self, n_event_types: int, n_stocks: int):
        super().__init__()
        self.affinity_logits = nn.Parameter(torch.zeros(n_event_types, n_stocks))
        nn.init.normal_(self.affinity_logits, mean=0.0, std=0.1)

    def forward(self, event_states: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.affinity_logits)
        return torch.einsum("kn,bkd->bnd", weights, event_states)


class StockGraphPropagation(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(state_dim, hidden_dim)
        self.src_att = nn.Linear(hidden_dim, 1, bias=False)
        self.dst_att = nn.Linear(hidden_dim, 1, bias=False)
        self.gate = nn.Linear(state_dim + hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, state_dim)

    def forward(self, stock_states: torch.Tensor) -> torch.Tensor:
        proj = self.proj(stock_states)
        src = self.src_att(proj)
        dst = self.dst_att(proj)
        scores = F.leaky_relu(src + dst.transpose(1, 2), negative_slope=0.2)
        attn = torch.softmax(scores, dim=-1)
        neighbors = torch.matmul(attn, proj)
        gate = torch.sigmoid(self.gate(torch.cat([stock_states, neighbors], dim=-1)))
        updated = stock_states + self.out(gate * neighbors)
        return updated


class MultiSignalAggregator(nn.Module):
    def __init__(self, impact_dim: int, price_hidden_dim: int, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(impact_dim, 1)
        self.price_gru = nn.GRU(
            input_size=5,
            hidden_size=price_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.fusion = nn.Sequential(
            nn.Linear(impact_dim * 2 + price_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        stock_states: torch.Tensor,
        price_history: torch.Tensor,
        target_stock_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn_scores = self.attn(stock_states).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.einsum("bn,bnd->bd", attn_weights, stock_states)

        batch_idx = torch.arange(stock_states.shape[0], device=stock_states.device)
        target_state = stock_states[batch_idx, target_stock_ids]
        _, hidden = self.price_gru(price_history)
        price_state = hidden[-1]
        return self.fusion(torch.cat([target_state, pooled, price_state], dim=-1))


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.direction = nn.Linear(hidden_dim, 3)
        self.magnitude = nn.Linear(hidden_dim, 1)

    def forward(self, fused_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.direction(fused_state), self.magnitude(fused_state).squeeze(-1)


class NeuralODEStockPredictor(nn.Module):
    def __init__(
        self,
        *,
        n_event_types: int,
        n_stocks: int,
        causal_matrix: torch.Tensor | None = None,
        lag_matrix: torch.Tensor | None = None,
        graph_mode: GraphMode = "full",
        config: NeuralODEConfig | None = None,
    ):
        super().__init__()
        self.config = config or NeuralODEConfig()
        self.n_event_types = n_event_types
        self.n_stocks = n_stocks
        self.initializer = ImpactInitializer(n_event_types, self.config)
        self.propagator = ODEEventPropagator(
            n_event_types=n_event_types,
            causal_matrix=causal_matrix,
            lag_matrix=lag_matrix,
            graph_mode=graph_mode,
            config=self.config,
        )
        self.mapper = EventToStockMapper(n_event_types, n_stocks)
        self.stock_prop = StockGraphPropagation(self.config.impact_dim, self.config.stock_hidden_dim)
        self.aggregator = MultiSignalAggregator(
            impact_dim=self.config.impact_dim,
            price_hidden_dim=self.config.price_hidden_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.head = PredictionHead(self.config.hidden_dim)

    def build_initial_state(
        self,
        text_embeddings: torch.Tensor,
        event_types: torch.Tensor,
        timestamps: torch.Tensor,
        magnitudes: torch.Tensor,
        event_profiles: torch.Tensor,
    ) -> torch.Tensor:
        impacts = self.initializer(text_embeddings, event_types, magnitudes, event_profiles)
        event_one_hot = F.one_hot(event_types, num_classes=self.n_event_types).float()
        recency = 0.5 + timestamps
        weighted_impacts = impacts * recency.unsqueeze(-1)
        state_sum = torch.einsum("blk,bld->bkd", event_one_hot, weighted_impacts)
        denom = torch.einsum("blk,bl->bk", event_one_hot, recency).unsqueeze(-1).clamp_min(1e-6)
        return state_sum / denom

    def forward(
        self,
        text_embeddings: torch.Tensor,
        event_types: torch.Tensor,
        timestamps: torch.Tensor,
        magnitudes: torch.Tensor,
        event_profiles: torch.Tensor,
        price_history: torch.Tensor,
        target_stock_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_state = self.build_initial_state(text_embeddings, event_types, timestamps, magnitudes, event_profiles)
        ode_steps = max(int(self.config.ode_steps), 2)
        time_points = torch.linspace(
            0.0,
            1.0,
            steps=ode_steps,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        state_path = self.propagator(initial_state, time_points)
        final_event_state = state_path[-1]
        stock_states = self.mapper(final_event_state)
        stock_states = self.stock_prop(stock_states)
        fused = self.aggregator(stock_states, price_history, target_stock_ids)
        return self.head(fused)
