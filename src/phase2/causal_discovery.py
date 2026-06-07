"""
Phase 2: 可微分时滞因果发现网络 (STACD)
=========================================
Sparse Temporal Attention Causal Discovery
- 可学习的时间编码器 (Fourier特征)
- 稀疏因果注意力 (带时滞参数)
- DAG约束 (NOTEARS)
- 因果图提取与可视化
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase1.event_types import NUM_EVENT_TYPES


# ============================================================
# 时间编码器 (Learnable Fourier Features)
# ============================================================
class TemporalEncoder(nn.Module):
    """
    将连续时间戳编码为向量表示。
    使用可学习的Fourier特征 (而非固定sinusoidal)，
    让模型自动发现金融数据中的重要时间周期。
    """

    def __init__(self, d_model: int = 128, num_freqs: int = 64):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_freqs) * 0.01)
        self.phases = nn.Parameter(torch.zeros(num_freqs))
        self.linear = nn.Linear(num_freqs * 2, d_model)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (batch, seq_len) - 归一化到[0,1]的时间戳
        Returns:
            temporal_encoding: (batch, seq_len, d_model)
        """
        t = timestamps.unsqueeze(-1)  # (batch, seq_len, 1)
        cos_feat = torch.cos(2 * math.pi * self.freqs * t + self.phases)
        sin_feat = torch.sin(2 * math.pi * self.freqs * t + self.phases)
        feat = torch.cat([cos_feat, sin_feat], dim=-1)  # (batch, seq_len, 2*num_freqs)
        return self.linear(feat)


# ============================================================
# 稀疏因果注意力层
# ============================================================
class SparseCausalAttentionLayer(nn.Module):
    """
    核心创新组件: 时滞感知的稀疏因果注意力。

    与标准Transformer Attention的区别:
    1. 因果mask: 只允许过去→未来的信息流
    2. 时滞参数: 为每对事件类型学习最优时滞 τ
    3. 高斯核: 时间差越接近最优时滞，attention权重越高
    4. 因果先验: 可学习的因果强度矩阵
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_event_types: int = NUM_EVENT_TYPES,
                 sigma: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sigma = sigma

        # 标准Multi-Head Attention组件
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # 核心: 可学习的时滞参数矩阵
        # lag_raw[i][j] → softplus → 实际时滞 (保证正值)
        initial_lag = torch.empty(n_event_types, n_event_types).uniform_(1.0, 10.0)
        self.lag_raw = nn.Parameter(torch.log(torch.expm1(initial_lag)))

        # 核心: 可学习的因果强度先验
        # causal_raw[i][j] → sigmoid → 因果概率
        self.causal_raw = nn.Parameter(torch.empty(n_event_types, n_event_types))
        nn.init.normal_(self.causal_raw, mean=-1.0, std=0.25)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    @property
    def lag_matrix(self):
        """获取时滞矩阵 (正值)"""
        return F.softplus(self.lag_raw)

    @property
    def causal_matrix(self):
        """获取因果强度矩阵 [0,1]"""
        return torch.sigmoid(self.causal_raw)

    def forward(self, event_reprs, event_types, timestamps):
        """
        Args:
            event_reprs: (batch, seq_len, d_model) - 事件表示
            event_types: (batch, seq_len) - 事件类型ID (0-19)
            timestamps:  (batch, seq_len) - 归一化时间戳
        Returns:
            output:      (batch, seq_len, d_model)
            attn_weights: (batch, n_heads, seq_len, seq_len) - 注意力权重
        """
        B, L, D = event_reprs.shape
        residual = event_reprs

        # Multi-Head Q, K, V
        Q = self.W_Q(event_reprs).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(event_reprs).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(event_reprs).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # 标准attention分数
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        # (B, n_heads, L, L)

        # === 因果约束 ===

        # 1. 时间方向性mask: 只有时间在前的事件可以影响后面的
        time_diff = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)  # (B, L, L)
        causal_mask = (time_diff > 0).float().unsqueeze(1)  # (B, 1, L, L)

        # 2. 时滞匹配分数
        # 获取每对事件的类型索引
        type_i = event_types.unsqueeze(-1).expand(-1, -1, L)  # (B, L, L)
        type_j = event_types.unsqueeze(-2).expand(-1, L, -1)  # (B, L, L)

        # 查表获取最优时滞
        optimal_lag = self.lag_matrix[type_i, type_j]  # (B, L, L)

        # 高斯核: 时间差越接近最优时滞，分数越高
        lag_score = torch.exp(
            -0.5 * ((time_diff.abs() - optimal_lag) ** 2) / (self.sigma ** 2)
        )
        lag_score = lag_score.unsqueeze(1)  # (B, 1, L, L)

        # 3. 因果强度先验
        causal_strength = self.causal_matrix[type_j, type_i]  # (B, L, L), source -> target
        causal_strength = causal_strength.unsqueeze(1)  # (B, 1, L, L)

        # 4. 最终: 标准attention × 因果mask × 时滞匹配 × 因果强度
        modulated_scores = attn_scores + torch.log(causal_mask + 1e-8) + \
                           torch.log(lag_score + 1e-8) + \
                           torch.log(causal_strength + 1e-8)

        attn_weights = F.softmax(modulated_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权聚合
        output = torch.matmul(attn_weights, V)  # (B, n_heads, L, d_k)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.W_O(output)

        # 残差连接 + LayerNorm
        output = self.layer_norm(output + residual)

        return output, attn_weights


# ============================================================
# 完整的STACD模型
# ============================================================
class STACD(nn.Module):
    """
    Sparse Temporal Attention Causal Discovery
    完整的因果发现模块。
    """

    def __init__(self, text_emb_dim: int = 768, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 4,
                 n_event_types: int = NUM_EVENT_TYPES,
                 sigma: float = 2.0, dropout: float = 0.1):
        super().__init__()

        self.n_event_types = n_event_types

        # 事件类型 embedding
        self.type_embedding = nn.Embedding(n_event_types, 128)

        # 时间编码器
        self.temporal_encoder = TemporalEncoder(d_model=128, num_freqs=64)

        # 输入投影: text_emb(768) + type_emb(128) + time_enc(128) → d_model
        self.input_projection = nn.Linear(text_emb_dim + 128 + 128, d_model)

        # 多层稀疏因果注意力 (共享因果参数)
        self.layers = nn.ModuleList([
            SparseCausalAttentionLayer(d_model, n_heads, n_event_types, sigma, dropout)
            for _ in range(n_layers)
        ])

        # 让所有层共享同一组因果参数
        shared_lag = self.layers[0].lag_raw
        shared_causal = self.layers[0].causal_raw
        for layer in self.layers[1:]:
            layer.lag_raw = shared_lag
            layer.causal_raw = shared_causal

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, text_embeddings, event_types, timestamps):
        """
        Args:
            text_embeddings: (batch, seq_len, 768) - FinBERT [CLS] embeddings
            event_types:     (batch, seq_len) - 事件类型ID
            timestamps:      (batch, seq_len) - 归一化时间戳
        Returns:
            event_reprs: (batch, seq_len, d_model) - 因果感知的事件表示
            causal_info: dict with causal_matrix, lag_matrix
        """
        # 构建输入表示
        type_emb = self.type_embedding(event_types)
        time_enc = self.temporal_encoder(timestamps)
        combined = torch.cat([text_embeddings, type_emb, time_enc], dim=-1)
        event_reprs = self.input_projection(combined)

        # 多层因果注意力
        all_attn_weights = []
        for layer in self.layers:
            event_reprs, attn_w = layer(event_reprs, event_types, timestamps)
            all_attn_weights.append(attn_w)

        # FFN
        ffn_out = self.ffn(event_reprs)
        event_reprs = self.ffn_norm(event_reprs + ffn_out)

        # 提取因果图信息
        causal_info = {
            "causal_matrix": self.layers[0].causal_matrix,  # (K, K)
            "lag_matrix": self.layers[0].lag_matrix,          # (K, K)
            "attention_weights": all_attn_weights,
        }

        return event_reprs, causal_info


# ============================================================
# 因果图正则化损失
# ============================================================
class CausalRegularizationLoss(nn.Module):
    """因果图的正则化约束"""

    def __init__(self, lambda_sparse: float = 0.0005, lambda_dag: float = 0.05, lambda_variance: float = 0.001):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_dag = lambda_dag
        self.lambda_variance = lambda_variance

    def sparsity_loss(self, causal_matrix: torch.Tensor) -> torch.Tensor:
        """Group Lasso稀疏性惩罚"""
        return self.lambda_sparse * torch.sum(torch.abs(causal_matrix))

    def dag_loss(self, causal_matrix: torch.Tensor) -> torch.Tensor:
        """
        NOTEARS DAG约束:
        h(A) = tr(e^{A ⊙ A}) - K = 0 iff A is DAG
        """
        K = causal_matrix.shape[0]
        M = causal_matrix * causal_matrix  # element-wise square
        # 矩阵指数的截断级数近似 (10阶)
        device = causal_matrix.device
        E = torch.eye(K, device=device)
        power = E.clone()
        expm = E.clone()
        for i in range(1, 10):
            power = torch.mm(power, M) / i
            expm = expm + power
        h = torch.trace(expm) - K
        return self.lambda_dag * (h ** 2)

    def forward(self, causal_info: dict) -> dict:
        """计算所有因果正则化损失"""
        A = causal_info["causal_matrix"]
        losses = {
            "sparsity": self.sparsity_loss(A),
            "dag": self.dag_loss(A),
            "variance": -self.lambda_variance * torch.var(A),
        }
        losses["total_reg"] = losses["sparsity"] + losses["dag"] + losses["variance"]
        return losses


# ============================================================
# 因果图提取与分析工具
# ============================================================
class CausalGraphAnalyzer:
    """从训练好的STACD模型中提取和分析因果图"""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    @torch.no_grad()
    def extract_graph(self, model: STACD):
        """提取因果邻接矩阵和时滞矩阵"""
        A = model.layers[0].causal_matrix.cpu().numpy()
        T = model.layers[0].lag_matrix.cpu().numpy()

        # 稀疏化
        A_sparse = A.copy()
        A_sparse[A_sparse < self.threshold] = 0

        return {
            "adjacency_raw": A,
            "adjacency_sparse": A_sparse,
            "lag_matrix": T,
            "n_edges": (A_sparse > 0).sum(),
            "density": (A_sparse > 0).mean(),
        }

    def get_top_causal_edges(self, model: STACD, top_k: int = 20):
        """获取因果强度最大的top-K条边"""
        graph = self.extract_graph(model)
        A = graph["adjacency_raw"]
        T = graph["lag_matrix"]

        from .event_extractor import EVENT_TYPE_LIST
        edges = []
        K = A.shape[0]
        for i in range(K):
            for j in range(K):
                if A[i, j] > self.threshold:
                    edges.append({
                        "source": EVENT_TYPE_LIST[i],
                        "target": EVENT_TYPE_LIST[j],
                        "strength": float(A[i, j]),
                        "lag_days": float(T[i, j]),
                    })
        edges.sort(key=lambda x: x["strength"], reverse=True)
        return edges[:top_k]
