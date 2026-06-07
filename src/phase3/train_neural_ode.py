from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Mapping, cast

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.phase3.neural_ode_model import GraphMode, NeuralODEConfig, NeuralODEStockPredictor
except ImportError:
    from neural_ode_model import GraphMode, NeuralODEConfig, NeuralODEStockPredictor


class Phase3TensorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, payload: Mapping[str, object]):
        self.text_embeddings = torch.as_tensor(payload["text_embeddings"]).float()
        self.event_types = torch.as_tensor(payload["event_types"]).long()
        self.timestamps = torch.as_tensor(payload["timestamps"]).float()
        self.magnitudes = torch.as_tensor(payload["magnitudes"]).float()
        profile_obj = payload.get("event_profiles")
        if profile_obj is None:
            self.event_profiles = torch.zeros((*self.magnitudes.shape, 4), dtype=torch.float32)
        else:
            self.event_profiles = torch.as_tensor(profile_obj).float()
        self.price_history = torch.as_tensor(payload["price_history"]).float()
        self.direction_labels = torch.as_tensor(payload["direction_labels"]).long()
        self.magnitude_labels = torch.as_tensor(payload["magnitude_labels"]).float()
        self.prediction_timestamps = torch.as_tensor(payload["prediction_timestamps"]).double()
        label_end_obj = payload.get("label_end_timestamps")
        self.label_end_timestamps = torch.as_tensor(label_end_obj).double() if label_end_obj is not None else None
        self.target_stock_ids = torch.as_tensor(payload["target_stock_ids"]).long()
        self.causal_matrix = torch.as_tensor(payload["causal_matrix"]).float()
        self.lag_matrix = torch.as_tensor(payload["lag_matrix"]).float()
        valid_mask = self._valid_example_mask()
        self.dropped_invalid_examples = int((~valid_mask).sum().item())
        if self.dropped_invalid_examples > 0:
            self.text_embeddings = self.text_embeddings[valid_mask]
            self.event_types = self.event_types[valid_mask]
            self.timestamps = self.timestamps[valid_mask]
            self.magnitudes = self.magnitudes[valid_mask]
            self.event_profiles = self.event_profiles[valid_mask]
            self.price_history = self.price_history[valid_mask]
            self.direction_labels = self.direction_labels[valid_mask]
            self.magnitude_labels = self.magnitude_labels[valid_mask]
            self.prediction_timestamps = self.prediction_timestamps[valid_mask]
            if self.label_end_timestamps is not None:
                self.label_end_timestamps = self.label_end_timestamps[valid_mask]
            self.target_stock_ids = self.target_stock_ids[valid_mask]
        self.stage2_graph_path = str(payload.get("stage2_graph_path", ""))
        stock_tickers_obj = payload.get("stock_tickers", [])
        if isinstance(stock_tickers_obj, list):
            self.stock_tickers = [str(item) for item in stock_tickers_obj]
        else:
            self.stock_tickers = []

    def _valid_example_mask(self) -> torch.Tensor:
        masks = [
            torch.isfinite(self.text_embeddings).all(dim=(1, 2)),
            torch.isfinite(self.timestamps).all(dim=1),
            torch.isfinite(self.magnitudes).all(dim=1),
            torch.isfinite(self.event_profiles).all(dim=(1, 2)),
            torch.isfinite(self.price_history).all(dim=(1, 2)),
            torch.isfinite(self.magnitude_labels),
        ]
        if self.label_end_timestamps is not None:
            masks.append(torch.isfinite(self.label_end_timestamps))
        valid_mask = masks[0]
        for mask in masks[1:]:
            valid_mask = valid_mask & mask
        return valid_mask

    def __len__(self) -> int:
        return int(self.direction_labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "text_embeddings": self.text_embeddings[idx],
            "event_types": self.event_types[idx],
            "timestamps": self.timestamps[idx],
            "magnitudes": self.magnitudes[idx],
            "event_profiles": self.event_profiles[idx],
            "price_history": self.price_history[idx],
            "direction_labels": self.direction_labels[idx],
            "magnitude_labels": self.magnitude_labels[idx],
            "prediction_timestamps": self.prediction_timestamps[idx],
            "target_stock_ids": self.target_stock_ids[idx],
        }


def load_payload(path: Path) -> Mapping[str, object]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError("Phase 3 dataset must be a dict payload")
    return data


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "pred_counts": np.bincount(y_pred, minlength=3).tolist(),
        "true_counts": np.bincount(y_true, minlength=3).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def stratified_random_split(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = torch.Generator().manual_seed(int(seed))
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    y = torch.tensor(labels, dtype=torch.long)
    for cls in range(int(y.max().item()) + 1):
        cls_idx = torch.where(y == cls)[0]
        cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=generator)].tolist()
        n_train = int(len(cls_idx) * float(train_ratio))
        n_val = int(len(cls_idx) * float(val_ratio))
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train : n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val :])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def chronological_split(
    timestamps: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    *,
    label_end_timestamps: np.ndarray | None = None,
    embargo_days: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_ratio <= 0.0 or val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")
    order = np.argsort(timestamps, kind="stable")
    ordered_ts = timestamps[order]
    _, first = np.unique(ordered_ts, return_index=True)
    boundaries = np.r_[first, len(order)]
    candidates = boundaries[1:-1]
    if not len(candidates):
        raise ValueError("Need at least three unique timestamps for train/val/test")
    train_target = int(len(order) * float(train_ratio))
    val_target = int(len(order) * float(train_ratio + val_ratio))
    train_pos = int(candidates[np.argmin(np.abs(candidates - train_target))])
    val_candidates = candidates[candidates > train_pos]
    if not len(val_candidates):
        raise ValueError("Could not place a validation boundary after the train boundary")
    val_pos = int(val_candidates[np.argmin(np.abs(val_candidates - val_target))])

    train_idx = order[:train_pos]
    val_idx = order[train_pos:val_pos]
    test_idx = order[val_pos:]
    val_start = float(timestamps[val_idx].min())
    test_start = float(timestamps[test_idx].min())

    if label_end_timestamps is not None:
        train_idx = train_idx[label_end_timestamps[train_idx] < val_start]
        val_idx = val_idx[label_end_timestamps[val_idx] < test_start]
    elif embargo_days > 0.0:
        embargo_seconds = float(embargo_days) * 86400.0
        train_idx = train_idx[timestamps[train_idx] < val_start - embargo_seconds]
        val_idx = val_idx[timestamps[val_idx] < test_start - embargo_seconds]

    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError("Purged chronological split produced an empty partition")
    return train_idx, val_idx, test_idx


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def balanced_class_weights(labels: np.ndarray, indices: np.ndarray, n_classes: int = 3, power: float = 1.0) -> torch.Tensor:
    y = labels[indices]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    if float(power) != 1.0:
        weights[present] = np.power(weights[present], float(power))
        weights[present] = weights[present] / weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32)


def permute_matrix_values(matrix: torch.Tensor, seed: int) -> torch.Tensor:
    flat = matrix.reshape(-1)
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(flat.numel(), generator=generator)
    return flat[order].reshape_as(matrix)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp().clamp(min=1e-8, max=1.0)
    return (((1.0 - pt) ** float(gamma)) * ce).mean()


def run_epoch(
    model: NeuralODEStockPredictor,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    beta_magnitude: float,
    class_weights: torch.Tensor | None,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
    desc: str,
) -> dict[str, object]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    cls_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(label_smoothing))
    reg_loss_fn = torch.nn.MSELoss()

    total_loss = 0.0
    total_cls = 0.0
    total_reg = 0.0
    count = 0
    direction_true: list[np.ndarray] = []
    direction_pred: list[np.ndarray] = []
    magnitude_true: list[np.ndarray] = []
    magnitude_pred: list[np.ndarray] = []

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=desc, leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            dir_logits, mag_pred = model(
                batch["text_embeddings"],
                batch["event_types"],
                batch["timestamps"],
                batch["magnitudes"],
                batch["event_profiles"],
                batch["price_history"],
                batch["target_stock_ids"],
            )
            if loss_type == "focal":
                cls_loss = focal_loss(
                    dir_logits,
                    batch["direction_labels"],
                    class_weights=class_weights,
                    gamma=focal_gamma,
                    label_smoothing=label_smoothing,
                )
            else:
                cls_loss = cls_loss_fn(dir_logits, batch["direction_labels"])
            reg_loss = reg_loss_fn(mag_pred, batch["magnitude_labels"])
            loss = cls_loss + float(beta_magnitude) * reg_loss

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = int(batch["direction_labels"].shape[0])
            total_loss += float(loss.detach().cpu()) * bs
            total_cls += float(cls_loss.detach().cpu()) * bs
            total_reg += float(reg_loss.detach().cpu()) * bs
            count += bs

            progress.set_postfix(
                loss=f"{(total_loss / count):.4f}",
                cls=f"{(total_cls / count):.4f}",
                reg=f"{(total_reg / count):.4f}",
            )

            direction_true.append(batch["direction_labels"].detach().cpu().numpy())
            direction_pred.append(dir_logits.argmax(dim=1).detach().cpu().numpy())
            magnitude_true.append(batch["magnitude_labels"].detach().cpu().numpy())
            magnitude_pred.append(mag_pred.detach().cpu().numpy())

    y_true = np.concatenate(direction_true)
    y_pred = np.concatenate(direction_pred)
    r_true = np.concatenate(magnitude_true)
    r_pred = np.concatenate(magnitude_pred)
    return {
        "loss": total_loss / count,
        "cls_loss": total_cls / count,
        "reg_loss": total_reg / count,
        "direction": macro_metrics(y_true, y_pred),
        "magnitude": regression_metrics(r_true, r_pred),
    }


def evaluate_predictions(
    model: NeuralODEStockPredictor,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[dict[str, object], list[dict[str, float | int]]]:
    model.eval()
    rows: list[dict[str, float | int]] = []
    direction_true: list[np.ndarray] = []
    direction_pred: list[np.ndarray] = []
    magnitude_true: list[np.ndarray] = []
    magnitude_pred: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            dir_logits, mag_pred = model(
                batch["text_embeddings"],
                batch["event_types"],
                batch["timestamps"],
                batch["magnitudes"],
                batch["event_profiles"],
                batch["price_history"],
                batch["target_stock_ids"],
            )
            dir_hat = dir_logits.argmax(dim=1)
            direction_true.append(batch["direction_labels"].detach().cpu().numpy())
            direction_pred.append(dir_hat.detach().cpu().numpy())
            magnitude_true.append(batch["magnitude_labels"].detach().cpu().numpy())
            magnitude_pred.append(mag_pred.detach().cpu().numpy())
            for true_dir, pred_dir, true_mag, pred_mag in zip(
                batch["direction_labels"].detach().cpu().tolist(),
                dir_hat.detach().cpu().tolist(),
                batch["magnitude_labels"].detach().cpu().tolist(),
                mag_pred.detach().cpu().tolist(),
            ):
                rows.append(
                    {
                        "direction_true": int(true_dir),
                        "direction_pred": int(pred_dir),
                        "magnitude_true": float(true_mag),
                        "magnitude_pred": float(pred_mag),
                    }
                )
    y_true = np.concatenate(direction_true)
    y_pred = np.concatenate(direction_pred)
    r_true = np.concatenate(magnitude_true)
    r_pred = np.concatenate(magnitude_pred)
    metrics: dict[str, object] = {
        "direction": macro_metrics(y_true, y_pred),
        "magnitude": regression_metrics(r_true, r_pred),
    }
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a standalone Phase 3 Neural ODE stock predictor")
    parser.add_argument("--data", default="data/stage3/phase3_dataset_ticker_precomputed.pt")
    parser.add_argument("--output-dir", default="outputs/phase3/neural_ode")
    parser.add_argument("--split", choices=["stratified_random", "chronological"], default="chronological")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--embargo-days", type=float, default=30.0, help="Fallback purge for datasets without label_end_timestamps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta-magnitude", type=float, default=0.1)
    parser.add_argument("--impact-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--price-hidden-dim", type=int, default=64)
    parser.add_argument("--stock-hidden-dim", type=int, default=64)
    parser.add_argument("--ode-steps", type=int, default=8)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--class-weight-power", type=float, default=0.6)
    parser.add_argument("--loss-type", choices=["ce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=["macro_f1", "mcc"], default="mcc")
    parser.add_argument("--graph-mode", choices=["full", "no_graph", "a_only", "t_only", "random"], default="full")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    payload = load_payload(Path(args.data))
    dataset = Phase3TensorDataset(payload)
    labels = dataset.direction_labels.numpy()
    if args.split == "stratified_random":
        train_idx, val_idx, test_idx = stratified_random_split(labels, args.train_ratio, args.val_ratio, args.seed)
    else:
        train_idx, val_idx, test_idx = chronological_split(
            dataset.prediction_timestamps.numpy(),
            args.train_ratio,
            args.val_ratio,
            label_end_timestamps=(
                dataset.label_end_timestamps.numpy() if dataset.label_end_timestamps is not None else None
            ),
            embargo_days=args.embargo_days,
        )

    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=args.batch_size, shuffle=False)
    class_weights = None
    if args.class_weight == "balanced":
        class_weights = balanced_class_weights(labels, train_idx, power=args.class_weight_power).to(device)

    config = NeuralODEConfig(
        text_emb_dim=int(dataset.text_embeddings.shape[-1]),
        impact_dim=int(args.impact_dim),
        hidden_dim=int(args.hidden_dim),
        price_hidden_dim=int(args.price_hidden_dim),
        stock_hidden_dim=int(args.stock_hidden_dim),
        ode_steps=int(args.ode_steps),
    )
    graph_mode = cast(GraphMode, args.graph_mode)
    causal_matrix: torch.Tensor | None = dataset.causal_matrix
    lag_matrix: torch.Tensor | None = dataset.lag_matrix
    if graph_mode == "random":
        causal_matrix = permute_matrix_values(dataset.causal_matrix, int(args.seed))
        lag_matrix = permute_matrix_values(dataset.lag_matrix, int(args.seed) + 1009)
    elif graph_mode == "no_graph":
        causal_matrix = None
        lag_matrix = None
    elif graph_mode == "a_only":
        lag_matrix = None
    elif graph_mode == "t_only":
        causal_matrix = None

    model = NeuralODEStockPredictor(
        n_event_types=int(dataset.event_types.max().item()) + 1,
        n_stocks=len(dataset.stock_tickers),
        causal_matrix=causal_matrix,
        lag_matrix=lag_matrix,
        graph_mode=graph_mode,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_score = float("-inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in tqdm(range(1, int(args.epochs) + 1), desc="phase3 neural ode"):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            beta_magnitude=args.beta_magnitude,
            class_weights=class_weights,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            desc=f"epoch {epoch:03d} train",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            beta_magnitude=args.beta_magnitude,
            class_weights=class_weights,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            desc=f"epoch {epoch:03d} val",
        )
        train_direction_metrics = train_metrics["direction"]
        direction_metrics = val_metrics["direction"]
        if not isinstance(train_direction_metrics, dict):
            raise TypeError("train direction metrics must be a dict")
        if not isinstance(direction_metrics, dict):
            raise TypeError("direction metrics must be a dict")
        val_score = float(direction_metrics[args.selection_metric])
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        tqdm.write(
            f"epoch={epoch:03d} train_loss={float(train_metrics['loss']):.4f} "
            f"train_f1={float(train_direction_metrics['macro_f1']):.4f} "
            f"train_acc={float(train_direction_metrics['accuracy']):.4f} "
            f"val_loss={float(val_metrics['loss']):.4f} "
            f"val_f1={float(direction_metrics['macro_f1']):.4f} "
            f"val_acc={float(direction_metrics['accuracy']):.4f} "
            f"val_mcc={float(direction_metrics['mcc']):.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training produced no model state")
    model.load_state_dict(best_state)

    val_eval, _ = evaluate_predictions(model, val_loader, device=device)
    test_eval, rows = evaluate_predictions(model, test_loader, device=device)

    torch.save(best_state, out_dir / "best_model.pt")
    torch.save(model.state_dict(), out_dir / "last_model.pt")

    with (out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["direction_true", "direction_pred", "magnitude_true", "magnitude_pred"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "data": args.data,
        "split": args.split,
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "device": str(device),
        "dropped_invalid_examples": int(dataset.dropped_invalid_examples),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "embargo_days": float(args.embargo_days),
        "used_exact_label_end_purge": dataset.label_end_timestamps is not None and args.split == "chronological",
        "n_stocks": int(len(dataset.stock_tickers)),
        "n_event_types": int(dataset.causal_matrix.shape[0]),
        "stage2_graph_path": dataset.stage2_graph_path,
        "graph_mode": graph_mode,
        "registered_graph_buffers": {
            "causal_matrix": "propagator.causal_matrix" in dict(model.named_buffers()),
            "lag_matrix": "propagator.lag_matrix" in dict(model.named_buffers()),
        },
        "config": asdict(config),
        "optimizer": {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "beta_magnitude": float(args.beta_magnitude),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "class_weight": args.class_weight,
            "class_weight_power": float(args.class_weight_power),
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
            "loss_type": args.loss_type,
            "focal_gamma": float(args.focal_gamma),
            "label_smoothing": float(args.label_smoothing),
            "selection_metric": args.selection_metric,
        },
        "best_epoch": best_epoch,
        "best_val_selection_metric": args.selection_metric,
        "best_val_selection_score": float(best_val_score),
        "val": val_eval,
        "test": test_eval,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    concise = {
        "model": "neural_ode",
        "graph_mode": graph_mode,
        "best_val_selection_metric": args.selection_metric,
        "best_val_selection_score": float(best_val_score),
        "optimizer": summary["optimizer"],
        "test": test_eval["direction"],
    }
    (out_dir / "concise_metrics.json").write_text(json.dumps(concise, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "confusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", "up_0", "down_1", "flat_2"])
        direction_eval = test_eval["direction"]
        if not isinstance(direction_eval, dict):
            raise TypeError("direction evaluation must be a dict")
        for label, row in zip(["up_0", "down_1", "flat_2"], direction_eval["confusion"]):
            writer.writerow([label, *row])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
