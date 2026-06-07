from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from causal_discovery import CausalRegularizationLoss
from dataset import FastEventDataset
from model_wrapper import FastModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def chronological_split(
    timestamps: list[float],
    label_end_timestamps: list[float],
    train_ratio: float,
) -> tuple[list[int], list[int]]:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    order = np.argsort(np.asarray(timestamps), kind="stable")
    ordered_ts = np.asarray(timestamps)[order]
    unique_ts, first = np.unique(ordered_ts, return_index=True)
    if len(unique_ts) < 2:
        raise ValueError("Need at least two unique timestamps for train/validation")

    train_target = int(len(order) * train_ratio)
    boundaries = np.r_[first, len(order)]
    candidates = boundaries[1:-1]
    train_pos = int(candidates[np.argmin(np.abs(candidates - train_target))])
    train_idx = order[:train_pos]
    val_idx = order[train_pos:]
    val_start = float(ordered_ts[train_pos])
    label_ends = np.asarray(label_end_timestamps)
    train_idx = train_idx[label_ends[train_idx] < val_start]
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Purged chronological split produced an empty partition")
    return train_idx.tolist(), val_idx.tolist()


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("--horizons must contain positive integers, e.g. 1,3,5,7,10")
    return horizons


def class_weights(dataset: FastEventDataset, indices: list[int], device: torch.device) -> torch.Tensor:
    labels = torch.tensor([dataset.examples[i][1] for i in indices], dtype=torch.long)
    weights = []
    for horizon_idx in range(labels.shape[1]):
        valid = labels[:, horizon_idx] != -100
        if not bool(valid.any()):
            weights.append(torch.ones(3))
            continue
        counts = torch.bincount(labels[valid, horizon_idx], minlength=3).float()
        horizon_weights = counts.sum() / counts.clamp_min(1.0)
        weights.append(horizon_weights / horizon_weights.mean())
    return torch.stack(weights).to(device)


def lag_horizon_weights(
    event_types: torch.Tensor,
    causal_info: dict,
    horizons: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    causal_matrix = causal_info["causal_matrix"]
    lag_matrix = causal_info["lag_matrix"]
    target_types = event_types[:, -1]
    target_lookup = target_types.unsqueeze(1).expand_as(event_types)

    pair_lags = lag_matrix[target_lookup, event_types]
    pair_strength = causal_matrix[event_types, target_lookup]
    lag_distance = horizons.view(1, 1, -1) - pair_lags.unsqueeze(-1)
    gates = torch.exp(-0.5 * (lag_distance ** 2) / (float(sigma) ** 2))
    scores = (pair_strength.unsqueeze(-1) * gates).sum(dim=1)
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)


def multi_horizon_cls_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    event_types: torch.Tensor,
    causal_info: dict,
    *,
    horizons: torch.Tensor,
    horizon_class_weights: torch.Tensor,
    lag_loss_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = labels != -100
    if not bool(valid.any()):
        zero = logits.sum() * 0.0
        return zero, valid, torch.zeros_like(labels, dtype=torch.float32)

    horizon_weights = lag_horizon_weights(event_types, causal_info, horizons, lag_loss_sigma)
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clamp_min(0)
    per_loss = -log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    per_class_weight = horizon_class_weights.unsqueeze(0).expand(labels.shape[0], -1, -1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    weighted = per_loss * horizon_weights * per_class_weight * valid.float()
    denom = (horizon_weights * per_class_weight * valid.float()).sum().clamp_min(1e-6)
    return weighted.sum() / denom, valid, horizon_weights


def best_horizon_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    valid = labels != -100
    has_valid = valid.any(dim=1)
    if not bool(has_valid.any()):
        return 0, 0

    probs = F.softmax(logits, dim=-1)
    confidence, pred = probs.max(dim=-1)
    confidence = confidence.masked_fill(~valid, -1.0)
    best_horizon_idx = confidence.argmax(dim=1)

    batch_idx = torch.arange(labels.shape[0], device=labels.device)
    selected = has_valid
    selected_pred = pred[batch_idx[selected], best_horizon_idx[selected]]
    selected_label = labels[batch_idx[selected], best_horizon_idx[selected]]
    correct = int((selected_pred == selected_label).sum().item())
    return correct, int(selected.sum().item())


def graph_payload(model: FastModel, epoch: int, metrics: dict[str, float]) -> dict[str, object]:
    layer = model.stacd.layers[0]
    causal_matrix = layer.causal_matrix.detach().cpu()
    lag_matrix = layer.lag_matrix.detach().cpu()
    return {
        "epoch": int(epoch),
        "metrics": metrics,
        "causal_matrix": causal_matrix,
        "lag_matrix": lag_matrix,
        "adjacency_raw": causal_matrix,
    }


def summarize_graph(model: FastModel) -> dict[str, float]:
    A = model.stacd.layers[0].causal_matrix.detach().cpu()
    off = A[~torch.eye(A.shape[0], dtype=torch.bool)]
    return {
        "causal_mean": float(A.mean()),
        "causal_std": float(A.std()),
        "causal_offdiag_mean": float(off.mean()),
        "causal_offdiag_std": float(off.std()),
        "causal_min": float(A.min()),
        "causal_max": float(A.max()),
    }


def run_epoch(
    model: FastModel,
    loader: DataLoader,
    *,
    device: torch.device,
    reg_fn: CausalRegularizationLoss,
    optimizer: torch.optim.Optimizer | None,
    reg_weight: float,
    horizons: torch.Tensor,
    horizon_class_weights: torch.Tensor,
    lag_loss_sigma: float,
    desc: str,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(mode=train)
    total = 0.0
    cls_total = 0.0
    reg_total = 0.0
    correct = 0
    n = 0
    n_valid = 0
    best_correct = 0
    best_count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            emb = batch["text_emb"].to(device)
            types = batch["event_types"].to(device)
            ts = batch["timestamps"].to(device)
            y = batch["labels"].to(device)

            logits, causal_info = model(emb, types, ts)
            cls_loss, valid, _ = multi_horizon_cls_loss(
                logits,
                y,
                types,
                causal_info,
                horizons=horizons,
                horizon_class_weights=horizon_class_weights,
                lag_loss_sigma=lag_loss_sigma,
            )
            reg_loss = reg_fn(causal_info)["total_reg"]
            loss = cls_loss + float(reg_weight) * reg_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = int(y.shape[0])
            total += float(loss.detach().cpu()) * bs
            cls_total += float(cls_loss.detach().cpu()) * bs
            reg_total += float(reg_loss.detach().cpu()) * bs
            pred = logits.argmax(dim=-1)
            correct += int(((pred == y) & valid).sum().item())
            batch_best_correct, batch_best_count = best_horizon_accuracy(logits, y)
            best_correct += batch_best_correct
            best_count += batch_best_count
            n += bs
            n_valid += int(valid.sum().item())

    return {
        "loss": total / n,
        "cls_loss": cls_total / n,
        "reg_loss": reg_total / n,
        "accuracy": correct / max(n_valid, 1),
        "best_horizon_accuracy": best_correct / max(best_count, 1),
    }


def write_row(path: Path, row: dict[str, object]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_cls_loss",
        "val_cls_loss",
        "train_reg_loss",
        "val_reg_loss",
        "train_accuracy",
        "val_accuracy",
        "train_best_horizon_accuracy",
        "val_best_horizon_accuracy",
        "causal_mean",
        "causal_std",
        "causal_offdiag_mean",
        "causal_offdiag_std",
        "causal_min",
        "causal_max",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain repaired Direction A Phase 2 STACD")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--embeddings", default=str(project_root / "data/embeddings/emb.pt"))
    parser.add_argument("--meta", default=str(project_root / "data/embeddings/meta.json"))
    parser.add_argument("--price-dir", default=str(project_root / "data/sp100_prices"))
    parser.add_argument("--output-dir", default=str(project_root / "data/stage2_repaired"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--label-threshold", type=float, default=0.005)
    parser.add_argument("--horizons", default="1,3,5,7,10")
    parser.add_argument("--lag-loss-sigma", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--reg-weight", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    horizons = parse_horizons(args.horizons)

    set_seed(args.seed)
    device = safe_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"
    metrics_csv.unlink(missing_ok=True)

    dataset = FastEventDataset(
        args.embeddings,
        args.meta,
        args.price_dir,
        seq_len=args.seq_len,
        label_threshold=args.label_threshold,
        horizons=horizons,
    )
    train_idx, val_idx = chronological_split(
        dataset.example_timestamps,
        dataset.example_label_end_timestamps,
        args.train_ratio,
    )
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = FastModel(num_horizons=len(horizons)).to(device)
    horizon_tensor = torch.tensor(horizons, dtype=torch.float32, device=device)
    horizon_class_weights = class_weights(dataset, train_idx, device)
    reg_fn = CausalRegularizationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = None
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc="stage2 training"):
        train = run_epoch(
            model,
            train_loader,
            device=device,
            reg_fn=reg_fn,
            optimizer=optimizer,
            reg_weight=args.reg_weight,
            horizons=horizon_tensor,
            horizon_class_weights=horizon_class_weights,
            lag_loss_sigma=args.lag_loss_sigma,
            desc=f"epoch {epoch:03d} train",
        )
        val = run_epoch(
            model,
            val_loader,
            device=device,
            reg_fn=reg_fn,
            optimizer=None,
            reg_weight=args.reg_weight,
            horizons=horizon_tensor,
            horizon_class_weights=horizon_class_weights,
            lag_loss_sigma=args.lag_loss_sigma,
            desc=f"epoch {epoch:03d} val",
        )
        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "val_loss": val["loss"],
            "train_cls_loss": train["cls_loss"],
            "val_cls_loss": val["cls_loss"],
            "train_reg_loss": train["reg_loss"],
            "val_reg_loss": val["reg_loss"],
            "train_accuracy": train["accuracy"],
            "val_accuracy": val["accuracy"],
            "train_best_horizon_accuracy": train["best_horizon_accuracy"],
            "val_best_horizon_accuracy": val["best_horizon_accuracy"],
            **summarize_graph(model),
        }
        write_row(metrics_csv, row)
        torch.save(graph_payload(model, epoch, val), out_dir / f"causal_epoch_{epoch}.pt")
        torch.save(model.state_dict(), out_dir / "last_model.pt")
        if best_acc is None or val["best_horizon_accuracy"] > best_acc:
            best_acc = val["best_horizon_accuracy"]
            best_epoch = epoch
            best_metrics = val
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            torch.save(graph_payload(model, epoch, val), out_dir / "best_graph.pt")
        else:
            epochs_without_improvement += 1
        tqdm.write(
            f"epoch={epoch:03d} train_acc={train['accuracy']:.4f} "
            f"val_acc={val['accuracy']:.4f} "
            f"val_best={val['best_horizon_accuracy']:.4f} A_std={row['causal_std']:.4f}"
        )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            tqdm.write(
                f"early stopping at epoch={epoch:03d}; "
                f"best_epoch={best_epoch:03d} best_val_best={best_acc:.4f}"
            )
            break

    meta = {
        "args": vars(args),
        "n_examples": len(dataset),
        "n_train": len(train_set),
        "n_val": len(val_set),
        "best_epoch": best_epoch,
        "best_val_best_horizon_accuracy": best_acc,
        "best_metrics": best_metrics,
        "graph": summarize_graph(model),
        "horizons": horizons,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
