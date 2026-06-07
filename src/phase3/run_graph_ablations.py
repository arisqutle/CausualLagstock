from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from tabular_baseline_with_graph import balanced_sample_weights, make_model


LABELS = [0, 1, 2]
PROFILE_DIM = 4


def load_data(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "true_counts": np.bincount(y_true, minlength=len(LABELS)).tolist(),
        "pred_counts": np.bincount(y_pred, minlength=len(LABELS)).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def base_components(data: dict, *, reverse_events: bool = False) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    price = data["price_history"].float().numpy()
    event_types = data["event_types"].long().numpy()
    magnitudes = data["magnitudes"].float().numpy()
    profile_obj = data.get("event_profiles")
    if profile_obj is None:
        profiles = np.zeros((*magnitudes.shape, PROFILE_DIM), dtype=np.float32)
    else:
        profiles = torch.as_tensor(profile_obj).float().numpy()
    stock_ids = data["target_stock_ids"].long().numpy()

    if reverse_events:
        event_types = event_types[:, ::-1].copy()
        magnitudes = magnitudes[:, ::-1].copy()
        profiles = profiles[:, ::-1, :].copy()

    n = price.shape[0]
    n_event_types = int(event_types.max()) + 1
    n_stocks = int(stock_ids.max()) + 1

    close = price[:, :, 3]
    returns = np.diff(close, axis=1) / np.maximum(np.abs(close[:, :-1]), 1e-8)
    volume = np.log1p(np.maximum(price[:, :, 4], 0.0))
    base_close = np.maximum(np.abs(price[:, 0:1, 3:4]), 1e-8)
    relative_ohlc = price[:, :, :4] / base_close - 1.0

    price_features = np.column_stack(
        [
            returns[:, -10:],
            returns.mean(axis=1),
            returns.std(axis=1),
            returns[:, -5:].mean(axis=1),
            returns[:, -5:].std(axis=1),
            volume[:, -10:].mean(axis=1),
            volume[:, -10:].std(axis=1),
            relative_ohlc[:, -1, :],
            relative_ohlc.mean(axis=1).reshape(n, -1),
            relative_ohlc.std(axis=1).reshape(n, -1),
        ]
    )

    counts = np.zeros((n, n_event_types), dtype=np.float32)
    magnitude_sum = np.zeros((n, n_event_types), dtype=np.float32)
    magnitude_abs = np.zeros((n, n_event_types), dtype=np.float32)
    last_type = np.zeros((n, n_event_types), dtype=np.float32)
    for i in range(n):
        for event_type, magnitude in zip(event_types[i], magnitudes[i]):
            counts[i, int(event_type)] += 1.0
            magnitude_sum[i, int(event_type)] += float(magnitude)
            magnitude_abs[i, int(event_type)] += abs(float(magnitude))
        last_type[i, int(event_types[i, -1])] = 1.0

    event_features = np.concatenate(
        [
            counts,
            magnitude_sum / event_types.shape[1],
            magnitude_abs / event_types.shape[1],
            last_type,
            magnitudes.mean(axis=1, keepdims=True),
            np.abs(magnitudes).mean(axis=1, keepdims=True),
            profiles.mean(axis=1),
            profiles[:, -1, :],
        ],
        axis=1,
    )
    stock_features = np.eye(n_stocks, dtype=np.float32)[stock_ids]
    meta = {
        "price_feature_dim": int(price_features.shape[1]),
        "event_feature_dim": int(event_features.shape[1]),
        "event_profile_dim": int(profiles.shape[2]),
        "event_profile_fields": data.get("event_profile_fields", ["surprise", "scope", "novelty", "credibility"]),
        "stock_feature_dim": int(stock_features.shape[1]),
        "n_event_types": int(n_event_types),
    }
    return {
        "price": price_features.astype(np.float32),
        "event": event_features.astype(np.float32),
        "stock": stock_features.astype(np.float32),
        "counts": counts,
        "magnitude_sum": magnitude_sum,
    }, meta


def graph_features(
    counts: np.ndarray,
    weighted_counts: np.ndarray,
    causal_matrix: np.ndarray,
    lag_matrix: np.ndarray,
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    causal = causal_matrix.astype(np.float32)
    lag = lag_matrix.astype(np.float32)

    if mode == "no_graph":
        return np.zeros((counts.shape[0], 0), dtype=np.float32)
    if mode == "random_graph":
        causal = rng.permutation(causal.reshape(-1)).reshape(causal.shape).astype(np.float32)
        lag = rng.permutation(lag.reshape(-1)).reshape(lag.shape).astype(np.float32)
    elif mode == "reverse_time":
        causal = causal.T.copy()
        lag = lag.T.copy()

    features: list[np.ndarray] = []
    if mode in {"full_graph", "random_graph", "reverse_time", "causal_only"}:
        features.extend([counts @ causal, weighted_counts @ causal])
    if mode in {"full_graph", "random_graph", "reverse_time", "lag_only"}:
        features.extend([counts @ lag, weighted_counts @ lag])
    if not features:
        return np.zeros((counts.shape[0], 0), dtype=np.float32)
    return np.concatenate(features, axis=1).astype(np.float32)


def build_features(data: dict, mode: str, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    reverse_events = mode == "reverse_event_order"
    parts, meta = base_components(data, reverse_events=reverse_events)
    graph = graph_features(
        parts["counts"],
        parts["magnitude_sum"],
        data["causal_matrix"].float().numpy(),
        data["lag_matrix"].float().numpy(),
        mode,
        seed,
    )
    features = np.concatenate([parts["price"], parts["event"], parts["stock"], graph], axis=1).astype(np.float32)
    meta["graph_feature_dim"] = int(graph.shape[1])
    meta["total_feature_dim"] = int(features.shape[1])
    return features, meta


def run_one(data: dict, mode: str, args: argparse.Namespace) -> dict[str, object]:
    labels = data["direction_labels"].long().numpy()
    features, feature_meta = build_features(data, mode, args.seed)
    train_idx, val_idx, test_idx = purged_chronological_split(
        data["prediction_timestamps"].numpy(),
        args.train_ratio,
        args.val_ratio,
        label_end_timestamps=(
            data["label_end_timestamps"].numpy()
            if "label_end_timestamps" in data
            else None
        ),
    )
    model = make_model(
        "hgb",
        args.seed,
        hgb_max_iter=args.hgb_max_iter,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_l2_regularization=args.hgb_l2_regularization,
        hgb_min_samples_leaf=args.hgb_min_samples_leaf,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
    )
    sample_weight = None
    if args.sample_weight == "balanced":
        sample_weight = balanced_sample_weights(labels, train_idx)
    model.fit(features[train_idx], labels[train_idx], sample_weight=sample_weight)
    pred = model.predict(features[test_idx])
    return {
        "mode": mode,
        "feature_meta": feature_meta,
        "test": metrics(labels[test_idx], pred),
    }


def purged_chronological_split(
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
    unique_ts, first = np.unique(ordered_ts, return_index=True)
    if len(unique_ts) < 3:
        raise ValueError("Need at least three unique timestamps for train/val/test")

    boundaries = np.r_[first, len(order)]
    candidates = boundaries[1:-1]
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


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def mean_std(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "formatted": f"{mean:.4f} +/- {std:.4f}",
    }


def aggregate_results(mode: str, runs: list[dict[str, object]]) -> dict[str, object]:
    metric_names = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    test_runs = [run["test"] for run in runs]
    confusion = np.asarray([test["confusion"] for test in test_runs], dtype=np.float64)
    confusion_mean = confusion.mean(axis=0)
    confusion_std = confusion.std(axis=0, ddof=1) if confusion.shape[0] > 1 else np.zeros_like(confusion_mean)
    return {
        "mode": mode,
        "n_seeds": len(runs),
        "feature_meta": runs[0]["feature_meta"],
        "metrics": {
            name: mean_std([float(test[name]) for test in test_runs])
            for name in metric_names
        },
        "true_counts": test_runs[0]["true_counts"],
        "pred_counts_mean": confusion.sum(axis=1).mean(axis=0).tolist(),
        "confusion_mean": confusion_mean.tolist(),
        "confusion_std": confusion_std.tolist(),
        "confusion_formatted": [
            [
                f"{confusion_mean[i, j]:.1f} +/- {confusion_std[i, j]:.1f}"
                for j in range(confusion_mean.shape[1])
            ]
            for i in range(confusion_mean.shape[0])
        ],
    }


def write_confusion_csv(path: Path, confusion: list[list[int]]) -> None:
    lines = ["true_class,pred_0,pred_1,pred_2"]
    for idx, row in enumerate(confusion):
        lines.append(",".join([str(idx), *[str(item) for item in row]]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_confusion_csv(path: Path, result: dict[str, object]) -> None:
    confusion = result["confusion_formatted"]
    lines = ["true_class,pred_0,pred_1,pred_2"]
    for idx, row in enumerate(confusion):
        lines.append(",".join([str(idx), *row]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_csv(path: Path, results: list[dict[str, object]]) -> None:
    lines = ["mode,accuracy,macro_f1,weighted_f1,macro_precision,macro_recall"]
    for result in results:
        metrics_blob = result["metrics"]
        lines.append(
            ",".join(
                [
                    str(result["mode"]),
                    metrics_blob["accuracy"]["formatted"],
                    metrics_blob["macro_f1"]["formatted"],
                    metrics_blob["weighted_f1"]["formatted"],
                    metrics_blob["macro_precision"]["formatted"],
                    metrics_blob["macro_recall"]["formatted"],
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 graph ablations and save concise final metrics")
    parser.add_argument("--data", default="data/stage3/phase3_dataset_ticker_3class_dayscale_horizon_selected.pt")
    parser.add_argument("--output-dir", default="outputs/newest")
    parser.add_argument("--modes", default="full_graph,no_graph,causal_only,lag_only,random_graph,reverse_time")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--hgb-max-iter", type=int, default=150)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.04)
    parser.add_argument("--hgb-l2-regularization", type=float, default=0.05)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=80)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--sample-weight", choices=["none", "balanced"], default="none")
    args = parser.parse_args()

    data = load_data(Path(args.data))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    results = []
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        runs = []
        for seed in seeds:
            args.seed = seed
            runs.append(run_one(data, mode, args))
        result = aggregate_results(mode, runs)
        results.append(result)
        write_aggregate_confusion_csv(out_dir / f"ablation_{mode}_confusion.csv", result)
        print(
            f"{mode}: accuracy={result['metrics']['accuracy']['formatted']} "
            f"macro_f1={result['metrics']['macro_f1']['formatted']}"
        )

    summary = {
        "data": args.data,
        "model": "hgb",
        "split": "chronological",
        "seeds": seeds,
        "model_params": {
            "hgb_max_iter": args.hgb_max_iter,
            "hgb_learning_rate": args.hgb_learning_rate,
            "hgb_l2_regularization": args.hgb_l2_regularization,
            "hgb_min_samples_leaf": args.hgb_min_samples_leaf,
            "hgb_max_leaf_nodes": args.hgb_max_leaf_nodes,
            "sample_weight": args.sample_weight,
        },
        "results": results,
    }
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_metrics_csv(out_dir / "ablation_metrics_mean_std.csv", results)


if __name__ == "__main__":
    main()
