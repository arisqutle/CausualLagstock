from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROFILE_DIM = 4


def load_data(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pred_counts": np.bincount(y_pred, minlength=3).tolist(),
        "true_counts": np.bincount(y_true, minlength=3).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def stratified_random_split(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = torch.Generator().manual_seed(int(seed))
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    y = torch.tensor(labels, dtype=torch.long)
    for cls in range(int(y.max().item()) + 1):
        cls_idx = torch.where(y == cls)[0]
        cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=g)].tolist()
        n_train = int(len(cls_idx) * float(train_ratio))
        n_val = int(len(cls_idx) * float(val_ratio))
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train : n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val :])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def chronological_split(timestamps: np.ndarray, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(timestamps)
    n_train = int(len(order) * float(train_ratio))
    n_val = int(len(order) * float(val_ratio))
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def build_graph_features(
    counts: np.ndarray,
    weighted_counts: np.ndarray,
    causal_matrix: np.ndarray,
    lag_matrix: np.ndarray,
) -> np.ndarray:
    if causal_matrix.shape != lag_matrix.shape:
        raise ValueError("causal_matrix and lag_matrix must have the same shape")
    if counts.shape[1] != causal_matrix.shape[0]:
        raise ValueError(
            f"event-type dimension mismatch: counts has {counts.shape[1]} types but graph has {causal_matrix.shape[0]}"
        )

    count_causal = counts @ causal_matrix
    count_lag = counts @ lag_matrix
    weighted_causal = weighted_counts @ causal_matrix
    weighted_lag = weighted_counts @ lag_matrix
    return np.concatenate([count_causal, count_lag, weighted_causal, weighted_lag], axis=1).astype(np.float32)


def build_features(data: dict, *, use_graph_features: bool = True) -> tuple[np.ndarray, dict[str, object]]:
    price = data["price_history"].float().numpy()
    event_types = data["event_types"].long().numpy()
    magnitudes = data["magnitudes"].float().numpy()
    profile_obj = data.get("event_profiles")
    if profile_obj is None:
        profiles = np.zeros((*magnitudes.shape, PROFILE_DIM), dtype=np.float32)
    else:
        profiles = torch.as_tensor(profile_obj).float().numpy()
    stock_ids = data["target_stock_ids"].long().numpy()

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
    if use_graph_features:
        causal_matrix = data["causal_matrix"].float().numpy()
        lag_matrix = data["lag_matrix"].float().numpy()
        graph_features = build_graph_features(counts, magnitude_sum, causal_matrix, lag_matrix)
    else:
        graph_features = np.zeros((n, 0), dtype=np.float32)

    features = np.concatenate([price_features, event_features, stock_features, graph_features], axis=1).astype(np.float32)
    feature_meta = {
        "price_feature_dim": int(price_features.shape[1]),
        "event_feature_dim": int(event_features.shape[1]),
        "event_profile_dim": int(profiles.shape[2]),
        "event_profile_fields": data.get("event_profile_fields", ["surprise", "scope", "novelty", "credibility"]),
        "stock_feature_dim": int(stock_features.shape[1]),
        "graph_feature_dim": int(graph_features.shape[1]),
        "use_graph_features": bool(use_graph_features),
        "n_event_types": int(n_event_types),
    }
    return features, feature_meta


def make_model(
    name: str,
    seed: int,
    *,
    hgb_max_iter: int = 250,
    hgb_learning_rate: float = 0.08,
    hgb_l2_regularization: float = 0.1,
    hgb_min_samples_leaf: int = 80,
    hgb_max_leaf_nodes: int | None = 15,
):
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=hgb_max_iter,
            learning_rate=hgb_learning_rate,
            l2_regularization=hgb_l2_regularization,
            min_samples_leaf=hgb_min_samples_leaf,
            max_leaf_nodes=hgb_max_leaf_nodes,
            random_state=seed,
        )
    if name == "hgb_regularized":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.025,
            l2_regularization=0.2,
            min_samples_leaf=30,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError("model must be hgb, hgb_regularized, extra_trees, or random_forest")


def balanced_sample_weights(labels: np.ndarray, indices: np.ndarray, n_classes: int = 3) -> np.ndarray:
    y = labels[indices]
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    return weights[y].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 tabular baseline with Phase 2 graph-derived features")
    parser.add_argument("--data", default="data/stage3/phase3_dataset_ticker_precomputed.pt")
    parser.add_argument("--output-dir", default="outputs/phase3/tabular_baseline_with_graph")
    parser.add_argument("--model", choices=["hgb", "hgb_regularized", "extra_trees", "random_forest"], default="hgb")
    parser.add_argument("--split", choices=["stratified_random", "chronological"], default="chronological")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hgb-max-iter", type=int, default=250)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.08)
    parser.add_argument("--hgb-l2-regularization", type=float, default=0.1)
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=80)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=15)
    parser.add_argument("--no-graph-features", action="store_true", help="Disable causal/lag matrix graph-derived features")
    parser.add_argument("--sample-weight", choices=["none", "balanced"], default="none", help="Apply per-sample class-balancing weights during fitting")
    parser.add_argument("--save-model", action="store_true", help="Save trained model with joblib")
    args = parser.parse_args()

    data = load_data(Path(args.data))
    labels = data["direction_labels"].long().numpy()
    use_graph_features = not args.no_graph_features
    features, feature_meta = build_features(data, use_graph_features=use_graph_features)

    if args.split == "stratified_random":
        train_idx, val_idx, test_idx = stratified_random_split(labels, args.train_ratio, args.val_ratio, args.seed)
    else:
        train_idx, val_idx, test_idx = chronological_split(data["prediction_timestamps"].numpy(), args.train_ratio, args.val_ratio)

    model = make_model(
        args.model,
        args.seed,
        hgb_max_iter=args.hgb_max_iter,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_l2_regularization=args.hgb_l2_regularization,
        hgb_min_samples_leaf=args.hgb_min_samples_leaf,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
    )
    train_sample_weight = None
    if args.sample_weight == "balanced":
        train_sample_weight = balanced_sample_weights(labels, train_idx)
    model.fit(features[train_idx], labels[train_idx], sample_weight=train_sample_weight)

    val_pred = model.predict(features[val_idx])
    test_pred = model.predict(features[test_idx])

    full_train_idx = np.concatenate([train_idx, val_idx])
    submission_model = make_model(
        args.model,
        args.seed,
        hgb_max_iter=args.hgb_max_iter,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_l2_regularization=args.hgb_l2_regularization,
        hgb_min_samples_leaf=args.hgb_min_samples_leaf,
        hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
    )
    full_sample_weight = None
    if args.sample_weight == "balanced":
        full_sample_weight = balanced_sample_weights(labels, full_train_idx)
    submission_model.fit(features[full_train_idx], labels[full_train_idx], sample_weight=full_sample_weight)

    summary = {
        "data": args.data,
        "model": args.model,
        "model_params": {
            "hgb_max_iter": args.hgb_max_iter,
            "hgb_learning_rate": args.hgb_learning_rate,
            "hgb_l2_regularization": args.hgb_l2_regularization,
            "hgb_min_samples_leaf": args.hgb_min_samples_leaf,
            "hgb_max_leaf_nodes": args.hgb_max_leaf_nodes,
            "sample_weight": args.sample_weight,
        },
        "split": args.split,
        "total_feature_dim": int(features.shape[1]),
        "feature_breakdown": feature_meta,
        "use_graph_features": bool(use_graph_features),
        "graph_source": data.get("stage2_graph_path") if use_graph_features else None,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "val": macro_metrics(labels[val_idx], val_pred),
        "test": macro_metrics(labels[test_idx], test_pred),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_meta.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    concise = {
        "model": args.model,
        "model_params": summary["model_params"],
        "use_graph_features": bool(use_graph_features),
        "val": summary["val"],
        "test": summary["test"],
    }
    (out_dir / "concise_metrics.json").write_text(json.dumps(concise, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "confusion.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", "up_0", "down_1", "flat_2"])
        for label, row in zip(["up_0", "down_1", "flat_2"], summary["test"]["confusion"]):
            writer.writerow([label, *row])
    with (out_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["direction_true", "direction_pred"])
        writer.writeheader()
        for y, p in zip(labels[test_idx], test_pred):
            writer.writerow({"direction_true": int(y), "direction_pred": int(p)})
    if args.save_model:
        joblib.dump(submission_model, out_dir / "model.joblib")
        print(f"Model saved to {out_dir / 'model.joblib'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
