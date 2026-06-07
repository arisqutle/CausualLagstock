from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.neural_ode_model import GraphMode, NeuralODEConfig, NeuralODEStockPredictor
from src.phase3.train_neural_ode import (
    Phase3TensorDataset,
    chronological_split,
    evaluate_predictions,
    load_payload,
    permute_matrix_values,
    stratified_random_split,
)

CLASS_NAMES = ["up", "down", "flat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Phase 3 Neural ODE checkpoint."
    )
    parser.add_argument(
        "--run-dir",
        help="Training output directory containing best_model.pt and run_meta.json",
    )
    parser.add_argument("--checkpoint", help="Checkpoint path; overrides --run-dir")
    parser.add_argument("--run-meta", help="run_meta.json path; overrides --run-dir")
    parser.add_argument("--data", help="Phase 3 dataset path; defaults to run metadata")
    parser.add_argument("--partition", choices=["test", "val", "train", "all"], default="test")
    parser.add_argument("--split", choices=["chronological", "stratified_random"])
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--embargo-days", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--graph-mode",
        choices=["full", "no_graph", "a_only", "t_only", "random"],
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", help="Optional metrics JSON output path")
    parser.add_argument("--predictions-csv", help="Optional prediction CSV output path")
    return parser.parse_args()


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Run metadata must be a JSON object: {path}")
    return data


def resolve_path(value: str | None, *, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        base / path,
        PROJECT_ROOT.parent / path,
        PROJECT_ROOT / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[1].resolve()


def select_indices(
    dataset: Phase3TensorDataset,
    *,
    split: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    embargo_days: float,
    partition: str,
) -> np.ndarray:
    if partition == "all":
        return np.arange(len(dataset))
    if split == "stratified_random":
        indices = stratified_random_split(
            dataset.direction_labels.numpy(), train_ratio, val_ratio, seed
        )
    else:
        indices = chronological_split(
            dataset.prediction_timestamps.numpy(),
            train_ratio,
            val_ratio,
            label_end_timestamps=(
                dataset.label_end_timestamps.numpy()
                if dataset.label_end_timestamps is not None
                else None
            ),
            embargo_days=embargo_days,
        )
    return indices[{"train": 0, "val": 1, "test": 2}[partition]]


def expanded_metrics(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    y_true = np.asarray([int(row["direction_true"]) for row in rows])
    y_pred = np.asarray([int(row["direction_pred"]) for row in rows])
    magnitude_true = np.asarray([float(row["magnitude_true"]) for row in rows])
    magnitude_pred = np.asarray([float(row["magnitude_pred"]) for row in rows])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2]
        ).tolist(),
        "class_metrics": {
            name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, name in enumerate(CLASS_NAMES)
        },
        "magnitude_mse": float(mean_squared_error(magnitude_true, magnitude_pred)),
        "magnitude_mae": float(mean_absolute_error(magnitude_true, magnitude_pred)),
    }


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a state dict: {path}")
    return state


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (run_dir / "best_model.pt" if run_dir else None)
    )
    meta_path = (
        Path(args.run_meta).expanduser().resolve()
        if args.run_meta
        else (run_dir / "run_meta.json" if run_dir else None)
    )
    if checkpoint is None:
        raise ValueError("Provide --run-dir or --checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    meta = load_json(meta_path)
    meta_base = meta_path.parent if meta_path else Path.cwd()
    data_path = resolve_path(args.data or meta.get("data"), base=meta_base)
    if data_path is None:
        raise ValueError("Provide --data or a run_meta.json containing the data path")
    if not data_path.is_file():
        raise FileNotFoundError(f"Phase 3 dataset not found: {data_path}")

    split = args.split or str(meta.get("split", "chronological"))
    train_ratio = float(
        args.train_ratio if args.train_ratio is not None else meta.get("train_ratio", 0.70)
    )
    val_ratio = float(
        args.val_ratio if args.val_ratio is not None else meta.get("val_ratio", 0.15)
    )
    embargo_days = float(
        args.embargo_days
        if args.embargo_days is not None
        else meta.get("embargo_days", 30.0)
    )
    seed = int(args.seed if args.seed is not None else meta.get("seed", 42))
    graph_mode = cast(GraphMode, args.graph_mode or meta.get("graph_mode", "full"))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    payload = load_payload(data_path)
    dataset = Phase3TensorDataset(payload)
    indices = select_indices(
        dataset,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        embargo_days=embargo_days,
        partition=args.partition,
    )
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    config_values = meta.get("config", {})
    if not isinstance(config_values, dict):
        config_values = {}
    valid_config_names = {field.name for field in fields(NeuralODEConfig)}
    config_values = {
        key: value for key, value in config_values.items() if key in valid_config_names
    }
    config_values["text_emb_dim"] = int(dataset.text_embeddings.shape[-1])
    config_values["event_profile_dim"] = int(dataset.event_profiles.shape[-1])
    config = NeuralODEConfig(**config_values)

    causal_matrix: torch.Tensor | None = dataset.causal_matrix
    lag_matrix: torch.Tensor | None = dataset.lag_matrix
    if graph_mode == "random":
        causal_matrix = permute_matrix_values(dataset.causal_matrix, seed)
        lag_matrix = permute_matrix_values(dataset.lag_matrix, seed + 1009)
    elif graph_mode == "no_graph":
        causal_matrix = None
        lag_matrix = None
    elif graph_mode == "a_only":
        lag_matrix = None
    elif graph_mode == "t_only":
        causal_matrix = None

    n_stocks = int(meta.get("n_stocks", len(dataset.stock_tickers)))
    if n_stocks <= 0:
        n_stocks = int(dataset.target_stock_ids.max().item()) + 1
    n_event_types = int(meta.get("n_event_types", dataset.causal_matrix.shape[0]))
    model = NeuralODEStockPredictor(
        n_event_types=n_event_types,
        n_stocks=n_stocks,
        causal_matrix=causal_matrix,
        lag_matrix=lag_matrix,
        graph_mode=graph_mode,
        config=config,
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint))

    base_metrics, rows = evaluate_predictions(model, loader, device=device)
    result = {
        "checkpoint": str(checkpoint),
        "data": str(data_path),
        "partition": args.partition,
        "split": split,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "embargo_days": embargo_days,
        "seed": seed,
        "graph_mode": graph_mode,
        "device": str(device),
        "num_examples": len(rows),
        "dropped_invalid_examples": dataset.dropped_invalid_examples,
        "metrics": expanded_metrics(rows),
        "model_metrics": base_metrics,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.predictions_csv:
        predictions_csv = Path(args.predictions_csv).expanduser().resolve()
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
