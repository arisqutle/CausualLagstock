from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DirectionA Stage 1, Stage 2, Stage 3, and evaluation end to end."
    )
    parser.add_argument("--news-input", help="Raw news input for Stage 1")
    parser.add_argument("--price-dir", default=str(DEFAULT_WORKSPACE / "data" / "meta_prices"))
    parser.add_argument("--run-dir", default=str(DEFAULT_WORKSPACE / "pipeline_run"))
    parser.add_argument("--labels", help="Stage 1 output or existing labels JSONL")
    parser.add_argument("--embeddings-dir", help="Stage 2 embedding output directory")
    parser.add_argument("--stage2-dir", help="Stage 2 graph output directory")
    parser.add_argument("--stage3-data", help="Stage 3 dataset .pt path")
    parser.add_argument("--stage3-output", help="Stage 3 training output directory")

    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "gpt-5.4"))
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--api-key-file",
        default=os.environ.get("LLM_API_KEY_FILE"),
        help="Text file containing one API key per line",
    )
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--llm-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--use-article", action="store_true")
    parser.add_argument("--overwrite-labels", action="store_true")

    default_finbert = DEFAULT_WORKSPACE / "transformers" / "FinBERT"
    parser.add_argument(
        "--finbert-model",
        default=str(default_finbert if default_finbert.exists() else "ProsusAI/finbert"),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-limit", type=int)

    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--horizons", default="1,3,5,7,10")
    parser.add_argument("--label-threshold", type=float, default=0.005)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--stage2-device", default="cuda")
    parser.add_argument("--stage2-seed", type=int, default=42)

    parser.add_argument("--price-window", type=int, default=30)
    parser.add_argument("--context-mode", choices=["ticker", "global"], default="ticker")
    parser.add_argument("--dataset-workers", type=int, default=1)
    parser.add_argument("--stage3-epochs", type=int, default=20)
    parser.add_argument("--stage3-batch-size", type=int, default=128)
    parser.add_argument("--stage3-device", default="auto")
    parser.add_argument("--stage3-seed", type=int, default=42)
    parser.add_argument(
        "--graph-mode",
        choices=["full", "no_graph", "a_only", "t_only", "random"],
        default="full",
    )
    parser.add_argument(
        "--split",
        choices=["chronological", "stratified_random"],
        default="chronological",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--embargo-days", type=float, default=30.0)

    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-stage3-build", action="store_true")
    parser.add_argument("--skip-stage3-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_step(name: str, command: list[str], *, dry_run: bool) -> None:
    print(f"\n[{name}]\n{display_command(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def require_path(path: Path, description: str, *, dry_run: bool) -> None:
    if not dry_run and not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    labels = Path(args.labels).expanduser().resolve() if args.labels else run_dir / "labels.jsonl"
    embeddings_dir = (
        Path(args.embeddings_dir).expanduser().resolve()
        if args.embeddings_dir
        else run_dir / "embeddings"
    )
    stage2_dir = (
        Path(args.stage2_dir).expanduser().resolve()
        if args.stage2_dir
        else run_dir / "stage2"
    )
    stage3_data = (
        Path(args.stage3_data).expanduser().resolve()
        if args.stage3_data
        else run_dir / "stage3" / "phase3_dataset.pt"
    )
    stage3_output = (
        Path(args.stage3_output).expanduser().resolve()
        if args.stage3_output
        else run_dir / "stage3_model"
    )
    news_input = Path(args.news_input).expanduser().resolve() if args.news_input else None
    price_dir = Path(args.price_dir).expanduser().resolve()
    python = sys.executable

    require_path(price_dir, "Price directory", dry_run=args.dry_run)
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_llm:
        if news_input is None:
            raise ValueError("Stage 1 requires --news-input")
        require_path(news_input, "News input", dry_run=args.dry_run)
        if not args.api_key_file:
            raise ValueError(
                "Stage 1 requires --api-key-file or the LLM_API_KEY_FILE environment variable"
            )
        api_key_file = Path(args.api_key_file).expanduser().resolve()
        require_path(api_key_file, "API key file", dry_run=args.dry_run)
        command = [
            python,
            str(PROJECT_ROOT / "src" / "phase1" / "llm_event_labeling.py"),
            "--input",
            str(news_input),
            "--output",
            str(labels),
            "--model",
            args.llm_model,
            "--base-url",
            args.llm_base_url,
            "--api-key-file",
            str(api_key_file),
            "--batch-size",
            str(args.llm_batch_size),
            "--max-workers",
            str(args.llm_workers),
            "--max-samples",
            str(args.max_samples),
        ]
        command.append("--overwrite" if args.overwrite_labels else "--resume")
        if args.use_article:
            command.append("--use-article")
        run_step("Stage 1: LLM event extraction", command, dry_run=args.dry_run)
    else:
        require_path(labels, "Existing Stage 1 labels", dry_run=args.dry_run)

    if not args.skip_embeddings:
        command = [
            python,
            str(PROJECT_ROOT / "src" / "phase2" / "preprocess.py"),
            "--labels",
            str(labels),
            "--output-dir",
            str(embeddings_dir),
            "--model-name",
            args.finbert_model,
            "--batch-size",
            str(args.embedding_batch_size),
        ]
        if args.embedding_device:
            command.extend(["--device", args.embedding_device])
        if args.embedding_limit is not None:
            command.extend(["--limit", str(args.embedding_limit)])
        run_step("Stage 2: FinBERT embeddings", command, dry_run=args.dry_run)
    else:
        require_path(embeddings_dir / "emb.pt", "Existing embeddings", dry_run=args.dry_run)
        require_path(embeddings_dir / "meta.json", "Existing embedding metadata", dry_run=args.dry_run)

    if not args.skip_stage2:
        command = [
            python,
            str(PROJECT_ROOT / "src" / "phase2" / "train.py"),
            "--embeddings",
            str(embeddings_dir / "emb.pt"),
            "--meta",
            str(embeddings_dir / "meta.json"),
            "--price-dir",
            str(price_dir),
            "--output-dir",
            str(stage2_dir),
            "--epochs",
            str(args.stage2_epochs),
            "--batch-size",
            str(args.stage2_batch_size),
            "--seq-len",
            str(args.seq_len),
            "--label-threshold",
            str(args.label_threshold),
            "--horizons",
            args.horizons,
            "--seed",
            str(args.stage2_seed),
            "--device",
            args.stage2_device,
        ]
        run_step("Stage 2: causal-lag graph training", command, dry_run=args.dry_run)
    else:
        require_path(stage2_dir, "Existing Stage 2 output", dry_run=args.dry_run)

    if not args.skip_stage3_build:
        command = [
            python,
            str(PROJECT_ROOT / "src" / "phase3" / "build_dataset.py"),
            "--labels",
            str(labels),
            "--embeddings",
            str(embeddings_dir / "emb.pt"),
            "--price-dir",
            str(price_dir),
            "--stage2-dir",
            str(stage2_dir),
            "--output",
            str(stage3_data),
            "--seq-len",
            str(args.seq_len),
            "--price-window",
            str(args.price_window),
            "--label-threshold",
            str(args.label_threshold),
            "--context-mode",
            args.context_mode,
            "--horizons",
            args.horizons,
            "--num-workers",
            str(args.dataset_workers),
        ]
        run_step("Stage 3: dataset construction", command, dry_run=args.dry_run)
    else:
        require_path(stage3_data, "Existing Stage 3 dataset", dry_run=args.dry_run)

    if not args.skip_stage3_train:
        command = [
            python,
            str(PROJECT_ROOT / "src" / "phase3" / "train_neural_ode.py"),
            "--data",
            str(stage3_data),
            "--output-dir",
            str(stage3_output),
            "--split",
            args.split,
            "--train-ratio",
            str(args.train_ratio),
            "--val-ratio",
            str(args.val_ratio),
            "--embargo-days",
            str(args.embargo_days),
            "--seed",
            str(args.stage3_seed),
            "--epochs",
            str(args.stage3_epochs),
            "--batch-size",
            str(args.stage3_batch_size),
            "--graph-mode",
            args.graph_mode,
            "--device",
            args.stage3_device,
        ]
        run_step("Stage 3: Neural ODE training", command, dry_run=args.dry_run)
    else:
        require_path(stage3_output / "best_model.pt", "Existing Stage 3 checkpoint", dry_run=args.dry_run)

    if not args.skip_evaluate:
        command = [
            python,
            str(PROJECT_ROOT / "evaluate.py"),
            "--run-dir",
            str(stage3_output),
            "--data",
            str(stage3_data),
            "--output-json",
            str(stage3_output / "evaluation.json"),
            "--predictions-csv",
            str(stage3_output / "evaluation_predictions.csv"),
            "--device",
            args.stage3_device,
        ]
        run_step("Evaluation", command, dry_run=args.dry_run)

    print(f"\nPipeline completed. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
