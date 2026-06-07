# pyright: reportMissingImports=false

import argparse
import importlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_char_span(text: str, phrase: str | None) -> tuple[int | None, int | None]:
    if phrase is None:
        return None, None
    target = phrase.strip()
    if not target:
        return None, None

    lo_text = text.lower()
    lo_target = target.lower()

    idx = lo_text.find(lo_target)
    if idx >= 0:
        return idx, idx + len(target)

    return None, None


def char_to_token_span(
    offsets: list[tuple[int, int]],
    char_start: int | None,
    char_end: int | None,
) -> tuple[int | None, int | None]:
    if char_start is None or char_end is None or char_end <= char_start:
        return None, None

    token_start: int | None = None
    token_end: int | None = None
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if token_start is None and s <= char_start < e:
            token_start = i
        if s < char_end <= e:
            token_end = i
            break
        if s >= char_start and e <= char_end:
            if token_start is None:
                token_start = i
            token_end = i

    return token_start, token_end


@dataclass
class EncodedSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None
    event_type_label: int
    subject_start: int
    subject_end: int
    object_start: int
    object_end: int
    subject_mask: float
    object_mask: float
    magnitude_label: float
    magnitude_mask: float


class EventDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        model: Any,
        event_type_to_id: dict[str, int],
        max_length: int = 128,
    ):
        self.samples: list[EncodedSample] = []
        tokenizer = model.tokenizer

        for row in rows:
            event_type = str(row.get("event_type", "")).strip().upper()
            if event_type not in event_type_to_id:
                continue

            text = str(row.get("headline", "")).strip()
            if not text:
                continue

            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = [
                (int(a), int(b)) for a, b in encoded["offset_mapping"][0].tolist()
            ]

            subject = row.get("subject")
            object_ = row.get("object")

            s_char_start, s_char_end = find_char_span(
                text, None if subject is None else str(subject)
            )
            o_char_start, o_char_end = find_char_span(
                text, None if object_ is None else str(object_)
            )

            s_tok_start, s_tok_end = char_to_token_span(
                offsets, s_char_start, s_char_end
            )
            o_tok_start, o_tok_end = char_to_token_span(
                offsets, o_char_start, o_char_end
            )

            subject_mask = (
                1.0 if s_tok_start is not None and s_tok_end is not None else 0.0
            )
            object_mask = (
                1.0 if o_tok_start is not None and o_tok_end is not None else 0.0
            )

            magnitude = row.get("magnitude")
            if isinstance(magnitude, (int, float)):
                magnitude_label = max(-10.0, min(10.0, float(magnitude)))
                magnitude_mask = 1.0
            else:
                magnitude_label = 0.0
                magnitude_mask = 0.0

            token_type_ids = encoded.get("token_type_ids")

            self.samples.append(
                EncodedSample(
                    input_ids=encoded["input_ids"][0].to(dtype=torch.long),
                    attention_mask=encoded["attention_mask"][0].to(dtype=torch.long),
                    token_type_ids=(
                        token_type_ids[0].to(dtype=torch.long)
                        if token_type_ids is not None
                        else None
                    ),
                    event_type_label=event_type_to_id[event_type],
                    subject_start=0 if s_tok_start is None else int(s_tok_start),
                    subject_end=0 if s_tok_end is None else int(s_tok_end),
                    object_start=0 if o_tok_start is None else int(o_tok_start),
                    object_end=0 if o_tok_end is None else int(o_tok_end),
                    subject_mask=subject_mask,
                    object_mask=object_mask,
                    magnitude_label=magnitude_label,
                    magnitude_mask=magnitude_mask,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EncodedSample:
        return self.samples[idx]


def collate_batch(batch: list[EncodedSample], pad_id: int) -> dict[str, Any]:
    input_ids = pad_sequence(
        [x.input_ids for x in batch], batch_first=True, padding_value=pad_id
    )
    attention_mask = pad_sequence(
        [x.attention_mask for x in batch], batch_first=True, padding_value=0
    )

    any_token_type = any(x.token_type_ids is not None for x in batch)
    if any_token_type:
        token_type_ids = pad_sequence(
            [
                x.token_type_ids
                if x.token_type_ids is not None
                else torch.zeros_like(x.input_ids)
                for x in batch
            ],
            batch_first=True,
            padding_value=0,
        )
    else:
        token_type_ids = None

    targets = {
        "event_type_labels": torch.tensor(
            [x.event_type_label for x in batch], dtype=torch.long
        ),
        "subject_start": torch.tensor(
            [x.subject_start for x in batch], dtype=torch.long
        ),
        "subject_end": torch.tensor([x.subject_end for x in batch], dtype=torch.long),
        "object_start": torch.tensor([x.object_start for x in batch], dtype=torch.long),
        "object_end": torch.tensor([x.object_end for x in batch], dtype=torch.long),
        "subject_mask": torch.tensor(
            [x.subject_mask for x in batch], dtype=torch.float32
        ),
        "object_mask": torch.tensor(
            [x.object_mask for x in batch], dtype=torch.float32
        ),
        "magnitude_labels": torch.tensor(
            [x.magnitude_label for x in batch], dtype=torch.float32
        ),
        "magnitude_mask": torch.tensor(
            [x.magnitude_mask for x in batch], dtype=torch.float32
        ),
    }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "targets": targets,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def split_rows(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [str(row.get("event_type", "")).strip().upper() for row in rows]
    train_rows, temp_rows = train_test_split(
        rows,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [str(row.get("event_type", "")).strip().upper() for row in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=(1.0 - train_ratio - val_ratio) / (1.0 - train_ratio),
        random_state=seed,
        stratify=temp_labels,
    )
    return train_rows, val_rows, test_rows


def summarize_event_types(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {}

    counts = Counter(str(row.get("event_type", "")).strip().upper() for row in rows)
    return {
        event_type: round(count / total, 4)
        for event_type, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    }


def move_targets_to_device(
    targets: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in targets.items()}


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: torch.device,
    optimizer: Any,
    grad_clip: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total = 0
    sums = {
        "total": 0.0,
        "type": 0.0,
        "argument": 0.0,
        "magnitude": 0.0,
        "type_acc": 0.0,
    }

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"]
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        targets = move_targets_to_device(batch["targets"], device)

        if train_mode:
            optimizer.zero_grad()

        predictions = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            event_type_ids=targets["event_type_labels"],
        )
        losses = criterion(predictions, targets)

        if train_mode:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bsz = int(input_ids.size(0))
        total += bsz
        for key in ("total", "type", "argument", "magnitude"):
            sums[key] += float(losses[key].item()) * bsz

        pred_type = predictions["event_type_logits"].argmax(dim=-1)
        type_acc = (pred_type == targets["event_type_labels"]).float().mean().item()
        sums["type_acc"] += float(type_acc) * bsz

    if total == 0:
        return {
            "total": 0.0,
            "type": 0.0,
            "argument": 0.0,
            "magnitude": 0.0,
            "type_acc": 0.0,
        }

    return {k: v / total for k, v in sums.items()}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    ee = importlib.import_module("src.phase1.event_extractor")
    EVENT_TYPE_TO_ID = ee.EVENT_TYPE_TO_ID
    EventExtractionLoss = ee.EventExtractionLoss
    FinBERTEventExtractor = ee.FinBERTEventExtractor

    parser = argparse.ArgumentParser(description="Train Phase1 event extractor")
    parser.add_argument("--labels", required=True, help="Input labels JSONL path")
    parser.add_argument("--output-dir", default="outputs/phase1")
    parser.add_argument("--model-name", default="ProsusAI/finbert")
    parser.add_argument("--freeze-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-type", type=float, default=1.0)
    parser.add_argument("--lambda-arg", type=float, default=1.0)
    parser.add_argument("--lambda-mag", type=float, default=0.5)
    parser.add_argument(
        "--drop-none",
        action="store_true",
        help="Drop rows with event_type == NONE before split",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = load_jsonl(Path(args.labels))
    if args.drop_none:
        rows = [
            r for r in rows if str(r.get("event_type", "")).strip().upper() != "NONE"
        ]

    train_rows, val_rows, test_rows = split_rows(
        rows, args.train_ratio, args.val_ratio, args.seed
    )

    print(
        json.dumps(
            {
                "train_event_type_ratio": summarize_event_types(train_rows),
                "val_event_type_ratio": summarize_event_types(val_rows),
                "test_event_type_ratio": summarize_event_types(test_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    model = FinBERTEventExtractor(
        finbert_model_name=args.model_name,
        freeze_layers=args.freeze_layers,
        dropout=args.dropout,
    ).to(device)

    train_dataset = EventDataset(
        train_rows,
        model,
        EVENT_TYPE_TO_ID,
        max_length=args.max_length,
    )
    val_dataset = EventDataset(
        val_rows,
        model,
        EVENT_TYPE_TO_ID,
        max_length=args.max_length,
    )
    test_dataset = EventDataset(
        test_rows,
        model,
        EVENT_TYPE_TO_ID,
        max_length=args.max_length,
    )

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty after preprocessing.")

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id),
    )

    criterion = EventExtractionLoss(
        lambda_type=args.lambda_type,
        lambda_arg=args.lambda_arg,
        lambda_mag=args.lambda_mag,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                grad_clip=args.grad_clip,
            )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"train_total={train_metrics['total']:.4f} val_total={val_metrics['total']:.4f} | "
            f"train_type_acc={train_metrics['type_acc']:.4f} val_type_acc={val_metrics['type_acc']:.4f}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_total": best_val,
                    "args": vars(args),
                },
                best_path,
            )

    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_clip=args.grad_clip,
        )

    metrics_out = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_val_total": best_val,
        "test": test_metrics,
        "history": history,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
