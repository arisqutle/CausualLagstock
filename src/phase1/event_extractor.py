"""
Phase 1: 金融事件结构化抽取模型
================================
FinBERT + Multi-Head Event Decoder
- Head 1: Event Type Classifier (20类)
- Head 2: Argument Extractor (Subject/Object span extraction)
- Head 3: Magnitude Regressor (事件强度回归到 [-10, 10])
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


from .event_types import (
    ID_TO_EVENT_TYPE,
    NUM_EVENT_TYPES,
)

@dataclass
class SpanPrediction:
    start: int | None
    end: int | None
    score: float
    text: str | None = None


@dataclass
class EventPrediction:
    event_type_id: int
    event_type: str
    event_type_confidence: float
    subject: SpanPrediction
    object: SpanPrediction
    magnitude: float


# ============================================================
# Head 1: 事件类型分类器
# ============================================================
class EventTypeClassifier(nn.Module):
    """将[CLS] token映射到20类事件类型"""

    def __init__(
        self,
        hidden_size: int = 768,
        num_types: int = NUM_EVENT_TYPES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_types),
        )

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(cls_hidden)


# ============================================================
# Head 2: 论元抽取器 (Span Extraction)
# ============================================================
class ArgumentExtractor(nn.Module):
    """抽取 Subject 和 Object 的文本 span。"""

    SUBJECT = 0
    OBJECT = 1
    NUM_ROLES = 2

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.start_predictor = nn.Linear(hidden_size, self.NUM_ROLES)
        self.end_predictor = nn.Linear(hidden_size, self.NUM_ROLES)

    def forward(
        self,
        sequence_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_logits = self.start_predictor(sequence_hidden)
        end_logits = self.end_predictor(sequence_hidden)

        mask = attention_mask.unsqueeze(-1).expand_as(start_logits)
        start_logits = start_logits.masked_fill(mask == 0, -1e9)
        end_logits = end_logits.masked_fill(mask == 0, -1e9)
        return start_logits, end_logits

    @staticmethod
    def _best_span_for_role(
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_span_mask: torch.Tensor | None,
        role_idx: int,
        max_span_length: int | None = None,
    ) -> tuple[list[int | None], list[int | None], list[float]]:
        batch_size, seq_len, _ = start_logits.shape
        start_scores = start_logits[:, :, role_idx]
        end_scores = end_logits[:, :, role_idx]

        best_starts: list[int | None] = []
        best_ends: list[int | None] = []
        best_scores: list[float] = []

        for batch_idx in range(batch_size):
            valid_positions = attention_mask[batch_idx].bool()
            if valid_span_mask is not None:
                valid_positions = valid_positions & valid_span_mask[batch_idx].bool()
            best_score = float("-inf")
            best_start: int | None = None
            best_end: int | None = None

            for start_idx in range(seq_len):
                if not valid_positions[start_idx]:
                    continue

                max_end = seq_len - 1
                if max_span_length is not None:
                    max_end = min(max_end, start_idx + max_span_length - 1)

                for end_idx in range(start_idx, max_end + 1):
                    if not valid_positions[end_idx]:
                        continue
                    score = (
                        start_scores[batch_idx, start_idx]
                        + end_scores[batch_idx, end_idx]
                    ).item()
                    if score > best_score:
                        best_score = score
                        best_start = start_idx
                        best_end = end_idx

            best_starts.append(best_start)
            best_ends.append(best_end)
            best_scores.append(best_score)

        return best_starts, best_ends, best_scores

    @staticmethod
    def _span_text_from_offsets(
        text: str,
        offsets: Sequence[tuple[int, int]],
        start_idx: int | None,
        end_idx: int | None,
    ) -> str | None:
        if start_idx is None or end_idx is None or end_idx < start_idx:
            return None

        start_char, _ = offsets[start_idx]
        _, end_char = offsets[end_idx]
        if end_char <= start_char:
            return None
        return text[start_char:end_char]

    def decode(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_span_mask: torch.Tensor | None = None,
        texts: Sequence[str] | None = None,
        offset_mappings: Sequence[Sequence[tuple[int, int]]] | None = None,
        max_span_length: int | None = 12,
    ) -> dict[str, list[SpanPrediction]]:
        role_to_name = {
            self.SUBJECT: "subject",
            self.OBJECT: "object",
        }

        decoded: dict[str, list[SpanPrediction]] = {"subject": [], "object": []}
        for role_idx, role_name in role_to_name.items():
            starts, ends, scores = self._best_span_for_role(
                start_logits,
                end_logits,
                attention_mask,
                valid_span_mask,
                role_idx,
                max_span_length=max_span_length,
            )

            for batch_idx, (start_idx, end_idx, score) in enumerate(
                zip(starts, ends, scores)
            ):
                span_text: str | None = None
                if texts is not None and offset_mappings is not None:
                    span_text = self._span_text_from_offsets(
                        texts[batch_idx],
                        offset_mappings[batch_idx],
                        start_idx,
                        end_idx,
                    )
                decoded[role_name].append(
                    SpanPrediction(
                        start=start_idx,
                        end=end_idx,
                        score=score,
                        text=span_text,
                    )
                )

        return decoded


# ============================================================
# Head 3: 幅度回归器
# ============================================================
class MagnitudeRegressor(nn.Module):
    """预测事件强度，输出范围固定在 [-10, 10]。"""

    def __init__(
        self,
        hidden_size: int = 768,
        event_emb_size: int = 64,
        num_types: int = NUM_EVENT_TYPES,
    ):
        super().__init__()
        self.event_type_emb = nn.Embedding(num_types, event_emb_size)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size + event_emb_size, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(
        self, cls_hidden: torch.Tensor, event_type_ids: torch.Tensor
    ) -> torch.Tensor:
        type_emb = self.event_type_emb(event_type_ids)
        combined = torch.cat([cls_hidden, type_emb], dim=-1)
        raw = self.regressor(combined)
        return 10.0 * torch.tanh(raw)


# ============================================================
# 完整模型: FinBERT + Multi-Head Event Decoder
# ============================================================
class FinBERTEventExtractor(nn.Module):
    """完整的事件抽取模型。"""

    def __init__(
        self,
        finbert_model_name: str = "ProsusAI/finbert",
        freeze_layers: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
        self.encoder = AutoModel.from_pretrained(finbert_model_name)

        for layer_idx, layer in enumerate(self.encoder.encoder.layer):
            if layer_idx < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.event_type_head = EventTypeClassifier(hidden_size, dropout=dropout)
        self.argument_head = ArgumentExtractor(hidden_size)
        self.magnitude_head = MagnitudeRegressor(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        event_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_inputs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)

        sequence_hidden = outputs.last_hidden_state
        cls_hidden = sequence_hidden[:, 0, :]
        event_type_logits = self.event_type_head(cls_hidden)
        start_logits, end_logits = self.argument_head(sequence_hidden, attention_mask)

        if event_type_ids is None:
            magnitude_type_ids = event_type_logits.argmax(dim=-1)
        else:
            magnitude_type_ids = event_type_ids

        magnitude_pred = self.magnitude_head(cls_hidden, magnitude_type_ids)

        return {
            "event_type_logits": event_type_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "magnitude_pred": magnitude_pred,
            "cls_embedding": cls_hidden,
        }

    @torch.no_grad()
    def predict(
        self,
        texts: Sequence[str],
        max_length: int = 128,
        device: torch.device | None = None,
        max_span_length: int = 12,
    ) -> list[EventPrediction]:
        if device is None:
            device = next(self.parameters()).device
        was_training = self.training

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping")
        valid_span_mask = (offset_mapping[..., 1] > offset_mapping[..., 0]).to(device)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        self.eval()
        outputs = self(**encoded)
        event_type_probs = torch.softmax(outputs["event_type_logits"], dim=-1)
        event_type_ids = event_type_probs.argmax(dim=-1)
        span_predictions = self.argument_head.decode(
            outputs["start_logits"],
            outputs["end_logits"],
            encoded["attention_mask"],
            valid_span_mask=valid_span_mask,
            texts=texts,
            offset_mappings=offset_mapping.tolist(),
            max_span_length=max_span_length,
        )

        predictions: list[EventPrediction] = []
        for batch_idx in range(len(texts)):
            event_type_id = int(event_type_ids[batch_idx].item())
            predictions.append(
                EventPrediction(
                    event_type_id=event_type_id,
                    event_type=ID_TO_EVENT_TYPE[event_type_id],
                    event_type_confidence=float(
                        event_type_probs[batch_idx, event_type_id].item()
                    ),
                    subject=span_predictions["subject"][batch_idx],
                    object=span_predictions["object"][batch_idx],
                    magnitude=float(outputs["magnitude_pred"][batch_idx].item()),
                )
            )
        if was_training:
            self.train()
        return predictions


# ============================================================
# 损失函数
# ============================================================
class EventExtractionLoss(nn.Module):
    """联合训练损失。"""

    def __init__(
        self,
        lambda_type: float = 1.0,
        lambda_arg: float = 1.0,
        lambda_mag: float = 0.5,
    ):
        super().__init__()
        self.lambda_type = lambda_type
        self.lambda_arg = lambda_arg
        self.lambda_mag = lambda_mag

        self.type_loss_fn = nn.CrossEntropyLoss()
        self.arg_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.mag_loss_fn = nn.SmoothL1Loss(reduction="none")

    def _masked_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = self.arg_loss_fn(logits, labels)
        mask = mask.float()
        denom = mask.sum().clamp_min(1.0)
        return (losses * mask).sum() / denom

    def _get_mask(
        self,
        targets: dict[str, torch.Tensor],
        key: str,
        fallback_from: str,
    ) -> torch.Tensor:
        if key in targets:
            return targets[key].to(dtype=torch.float32)

        reference = targets[fallback_from]
        return torch.ones_like(reference, dtype=torch.float32)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        losses["type"] = self.type_loss_fn(
            predictions["event_type_logits"],
            targets["event_type_labels"],
        )

        subject_mask = self._get_mask(targets, "subject_mask", "subject_start")
        object_mask = self._get_mask(targets, "object_mask", "object_start")

        losses["subject_start"] = self._masked_cross_entropy(
            predictions["start_logits"][:, :, ArgumentExtractor.SUBJECT],
            targets["subject_start"],
            subject_mask,
        )
        losses["subject_end"] = self._masked_cross_entropy(
            predictions["end_logits"][:, :, ArgumentExtractor.SUBJECT],
            targets["subject_end"],
            subject_mask,
        )
        losses["object_start"] = self._masked_cross_entropy(
            predictions["start_logits"][:, :, ArgumentExtractor.OBJECT],
            targets["object_start"],
            object_mask,
        )
        losses["object_end"] = self._masked_cross_entropy(
            predictions["end_logits"][:, :, ArgumentExtractor.OBJECT],
            targets["object_end"],
            object_mask,
        )

        arg_loss = (
            losses["subject_start"]
            + losses["subject_end"]
            + losses["object_start"]
            + losses["object_end"]
        ) / 4.0
        losses["argument"] = arg_loss

        magnitude_mask = self._get_mask(targets, "magnitude_mask", "magnitude_labels")
        mag_pred = predictions["magnitude_pred"].squeeze(-1)
        mag_true = targets["magnitude_labels"].float()
        mag_loss = self.mag_loss_fn(mag_pred, mag_true)
        magnitude_denom = magnitude_mask.sum().clamp_min(1.0)
        losses["magnitude"] = (mag_loss * magnitude_mask).sum() / magnitude_denom

        losses["total"] = (
            self.lambda_type * losses["type"]
            + self.lambda_arg * losses["argument"]
            + self.lambda_mag * losses["magnitude"]
        )
        return losses
