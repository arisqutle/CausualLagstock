"""
LLM自动事件标注脚本
====================
使用DeepSeek/GPT-4对FNSPID新闻进行结构化事件标注，
生成银标数据用于训练FinBERT事件抽取模型。

用法:
    python scripts/llm_event_labeling.py \
        --input data/fnspid/All_external.csv \
        --output data/labels/silver_labels.jsonl \
        --model deepseek-chat \
        --batch-size 10 \
        --max-samples 50000 \
        --use-article \
        --max-article-chars 1200 \
        --start-date 2020-01-01 \
        --end-date 2020-12-31
"""

"""
python scripts/llm_event_labeling.py --input data/fnspid/nasdaq_exteral_data.csv --output data/labels/silver_labels_test_newest.jsonl --api-key-file configs/keys.txt --use-article --market-data-file data/market_data.csv --max-workers 10 --max-samples 100 --inter-request-delay 3 --resume
"""
import argparse
import csv
import importlib
import importlib.util
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# 在这里填写你的API Key
API_KEY = ""  # 或者通过环境变量、命令行参数等方式提供
API_URL = "https://www.fhl.mom/v1"

EVENT_TYPE_SET = {
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "K1",
    "K2",
    "K3",
    "K4",
    "G1",
    "G2",
    "NONE",
}

DEFAULT_DATE_CANDIDATES = [
    "Date",
    "date",
    "pub_time",
    "timestamp",
    "time",
    "created_at",
]

DEFAULT_RETRY_BACKOFF = 1.5
LOCAL_FINBERT_PATH = Path(__file__).resolve().parent.parent / "transformers" / "FinBERT"
NOVELTY_SIMILARITY_THRESHOLD = 0.85

LABELING_PROMPT = """You are a financial event extraction expert. Given financial news content, extract structured event information.

News content:
{news_input}

Event type guide (pick the best match):
- M1 rate changes; M2 GDP/jobs releases; M3 trade policy/tariffs; M4 monetary policy signals; M5 inflation data; M6 regulation/policy changes
- C1 earnings reports; C2 M&A; C3 executive changes; C4 product launches; C5 buybacks; C6 stock splits; C7 lawsuits/fines; C8 bankruptcy/default
- K1 analyst rating changes; K2 insider/large block trades; K3 index rebalancing; K4 technical breakout
- G1 geopolitical conflict; G2 disaster/public health crisis; NONE no clear financial event

Extract the following in JSON format:
{{
  "event_type": "one of: M1, M2, M3, M4, M5, M6, C1, C2, C3, C4, C5, C6, C7, C8, K1, K2, K3, K4, G1, G2, or NONE",
  "subject": "the entity performing the action (e.g., 'Federal Reserve', 'Apple Inc.')",
  "action": "the core action verb (e.g., 'raise', 'acquire', 'report')",
  "object": "the entity/concept being acted upon (e.g., 'interest_rate', 'Activision')",
  "magnitude": "a normalized float in [-10, 10] representing legacy event intensity/size, null if not enough information",
  "polarity": "one of: positive, negative, neutral, mixed, or null",
  "surprise": "a normalized float in [-1, 1] representing the expectation gap, or null if unavailable",
  "scope": "one of: single_stock, sector, market, global, or null",
  "novelty": "a normalized float in [0, 1] representing how new the information is, or null if unavailable",
  "credibility": "a normalized float in [0, 1] representing source credibility, or null if unavailable",
  "affected_tickers": ["list of stock tickers directly affected, e.g., ['AAPL', 'MSFT']"]
}}

Important rules:
- If the text describes multiple events, extract the MOST IMPORTANT one
- For magnitude, keep it as a numeric legacy intensity score when possible
- For polarity, use a direction label rather than a numeric value
- For surprise, novelty, and credibility, output a numeric score when possible
- Surprise calculation rule:
  - Prefer an expectation-gap style score when the news gives both an actual/reported value and an expected/forecast/consensus/estimate value
  - Compute surprise as (actual - expected) / abs(expected)
  - If expected is exactly 0, use actual - expected instead of dividing
  - Clamp the final surprise score to the range [-1, 1]
  - If no expectation information is available in the news, use null rather than inventing a value from magnitude or sentiment
- For scope, choose the smallest scope that still matches the news impact
- For credibility, judge it from the publishing source and article provenance, not from the market impact itself
- In this dataset, Publisher is often missing, so infer credibility mainly from URL domain, article byline, and self-attribution phrases like "This story originally appeared on ..."
- Credibility scoring rubric:
  - 0.95-1.00: top-tier financial/official sources such as Bloomberg, Reuters, SEC filings, official company releases, exchange notices
  - 0.80-0.94: established financial media/platforms such as Nasdaq, WSJ, FT, CNBC, MarketWatch
  - 0.60-0.79: specialist investment research or syndicated market commentary such as Fintel, analyst writeups, screened summaries on credible platforms
  - 0.40-0.59: general news, secondary reposts, unclear original sourcing, or mixed editorial quality
  - 0.00-0.39: low-trust, off-domain, anonymous, social/forum/blog style, or clearly non-financial / irrelevant source for stock-event labeling
- If the host site is a syndication page (for example Nasdaq) but the article explicitly says it originally appeared on another source, score credibility based on the ORIGINAL source with a slight discount for syndication
- If the article source is missing or ambiguous, use the URL domain plus byline/article wording to make the best estimate
- affected_tickers should be US stock tickers only
- event_type must be exactly one token from the allowed set (no explanations)
- affected_tickers must always be an array (use [] if unknown)
- If event_type is not NONE, subject/action/object should not be null (use "UNKNOWN" if needed)
- magnitude must be a number in [-10, 10] when available; positive means bullish impact, negative means bearish impact
- If event_type is NONE, set all other fields to null and affected_tickers to []

Output ONLY the JSON, no explanation."""

_CLIENTS: dict[tuple[str, str], Any] = {}
_CLIENTS_LOCK = threading.Lock()


def set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


set_csv_field_size_limit()


def parse_datetime_safe(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None

    normalized = text.replace(" UTC", "").replace("Z", "+00:00")
    iso_candidate = (
        normalized.replace(" ", "T") if "T" not in normalized else normalized
    )

    for candidate in (normalized, iso_candidate):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%a %b %d %H:%M:%S %z %Y",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def in_date_window(
    dt: datetime | None, start_dt: datetime | None, end_dt: datetime | None
) -> bool:
    if start_dt is None and end_dt is None:
        return True
    if dt is None:
        return False
    if start_dt is not None and dt < start_dt:
        return False
    if end_dt is not None and dt > end_dt:
        return False
    return True


def pick_field(data: dict[str, Any], candidates: list[str]) -> str:
    for key in candidates:
        value = data.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return ""


def parse_first_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        return num

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        num = float(match.group(0))
    except ValueError:
        return None

    if math.isnan(num) or math.isinf(num):
        return None
    return num


def clamp_float(value: Any, low: float, high: float) -> float | None:
    num = parse_first_float(value)
    if num is None:
        return None
    return max(low, min(high, num))


def normalize_polarity(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    mapping = {
        "positive": "positive",
        "bullish": "positive",
        "up": "positive",
        "利好": "positive",
        "+": "positive",
        "negative": "negative",
        "bearish": "negative",
        "down": "negative",
        "利空": "negative",
        "-": "negative",
        "neutral": "neutral",
        "mixed": "mixed",
    }

    if text in mapping:
        return mapping[text]
    if "posit" in text or "bull" in text:
        return "positive"
    if "negat" in text or "bear" in text:
        return "negative"
    if "neutral" in text:
        return "neutral"
    if "mix" in text:
        return "mixed"
    return None


def normalize_scope(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    mapping = {
        "single": "single_stock",
        "single_stock": "single_stock",
        "stock": "single_stock",
        "company": "single_stock",
        "sector": "sector",
        "industry": "sector",
        "market": "market",
        "macro": "market",
        "global": "global",
        "world": "global",
    }

    if text in mapping:
        return mapping[text]
    if "single" in text or "company" in text or "stock" in text:
        return "single_stock"
    if "sector" in text or "industry" in text:
        return "sector"
    if "market" in text or "macro" in text:
        return "market"
    if "global" in text or "world" in text:
        return "global"
    return None


def normalize_profile_field(
    label: dict[str, Any],
    field: str,
    nested: dict[str, Any],
    kind: str,
) -> Any:
    raw_value = label.get(field)
    if raw_value is None:
        raw_value = nested.get(field)

    if kind == "polarity":
        return normalize_polarity(raw_value)
    if kind == "scope":
        return normalize_scope(raw_value)
    if kind == "surprise":
        return clamp_float(raw_value, -1.0, 1.0)
    if kind == "novelty":
        return clamp_float(raw_value, 0.0, 1.0)
    if kind == "credibility":
        return clamp_float(raw_value, 0.0, 1.0)
    if kind == "magnitude":
        return clamp_float(raw_value, -10.0, 10.0)
    return raw_value


POSITIVE_HINT_RE = re.compile(
    r"\b(beat|beats|beating|raise|raises|raised|upgrade|upgraded|buyback|buybacks|growth|growing|record|approval|approved|profit|profitable|strong|surge|surges|surged|increase|increases|increased|expand|expands|expanded|outperform)\b",
    re.IGNORECASE,
)
NEGATIVE_HINT_RE = re.compile(
    r"\b(miss|misses|missed|cut|cuts|cutting|downgrade|downgraded|lawsuit|fine|fined|investigation|investigated|recall|warn|warning|loss|losses|weak|decline|declines|declined|delay|delays|bankrupt|fraud|charge|charged)\b",
    re.IGNORECASE,
)
MACRO_HINT_RE = re.compile(
    r"\b(fed|fomc|cpi|ppi|inflation|rates?|interest rate|tariff|war|geopolitical|recession|gdp|unemployment|macro)\b",
    re.IGNORECASE,
)


def get_raw_field(item: dict[str, Any], *names: str) -> Any:
    raw_fields = item.get("raw_fields")
    if not isinstance(raw_fields, dict):
        raw_fields = {}

    for name in names:
        value = item.get(name)
        if value is not None and str(value).strip() != "":
            return value
        value = raw_fields.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def text_blob(item: dict[str, Any]) -> str:
    headline = str(item.get("headline", "")).strip()
    article = str(item.get("article", "")).strip()
    if article:
        return f"{headline}\n{article}"
    return headline


def tokenize_for_similarity(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}", text.lower())
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "will",
        "have",
        "has",
        "was",
        "were",
        "are",
        "been",
        "into",
        "over",
        "after",
        "before",
        "news",
        "stock",
        "stocks",
    }
    return {tok for tok in tokens if tok not in stopwords}


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else inter / union


class LocalFinBERTEncoder:
    """L2-normalized FinBERT embeddings for semantic similarity."""

    def __init__(
        self, model_path: Path = LOCAL_FINBERT_PATH, device: str | None = None
    ):
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None
        ):
            raise ImportError("transformers + torch required for semantic novelty")
        if not model_path.exists():
            raise FileNotFoundError(f"FinBERT model path not found: {model_path}")

        self._torch = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
        tokenizer_cls = getattr(transformers_module, "AutoTokenizer")
        model_cls = getattr(transformers_module, "AutoModel")

        self.model_path = model_path
        self.device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer_cls.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = model_cls.from_pretrained(
            str(model_path), local_files_only=True
        ).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        all_embs: list[np.ndarray] = []
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)
                out = self.model(**enc)
                emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / np.maximum(norms, 1e-8)
                all_embs.append(emb.astype(np.float32))
        return np.vstack(all_embs)


_novelty_encoder: LocalFinBERTEncoder | None = None
_novelty_encoder_lock = threading.Lock()


def get_novelty_encoder() -> LocalFinBERTEncoder | None:
    global _novelty_encoder
    if _novelty_encoder is not None:
        return _novelty_encoder

    with _novelty_encoder_lock:
        if _novelty_encoder is not None:
            return _novelty_encoder
        try:
            _novelty_encoder = LocalFinBERTEncoder()
        except Exception:
            _novelty_encoder = None
        return _novelty_encoder


def parse_numeric_field(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        num = float(match.group(0))
    except ValueError:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def load_context_index(input_path: str | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not input_path:
        return {}

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"market context file not found: {input_path}")

    suffix = path.suffix.lower()
    index: dict[tuple[str, str], dict[str, Any]] = {}

    def ingest_row(row: dict[str, Any]) -> None:
        ticker = pick_field(row, ["ticker", "symbol", "Stock_symbol", "StockSymbol"])
        date_value = pick_field(row, DEFAULT_DATE_CANDIDATES)
        key = (ticker.strip().upper(), normalize_date_key(date_value))
        if not key[0] or not key[1]:
            return
        index[key] = dict(row)

    def attach_prior_return_features() -> None:
        grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for (ticker, date_key), row in index.items():
            grouped.setdefault(ticker, []).append((date_key, row))

        for ticker_rows in grouped.values():
            ticker_rows.sort(key=lambda item: item[0])
            closes = [parse_numeric_field(row.get("close") or row.get("Close")) for _, row in ticker_rows]

            for idx, (_, row) in enumerate(ticker_rows):
                prior_close = parse_numeric_field(row.get("prev_close"))
                if prior_close is None and idx >= 1:
                    prior_close = closes[idx - 1]
                if prior_close is not None:
                    row["prior_1d_close"] = prior_close

                # Prior 3-day return ending before the current day's session.
                # Uses only information available up to the previous trading close.
                prior_3d_return: float | None = None
                if prior_close is not None and idx >= 3:
                    anchor_close = closes[idx - 3]
                    if anchor_close is not None and abs(anchor_close) >= 1e-8:
                        prior_3d_return = (prior_close - anchor_close) / abs(anchor_close)

                row["prior_3d_return"] = clamp_float(prior_3d_return, -1.0, 1.0)

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ingest_row(dict(row))
        attach_prior_return_features()
        return index

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                ingest_row(data)
    attach_prior_return_features()
    return index


def lookup_context(
    item: dict[str, Any], context_index: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    ticker = str(item.get("ticker", "")).strip().upper()
    date_key = normalize_date_key(str(item.get("date", "")))
    if not ticker:
        return {}

    exact = context_index.get((ticker, date_key))
    if exact is not None:
        return exact

    return context_index.get((ticker, ""), {})


def infer_surprise(
    item: dict[str, Any], market_ctx: dict[str, Any]
) -> tuple[float | None, str]:
    candidates = [
        ("actual", "expected"),
        ("actual", "forecast"),
        ("actual", "consensus"),
        ("actual", "estimate"),
        ("value", "expected"),
        ("reported", "expected"),
        ("realized", "expected"),
    ]
    for actual_key, expected_key in candidates:
        actual = parse_numeric_field(get_raw_field(item, actual_key))
        if actual is None:
            actual = parse_numeric_field(market_ctx.get(actual_key))
        expected = parse_numeric_field(get_raw_field(item, expected_key))
        if expected is None:
            expected = parse_numeric_field(market_ctx.get(expected_key))
        if actual is None or expected is None:
            continue
        if abs(expected) < 1e-8:
            gap = actual - expected
        else:
            gap = (actual - expected) / abs(expected)
        return clamp_float(gap, -1.0, 1.0), f"{actual_key}-{expected_key}"

    # fallback: use prior 3-day return ending before the current session
    prior_3d_return = parse_numeric_field(get_raw_field(item, "prior_3d_return"))
    if prior_3d_return is None:
        prior_3d_return = parse_numeric_field(market_ctx.get("prior_3d_return"))
    if prior_3d_return is not None:
        return clamp_float(prior_3d_return, -1.0, 1.0), "prior_3d_return"

    return None, "none"


def infer_novelty(
    item: dict[str, Any],
    prior_same_ticker: list[dict[str, Any]],
    encoder: LocalFinBERTEncoder | None = None,
) -> tuple[float | None, str]:
    current_text = text_blob(item).strip()
    if not current_text:
        return None, "no_text"

    prior_texts = [text_blob(prior).strip() for prior in prior_same_ticker[-10:]]
    prior_texts = [text for text in prior_texts if text]
    if not prior_texts:
        return 1.0, "no_history"

    if encoder is None:
        encoder = get_novelty_encoder()

    if encoder is None:
        current_tokens = tokenize_for_similarity(current_text)
        if not current_tokens:
            return None, "no_text"

        similar_count = 0
        for prior_text in prior_texts:
            sim = jaccard_similarity(
                current_tokens, tokenize_for_similarity(prior_text)
            )
            if sim >= NOVELTY_SIMILARITY_THRESHOLD:
                similar_count += 1
        novelty = clamp_float(1.0 / (1.0 + similar_count), 0.0, 1.0)
        return (
            novelty,
            f"1/(1+count_ge_{NOVELTY_SIMILARITY_THRESHOLD:.2f}):{similar_count}",
        )

    try:
        embs = encoder.encode([current_text] + prior_texts)
        if len(embs) < 2:
            return 1.0, "no_history"
        query = embs[0]
        history = embs[1:]
        similarities = (
            history @ query if len(history) else np.array([], dtype=np.float32)
        )
        similar_count = int(np.sum(similarities >= NOVELTY_SIMILARITY_THRESHOLD))
        novelty = clamp_float(1.0 / (1.0 + similar_count), 0.0, 1.0)
        return (
            novelty,
            f"1/(1+count_ge_{NOVELTY_SIMILARITY_THRESHOLD:.2f}):{similar_count}",
        )
    except Exception:
        current_tokens = tokenize_for_similarity(current_text)
        if not current_tokens:
            return None, "no_text"

        similar_count = 0
        for prior_text in prior_texts:
            sim = jaccard_similarity(
                current_tokens, tokenize_for_similarity(prior_text)
            )
            if sim >= NOVELTY_SIMILARITY_THRESHOLD:
                similar_count += 1
        novelty = clamp_float(1.0 / (1.0 + similar_count), 0.0, 1.0)
        return (
            novelty,
            f"1/(1+count_ge_{NOVELTY_SIMILARITY_THRESHOLD:.2f}):{similar_count}",
        )


def build_hybrid_signals(
    items: list[dict[str, Any]],
    context_index: dict[tuple[str, str], dict[str, Any]],
    novelty_encoder: LocalFinBERTEncoder | None = None,
) -> list[dict[str, Any]]:
    history_by_ticker: dict[str, list[dict[str, Any]]] = {}
    enriched: list[dict[str, Any] | None] = [None] * len(items)

    def time_key(item: dict[str, Any]) -> float:
        dt = parse_datetime_safe(str(item.get("date", "")))
        return dt.timestamp() if dt is not None else float("-inf")

    ordered = sorted(
        enumerate(items),
        key=lambda pair: (time_key(pair[1]), pair[0]),
    )

    start = 0
    while start < len(ordered):
        current_time = time_key(ordered[start][1])
        end = start + 1
        while end < len(ordered):
            next_time = time_key(ordered[end][1])
            if next_time != current_time:
                break
            end += 1

        pending_history: list[tuple[str, dict[str, Any]]] = []
        for original_index, item in ordered[start:end]:
            ticker = str(item.get("ticker", "")).strip().upper()
            market_ctx = lookup_context(item, context_index)
            prior_same_ticker = history_by_ticker.get(ticker, [])

            surprise, surprise_source = infer_surprise(item, market_ctx)
            novelty, novelty_source = infer_novelty(
                item, prior_same_ticker, encoder=novelty_encoder
            )

            raw_affected = item.get("affected_tickers")
            if not isinstance(raw_affected, list):
                raw_affected = []
            affected_tickers = [
                str(x).strip().upper() for x in raw_affected if str(x).strip()
            ]
            if not affected_tickers and ticker:
                affected_tickers = [ticker]

            magnitude = item.get("magnitude")

            hybrid_signals = {
                "rule_hints": {
                    "novelty": novelty,
                },
                "market_hints": {
                    "surprise": surprise,
                    "surprise_source": surprise_source,
                    "raw_context_hit": bool(market_ctx),
                },
                "provenance": {
                    "novelty_source": novelty_source,
                    "ticker": ticker,
                    "date": normalize_date_key(str(item.get("date", ""))),
                },
            }

            enriched_item = dict(item)
            enriched_item["affected_tickers"] = affected_tickers
            enriched_item["magnitude"] = magnitude
            enriched_item["hybrid_signals"] = hybrid_signals
            enriched_item["hybrid_hints"] = {
                "surprise": surprise,
                "novelty": novelty,
            }
            enriched[original_index] = enriched_item

            if ticker:
                pending_history.append((ticker, item))

        # Items with the same timestamp must not become history for one another.
        for ticker, item in pending_history:
            history_by_ticker.setdefault(ticker, []).append(item)
        start = end

    if any(item is None for item in enriched):
        raise RuntimeError("Hybrid signal enrichment did not preserve every input item")
    return [item for item in enriched if item is not None]


def build_item_key(
    headline: str, source_date: str, source_ticker: str
) -> tuple[str, str, str]:
    return (headline.strip(), source_date.strip(), source_ticker.strip())


def normalize_date_key(value: str | None) -> str:
    dt = parse_datetime_safe(value)
    if dt is not None:
        return dt.date().isoformat()
    return "" if value is None else str(value).strip()


def load_news_items(
    input_path: Path,
    max_samples: int,
    start_date: str | None,
    end_date: str | None,
    date_field: str | None,
) -> list[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    items: list[dict[str, Any]] = []

    start_dt = parse_datetime_safe(start_date)
    end_dt = parse_datetime_safe(end_date)
    date_candidates = [date_field] if date_field else DEFAULT_DATE_CANDIDATES

    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_headline = (
                    row.get("headline") or row.get("title") or row.get("Article_title")
                )
                headline = "" if raw_headline is None else str(raw_headline).strip()
                if not headline:
                    continue

                date_value = pick_field(row, date_candidates)
                dt = parse_datetime_safe(date_value)
                if not in_date_window(dt, start_dt, end_dt):
                    continue

                article = pick_field(
                    row, ["Article", "article", "text", "content", "body"]
                )
                ticker = pick_field(row, ["Stock_symbol", "ticker", "symbol"])
                items.append(
                    {
                        "headline": headline,
                        "article": article,
                        "date": date_value,
                        "ticker": ticker,
                        "raw_fields": dict(row),
                    }
                )
                if len(items) >= max_samples:
                    break
        return items

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            raw_headline = data.get("headline") or data.get("title")
            headline = "" if raw_headline is None else str(raw_headline).strip()
            if not headline:
                continue

            date_value = pick_field(data, date_candidates)
            dt = parse_datetime_safe(date_value)
            if not in_date_window(dt, start_dt, end_dt):
                continue

            article = pick_field(
                data, ["Article", "article", "text", "content", "body"]
            )
            ticker = pick_field(data, ["Stock_symbol", "ticker", "symbol"])
            items.append(
                {
                    "headline": headline,
                    "article": article,
                    "date": date_value,
                    "ticker": ticker,
                    "raw_fields": dict(data),
                }
            )
            if len(items) >= max_samples:
                break
    return items


def load_api_keys(api_keys_arg: str | None, api_key_file: str | None) -> list[str]:
    keys: list[str] = []

    if api_keys_arg:
        keys.extend(part.strip() for part in api_keys_arg.split(",") if part.strip())

    if api_key_file:
        key_file_path = Path(api_key_file)
        with key_file_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                key = line.strip()
                if key:
                    keys.append(key)

    if not keys and API_KEY.strip():
        keys.extend(part.strip() for part in API_KEY.split(",") if part.strip())

    unique_keys: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    return unique_keys


def load_completed_keys(output_path: Path) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("event_type", "")).strip().upper() == "ERROR":
                continue
            completed.add(
                build_item_key(
                    str(row.get("headline", "")),
                    str(row.get("source_date", "")),
                    str(row.get("source_ticker", "")),
                )
            )
    return completed


def filter_pending_items(
    items: list[dict[str, Any]], completed_keys: set[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    if not completed_keys:
        return items

    pending: list[dict[str, Any]] = []
    for item in items:
        key = build_item_key(
            str(item.get("headline", "")),
            str(item.get("date", "")),
            str(item.get("ticker", "")),
        )
        if key not in completed_keys:
            pending.append(item)
    return pending


def chunk_items(
    items: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def get_openai_client(api_key: str, base_url: str):
    cache_key = (api_key, base_url)
    with _CLIENTS_LOCK:
        if cache_key in _CLIENTS:
            return _CLIENTS[cache_key]

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("请安装openai库: pip install openai") from exc

        client = OpenAI(api_key=api_key, base_url=base_url)
        _CLIENTS[cache_key] = client
        return client


def build_news_input(
    item: dict[str, Any], use_article: bool, max_article_chars: int
) -> str:
    headline = str(item.get("headline", ""))
    article = str(item.get("article", ""))
    url = get_raw_field(item, "Url", "url")
    publisher = get_raw_field(item, "Publisher", "publisher")
    author = get_raw_field(item, "Author", "author")

    news_input = f"Headline: {headline}"
    if item.get("date"):
        news_input += f"\nDate: {item.get('date')}"
    if item.get("ticker"):
        news_input += f"\nTicker: {item.get('ticker')}"
    if url:
        news_input += f"\nURL: {url}"
    if publisher:
        news_input += f"\nPublisher: {publisher}"
    if author:
        news_input += f"\nAuthor: {author}"
    if use_article and article:
        news_input += f"\nArticle excerpt: {article[:max_article_chars].strip()}"
    return news_input


def extract_json_text(content: str) -> str:
    text = content.strip()
    if not text:
        return text

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
        else:
            text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        text = text[first : last + 1]
    return text.strip()


def normalize_label_payload(
    label: dict[str, Any], item: dict[str, Any], model: str
) -> dict[str, Any]:
    raw_type = str(label.get("event_type", "NONE"))
    normalized_type = raw_type.split("(")[0].strip().upper()
    if normalized_type not in EVENT_TYPE_SET:
        normalized_type = "NONE"
    label["event_type"] = normalized_type

    if not isinstance(label.get("affected_tickers"), list):
        label["affected_tickers"] = []

    nested_profile = label.get("impact_profile")
    if not isinstance(nested_profile, dict):
        nested_profile = {}

    label["magnitude"] = normalize_profile_field(
        label, "magnitude", nested_profile, "magnitude"
    )
    label["polarity"] = normalize_profile_field(
        label, "polarity", nested_profile, "polarity"
    )
    label["surprise"] = normalize_profile_field(
        label, "surprise", nested_profile, "surprise"
    )
    label["scope"] = normalize_profile_field(label, "scope", nested_profile, "scope")
    label["novelty"] = normalize_profile_field(
        label, "novelty", nested_profile, "novelty"
    )
    label["credibility"] = normalize_profile_field(
        label, "credibility", nested_profile, "credibility"
    )

    label["impact_profile"] = {
        "polarity": label["polarity"],
        "surprise": label["surprise"],
        "scope": label["scope"],
        "novelty": label["novelty"],
        "credibility": label["credibility"],
    }

    if normalized_type == "NONE":
        label["subject"] = None
        label["action"] = None
        label["object"] = None
        label["magnitude"] = None
        label["polarity"] = None
        label["surprise"] = None
        label["scope"] = None
        label["novelty"] = None
        label["credibility"] = None
        label["impact_profile"] = {
            "polarity": None,
            "surprise": None,
            "scope": None,
            "novelty": None,
            "credibility": None,
        }
        label["affected_tickers"] = []
    else:
        for key in ("subject", "action", "object"):
            value = label.get(key)
            if value is None or str(value).strip() == "":
                label[key] = "UNKNOWN"

    label["headline"] = str(item.get("headline", ""))
    label["article"] = str(item.get("article", ""))
    label["source_date"] = str(item.get("date", ""))
    label["source_ticker"] = str(item.get("ticker", ""))
    label["labeling_model"] = model
    return label


def apply_hybrid_corrections(
    label: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    hybrid_hints = item.get("hybrid_hints")
    if not isinstance(hybrid_hints, dict):
        hybrid_hints = {}

    rule_hints = hybrid_hints.get("rule_hints", {})
    market_hints = hybrid_hints.get("market_hints", {})

    if label.get("event_type") not in {"NONE", "ERROR"}:
        # surprise: prefer market-derived surprise, then LLM hint
        market_surprise = market_hints.get("surprise")
        if market_surprise is not None:
            label["surprise"] = clamp_float(market_surprise, -1.0, 1.0)
        elif label.get("surprise") is None and rule_hints.get("novelty") is not None:
            # leave surprise empty unless market evidence exists; do not invent it from novelty
            pass

        # novelty: rule-based when LLM omitted it
        for field in ("novelty",):
            if label.get(field) is None and rule_hints.get(field) is not None:
                label[field] = rule_hints.get(field)

    label["impact_profile"] = {
        "polarity": label.get("polarity"),
        "surprise": label.get("surprise"),
        "scope": label.get("scope"),
        "novelty": label.get("novelty"),
        "credibility": label.get("credibility"),
    }

    if isinstance(hybrid_hints, dict) and hybrid_hints:
        label["hybrid_signals"] = item.get("hybrid_signals", {})

    return label


def make_error_label(
    item: dict[str, Any], model: str, error: str, api_key_index: int
) -> dict[str, Any]:
    return {
        "headline": str(item.get("headline", "")),
        "article": str(item.get("article", "")),
        "source_date": str(item.get("date", "")),
        "source_ticker": str(item.get("ticker", "")),
        "event_type": "ERROR",
        "subject": None,
        "action": None,
        "object": None,
        "magnitude": None,
        "polarity": None,
        "surprise": None,
        "scope": None,
        "novelty": None,
        "credibility": None,
        "affected_tickers": [],
        "impact_profile": {
            "polarity": None,
            "surprise": None,
            "scope": None,
            "novelty": None,
            "credibility": None,
        },
        "hybrid_signals": item.get("hybrid_signals", {}),
        "error": error,
        "labeling_model": model,
        "api_key_index": api_key_index,
    }


def label_one_item(
    item: dict[str, Any],
    api_key: str,
    api_key_index: int,
    model: str,
    base_url: str,
    use_article: bool,
    max_article_chars: int,
    max_retries: int,
    retry_backoff: float,
    request_timeout: float,
) -> dict[str, Any]:
    client = get_openai_client(api_key=api_key, base_url=base_url)
    news_input = build_news_input(item, use_article, max_article_chars)
    prompt = LABELING_PROMPT.format(news_input=news_input)
    hybrid_hints = item.get("hybrid_hints")
    if isinstance(hybrid_hints, dict) and hybrid_hints:
        prompt += "\n\nHybrid hints (soft evidence; do not ignore the article):\n"
        prompt += json.dumps(hybrid_hints, ensure_ascii=False, indent=2)
        prompt += (
            "\nInstruction: polarity and scope must be judged directly from the news text; "
            "use the hints only for surprise/novelty calibration. Credibility must be judged from source metadata and article provenance."
        )

    last_error = "unknown error"
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                timeout=request_timeout,
            )
            raw_content = response.choices[0].message.content
            content = "" if raw_content is None else str(raw_content)
            json_text = extract_json_text(content)
            if not json_text:
                raise ValueError("empty response content")

            label = json.loads(json_text)
            if not isinstance(label, dict):
                raise ValueError("response is not a JSON object")
            normalized = normalize_label_payload(label, item, model)
            normalized = apply_hybrid_corrections(normalized, item)
            normalized["api_key_index"] = api_key_index
            return normalized
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            sleep_seconds = retry_backoff * math.pow(2, attempt)
            time.sleep(sleep_seconds)

    return make_error_label(item, model, last_error, api_key_index)


def call_llm_api(
    items: list[dict[str, Any]],
    api_key: str,
    api_key_index: int,
    model: str = "gpt-5.4",
    base_url: str = API_URL,
    use_article: bool = False,
    max_article_chars: int = 1200,
    max_retries: int = 2,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    request_timeout: float = 60.0,
    inter_request_delay: float = 0.0,
) -> list[dict[str, Any]]:
    """调用LLM API进行批量标注"""
    results: list[dict[str, Any]] = []
    for item in items:
        results.append(
            label_one_item(
                item=item,
                api_key=api_key,
                api_key_index=api_key_index,
                model=model,
                base_url=base_url,
                use_article=use_article,
                max_article_chars=max_article_chars,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                request_timeout=request_timeout,
            )
        )
        if inter_request_delay > 0:
            time.sleep(inter_request_delay)
    return results


def process_batches_parallel(
    batches: list[list[dict[str, Any]]],
    api_keys: list[str],
    model: str,
    base_url: str,
    use_article: bool,
    max_article_chars: int,
    max_retries: int,
    retry_backoff: float,
    request_timeout: float,
    inter_request_delay: float,
    max_workers: int,
):
    if not batches:
        return

    safe_max_workers = max(1, min(max_workers, len(api_keys), len(batches)))
    with ThreadPoolExecutor(max_workers=safe_max_workers) as executor:
        future_to_meta = {}
        for batch_index, batch in enumerate(batches):
            api_key_index = batch_index % len(api_keys)
            future = executor.submit(
                call_llm_api,
                batch,
                api_keys[api_key_index],
                api_key_index,
                model,
                base_url,
                use_article,
                max_article_chars,
                max_retries,
                retry_backoff,
                request_timeout,
                inter_request_delay,
            )
            future_to_meta[future] = (batch_index, batch)

        for future in as_completed(future_to_meta):
            batch_index, batch = future_to_meta[future]
            yield batch_index, batch, future


def main():
    parser = argparse.ArgumentParser(description="LLM事件标注")
    parser.add_argument("--input", required=True, help="输入新闻文件路径")
    parser.add_argument("--output", required=True, help="输出标注文件路径")
    parser.add_argument("--model", default="gpt-5.4", help="LLM模型名")
    parser.add_argument("--base-url", default=API_URL)
    parser.add_argument("--api-keys", default=None, help="逗号分隔的多个API Key")
    parser.add_argument(
        "--api-key-file",
        default=None,
        help="每行一个API Key的文本文件路径",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并发批次数；建议从API key数量或较小值开始压测",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="单条样本最大重试次数",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help="失败重试的基础退避秒数，后续重试按指数退避",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="单次请求超时时间（秒）",
    )
    parser.add_argument(
        "--inter-request-delay",
        type=float,
        default=0.0,
        help="同一worker内相邻请求的间隔秒数",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="输出文件已存在时跳过已完成样本并以追加模式续跑",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="忽略已有输出文件并重新覆盖写出",
    )
    parser.add_argument(
        "--use-article",
        action="store_true",
        help="启用标题+正文片段模式（默认仅标题）",
    )
    parser.add_argument(
        "--max-article-chars",
        type=int,
        default=1200,
        help="正文截断长度（字符）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="起始时间(包含)，如 2020-01-01 或 2020-01-01 00:00:00",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束时间(包含)，如 2020-12-31 或 2020-12-31 23:59:59",
    )
    parser.add_argument(
        "--date-field",
        default=None,
        help="指定时间字段名；默认自动尝试 Date/date/pub_time/timestamp/time/created_at",
    )
    parser.add_argument(
        "--market-data-file",
        default=None,
        help="可选市场数据文件（csv/jsonl），用于按 ticker/date 注入实际值、预期值、来源可信度等校正信号",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="关闭 LLM + 规则/市场数据混合模式，只保留纯 LLM 标注",
    )
    args = parser.parse_args()

    api_keys = load_api_keys(args.api_keys, args.api_key_file)
    if not api_keys:
        raise ValueError(
            "请先在脚本顶部设置 API_KEY，或通过 --api-keys / --api-key-file 提供"
        )

    input_path = Path(args.input)
    items = load_news_items(
        input_path=input_path,
        max_samples=args.max_samples,
        start_date=args.start_date,
        end_date=args.end_date,
        date_field=args.date_field,
    )
    print(f"共 {len(items)} 条新闻待标注")

    hybrid_enabled = not args.no_hybrid
    context_index = load_context_index(args.market_data_file) if hybrid_enabled else {}
    if hybrid_enabled:
        novelty_encoder = get_novelty_encoder()
        items = build_hybrid_signals(
            items,
            context_index,
            novelty_encoder=novelty_encoder,
        )
        print(
            f"混合模式已启用: market_context={len(context_index)} 条, "
            f"可用于规则/市场数据校正"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and args.resume:
        raise ValueError("--overwrite 和 --resume 不能同时使用")

    completed_keys: set[tuple[str, str, str]] = set()
    if args.resume and output_path.exists():
        completed_keys = load_completed_keys(output_path)
        pending_items = filter_pending_items(items, completed_keys)
        print(f"检测到续跑模式，跳过 {len(completed_keys)} 条已完成样本")
    else:
        pending_items = items

    print(f"本次实际待标注: {len(pending_items)} 条")
    safe_max_workers = max(
        1,
        min(
            args.max_workers,
            len(api_keys),
            len(batches := chunk_items(pending_items, args.batch_size)),
        ),
    )
    print(f"使用 {len(api_keys)} 个API key, max_workers={safe_max_workers}")

    if not pending_items:
        print("无需新增标注，任务结束")
        return

    file_mode = "a" if args.resume and output_path.exists() else "w"
    total_labeled = 0
    total_errors = 0
    write_lock = threading.Lock()

    with output_path.open(file_mode, encoding="utf-8") as fout:
        for batch_index, batch, future in process_batches_parallel(
            batches=batches,
            api_keys=api_keys,
            model=args.model,
            base_url=args.base_url,
            use_article=args.use_article,
            max_article_chars=args.max_article_chars,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            request_timeout=args.request_timeout,
            inter_request_delay=args.inter_request_delay,
            max_workers=safe_max_workers,
        ):
            try:
                results = future.result()
            except Exception as exc:
                results = [
                    make_error_label(
                        item=item,
                        model=args.model,
                        error=f"batch_failed: {exc}",
                        api_key_index=batch_index % len(api_keys),
                    )
                    for item in batch
                ]

            batch_errors = sum(1 for row in results if row.get("event_type") == "ERROR")
            with write_lock:
                for row in results:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

            total_labeled += len(results)
            total_errors += batch_errors
            print(
                f"  已标注: {total_labeled}/{len(pending_items)} "
                f"(batch={batch_index + 1}/{len(batches)}, 本批错误: {batch_errors}/{len(batch)}, 累计错误: {total_errors})"
            )

    print(f"\n标注完成! 输出: {output_path}")
    print(f"总计新增: {total_labeled} 条")
    if args.resume:
        print(f"续跑前已完成: {len(completed_keys)} 条")
    print(f"累计错误: {total_errors} 条")


if __name__ == "__main__":
    main()
