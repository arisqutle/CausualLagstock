"""Build the Phase 3 tensor dataset from labeled events, embeddings, prices, and Stage 2 graphs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.phase1.event_types import EVENT_TYPE_TO_ID

PROFILE_FIELDS = ("surprise", "scope", "novelty", "credibility")
SCOPE_TO_VALUE = {
    "single_stock": 0.25,
    "single": 0.25,
    "stock": 0.25,
    "company": 0.25,
    "sector": 0.50,
    "industry": 0.50,
    "market": 0.75,
    "macro": 0.75,
    "global": 1.00,
    "world": 1.00,
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_price_tables(price_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for path in price_dir.glob("*.csv"):
        df = pd.read_csv(path)
        df.columns = [col.strip().lower().replace("_", " ") for col in df.columns]
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
        df = df.sort_values("date").set_index("date")
        tables[path.stem] = df
    return tables


def latest_causal_snapshot(stage2_dir: Path) -> Path:
    paths = list(stage2_dir.glob("causal_epoch_*.pt"))
    keyed = []
    for path in paths:
        match = re.search(r"causal_epoch_(\d+)\.pt$", path.name)
        if match:
            keyed.append((int(match.group(1)), path))
    if not keyed:
        raise FileNotFoundError(f"No causal_epoch_*.pt files found in {stage2_dir}")
    return max(keyed, key=lambda x: x[0])[1]


def extract_graph(payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if "causal_matrix" in payload and "lag_matrix" in payload:
        return payload["causal_matrix"].float(), payload["lag_matrix"].float()
    if "causal_info" in payload:
        info = payload["causal_info"]
        return info["causal_matrix"].float(), info["lag_matrix"].float()
    if "adjacency_raw" in payload and "lag_matrix" in payload:
        return torch.as_tensor(payload["adjacency_raw"]).float(), torch.as_tensor(payload["lag_matrix"]).float()
    if "stacd.layers.0.causal_raw" in payload and "stacd.layers.0.lag_raw" in payload:
        causal_matrix = torch.sigmoid(payload["stacd.layers.0.causal_raw"]).float()
        lag_matrix = torch.nn.functional.softplus(payload["stacd.layers.0.lag_raw"]).float()
        return causal_matrix, lag_matrix
    if "causal_raw" in payload and "lag_raw" in payload:
        causal_matrix = torch.sigmoid(payload["causal_raw"]).float()
        lag_matrix = torch.nn.functional.softplus(payload["lag_raw"]).float()
        return causal_matrix, lag_matrix
    if "T" in payload:
        lag_matrix = payload["T"].float()
        causal_matrix = torch.ones_like(lag_matrix)
        causal_matrix.fill_diagonal_(0.0)
        return causal_matrix, lag_matrix
    raise KeyError("Causal snapshot needs causal_matrix/lag_matrix or causal_info with both tensors.")


def load_stage2_graph(stage2_dir: Path, graph_path: Path | None) -> tuple[torch.Tensor, torch.Tensor, Path]:
    best_graph = stage2_dir / "best_graph.pt"
    best_model = stage2_dir / "best_model.pt"
    candidates = [graph_path] if graph_path is not None else [best_graph, latest_causal_snapshot(stage2_dir), best_model]
    last_error: Exception | None = None
    for path in candidates:
        if path is None or not path.exists():
            continue
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        try:
            causal_matrix, lag_matrix = extract_graph(payload)
            return causal_matrix, lag_matrix, path
        except KeyError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise KeyError("No usable Stage 2 causal graph found.")


def event_magnitude(row: dict) -> float:
    magnitude = row["magnitude"]
    if magnitude is None:
        return 0.0
    polarity = row["polarity"]
    sign = -1.0 if polarity == "negative" else 1.0
    return sign * float(magnitude)


def numeric_profile(row: dict) -> torch.Tensor:
    profile = row.get("impact_profile")
    if not isinstance(profile, dict):
        profile = {}
    values = []
    for field in PROFILE_FIELDS:
        value = row.get(field)
        if value is None:
            value = profile.get(field)
        if field == "scope" and isinstance(value, str):
            text = value.strip().lower()
            if text not in SCOPE_TO_VALUE:
                if "single" in text or "company" in text or "stock" in text:
                    text = "single_stock"
                elif "sector" in text or "industry" in text:
                    text = "sector"
                elif "market" in text or "macro" in text:
                    text = "market"
                elif "global" in text or "world" in text:
                    text = "global"
            value = SCOPE_TO_VALUE.get(text, 0.0)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


def price_window_and_label(
    table: pd.DataFrame,
    event_time: pd.Timestamp,
    price_window: int,
    label_threshold: float,
    horizons: tuple[int, ...],
) -> tuple[torch.Tensor, list[int], list[float], list[float]]:
    dates = table.index[table.index <= event_time]
    if len(dates) < price_window:
        raise IndexError("Not enough price history before event timestamp")
    current_dt = dates[-1]
    loc = table.index.get_loc(current_dt)
    window = table.iloc[loc - price_window + 1 : loc + 1][["open", "high", "low", "close", "volume"]].astype("float32")
    current_close = float(table.iloc[loc]["close"])
    directions = []
    returns = []
    label_end_timestamps = []
    for horizon in horizons:
        if loc + horizon >= len(table):
            directions.append(-100)
            returns.append(float("nan"))
            label_end_timestamps.append(float("nan"))
            continue
        future_close = float(table.iloc[loc + horizon]["close"])
        label_end_timestamps.append(float(table.index[loc + horizon].timestamp()))
        ret = (future_close - current_close) / current_close
        if ret > label_threshold:
            direction = 0
        elif ret < -label_threshold:
            direction = 1
        else:
            direction = 2
        directions.append(direction)
        returns.append(float(ret))
    if all(direction == -100 for direction in directions):
        raise IndexError("No future trading day for any requested horizon")
    return (
        torch.tensor(window.to_numpy(), dtype=torch.float32),
        directions,
        returns,
        label_end_timestamps,
    )


def select_horizon_labels(
    event_types: torch.Tensor,
    causal_matrix: torch.Tensor,
    lag_matrix: torch.Tensor,
    horizons: tuple[int, ...],
    directions: list[int],
    returns: list[float],
    sigma: float,
    label_end_timestamps: list[float],
) -> tuple[int, float, int, float, torch.Tensor]:
    target_type = int(event_types[-1].item())
    source_types = event_types.long()
    horizon_tensor = torch.tensor(horizons, dtype=torch.float32)
    source_lags = lag_matrix[target_type, source_types]
    source_strength = causal_matrix[source_types, target_type]
    lag_distance = horizon_tensor.view(1, -1) - source_lags.view(-1, 1)
    gates = torch.exp(-0.5 * (lag_distance ** 2) / (float(sigma) ** 2))
    weights = (source_strength.view(-1, 1) * gates).sum(dim=0)
    valid = torch.tensor([direction != -100 for direction in directions], dtype=torch.bool)
    weights = weights.masked_fill(~valid, -1.0)
    selected_idx = int(weights.argmax().item())
    return (
        directions[selected_idx],
        returns[selected_idx],
        horizons[selected_idx],
        label_end_timestamps[selected_idx],
        weights.clamp_min(0.0),
    )


def build_examples(
    labels: list[dict],
    embeddings: torch.Tensor,
    prices: dict[str, pd.DataFrame],
    seq_len: int,
    price_window: int,
    label_threshold: float,
    context_mode: str,
    horizons: tuple[int, ...],
    causal_matrix: torch.Tensor,
    lag_matrix: torch.Tensor,
    horizon_select_sigma: float,
    num_workers: int,
) -> dict[str, torch.Tensor]:
    valid = []
    for i, row in enumerate(labels):
        if row["event_type"] not in EVENT_TYPE_TO_ID or not row["affected_tickers"]:
            continue
        event_time = pd.to_datetime(row["source_date"], utc=True).tz_convert(None)
        row["_event_time"] = event_time
        row["_event_timestamp"] = event_time.timestamp()
        valid.append((i, row))
    valid.sort(key=lambda item: item[1]["_event_time"])

    if context_mode == "global":
        targets = []
        for end in range(seq_len - 1, len(valid)):
            seq = valid[end - seq_len + 1 : end + 1]
            ticker = seq[-1][1]["affected_tickers"][0]
            targets.append((ticker, seq))
    elif context_mode == "ticker":
        by_ticker: dict[str, list[tuple[int, dict]]] = {}
        for item in valid:
            _, row = item
            for ticker in row["affected_tickers"]:
                by_ticker.setdefault(ticker, []).append(item)
        targets = []
        for ticker in sorted(by_ticker):
            rows = by_ticker[ticker]
            for end in range(seq_len - 1, len(rows)):
                targets.append((ticker, rows[end - seq_len + 1 : end + 1]))
        targets.sort(key=lambda item: item[1][-1][1]["_event_time"])
    else:
        raise ValueError("context_mode must be 'global' or 'ticker'")

    stock_tickers = sorted({ticker for ticker, _ in targets if ticker in prices})
    stock_to_id = {ticker: i for i, ticker in enumerate(stock_tickers)}

    text_embeddings = []
    event_types = []
    timestamps = []
    magnitudes = []
    event_profiles = []
    event_age_days = []
    price_history = []
    direction_labels = []
    magnitude_labels = []
    direction_labels_by_horizon = []
    magnitude_labels_by_horizon = []
    selected_horizons = []
    horizon_selection_weights = []
    prediction_timestamps = []
    label_end_timestamps = []
    tickers = []
    target_stock_ids = []
    label_row_indices = []

    def build_one(target_item: tuple[str, list[tuple[int, dict]]]):
        ticker, seq = target_item
        idxs = [idx for idx, _ in seq]
        rows = [row for _, row in seq]
        target = rows[-1]
        if ticker not in prices:
            return None

        event_time = target["_event_time"]
        try:
            px, directions, returns, future_timestamps = price_window_and_label(
                prices[ticker],
                event_time,
                price_window,
                label_threshold,
                horizons,
            )
        except IndexError:
            return None

        ts = torch.tensor([row["_event_timestamp"] for row in rows], dtype=torch.float32)
        ts = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
        raw_event_times = [row["_event_time"] for row in rows]
        ages = torch.tensor([(event_time - row_time).total_seconds() / 86400.0 for row_time in raw_event_times], dtype=torch.float32)
        event_type_tensor = torch.tensor([EVENT_TYPE_TO_ID[row["event_type"]] for row in rows], dtype=torch.long)
        direction, ret, selected_horizon, label_end_timestamp, horizon_weights = select_horizon_labels(
            event_type_tensor,
            causal_matrix,
            lag_matrix,
            horizons,
            directions,
            returns,
            horizon_select_sigma,
            future_timestamps,
        )
        return {
            "text_embeddings": embeddings[idxs].float(),
            "event_types": event_type_tensor,
            "timestamps": ts,
            "magnitudes": torch.tensor([event_magnitude(row) for row in rows], dtype=torch.float32),
            "event_profiles": torch.stack([numeric_profile(row) for row in rows]),
            "event_age_days": ages,
            "price_history": px,
            "direction_label": direction,
            "magnitude_label": ret,
            "direction_labels_by_horizon": torch.tensor(directions, dtype=torch.long),
            "magnitude_labels_by_horizon": torch.tensor(returns, dtype=torch.float32),
            "selected_horizon": selected_horizon,
            "horizon_selection_weights": horizon_weights.float(),
            "prediction_timestamp": event_time.timestamp(),
            "label_end_timestamp": label_end_timestamp,
            "ticker": ticker,
            "target_stock_id": stock_to_id[ticker],
            "label_row_index": idxs[-1],
        }

    def append_result(result: dict | None) -> None:
        if result is None:
            return
        text_embeddings.append(result["text_embeddings"])
        event_types.append(result["event_types"])
        timestamps.append(result["timestamps"])
        magnitudes.append(result["magnitudes"])
        event_profiles.append(result["event_profiles"])
        event_age_days.append(result["event_age_days"])
        price_history.append(result["price_history"])
        direction_labels.append(result["direction_label"])
        magnitude_labels.append(result["magnitude_label"])
        direction_labels_by_horizon.append(result["direction_labels_by_horizon"])
        magnitude_labels_by_horizon.append(result["magnitude_labels_by_horizon"])
        selected_horizons.append(result["selected_horizon"])
        horizon_selection_weights.append(result["horizon_selection_weights"])
        prediction_timestamps.append(result["prediction_timestamp"])
        label_end_timestamps.append(result["label_end_timestamp"])
        tickers.append(result["ticker"])
        target_stock_ids.append(result["target_stock_id"])
        label_row_indices.append(result["label_row_index"])

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(build_one, item) for item in targets]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"building phase3 dataset ({context_mode}, workers={num_workers})"):
                append_result(future.result())
    else:
        for item in tqdm(targets, desc=f"building phase3 dataset ({context_mode})"):
            append_result(build_one(item))

    if not text_embeddings:
        raise RuntimeError("No Phase 3 examples were built. Check label dates, tickers, prices, and embeddings.")

    return {
        "text_embeddings": torch.stack(text_embeddings),
        "event_types": torch.stack(event_types),
        "timestamps": torch.stack(timestamps),
        "magnitudes": torch.stack(magnitudes),
        "event_profiles": torch.stack(event_profiles),
        "event_profile_fields": list(PROFILE_FIELDS),
        "event_age_days": torch.stack(event_age_days),
        "price_history": torch.stack(price_history),
        "direction_labels": torch.tensor(direction_labels, dtype=torch.long),
        "magnitude_labels": torch.tensor(magnitude_labels, dtype=torch.float32),
        "direction_labels_by_horizon": torch.stack(direction_labels_by_horizon),
        "magnitude_labels_by_horizon": torch.stack(magnitude_labels_by_horizon),
        "selected_horizons": torch.tensor(selected_horizons, dtype=torch.long),
        "horizon_selection_weights": torch.stack(horizon_selection_weights),
        "horizons": torch.tensor(horizons, dtype=torch.long),
        "prediction_timestamps": torch.tensor(prediction_timestamps, dtype=torch.float64),
        "label_end_timestamps": torch.tensor(label_end_timestamps, dtype=torch.float64),
        "tickers": tickers,
        "target_stock_ids": torch.tensor(target_stock_ids, dtype=torch.long),
        "stock_tickers": stock_tickers,
        "label_row_indices": torch.tensor(label_row_indices, dtype=torch.long),
        "context_mode": context_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Direction A Phase 3 dataset")
    parser.add_argument("--labels", default="data/labels/silver_labels_test_upgraded.jsonl")
    parser.add_argument("--embeddings", default="data/embeddings/emb.pt")
    parser.add_argument("--price-dir", default="data/sp100_prices")
    parser.add_argument("--stage2-dir", default="data/stage2")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--output", default="data/stage3/phase3_dataset.pt")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--price-window", type=int, default=30)
    parser.add_argument("--label-threshold", type=float, default=0.005)
    parser.add_argument("--context-mode", choices=["ticker", "global"], default="ticker")
    parser.add_argument("--horizons", default="1,3,5,7,10")
    parser.add_argument("--horizon-select-sigma", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())

    labels = load_jsonl(Path(args.labels))
    embeddings = torch.load(args.embeddings, map_location="cpu")
    prices = load_price_tables(Path(args.price_dir))
    causal_matrix, lag_matrix, graph_path = load_stage2_graph(Path(args.stage2_dir), Path(args.graph) if args.graph else None)

    data = build_examples(
        labels,
        embeddings,
        prices,
        seq_len=args.seq_len,
        price_window=args.price_window,
        label_threshold=args.label_threshold,
        context_mode=args.context_mode,
        horizons=horizons,
        causal_matrix=causal_matrix,
        lag_matrix=lag_matrix,
        horizon_select_sigma=args.horizon_select_sigma,
        num_workers=max(1, int(args.num_workers)),
    )
    data["causal_matrix"] = causal_matrix
    data["lag_matrix"] = lag_matrix
    data["stage2_graph_path"] = str(graph_path)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output, _use_new_zipfile_serialization=False)
    print(f"wrote {output}")
    print(f"n={data['direction_labels'].shape[0]} graph={graph_path}")
    print(f"context_mode={data['context_mode']} n_stocks={len(data['stock_tickers'])}")


if __name__ == "__main__":
    main()
