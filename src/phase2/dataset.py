from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

EVENT_TYPE_MAP = {
    "M1": 0,
    "M2": 1,
    "M3": 2,
    "M4": 3,
    "M5": 4,
    "M6": 5,
    "C1": 6,
    "C2": 7,
    "C3": 8,
    "C4": 9,
    "C5": 10,
    "C6": 11,
    "C7": 12,
    "C8": 13,
    "K1": 14,
    "K2": 15,
    "K3": 16,
    "K4": 17,
    "G1": 18,
    "G2": 19,
}


class FastEventDataset(Dataset):
    def __init__(
        self,
        emb_path: str | Path,
        meta_path: str | Path,
        price_dir: str | Path,
        seq_len: int = 32,
        label_threshold: float = 0.005,
        horizons: list[int] | tuple[int, ...] = (1, 3, 5, 7, 10),
    ):
        self.emb = torch.load(Path(emb_path), map_location="cpu").float()
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.seq_len = int(seq_len)
        self.label_threshold = float(label_threshold)
        self.horizons = tuple(int(h) for h in horizons)
        self.price_data = self._load_prices(Path(price_dir))
        (
            self.examples,
            self.example_timestamps,
            self.example_label_end_timestamps,
        ) = self._build_examples()

    @staticmethod
    def _parse_time(value: str) -> pd.Timestamp:
        return pd.to_datetime(value, utc=True).tz_convert(None)

    @staticmethod
    def _load_prices(price_dir: Path) -> dict[str, pd.DataFrame]:
        tables = {}
        for path in price_dir.glob("*.csv"):
            df = pd.read_csv(path)
            df.columns = [col.strip().lower().replace("_", " ") for col in df.columns]
            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
            df = df.sort_values("date").set_index("date")
            tables[path.stem] = df
        return tables

    def _labels_for(self, ticker: str, timestamp: pd.Timestamp) -> tuple[list[int], list[float]]:
        table = self.price_data.get(ticker)
        if table is None:
            return [-100 for _ in self.horizons], [float("nan") for _ in self.horizons]

        dates = table.index[table.index <= timestamp]
        if len(dates) == 0:
            return [-100 for _ in self.horizons], [float("nan") for _ in self.horizons]
        current_dt = dates[-1]
        loc = table.index.get_loc(current_dt)

        current_close = float(table.iloc[loc]["close"])
        labels = []
        label_end_timestamps = []
        for horizon in self.horizons:
            if loc + horizon >= len(table):
                labels.append(-100)
                label_end_timestamps.append(float("nan"))
                continue
            future_close = float(table.iloc[loc + horizon]["close"])
            label_end_timestamps.append(float(table.index[loc + horizon].timestamp()))
            ret = (future_close - current_close) / current_close
            if ret > self.label_threshold:
                labels.append(0)
            elif ret < -self.label_threshold:
                labels.append(1)
            else:
                labels.append(2)
        return labels, label_end_timestamps

    def _build_examples(
        self,
    ) -> tuple[list[tuple[list[int], list[int]]], list[float], list[float]]:
        by_ticker: dict[str, list[tuple[pd.Timestamp, int]]] = {}
        for idx, row in enumerate(self.meta):
            event_type = row["event_type"]
            ticker = row["ticker"]
            if event_type not in EVENT_TYPE_MAP or ticker not in self.price_data:
                continue
            by_ticker.setdefault(ticker, []).append((self._parse_time(row["timestamp"]), idx))

        examples = []
        example_timestamps = []
        example_label_end_timestamps = []
        for ticker in sorted(by_ticker):
            rows = sorted(by_ticker[ticker], key=lambda x: x[0])
            for end in range(self.seq_len - 1, len(rows)):
                seq = rows[end - self.seq_len + 1 : end + 1]
                target_time, _ = seq[-1]
                labels, label_end_timestamps = self._labels_for(ticker, target_time)
                if all(label == -100 for label in labels):
                    continue
                examples.append(([idx for _, idx in seq], labels))
                example_timestamps.append(float(target_time.timestamp()))
                valid_label_ends = [
                    value
                    for label, value in zip(labels, label_end_timestamps)
                    if label != -100
                ]
                example_label_end_timestamps.append(max(valid_label_ends))
        return examples, example_timestamps, example_label_end_timestamps

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def encode_time(value: str) -> float:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC").timestamp()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idxs, labels = self.examples[idx]
        rows = [self.meta[i] for i in idxs]
        timestamps = torch.tensor([self.encode_time(row["timestamp"]) for row in rows], dtype=torch.float32)
        timestamps = (timestamps - timestamps.min()) / 86400.0

        return {
            "text_emb": self.emb[idxs],
            "event_types": torch.tensor([EVENT_TYPE_MAP[row["event_type"]] for row in rows], dtype=torch.long),
            "timestamps": timestamps,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
