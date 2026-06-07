"""Download Yahoo price histories for tickers referenced by the Phase 3 labels."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import yfinance as yf
from tqdm import tqdm


def label_tickers(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["event_type"] in {"NONE", "ERROR"}:
                continue
            for ticker in row["affected_tickers"]:
                counts[ticker] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download prices for Phase 3 label tickers")
    parser.add_argument("--labels", default="data/labels/silver_labels_test_upgraded.jsonl")
    parser.add_argument("--output-dir", default="data/sp100_prices")
    parser.add_argument("--start-date", default="2009-01-01")
    parser.add_argument("--end-date", default="2024-01-10")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    tickers = label_tickers(Path(args.labels))
    selected = [ticker for ticker, count in tickers.most_common() if count >= args.min_count]
    if args.top_k is not None:
        selected = selected[:args.top_k]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending = [ticker for ticker in selected if not (output_dir / f"{ticker}.csv").exists()]
    failed = []
    if pending:
        data = yf.download(
            pending,
            start=args.start_date,
            end=args.end_date,
            group_by="ticker",
            auto_adjust=False,
            threads=False,
            progress=True,
        )
        for ticker in tqdm(pending, desc="writing price csv"):
            output_path = output_dir / f"{ticker}.csv"
            if len(pending) == 1:
                df = data
            elif ticker in data.columns.get_level_values(0):
                df = data[ticker]
            else:
                failed.append(ticker)
                continue
            if len(df.dropna(how="all")) == 0:
                failed.append(ticker)
                continue
            df.reset_index().to_csv(output_path, index=False)
    for ticker in selected:
        output_path = output_dir / f"{ticker}.csv"
        if output_path.exists():
            continue
        try:
            df = yf.Ticker(ticker).history(start=args.start_date, end=args.end_date, auto_adjust=False)
        except Exception:
            failed.append(ticker)
            continue
        if len(df) == 0:
            failed.append(ticker)
            continue
        df.reset_index().to_csv(output_path, index=False)
        time.sleep(args.sleep)

    if failed:
        (output_dir / "failed_tickers.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"requested={len(selected)} saved={len(list(output_dir.glob('*.csv')))} failed={len(failed)}")


if __name__ == "__main__":
    main()
