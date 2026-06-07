from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import yfinance as yf
from tqdm import tqdm


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


def meta_tickers(path: Path, *, valid_only: bool) -> Counter[str]:
    rows: list[dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    for row in rows:
        ticker = row.get("ticker")
        event_type = row.get("event_type")
        if not isinstance(ticker, str) or not ticker:
            continue
        if valid_only and (not isinstance(event_type, str) or event_type not in EVENT_TYPE_MAP):
            continue
        counts[ticker] += 1
    return counts


def write_ticker_manifest(path: Path, counts: Counter[str]) -> None:
    lines = [f"{ticker}\t{count}" for ticker, count in counts.most_common()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_price_csv(df, output_path: Path) -> None:
    out = df.reset_index()
    first = str(out.columns[0]).strip().lower()
    if first == "index":
        out = out.rename(columns={out.columns[0]: "Date"})
    out.to_csv(output_path, index=False)


def download_batch(
    tickers: list[str],
    *,
    output_dir: Path,
    start_date: str,
    end_date: str,
) -> list[str]:
    failed: list[str] = []
    if not tickers:
        return failed

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        threads=False,
        progress=True,
    )
    if data is None:
        return list(tickers)
    for ticker in tqdm(tickers, desc="writing price csv"):
        output_path = output_dir / f"{ticker}.csv"
        if len(tickers) == 1:
            df = data
        elif ticker in data.columns.get_level_values(0):
            df = data[ticker]
        else:
            failed.append(ticker)
            continue
        if len(df.dropna(how="all")) == 0:
            failed.append(ticker)
            continue
        write_price_csv(df, output_path)
    return failed


def fallback_download(
    tickers: list[str],
    *,
    output_dir: Path,
    start_date: str,
    end_date: str,
    sleep_s: float,
) -> list[str]:
    failed: list[str] = []
    for ticker in tickers:
        output_path = output_dir / f"{ticker}.csv"
        if output_path.exists():
            continue
        try:
            df = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
            )
        except Exception:
            failed.append(ticker)
            continue
        if len(df) == 0:
            failed.append(ticker)
            continue
        write_price_csv(df, output_path)
        time.sleep(sleep_s)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download prices for tickers referenced by Phase 2 meta.json")
    parser.add_argument("--meta", default="data/embeddings/meta.json")
    parser.add_argument("--output-dir", default="data/meta_prices")
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--all-events", action="store_true", help="Include all tickers from meta.json, not just the 20 valid Phase 2 event types")
    parser.add_argument("--dry-run", action="store_true", help="Only write ticker manifests without downloading prices")
    args = parser.parse_args()

    counts = meta_tickers(Path(args.meta), valid_only=not args.all_events)
    selected = [ticker for ticker, count in counts.most_common() if count >= args.min_count]
    if args.top_k is not None:
        selected = selected[: args.top_k]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ticker_manifest(output_dir / "selected_tickers.txt", Counter({ticker: counts[ticker] for ticker in selected}))
    write_ticker_manifest(output_dir / "all_tickers.txt", counts)

    if args.dry_run:
        print(
            f"selected={len(selected)} existing={len(list(output_dir.glob('*.csv')))} mode={'all-events' if args.all_events else 'valid-only'}"
        )
        return

    pending = [ticker for ticker in selected if not (output_dir / f"{ticker}.csv").exists()]
    failed: list[str] = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        try:
            failed.extend(
                download_batch(
                    batch,
                    output_dir=output_dir,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
            )
        except Exception:
            failed.extend(batch)
        time.sleep(args.sleep)

    failed = sorted(set(failed))
    failed = fallback_download(
        failed,
        output_dir=output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        sleep_s=args.sleep,
    )

    if failed:
        (output_dir / "failed_tickers.txt").write_text("\n".join(sorted(set(failed))) + "\n", encoding="utf-8")
    print(
        f"requested={len(selected)} saved={len(list(output_dir.glob('*.csv')))} failed={len(set(failed))} mode={'all-events' if args.all_events else 'valid-only'}"
    )


if __name__ == "__main__":
    main()
