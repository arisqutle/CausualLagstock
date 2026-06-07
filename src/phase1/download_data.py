"""
CausalStock 数据下载脚本
========================
下载所有实验所需的数据集:
1. FNSPID (HuggingFace) - 金融新闻 + 价格
2. EDT (GitHub) - 事件标注数据
3. StockNet (GitHub) - Twitter + 价格
4. S&P 100 股价 (yfinance) - 最新价格补充
5. S&P 100 行业分类 (yfinance) - 构建股票关系图
"""

import subprocess
import argparse
from pathlib import Path


# ============================================================
# 配置
# ============================================================
DATA_ROOT = Path(__file__).parent.parent / "data"

FNSPID_HF_REPO = "Zihan1004/FNSPID"
EDT_GITHUB_URL = "https://github.com/Zhihan1996/TradeTheEvent.git"
STOCKNET_GITHUB_URL = "https://github.com/yumoxu/stocknet-dataset.git"

# S&P 100 成分股 (截至2024年)
SP100_TICKERS = [
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AIG",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "AVGO",
    "AXP",
    "BA",
    "BAC",
    "BK",
    "BKNG",
    "BLK",
    "BMY",
    "BRK-B",
    "C",
    "CAT",
    "CHTR",
    "CL",
    "CMCSA",
    "COF",
    "COP",
    "COST",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DOW",
    "DUK",
    "EMR",
    "EXC",
    "F",
    "FDX",
    "GD",
    "GE",
    "GILD",
    "GM",
    "GOOG",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KHC",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "MA",
    "MCD",
    "MDLZ",
    "MDT",
    "MET",
    "META",
    "MMM",
    "MO",
    "MRK",
    "MS",
    "MSFT",
    "NEE",
    "NFLX",
    "NKE",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PM",
    "PYPL",
    "QCOM",
    "RTX",
    "SBUX",
    "SCHW",
    "SO",
    "SPG",
    "T",
    "TGT",
    "TMO",
    "TMUS",
    "TXN",
    "UNH",
    "UNP",
    "UPS",
    "USB",
    "V",
    "VZ",
    "WBA",
    "WFC",
    "WMT",
    "XOM",
]


# ============================================================
# Step 1: 下载 FNSPID
# ============================================================
def download_fnspid():
    """从HuggingFace下载FNSPID数据集"""
    fnspid_dir = DATA_ROOT / "fnspid"
    fnspid_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [
        p for p in fnspid_dir.rglob("*") if p.is_file() and p.suffix.lower() != ".lock"
    ]
    if existing_files:
        print("=" * 60)
        print("[Step 1/5] 下载 FNSPID 数据集 (KDD 2024)")
        print("=" * 60)
        print(f"  检测到本地已有FNSPID数据 ({len(existing_files)} 个文件), 跳过下载")
        print(f"  数据目录: {fnspid_dir}")
        print()
        return

    print("=" * 60)
    print("[Step 1/5] 下载 FNSPID 数据集 (KDD 2024)")
    print("=" * 60)

    try:
        from datasets import load_dataset

        print("通过 HuggingFace datasets 库下载...")
        ds = load_dataset(FNSPID_HF_REPO, cache_dir=str(fnspid_dir / "hf_cache"))
        print(f"下载完成. 数据集信息: {ds}")
        # 保存为parquet便于后续处理
        for split_name in ds:
            output_path = fnspid_dir / f"{split_name}.parquet"
            ds[split_name].to_parquet(str(output_path))
            print(f"  已保存: {output_path}")
    except ImportError:
        print("datasets库未安装, 使用wget下载...")
        urls = [
            f"https://huggingface.co/datasets/{FNSPID_HF_REPO}/resolve/main/Stock_price/full_history.zip",
            f"https://huggingface.co/datasets/{FNSPID_HF_REPO}/resolve/main/Stock_news/nasdaq_exteral_data.csv",
        ]
        for url in urls:
            filename = url.split("/")[-1]
            output_path = fnspid_dir / filename
            if not output_path.exists():
                subprocess.run(["wget", "-O", str(output_path), url], check=True)
                print(f"  已下载: {output_path}")
            else:
                print(f"  已存在: {output_path}")

    print()


# ============================================================
# Step 2: 下载 EDT
# ============================================================
def download_edt():
    """从GitHub克隆EDT数据集"""
    edt_dir = DATA_ROOT / "edt"
    edt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[Step 2/5] 下载 EDT 数据集 (ACL 2021)")
    print("=" * 60)

    repo_dir = edt_dir / "TradeTheEvent"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", EDT_GITHUB_URL, str(repo_dir)], check=True
        )
        print(f"  克隆完成: {repo_dir}")
    else:
        print(f"  已存在: {repo_dir}")

    # 检查data目录
    data_subdir = repo_dir / "data"
    if data_subdir.exists():
        print(f"  事件标注数据位于: {data_subdir}")
        for f in data_subdir.iterdir():
            print(f"    - {f.name}")

    print()


# ============================================================
# Step 3: 下载 StockNet
# ============================================================
def download_stocknet():
    """从GitHub克隆StockNet数据集"""
    stocknet_dir = DATA_ROOT / "stocknet"
    stocknet_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[Step 3/5] 下载 StockNet 数据集")
    print("=" * 60)

    repo_dir = stocknet_dir / "stocknet-dataset"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", STOCKNET_GITHUB_URL, str(repo_dir)],
            check=True,
        )
        print(f"  克隆完成: {repo_dir}")
    else:
        print(f"  已存在: {repo_dir}")

    print()


# ============================================================
# Step 4: 下载 S&P 100 股价数据
# ============================================================
def download_sp100_prices(start_date="2018-01-01", end_date="2023-12-31"):
    """使用yfinance下载S&P 100股价数据"""
    prices_dir = DATA_ROOT / "sp100_prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[Step 4/5] 下载 S&P 100 股价数据 ({start_date} ~ {end_date})")
    print("=" * 60)

    try:
        import yfinance as yf

        all_data = {}
        failed = []

        for i, ticker in enumerate(SP100_TICKERS):
            output_path = prices_dir / f"{ticker}.csv"
            if output_path.exists():
                print(f"  [{i + 1}/{len(SP100_TICKERS)}] {ticker}: 已存在, 跳过")
                continue

            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                if len(df) > 0:
                    df.to_csv(str(output_path))
                    all_data[ticker] = df
                    print(
                        f"  [{i + 1}/{len(SP100_TICKERS)}] {ticker}: {len(df)} 条记录"
                    )
                else:
                    failed.append(ticker)
                    print(f"  [{i + 1}/{len(SP100_TICKERS)}] {ticker}: 无数据!")
            except Exception as e:
                failed.append(ticker)
                print(f"  [{i + 1}/{len(SP100_TICKERS)}] {ticker}: 错误 - {e}")

        if failed:
            print(f"\n  失败: {failed}")

    except ImportError:
        print("  yfinance未安装! 请运行: pip install yfinance")

    print()


# ============================================================
# Step 5: 获取 S&P 100 行业分类
# ============================================================
def download_sp100_sectors():
    """获取S&P 100的行业分类信息 (用于构建股票关系图)"""
    output_path = DATA_ROOT / "sp100_sectors.csv"

    print("=" * 60)
    print("[Step 5/5] 获取 S&P 100 行业分类信息")
    print("=" * 60)

    if output_path.exists():
        print(f"  已存在: {output_path}")
        print()
        return

    try:
        import yfinance as yf
        import pandas as pd

        records = []
        for i, ticker in enumerate(SP100_TICKERS):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                records.append(
                    {
                        "ticker": ticker,
                        "sector": info.get("sector", "Unknown"),
                        "industry": info.get("industry", "Unknown"),
                        "name": info.get("shortName", ticker),
                        "market_cap": info.get("marketCap", 0),
                    }
                )
                print(
                    f"  [{i + 1}/{len(SP100_TICKERS)}] {ticker}: "
                    f"{records[-1]['sector']} / {records[-1]['industry']}"
                )
            except Exception as e:
                records.append(
                    {
                        "ticker": ticker,
                        "sector": "Unknown",
                        "industry": "Unknown",
                        "name": ticker,
                        "market_cap": 0,
                    }
                )
                print(f"  [{i + 1}/{len(SP100_TICKERS)}] {ticker}: 错误 - {e}")

        df = pd.DataFrame(records)
        df.to_csv(str(output_path), index=False)
        print(f"\n  已保存: {output_path}")
        print(f"  行业分布:\n{df['sector'].value_counts().to_string()}")

    except ImportError:
        print("  yfinance未安装! 请运行: pip install yfinance")

    print()


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CausalStock 数据下载脚本")
    parser.add_argument("--all", action="store_true", help="下载所有数据集")
    parser.add_argument("--fnspid", action="store_true", help="下载FNSPID")
    parser.add_argument("--edt", action="store_true", help="下载EDT")
    parser.add_argument("--stocknet", action="store_true", help="下载StockNet")
    parser.add_argument("--prices", action="store_true", help="下载S&P100股价")
    parser.add_argument("--sectors", action="store_true", help="获取行业分类")
    parser.add_argument("--start-date", default="2018-01-01", help="价格数据起始日期")
    parser.add_argument("--end-date", default="2023-12-31", help="价格数据结束日期")
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if args.all or not any(
        [args.fnspid, args.edt, args.stocknet, args.prices, args.sectors]
    ):
        # 默认下载全部
        download_fnspid()
        download_edt()
        download_stocknet()
        download_sp100_prices(args.start_date, args.end_date)
        download_sp100_sectors()
    else:
        if args.fnspid:
            download_fnspid()
        if args.edt:
            download_edt()
        if args.stocknet:
            download_stocknet()
        if args.prices:
            download_sp100_prices(args.start_date, args.end_date)
        if args.sectors:
            download_sp100_sectors()

    print("=" * 60)
    print("数据下载完成!")
    print(f"数据目录: {DATA_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
