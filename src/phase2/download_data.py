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

import os
import argparse
from pathlib import Path

proxy = 'your proxy here' 
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy
# ============================================================
# 配置
# ============================================================
DATA_ROOT = Path(__file__).parent.parent / "data"

FNSPID_HF_REPO = "Zihan1004/FNSPID"
EDT_GITHUB_URL = "https://github.com/Zhihan1996/TradeTheEvent.git"
STOCKNET_GITHUB_URL = "https://github.com/yumoxu/stocknet-dataset.git"

# S&P 100 成分股 (截至2024年)
SP100_TICKERS = [
    "A",
    "AA",
    "AAAU",
    "AACG",
    "AADR",
    "AAL",
]


# ============================================================
# Step 3: 下载 StockNet
# ============================================================
def download_stocknet():
    # """从GitHub克隆StockNet数据集"""
    # stocknet_dir = DATA_ROOT / "stocknet"
    # stocknet_dir.mkdir(parents=True, exist_ok=True)

    # print("=" * 60)
    # print("[Step 3/5] 下载 StockNet 数据集")
    # print("=" * 60)

    # repo_dir = stocknet_dir / "stocknet-dataset"
    # if not repo_dir.exists():
    #     subprocess.run(
    #         ["git", "clone", "--depth", "1", STOCKNET_GITHUB_URL, str(repo_dir)],
    #         check=True,
    #     )
    #     print(f"  克隆完成: {repo_dir}")
    # else:
    #     print(f"  已存在: {repo_dir}")

    print()


# ============================================================
# Step 4: 下载 S&P 100 股价数据
# ============================================================
import time
import yfinance as yf

def download_sp100_prices(start_date="2018-01-01", end_date="2023-12-31"):
    prices_dir = DATA_ROOT / "sp100_prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    # 分批
    batch_size = 5

    for i in range(0, len(SP100_TICKERS), batch_size):
        batch = SP100_TICKERS[i:i+batch_size]

        print(f"下载 batch: {batch}")

        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                group_by="ticker",
                threads=False
            )

            for ticker in batch:
                if ticker in data:
                    df = data[ticker]
                    if len(df) > 0:
                        df.to_csv(prices_dir / f"{ticker}.csv")

        except Exception as e:
            print(f"batch失败: {e}")

        time.sleep(2)


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
        download_stocknet()
        download_sp100_prices(args.start_date, args.end_date)
        download_sp100_sectors()
    else:
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
