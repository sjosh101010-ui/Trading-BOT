import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_forex_m5(symbol: str, days: int = 14) -> pd.DataFrame:
    yahoo_symbol = symbol.replace("BTCUSD", "BTC-USD").replace("XAUUSD", "GC=F").replace("USDJPY", "USDJPY=X").replace("EURUSD", "EURUSD=X").replace("GBPUSD", "GBPUSD=X")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {symbol} M5 data (last {days} days) from Yahoo Finance...")
    df = yf.download(yahoo_symbol, period=f"{days}d", interval="5m", progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {yahoo_symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df.index.name = "time"
    df.index = pd.to_datetime(df.index)

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df["volume"] = df["volume"].clip(lower=1)

    out_path = RAW_DIR / f"{symbol}_M5_{days}d.csv"
    df.to_csv(out_path)
    print(f"Saved {len(df)} candles to {out_path}")
    print(f"  Range: {df.index[0]} to {df.index[-1]}")
    return df


def download_all(days: int = 14):
    from config import SYMBOLS
    for sym in SYMBOLS:
        try:
            download_forex_m5(sym, days)
        except Exception as e:
            print(f"  Failed {sym}: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSD")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        download_all(args.days)
    else:
        download_forex_m5(args.symbol, args.days)
