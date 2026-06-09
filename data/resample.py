import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def m5_to_h1(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Resample M5 OHLCV data to H1.

    Expects columns: timestamp, open, high, low, close, volume
    Returns DataFrame with hourly OHLCV.
    """
    df = df.copy()
    if timestamp_col in df.columns:
        df["time"] = pd.to_datetime(df[timestamp_col])
        df.set_index("time", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    h1 = df.resample("1h").agg(ohlc)
    h1.dropna(subset=["open", "high", "low", "close"], inplace=True)
    h1 = h1[["open", "high", "low", "close", "volume"]]

    # Ensure no zero-volume rows get through
    h1["volume"] = h1["volume"].clip(lower=1)
    h1.index.name = "time"
    return h1


def load_m5_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    if "timestamp" not in df.columns and "time" in df.columns:
        df.rename(columns={"time": "timestamp"}, inplace=True)
    return df


def load_and_resample(csv_path: str, save_h1: bool = False) -> pd.DataFrame:
    df = load_m5_csv(csv_path)
    print(f"  Loaded {len(df):,} M5 candles from {csv_path}")
    h1 = m5_to_h1(df)
    print(f"  Resampled to {len(h1):,} H1 candles")
    print(f"  Range: {h1.index[0]} to {h1.index[-1]}")

    if save_h1:
        out = RAW_DIR / f"H1_resampled.csv"
        h1.to_csv(out)
        print(f"  Saved to {out}")

    return h1
