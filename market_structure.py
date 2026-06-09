"""
Market Structure Analysis for 7 instruments across multiple timeframes.
Computes ATR, ADX, session analysis, volatility consistency.
Flags combinations where Spread/ATR <= 15%.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from analysis.feature_engineering import atr as compute_atr, adx as compute_adx, ema, add_indicators

SESSION_START = 0
SESSION_END = 24

# Vantage RAW ECN spreads (from web research, 2025-2026)
# For forex: pip = standard pip (0.0001 for most, 0.01 for JPY)
# For XAUUSD: pip = 0.01 (standard contract)
# For indices: point = 1.0
VANTAGE_SPREADS = {
    "XAUUSD": 0.15,    # pips (RAW ECN avg ~0.12-0.25, call it 0.15)
    "EURUSD": 0.12,    # pips
    "GBPUSD": 0.15,    # pips
    "US30":   1.2,     # points
    "NAS100": 1.0,     # points
    "USDCAD": 0.25,    # pips
    "NZDUSD": 0.25,    # pips
}

PIP_VALUES = {
    "XAUUSD": 0.01,
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "US30":   1.0,
    "NAS100": 1.0,
    "USDCAD": 0.0001,
    "NZDUSD": 0.0001,
}

# Existing Dukascopy M5 data paths
DUKASCOPY_DATA = {
    "EURUSD": "data/raw/EURUSD_M5_2023-01-01_2026-06-05.csv",
    "GBPUSD": "data/raw/GBPUSD_M5_2023-06-09_2026-06-09.csv",
    "USDCAD": None,
    "NZDUSD": None,
    "XAUUSD": None,
}

# Yahoo Finance symbols
YAHOO_SYMBOLS = {
    "XAUUSD": "GC=F",       # Gold futures
    "US30": "^DJI",         # Dow Jones Industrial Average
    "NAS100": "^IXIC",      # Nasdaq Composite
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
}


def load_dukascopy_m5(path: str) -> pd.DataFrame:
    """Load Dukascopy M5 CSV, add indicators."""
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = add_indicators(df)
    return df


def load_yahoo_m5(symbol: str, days: int = 60) -> pd.DataFrame:
    """Download M5 data from Yahoo Finance for a symbol."""
    yahoo_sym = YAHOO_SYMBOLS.get(symbol, symbol)
    df = yf.download(yahoo_sym, period=f"{days}d", interval="5m", progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol} ({yahoo_sym})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df["volume"] = df["volume"].clip(lower=1)  # avoid zero-volume breakage in indicators
    # Keep all hours (no session filter)
    df = add_indicators(df)
    return df


def resample_to(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample M5 OHLC to higher timeframe."""
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    resampled = df.resample(freq).agg(ohlc).dropna()
    resampled = add_indicators(resampled)
    return resampled


def compute_metrics(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    """Compute all requested metrics for one instrument+timeframe."""
    if df.empty or len(df) < 60:
        return None

    # ATR(50) in raw price units
    atr_50 = df["atr"].rolling(50).mean().iloc[-1] if len(df) >= 50 else df["atr"].mean()
    pip_val = PIP_VALUES[symbol]
    atr_pips = atr_50 / pip_val

    # Spread
    spread_pips = VANTAGE_SPREADS[symbol]
    spread_pct_atr = (spread_pips / atr_pips * 100) if atr_pips > 0 else 999

    # ADX average (over entire dataset)
    adx_vals = df["adx"].dropna()
    adx_mean = adx_vals.mean()
    adx_trending_pct = (adx_vals > 25).mean() * 100  # % of time trending (ADX > 25)

    # Session analysis - hourly ATR to find best session
    if timeframe == "M5":
        df["hour"] = df.index.hour
        hourly_atr = df.groupby("hour")["atr"].mean()
        best_hour = hourly_atr.idxmax()
        # Multi-hour rating: rank hours by ATR
        top_hours = hourly_atr.nlargest(4).index.sort_values().tolist()
        best_session_str = f"{top_hours[0]}-{top_hours[-1]} UTC" if top_hours else "N/A"

        # Volatility consistency (std dev of daily max ATR)
        df["date"] = df.index.date
        daily_max_atr = df.groupby("date")["atr"].max()
        daily_atr_std = daily_max_atr.std()
        daily_atr_mean = daily_max_atr.mean()
        vol_consistency = daily_atr_std / daily_atr_mean if daily_atr_mean > 0 else 0
    else:
        # For higher TFs, sample-based session analysis
        best_session_str = f"{SESSION_START}-{SESSION_END} UTC"
        df["date"] = df.index.date
        daily_max_atr = df.groupby("date")["atr"].max() if len(df) > 1 else pd.Series([df["atr"].iloc[-1]])
        daily_atr_std = daily_max_atr.std() if len(daily_max_atr) > 1 else 0
        daily_atr_mean = daily_max_atr.mean() if len(daily_max_atr) > 0 else df["atr"].mean()
        vol_consistency = daily_atr_std / daily_atr_mean if daily_atr_mean > 0 else 0

    return {
        "symbol": symbol,
        "tf": timeframe,
        "candles": len(df),
        "atr_pips": round(atr_pips, 1),
        "spread_pips": spread_pips,
        "spread_pct_atr": round(spread_pct_atr, 1),
        "adx_mean": round(adx_mean, 1),
        "adx_trending_pct": round(adx_trending_pct, 1),
        "best_session": best_session_str,
        "vol_consistency": round(vol_consistency, 2),
        "spread_flag": "✅" if spread_pct_atr <= 15 else "❌",
    }


def main():
    results = []

    # --- Process existing Dukascopy data ---
    print("Loading existing Dukascopy data...")
    for sym, path in DUKASCOPY_DATA.items():
        if path is None:
            continue
        try:
            df_m5 = load_dukascopy_m5(path)
            print(f"  {sym} M5: {len(df_m5)} candles")
            for tf, freq in [("M5", "5min"), ("M15", "15min"), ("H1", "1h")]:
                df = df_m5 if tf == "M5" else resample_to(df_m5, freq)
                m = compute_metrics(df, sym, tf)
                if m:
                    results.append(m)
        except Exception as e:
            print(f"  {sym}: ERROR - {e}")

    # --- Download Yahoo data for remaining instruments ---
    print("\nDownloading Yahoo Finance data (60 days)...")
    for sym in ["XAUUSD", "US30", "NAS100", "USDCAD", "NZDUSD"]:
        try:
            df_m5 = load_yahoo_m5(sym, days=60)
            print(f"  {sym} M5: {len(df_m5)} candles")
            for tf, freq in [("M5", "5min"), ("M15", "15min")]:
                df = df_m5 if tf == "M5" else resample_to(df_m5, freq)
                m = compute_metrics(df, sym, tf)
                if m:
                    results.append(m)
            # Also H1 for XAUUSD
            if sym == "XAUUSD":
                df_h1 = resample_to(df_m5, "1h")
                m = compute_metrics(df_h1, sym, "H1")
                if m:
                    results.append(m)
        except Exception as e:
            print(f"  {sym}: ERROR - {e}")

    # --- Print Report ---
    print("\n" + "=" * 130)
    print("  MARKET STRUCTURE ANALYSIS")
    print("  Source: Dukascopy (3yr EURUSD/GBPUSD) + Yahoo Finance (60d others)")
    print(f"  Session: {SESSION_START}-{SESSION_END} UTC")
    print(f"  Broker: Vantage RAW ECN spreads")
    print("=" * 130)

    header = f"{'Instrument':<10} {'TF':<5} {'ATR(50)':<10} {'Spread':<8} {'Spr/ATR%':<10} {'ADX avg':<8} {'Trend%':<8} {'Best Sess':<14} {'Vol Std':<8} {'Suitable?':<8}"
    print(header)
    print("-" * 130)

    suitable = [r for r in results if r["spread_flag"] == "✅"]
    not_suitable = [r for r in results if r["spread_flag"] == "❌"]

    for r in suitable + not_suitable:
        line = (
            f"{r['symbol']:<10} {r['tf']:<5} "
            f"{r['atr_pips']:<10} {r['spread_pips']:<8} "
            f"{r['spread_pct_atr']:<10} {r['adx_mean']:<8} "
            f"{r['adx_trending_pct']:<8} {r['best_session']:<14} "
            f"{r['vol_consistency']:<8} {r['spread_flag']:<8}"
        )
        print(line)

    print("-" * 130)
    print(f"\n  Suitable (Spread/ATR <= 15%): {len(suitable)}")
    for r in suitable:
        print(f"    ✅ {r['symbol']} {r['tf']} — ATR={r['atr_pips']}pips, Spread={r['spread_pips']}pip, Ratio={r['spread_pct_atr']}%")
    print(f"\n  Not Suitable (Spread/ATR > 15%): {len(not_suitable)}")
    for r in not_suitable:
        print(f"    ❌ {r['symbol']} {r['tf']} — ATR={r['atr_pips']}pips, Spread={r['spread_pips']}pip, Ratio={r['spread_pct_atr']}%")

    # --- Summary by instrument ---
    print("\n" + "=" * 130)
    print("  EXECUTIVE SUMMARY")
    print("=" * 130)
    for sym in ["EURUSD", "GBPUSD", "XAUUSD", "US30", "NAS100", "USDCAD", "NZDUSD"]:
        sym_results = [r for r in results if r["symbol"] == sym]
        if not sym_results:
            print(f"\n  {sym}: No data available")
            continue
        print(f"\n  {sym}:")
        for r in sym_results:
            flag = "VIABLE" if r["spread_flag"] == "✅" else "HIGH COST"
            print(f"    {r['tf']:5s}: ATR={r['atr_pips']:>6.1f}pips  Spread={r['spread_pips']}pip  Cost={r['spread_pct_atr']:>4.1f}%  ADX={r['adx_mean']:>4.1f}  VolCV={r['vol_consistency']:.2f}  [{flag}]")


if __name__ == "__main__":
    main()
