"""
Backtest the AI Forex system on historical data.

Usage:
  # Using synthetic data (quick demo)
  python run_backtest.py --demo

  # Using a CSV file
  python run_backtest.py --csv data/raw/BTCUSD_M5.csv --symbol BTCUSD

  # Download sample data first (if available)
  python run_backtest.py --download BTCUSD --days 30 --demo
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from backtest.runner import run_backtest
from data.download import download_forex_m5 as download


def generate_synthetic_data(symbol: str = "BTCUSD", n_candles: int = 5000) -> pd.DataFrame:
    print(f"Generating {n_candles} synthetic M5 candles for {symbol}...")
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n_candles, freq="5min")
    base = 70000.0 if symbol == "BTCUSD" else (1.08 if "USD" in symbol else 150.0)
    vol = base * 0.0003 if symbol == "BTCUSD" else (0.0003 if "USD" in symbol else 0.05)

    trend_cycles = 8
    cycle_len = n_candles // trend_cycles
    closes = []
    for c in range(trend_cycles):
        direction = 1 if c % 2 == 0 else -1
        for _ in range(cycle_len):
            trend = direction * np.random.uniform(-vol * 2, vol * 4)
            noise = np.random.normal(0, vol * 0.5)
            closes.append((closes[-1] if closes else base) * (1 + trend + noise))
    while len(closes) < n_candles:
        closes.append(closes[-1] * (1 + np.random.normal(0, vol * 0.5)))
    closes = np.array(closes)

    opens = closes * (1 + np.random.uniform(-vol * 0.3, vol * 0.3, n_candles))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, vol * 0.5, n_candles)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, vol * 0.5, n_candles)))

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": np.random.randint(100, 5000, n_candles),
    }, index=idx)
    return df


def main():
    parser = argparse.ArgumentParser(description="AI Forex Backtester")
    parser.add_argument("--csv", type=str, help="Path to CSV file with OHLCV data")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Trading symbol")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data for demo")
    parser.add_argument("--candles", type=int, default=5000, help="Number of candles")
    parser.add_argument("--balance", type=float, default=100.0, help="Initial balance")
    parser.add_argument("--download", type=str, help="Download data for symbol")
    parser.add_argument("--days", type=int, default=30, help="Days of data to download")
    parser.add_argument("--strategy", type=str, default="m5", choices=["m5", "pa", "hc", "mr", "rapid"],
                        help="m5=scalper (default), pa=candlestick patterns, hc=high conviction, mr=mean reversion")
    parser.add_argument("--date", type=str, help="Trade only on this date (YYYY-MM-DD)")
    parser.add_argument("--from-date", type=str, help="Start cumulative backtest from this date (YYYY-MM-DD)")
    args = parser.parse_args()

    kwargs = {"strategy": args.strategy}

    if args.download:
        df = download(args.download, args.days)

        if args.demo and df is not None:
            trades, metrics = run_backtest(df=df, symbol=args.download, initial_balance=args.balance, **kwargs)
        else:
            print("Use --demo to run backtest on downloaded data.")
        return

    if args.demo:
        df = generate_synthetic_data(args.symbol, args.candles)
        trades, metrics = run_backtest(df=df, symbol=args.symbol, initial_balance=args.balance, **kwargs)

    elif args.csv:
        if args.date:
            target = pd.Timestamp(args.date, tz="UTC")
            full = pd.read_csv(args.csv, parse_dates=True, index_col=0)
            full = full[full.index >= target - pd.Timedelta(days=3)]
            full = full[full.index <= target + pd.Timedelta(days=1)]
            trades, metrics = run_backtest(df=full, symbol=args.symbol, initial_balance=args.balance, trade_date=target, **kwargs)
        elif args.from_date:
            start = pd.Timestamp(args.from_date, tz="UTC")
            full = pd.read_csv(args.csv, parse_dates=True, index_col=0)
            full = full[full.index >= start - pd.Timedelta(days=3)]
            trades, metrics = run_backtest(df=full, symbol=args.symbol, initial_balance=args.balance, trade_start_date=start, **kwargs)
        else:
            trades, metrics = run_backtest(csv_path=args.csv, symbol=args.symbol, initial_balance=args.balance, **kwargs)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
