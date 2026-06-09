"""
Multi-Year H1 Validation — BTCUSD
Resamples M5 → H1, then runs backtest for all strategies over the full period.

Usage:
  python validate_h1.py                              # Full run
  python validate_h1.py --strategy hc                # Single strategy
  python validate_h1.py --walk-forward               # Walk-forward segments
  python validate_h1.py --save-h1                    # Save resampled H1 CSV
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone

from data.resample import load_and_resample, load_m5_csv
from backtest.runner import backtest
from backtest.metrics import calculate_metrics, print_metrics
from config import INITIAL_BALANCE, SYMBOLS

SYMBOL = SYMBOLS[0]
DATA_DIR = Path("data/raw")


def load_h1_data(save_h1: bool = False) -> pd.DataFrame:
    h1_csv = DATA_DIR / f"{SYMBOL}_H1_resampled.csv"
    if h1_csv.exists():
        print(f"Found cached H1 file: {h1_csv}")
        df = pd.read_csv(h1_csv, parse_dates=True, index_col=0)
        print(f"  Loaded {len(df):,} H1 candles ({df.index[0]} to {df.index[-1]})")
        return df

    m5_candidates = list(DATA_DIR.glob(f"{SYMBOL}_M5_*.csv"))
    if not m5_candidates:
        raise FileNotFoundError(
            f"No M5 CSV found for {SYMBOL} in {DATA_DIR}\n"
            f"Run: python data/download.py --symbol {SYMBOL} --days 90"
        )

    m5_csv = max(m5_candidates, key=lambda p: p.stat().st_size)
    print(f"Loading M5 data from {m5_csv}...")
    return load_and_resample(str(m5_csv), save_h1=save_h1)


def run_strategy(df: pd.DataFrame, strategy: str, balance: float) -> dict:
    label = {"m5": "M5 Scalper", "pa": "Price Action", "hc": "High Conviction",
             "mr": "Mean Reversion", "rapid": "Rapid M5 Scalper"}.get(strategy, strategy)
    print(f"\n  ── {label} ({strategy}) ──")

    trades, metrics = backtest(
        df=df, symbol=SYMBOL, initial_balance=balance,
        lookback=200, strategy=strategy,
    )
    return metrics


def run_all_strategies(df: pd.DataFrame):
    strategies = ["m5", "hc", "pa", "mr"]
    results = []

    print(f"\n{'='*55}")
    print(f"  H1 Validation — {len(df):,} candles")
    print(f"  Period: {df.index[0]} → {df.index[-1]}")
    print(f"{'='*55}")

    for strategy in strategies:
        metrics = run_strategy(df, strategy, balance=INITIAL_BALANCE)
        if metrics and "error" not in metrics:
            results.append((strategy, metrics))
            print_metrics(metrics)
        else:
            err = metrics.get("error", "unknown error") if metrics else "no metrics"
            print(f"  [SKIP] {strategy}: {err}")

    print(f"\n{'='*55}")
    print(f"  Summary: All Strategies on H1")
    print(f"{'='*55}")
    print(f"  {'Strategy':<15s} {'Trades':>7s} {'WR':>7s} {'P&L':>10s} {'PF':>7s} {'DD':>8s}")
    print(f"  {'─'*15} {'─'*7} {'─'*7} {'─'*10} {'─'*7} {'─'*8}")
    for strat, m in results:
        wr = f"{m['win_rate']*100:.1f}%"
        pnl = f"${m['total_pnl']:+.2f}"
        pf = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞"
        dd = f"{m['max_drawdown_pct']*100:.1f}%"
        print(f"  {strat:<15s} {m['total_trades']:>7d} {wr:>7s} {pnl:>10s} {pf:>7s} {dd:>8s}")
    print(f"{'='*55}")
    return results


def run_walk_forward(df: pd.DataFrame, strategy: str = "m5"):
    """
    Rolling walk-forward: train 30 days, test 5 days, step 5 days.
    Retrains XGBoost on each training window, then backtests on the test window.
    """
    from analysis.feature_engineering import add_indicators
    from analysis.ml_model import build_dataset, train_xgb, predict_xgb

    unique_dates = sorted(set(df.index.date))
    print(f"\nWalk-Forward H1 — {strategy.upper()}")
    print(f"  Trading days: {len(unique_dates)}")
    print(f"  Range: {unique_dates[0]}  →  {unique_dates[-1]}")

    train_days = 60   # ~2 months H1
    test_days = 10
    step_days = 10

    balance = INITIAL_BALANCE
    all_metrics = []

    for i in range(train_days, len(unique_dates) - test_days, step_days):
        train_start = unique_dates[i - train_days]
        train_end = unique_dates[i]
        test_start = unique_dates[i]
        test_end = unique_dates[min(i + test_days, len(unique_dates) - 1)]

        train_mask = (df.index.date >= train_start) & (df.index.date < train_end)
        test_mask = (df.index.date >= test_start) & (df.index.date < test_end)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) < 100 or len(test_df) < 50:
            continue

        try:
            train_xgb(SYMBOL, train_df)
        except Exception as e:
            print(f"  [SKIP] Train failed {train_start}→{test_end}: {e}")
            continue

        trades, metrics = backtest(
            df=test_df, symbol=SYMBOL, initial_balance=balance,
            lookback=100, strategy=strategy,
        )
        if metrics and "error" not in metrics:
            pnl = metrics.get("total_pnl", 0)
            balance += pnl
            all_metrics.append(metrics)
            print(f"  {train_start}→{test_end}: "
                  f"{metrics['total_trades']} trades, "
                  f"P&L=${pnl:+.2f}, WR={metrics['win_rate']*100:.1f}%")

    if not all_metrics:
        print("  No walk-forward segments had trades.")
        return

    total_pnl = sum(m.get("total_pnl", 0) for m in all_metrics)
    total_trades = sum(m.get("total_trades", 0) for m in all_metrics)
    wins = sum(m.get("wins", 0) for m in all_metrics)
    losses = sum(m.get("losses", 0) for m in all_metrics)
    wr = wins / total_trades * 100 if total_trades > 0 else 0

    print(f"\n  ── Walk-Forward Summary ({strategy.upper()} on H1) ──")
    print(f"  Windows:    {len(all_metrics)}")
    print(f"  Trades:     {total_trades} ({wins}W/{losses}L)")
    print(f"  Win Rate:   {wr:.1f}%")
    print(f"  Total P&L:  ${total_pnl:+.2f}")
    print(f"  Final Bal:  ${balance:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="H1 Multi-Year Validation")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["m5", "pa", "hc", "mr", None])
    parser.add_argument("--save-h1", action="store_true", help="Save resampled H1 CSV")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--balance", type=float, default=INITIAL_BALANCE)
    args = parser.parse_args()

    df = load_h1_data(save_h1=args.save_h1)

    if args.walk_forward:
        strat = args.strategy or "m5"
        run_walk_forward(df, strategy=strat)
    elif args.strategy:
        metrics = run_strategy(df, args.strategy, args.balance)
        if metrics and "error" not in metrics:
            print_metrics(metrics)
    else:
        run_all_strategies(df)


if __name__ == "__main__":
    main()
