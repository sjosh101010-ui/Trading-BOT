"""
Combined multi-pair backtest: trades BTCUSD, EURUSD, GBPUSD, AUDUSD simultaneously
from a single account balance. Runs each pair independently, then merges
trade logs chronologically for combined metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from backtest.runner import run_backtest

SYMBOLS = ["BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDILS", "USDZAR"]
INITIAL_BALANCE = 100.0
CSV_FILES = {s: f"data/raw/{s}_H1_yahoo_2y.csv" for s in SYMBOLS}

all_trades = []
for symbol in SYMBOLS:
    print(f"\nRunning {symbol}...")
    trades, metrics = run_backtest(
        csv_path=CSV_FILES[symbol],
        df=None,
        symbol=symbol,
        initial_balance=INITIAL_BALANCE,
        strategy="rapid",
    )
    if trades is not None and not trades.empty:
        trades["symbol"] = symbol
        all_trades.append(trades)

if not all_trades:
    print("No trades generated.")
    exit()

combined = pd.concat(all_trades, ignore_index=True)
combined = combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

print(f"\n{'='*50}")
print(f"  COMBINED MULTI-PAIR RESULTS")
print(f"{'='*50}")
print(f"  Pairs: {', '.join(SYMBOLS)}")
print(f"  Period: {combined['timestamp'].min()} to {combined['exit_time'].max()}")
print(f"  Total trades: {len(combined)}")
print(f"  Per pair:")
for s in SYMBOLS:
    mask = combined["symbol"] == s
    print(f"    {s}: {mask.sum()} trades")

closed = combined[combined["status"].isin(["WIN", "LOSS"])]
wins = closed[closed["status"] == "WIN"]
losses = closed[closed["status"] == "LOSS"]
total_wins = len(wins)
total_losses = len(losses)
win_rate = total_wins / len(closed) if len(closed) > 0 else 0
gross_profit = wins["pnl_usd"].sum() if not wins.empty else 0
gross_loss = abs(losses["pnl_usd"].sum()) if not losses.empty else 0
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
total_pnl = gross_profit - gross_loss
avg_win = wins["pnl_usd"].mean() if not wins.empty else 0
avg_loss = losses["pnl_usd"].mean() if not losses.empty else 0

# Equity curve from chronological trade list
equity = [INITIAL_BALANCE]
running = INITIAL_BALANCE
peak = INITIAL_BALANCE
max_dd = 0
for _, t in combined.iterrows():
    if t["status"] in ("WIN", "LOSS"):
        running += t["pnl_usd"]
    equity.append(running)
    peak = max(peak, running)
    dd = peak - running
    max_dd = max(max_dd, dd)
max_dd_pct = max_dd / peak if peak > 0 else 0

# Sharpe-like from daily returns
equity_series = pd.Series(equity)
daily_returns = equity_series.pct_change().dropna()
sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

final_balance = running
print(f"\n  Results: {len(closed)} closed trades")
print(f"  Wins / Losses: {total_wins} / {total_losses}")
print(f"  Win Rate:      {win_rate:.1%}")
print(f"  Profit Factor: {profit_factor:.2f}")
print(f"  Total P&L:     ${total_pnl:+.2f}")
print(f"  Final Balance: ${final_balance:.2f}")
print(f"  Avg Win/Loss:  ${avg_win:+.2f} / ${avg_loss:+.2f}")
print(f"  Max Drawdown:  ${max_dd:.2f} ({max_dd_pct:.1%})")
print(f"  Sharpe Ratio:  {sharpe:.2f}")
print(f"{'='*50}")

combined.to_csv("logs/backtest_combined.csv", index=False)
print(f"  Trade log saved: logs/backtest_combined.csv")
