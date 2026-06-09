"""
Analyze multi-pair backtest results from logs/backtest_combined_pnl_fix.csv
"""
import pandas as pd
import numpy as np

# Load data
combined = pd.read_csv("logs/backtest_combined_pnl_fix.csv", parse_dates=["timestamp", "exit_time"])
closed = combined[combined["status"].isin(["WIN", "LOSS"])].copy()
INITIAL = 10_000.0

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

print("=" * 70)
print("  PER-PAIR RESULTS (0.1 lot fixed, SL=1.0 ATR, TP=2.0 ATR, 1-9 UTC)")
print("=" * 70)

for sym in SYMBOLS:
    mask = closed["symbol"] == sym
    df = closed[mask].copy()
    if df.empty:
        print(f"\n  {sym}: NO TRADES")
        continue

    n = len(df)
    wins = df[df["status"] == "WIN"]
    losses = df[df["status"] == "LOSS"]
    n_wins = len(wins)
    n_losses = len(losses)
    wr = n_wins / n if n > 0 else 0

    gp = wins["pnl_usd"].sum()
    gl = abs(losses["pnl_usd"].sum())
    pf = gp / gl if gl > 0 else 0

    avg_win = wins["pnl_usd"].mean()
    avg_loss = losses["pnl_usd"].mean()
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    total_pnl = gp - gl

    # Date range
    start = df["timestamp"].min()
    end = df["exit_time"].max()
    days = (end - start).days
    trades_per_day = n / max(days, 1)

    # Max losing streak
    streak = 0
    max_streak = 0
    for pnl in df["pnl_usd"]:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Max drawdown (fixed 0.1 lot, no compounding)
    balance = INITIAL
    peak = INITIAL
    max_dd = 0
    for pnl in df["pnl_usd"]:
        balance += pnl
        peak = max(peak, balance)
        dd = peak - balance
        max_dd = max(max_dd, dd)
    max_dd_pct = max_dd / peak * 100 if peak > 0 else 0

    print(f"\n  {sym}:")
    print(f"    Total trades (period):     {n}")
    print(f"    Date range:                {start.date()} to {end.date()} ({days} days)")
    print(f"    Trades per day (avg):       {trades_per_day:.2f}")
    print(f"    Win Rate:                   {wr:.1%}")
    print(f"    Profit Factor:              {pf:.2f}")
    print(f"    Avg Win / Avg Loss:         ${avg_win:+.2f} / ${avg_loss:+.2f}")
    print(f"    Avg RR achieved:            {avg_rr:.2f}")
    print(f"    Total P&L:                  ${total_pnl:+.2f}")
    print(f"    Max losing streak:          {max_streak}")
    print(f"    Max drawdown (fixed):       {max_dd_pct:.1f}%")

    # PF per year
    df["year"] = df["timestamp"].dt.year
    print(f"    PF by year:")
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        yr_wins = yr_df[yr_df["status"] == "WIN"]["pnl_usd"].sum()
        yr_losses = abs(yr_df[yr_df["status"] == "LOSS"]["pnl_usd"].sum())
        yr_pf = yr_wins / yr_losses if yr_losses > 0 else 0
        yr_pnl = yr_wins - yr_losses
        print(f"      {yr}: PF={yr_pf:.2f}, P&L=${yr_pnl:+.2f}, {len(yr_df)} trades")

print("\n" + "=" * 70)
print("  COMBINED PORTFOLIO (all 4 pairs, FIFO chronological order)")
print("=" * 70)

# Sort chronologically
chrono = closed.sort_values(["timestamp", "exit_time"]).reset_index(drop=True)
n_total = len(chrono)
wins = chrono[chrono["status"] == "WIN"]
losses = chrono[chrono["status"] == "LOSS"]
n_wins = len(wins)
n_losses = len(losses)
wr = n_wins / n_total if n_total > 0 else 0
gp = wins["pnl_usd"].sum()
gl = abs(losses["pnl_usd"].sum())
pf = gp / gl if gl > 0 else 0
total_pnl = gp - gl

avg_win = wins["pnl_usd"].mean()
avg_loss = losses["pnl_usd"].mean()
avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

# Days in portfolio
start = chrono["timestamp"].min()
end = chrono["exit_time"].max()
portfolio_days = (end - start).days
trades_per_day = n_total / max(portfolio_days, 1)

# Portfolio equity curve + max drawdown
balance = INITIAL
peak = INITIAL
max_dd = 0
streak = 0
max_streak = 0
for pnl in chrono["pnl_usd"]:
    balance += pnl
    peak = max(peak, balance)
    dd = peak - balance
    max_dd = max(max_dd, dd)
    if pnl < 0:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0
max_dd_pct = max_dd / peak * 100 if peak > 0 else 0

# Check for overlapping trades
chrono_sorted = chrono.sort_values("timestamp")
overlaps = 0
for i in range(len(chrono_sorted)):
    for j in range(i + 1, len(chrono_sorted)):
        t1 = chrono_sorted.iloc[i]
        t2 = chrono_sorted.iloc[j]
        # Overlap if t2 entry before t1 exit
        if t2["timestamp"] < t1["exit_time"]:
            overlaps += 1
        else:
            break  # Since sorted by timestamp, no more overlaps possible for this t1

print(f"\n  Total closed trades:         {n_total}")
print(f"  Date range:                {start.date()} to {end.date()} ({portfolio_days} days)")
print(f"  Trades per day:             {trades_per_day:.2f}")
print(f"  Combined Win Rate:          {wr:.1%}")
print(f"  Combined PF:                {pf:.2f}")
print(f"  Total P&L:                  ${total_pnl:+.2f}")
print(f"  Avg RR achieved:            {avg_rr:.2f}")
print(f"  Max losing streak:          {max_streak}")
print(f"  Max portfolio drawdown:     {max_dd_pct:.1f}%")
print(f"  Simultaneous trades:         {overlaps} overlapping pairs found")

# PF per year combined
chrono["year"] = chrono["timestamp"].dt.year
print(f"  Combined PF by year:")
for yr in sorted(chrono["year"].unique()):
    yr_df = chrono[chrono["year"] == yr]
    yr_wins = yr_df[yr_df["status"] == "WIN"]["pnl_usd"].sum()
    yr_losses = abs(yr_df[yr_df["status"] == "LOSS"]["pnl_usd"].sum())
    yr_pf = yr_wins / yr_losses if yr_losses > 0 else 0
    yr_pnl = yr_wins - yr_losses
    print(f"      {yr}: PF={yr_pf:.2f}, P&L=${yr_pnl:+.2f}, {len(yr_df)} trades")

# Worst consecutive losing days (days where ALL pairs lost)
chrono["date"] = chrono["timestamp"].dt.date
daily_pnl = chrono.groupby("date")["pnl_usd"].sum()
daily_pnl = daily_pnl.sort_index()
max_consec_loss_days = 0
consec = 0
for d in daily_pnl.index:
    if daily_pnl[d] < 0:
        consec += 1
        max_consec_loss_days = max(max_consec_loss_days, consec)
    else:
        consec = 0

# Days where ALL pairs lost (not just combined negative)
all_lost_days = 0
for date, grp in chrono.groupby("date"):
    per_pair = grp.groupby("symbol")["pnl_usd"].sum()
    if len(per_pair) >= 2 and (per_pair < 0).all():
        all_lost_days += 1

print(f"\n  Worst consecutive losing DAYS:   {max_consec_loss_days}")
print(f"  Days where ALL pairs lost:       {all_lost_days}")

# Also show combined PF excluding USDJPY
ex_usdjpy = chrono[chrono["symbol"] != "USDJPY"]
ex_wins = ex_usdjpy[ex_usdjpy["status"] == "WIN"]["pnl_usd"].sum()
ex_losses = abs(ex_usdjpy[ex_usdjpy["status"] == "LOSS"]["pnl_usd"].sum())
ex_pf = ex_wins / ex_losses if ex_losses > 0 else 0
ex_pnl = ex_wins - ex_losses
ex_wr = len(ex_usdjpy[ex_usdjpy["status"] == "WIN"]) / len(ex_usdjpy) if len(ex_usdjpy) > 0 else 0
print(f"\n  Excluding USDJPY:")
print(f"    Trades:  {len(ex_usdjpy)}")
print(f"    Win Rate: {ex_wr:.1%}")
print(f"    PF:      {ex_pf:.2f}")
print(f"    P&L:     ${ex_pnl:+.2f}")

print("\n" + "=" * 70)
