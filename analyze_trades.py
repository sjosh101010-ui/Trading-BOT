#!/usr/bin/env python3
"""Analyze backtest trade log to find loss patterns."""
import pandas as pd
import numpy as np

df = pd.read_csv("logs/backtest_eurusd.csv", parse_dates=["timestamp", "exit_time"])
closed = df[df["status"].isin(["WIN", "LOSS"])].copy()
wins = closed[closed["status"] == "WIN"]
losses = closed[closed["status"] == "LOSS"]

print(f"Total closed trades: {len(closed)}")
print(f"Wins: {len(wins)} ({len(wins)/len(closed)*100:.1f}%)")
print(f"Losses: {len(losses)} ({len(losses)/len(closed)*100:.1f}%)")
print()

# Duration analysis
print("=== DURATION ANALYSIS ===")
for label, grp in [("ALL", closed), ("WINS", wins), ("LOSSES", losses)]:
    d = grp["duration_hours"]
    print(f"{label}: mean={d.mean():.2f}h, median={d.median():.2f}h, "
          f"<0.1h={(d<0.1).sum()/len(d)*100:.0f}%, "
          f"<0.25h={(d<0.25).sum()/len(d)*100:.0f}%, "
          f"<0.5h={(d<0.5).sum()/len(d)*100:.0f}%")
print()

# SL distance in pips
ps = 0.0001
closed["sl_dist"] = (closed["entry_price"] - closed["sl_price"]).abs() / ps
print(f"=== SL DISTANCE ===")
print(f"Mean: {closed['sl_dist'].mean():.1f} pips")
print(f"Min: {closed['sl_dist'].min():.1f} pips")
print(f"Max: {closed['sl_dist'].max():.1f} pips")
print()

# ADX at entry
print(f"=== ADX AT ENTRY ===")
for label, grp in [("ALL", closed), ("WINS", wins), ("LOSSES", losses)]:
    a = grp["adx_entry"]
    print(f"{label}: mean={a.mean():.1f}, median={a.median():.1f}, min={a.min():.1f}, max={a.max():.1f}")
print()

# Consecutive loss streaks
closed["is_loss"] = closed["status"] == "LOSS"
streak = 0
max_streak = 0
streaks = []
for v in closed["is_loss"]:
    if v:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        if streak > 0:
            streaks.append(streak)
        streak = 0
if streak > 0:
    streaks.append(streak)
print(f"=== LOSS STREAKS ===")
print(f"Max consecutive losses: {max_streak}")
print(f"Average streak: {np.mean(streaks):.1f}")
print(f"Streaks >= 10: {sum(1 for s in streaks if s >= 10)}")
print()

# Direction bias
print(f"=== DIRECTION BIAS ===")
for direction in ["BUY", "SELL"]:
    d = closed[closed["direction"] == direction]
    w = (d["status"] == "WIN").sum()
    l_ = (d["status"] == "LOSS").sum()
    t = w + l_
    wr = w / t * 100 if t else 0
    pnl = d["pnl_usd"].sum()
    print(f"{direction}: {t} trades, WR={wr:.1f}%, PnL=${pnl:.2f}")
print()

# Consecutive same-direction trades analysis
closed["prev_dir"] = closed["direction"].shift(1)
closed["direction_change"] = closed["direction"] != closed["prev_dir"]
same_dir = closed[closed["direction_change"] == False].dropna()
diff_dir = closed[closed["direction_change"] == True]
print(f"=== DIRECTION SWITCH ANALYSIS ===")
print(f"Same direction as previous: {len(same_dir)} trades, "
      f"WR={(same_dir['status']=='WIN').sum()/len(same_dir)*100:.1f}%")
print(f"Different direction: {len(diff_dir)} trades, "
      f"WR={(diff_dir['status']=='WIN').sum()/len(diff_dir)*100:.1f}%")

# How many trades reverse direction after a loss?
closed["prev_status"] = closed["status"].shift(1)
closed["prev_dir"] = closed["direction"].shift(1)
loss_then_reverse = closed[(closed["prev_status"] == "LOSS") & (closed["direction"] != closed["prev_dir"])]
loss_then_same = closed[(closed["prev_status"] == "LOSS") & (closed["direction"] == closed["prev_dir"])]
print(f"\nAfter a LOSS, reverse direction: {len(loss_then_reverse)} trades, "
      f"WR={(loss_then_reverse['status']=='WIN').sum()/len(loss_then_reverse)*100:.1f}%")
print(f"After a LOSS, same direction: {len(loss_then_same)} trades, "
      f"WR={(loss_then_same['status']=='WIN').sum()/len(loss_then_same)*100:.1f}%")

# Win analysis - how quickly do wins happen?
print(f"\n=== WIN SPEED ===")
w = wins.copy()
w["bars_to_exit"] = (w["duration_hours"] / 0.08333).round()
print(f"Wins by duration:")
for bins in [(0, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 24)]:
    mask = (w["duration_hours"] >= bins[0]) & (w["duration_hours"] < bins[1])
    cnt = mask.sum()
    if cnt:
        print(f"  {bins[0]:.1f}-{bins[1]:.1f}h: {cnt} wins, avg PnL=${w[mask]['pnl_usd'].mean():.1f}")

# Loss analysis - quick vs slow losses
print(f"\n=== LOSS SPEED ===")
l_ = losses.copy()
fast = l_[l_["duration_hours"] < 0.1]
slow = l_[l_["duration_hours"] >= 0.1]
print(f"Losses in <6 min: {len(fast)} ({len(fast)/len(l_)*100:.0f}%) avg=${fast['pnl_usd'].mean():.1f}")
print(f"Losses in >=6 min: {len(slow)} ({len(slow)/len(l_)*100:.0f}%) avg=${slow['pnl_usd'].mean():.1f}")
