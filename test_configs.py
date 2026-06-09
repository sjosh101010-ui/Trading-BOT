#!/usr/bin/env python3
"""Quick backtest comparison across config variants — proper deferred entry."""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from analysis.feature_engineering import add_indicators
from analysis.m5_scalper import compute_m5_score
from risk.lot_sizer import compute_lot_size_sim, _peak_balance
from backtest.metrics import calculate_metrics
from config import RAPID_PA_THRESHOLD, SPREAD_PIPS, LEVERAGE, pip_size

CONTRACT_SIZE = {"EURUSD": 100_000}

def backtest_config(df, use_adx=False, use_ema50=False, sl_mult=0.3, tp_mult_val=4.0):
    df = df.copy()
    df = add_indicators(df)
    balance = 100.0 * LEVERAGE
    _peak_balance["EURUSD"] = balance
    trades = []
    open_pos = {}
    pending_entry = None
    start_idx = max(100, 50)

    for i in range(start_idx, len(df)):
        current = df.iloc[i]
        balance += _check_sl_tp(current, open_pos, trades, df.index[i])

        if pending_entry is not None and "EURUSD" not in open_pos:
            entry = pending_entry
            pending_entry = None
            price = float(current["open"])
            atr_val = float(current["atr"]) if current["atr"] > 0 else 0.001
            ps = pip_size("EURUSD")
            spread_pts = SPREAD_PIPS * ps
            entry_price = price + spread_pts / 2 if entry["direction"] == "BUY" else price - spread_pts / 2
            sl_price = entry_price - sl_mult * atr_val if entry["direction"] == "BUY" else entry_price + sl_mult * atr_val
            tp_price = entry_price + tp_mult_val * atr_val if entry["direction"] == "BUY" else entry_price - tp_mult_val * atr_val
            sl_pips = abs(entry_price - sl_price) / ps
            lot = compute_lot_size_sim("EURUSD", balance, sl_pips, min(1.0, abs(entry["pa"])), price=price)
            if lot > 0:
                open_pos["EURUSD"] = {
                    "ticket": len(trades) + 1, "symbol": "EURUSD",
                    "direction": entry["direction"], "lot": lot,
                    "entry_price": entry_price, "sl": sl_price, "tp": tp_price,
                    "entry_time": df.index[i],
                }
            continue

        if "EURUSD" in open_pos:
            continue

        window = df.iloc[:i+1]
        pa_score = compute_m5_score(window)
        if abs(pa_score) < RAPID_PA_THRESHOLD:
            continue

        last = window.iloc[-1]
        price = float(last["close"])
        ema50 = float(last.get("ema_50", price))

        if use_adx:
            adx_val = float(last.get("adx", 0))
            if adx_val < 12:
                continue
        if use_ema50:
            if pa_score > 0 and price <= ema50:
                continue
            if pa_score < 0 and price >= ema50:
                continue

        direction = "BUY" if pa_score > 0 else "SELL"
        pending_entry = {"direction": direction, "pa": pa_score}

    for sym, pos in list(open_pos.items()):
        trades.append({
            "timestamp": pos["entry_time"], "symbol": sym,
            "direction": pos["direction"], "lot_size": pos["lot"],
            "entry_price": pos["entry_price"], "exit_price": pos["entry_price"],
            "sl_price": pos["sl"], "tp_price": pos["tp"],
            "pnl_usd": 0, "status": "OPEN",
            "exit_time": df.index[-1], "duration_hours": 0,
        })
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = calculate_metrics(trades_df, 100.0)
    return trades_df, metrics


def _check_sl_tp(current_row, open_positions, trades, current_time):
    total_pnl = 0.0
    for sym in list(open_positions.keys()):
        pos = open_positions[sym]
        high = current_row["high"]
        low = current_row["low"]
        sl, tp, entry, lot = pos["sl"], pos["tp"], pos["entry_price"], pos["lot"]
        csize = CONTRACT_SIZE.get(sym, 100_000)

        if pos["direction"] == "BUY":
            hit_sl, hit_tp = low <= sl, high >= tp
        else:
            hit_sl, hit_tp = high >= sl, low <= tp

        if hit_sl or hit_tp:
            exit_price = sl if hit_sl else tp
            pnl = (exit_price - entry) * lot * csize
            if pos["direction"] == "SELL":
                pnl = -pnl
            trades.append({
                "timestamp": pos["entry_time"], "symbol": sym,
                "direction": pos["direction"], "lot_size": lot,
                "entry_price": entry, "exit_price": exit_price,
                "sl_price": sl, "tp_price": tp,
                "pnl_usd": round(pnl, 2), "status": "LOSS" if hit_sl else "WIN",
                "exit_time": current_time, "duration_hours": 0,
            })
            total_pnl += pnl
            del open_positions[sym]
    return total_pnl


configs = [
    ("SL=0.3× TP=4×", dict(sl_mult=0.3, tp_mult_val=4.0, use_adx=True, use_ema50=True)),
    ("SL=0.5× TP=4×", dict(sl_mult=0.5, tp_mult_val=4.0, use_adx=True, use_ema50=True)),
    ("SL=1.0× TP=4×", dict(sl_mult=1.0, tp_mult_val=4.0, use_adx=True, use_ema50=True)),
    ("SL=1.5× TP=4×", dict(sl_mult=1.5, tp_mult_val=4.0, use_adx=True, use_ema50=True)),
    ("SL=0.3× TP=2×", dict(sl_mult=0.3, tp_mult_val=2.0, use_adx=True, use_ema50=True)),
    ("SL=1.0× TP=2×", dict(sl_mult=1.0, tp_mult_val=2.0, use_adx=True, use_ema50=True)),
    ("SL=0.75× TP=4×", dict(sl_mult=0.75, tp_mult_val=4.0, use_adx=True, use_ema50=True)),
]

df = pd.read_csv("data/raw/EURUSD_M5_30d.csv", parse_dates=True, index_col=0)

print(f"\n{'='*72}")
print(f"  EURUSD M5 — {len(df)} candles ({df.index[0].date()} to {df.index[-1].date()})")
print(f"  All configs: ADX≥12 + EMA50 guard | Balance=$100 (4x leverage)")
print(f"{'='*72}")
print(f"  {'Config':<22s} {'Trades':>6s} {'WR%':>5s} {'AvgW':>7s} {'AvgL':>7s} {'R:R':>6s} {'PnL':>10s} {'PF':>5s}")
print(f"  {'-'*22} {'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*6} {'-'*10} {'-'*5}")

results = []
for label, kwargs in configs:
    trades_df, metrics = backtest_config(df, **kwargs)
    if not trades_df.empty:
        closed = trades_df[trades_df["status"].isin(["WIN", "LOSS"])]
        wins = len(closed[closed["status"] == "WIN"])
        losses = len(closed[closed["status"] == "LOSS"])
        total = wins + losses
        wr = wins / total * 100 if total else 0
        avg_w = closed[closed["status"]=="WIN"]["pnl_usd"].mean() if wins else 0
        avg_l = abs(closed[closed["status"]=="LOSS"]["pnl_usd"].mean()) if losses else 0
        rr = avg_w / avg_l if avg_l > 0 else 0
        pnl = closed["pnl_usd"].sum()
    else:
        total = wins = losses = 0
        wr = avg_w = avg_l = rr = pnl = 0
    results.append((label, total, wr, avg_w, avg_l, rr, pnl, metrics.get("profit_factor", 0)))
    print(f"  {label:<22s} {total:>6d} {wr:>4.1f}% ${avg_w:>5.1f} ${avg_l:>5.1f} {rr:>5.1f}:1 ${pnl:>+7.2f} {metrics.get('profit_factor',0):>4.2f}")
