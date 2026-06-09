"""
Two-stage grid search for HC strategy parameters on H1 data.
Stage 1: SL/TP multipliers (fixed PA/ML thresholds at defaults)
Stage 2: PA/ML thresholds (fixed SL/TP at best from Stage 1)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import itertools
from validate_h1 import load_h1_data
from backtest.runner import backtest
from backtest.metrics import calculate_metrics
import config as cfg
import backtest.runner as runner_mod


def run_with_params(df, params, label="") -> dict:
    # Update config values
    cfg.HC_PA_SCORE_THRESHOLD = params["pa_thr"]
    cfg.HC_ML_PROB_BUY = params["ml_buy"]
    cfg.HC_ML_PROB_SELL = params["ml_sell"]
    cfg.HC_TP_ATR_MULT_DEFAULT = params["tp_def"]
    cfg.HC_TP_ATR_MULT_NEAR_SR = params["tp_sr"]
    cfg.HC_SL_ATR_MULT = params["sl_atr"]

    runner_mod.HC_PA_SCORE_THRESHOLD = params["pa_thr"]
    runner_mod.HC_ML_PROB_BUY = params["ml_buy"]
    runner_mod.HC_ML_PROB_SELL = params["ml_sell"]
    runner_mod.HC_TP_ATR_MULT_DEFAULT = params["tp_def"]
    runner_mod.HC_TP_ATR_MULT_NEAR_SR = params["tp_sr"]
    runner_mod.HC_SL_ATR_MULT = params["sl_atr"]

    trades, metrics = backtest(df=df, symbol="BTCUSD", strategy="hc",
                                lookback=200, progress=False)
    return metrics


def header():
    return (f"{'pa_thr':>6s} {'ml_buy':>6s} {'ml_sell':>6s} "
            f"{'sl_atr':>6s} {'tp_def':>6s} {'tp_sr':>6s} "
            f"{'trds':>5s} {'WR':>5s} {'P&L':>9s} {'PF':>6s} {'DD':>5s} {'Sharpe':>6s}")


def row(p, m):
    if not m or "error" in m:
        return None
    return (f"{p['pa_thr']:>6.2f} {p['ml_buy']:>6.2f} {p['ml_sell']:>6.2f} "
            f"{p['sl_atr']:>6.1f} {p['tp_def']:>6.1f} {p['tp_sr']:>6.1f} "
            f"{m['total_trades']:>5d} {m['win_rate']*100:>4.1f}% "
            f"{m['total_pnl']:>+9.2f} {m['profit_factor']:>6.2f} "
            f"{m['max_drawdown_pct']*100:>4.1f}% {m['sharpe_ratio'] or 0:>6.2f}")


def run_stage(df, label, param_list, sort_key):
    print(f"\n{'='*100}")
    print(f"  Stage: {label}")
    print(f"{'='*100}")
    print(header())
    print("-" * 100)

    results = []
    for p in param_list:
        m = run_with_params(df, p, label)
        r = row(p, m)
        if r:
            print(r)
            results.append((p, m))
    return results


def main():
    df = load_h1_data()
    print(f"\nHC Grid Search — {len(df)} H1 candles")

    # ── Stage 1: SL/TP grid (fix PA/ML at defaults) ──
    defaults = {"pa_thr": 0.30, "ml_buy": 0.60, "ml_sell": 0.40}

    stage1 = []
    for sl in [0.5, 0.8, 1.0]:
        for tp_def in [1.0, 1.5, 2.0, 3.0]:
            for tp_sr in [1.5, 2.0, 3.0, 4.0]:
                p = defaults.copy()
                p.update({"sl_atr": sl, "tp_def": tp_def, "tp_sr": tp_sr})
                stage1.append(p)

    r1 = run_stage(df, "1 — SL/TP Multipliers (Defaults: pa=0.30, ml_buy=0.60, ml_sell=0.40)", stage1, None)

    if not r1:
        print("No valid stage 1 results.")
        return

    # Sort by Sharpe (min 10 trades)
    valid = [(p, m) for p, m in r1 if m["total_trades"] >= 10]
    valid.sort(key=lambda x: x[1].get("sharpe_ratio", 0) or 0, reverse=True)

    if valid:
        best_sl_tp = valid[0][0]
        print(f"\n  Best SL/TP: sl_atr={best_sl_tp['sl_atr']}, tp_def={best_sl_tp['tp_def']}, tp_sr={best_sl_tp['tp_sr']} "
              f"(Sharpe={valid[0][1]['sharpe_ratio'] or 0:.2f}, PF={valid[0][1]['profit_factor']:.2f}, "
              f"P&L=${valid[0][1]['total_pnl']:+.2f}, {valid[0][1]['total_trades']} trades)")
    else:
        print("  No valid combos with >= 10 trades in stage 1. Using defaults.")
        best_sl_tp = {"sl_atr": 0.8, "tp_def": 1.5, "tp_sr": 2.0}

    # ── Stage 2: PA/ML threshold grid (fix SL/TP at best from Stage 1) ──
    stage2 = []
    for pa_thr in [0.15, 0.20, 0.30, 0.40, 0.50]:
        for ml_buy in [0.50, 0.55, 0.60, 0.70, 0.80]:
            for ml_sell in [0.20, 0.30, 0.40, 0.45, 0.50]:
                p = best_sl_tp.copy()
                p.update({"pa_thr": pa_thr, "ml_buy": ml_buy, "ml_sell": ml_sell})
                stage2.append(p)

    r2 = run_stage(df, f"2 — PA/ML Thresholds (SL/TP fixed: sl={best_sl_tp['sl_atr']}, tp={best_sl_tp['tp_def']}/{best_sl_tp['tp_sr']})", stage2, None)

    if not r2:
        print("No valid stage 2 results.")
        return

    # ── Final rankings ──
    all_results = r1 + r2
    valid_all = [(p, m) for p, m in all_results if m["total_trades"] >= 10]

    for sort_name, sort_fn in [("Sharpe Ratio", lambda x: x[1].get("sharpe_ratio", 0) or 0),
                                 ("Profit Factor", lambda x: x[1]["profit_factor"]),
                                 ("Total P&L", lambda x: x[1]["total_pnl"])]:
        valid_all.sort(key=sort_fn, reverse=True)
        print(f"\n{'='*100}")
        print(f"  Top 10 by {sort_name} (≥10 trades)")
        print(f"{'='*100}")
        print(header())
        print("-" * 100)
        for p, m in valid_all[:10]:
            print(row(p, m))

    # Best overall by Sharpe
    valid_all.sort(key=lambda x: x[1].get("sharpe_ratio", 0) or 0, reverse=True)
    best = valid_all[0]
    print(f"\n{'='*100}")
    print(f"  BEST CONFIG: pa_thr={best[0]['pa_thr']}, ml_buy={best[0]['ml_buy']}, "
          f"ml_sell={best[0]['ml_sell']}, sl_atr={best[0]['sl_atr']}, "
          f"tp_def={best[0]['tp_def']}, tp_sr={best[0]['tp_sr']}")
    print(f"  Trades={best[1]['total_trades']}, WR={best[1]['win_rate']*100:.1f}%, "
          f"P&L=${best[1]['total_pnl']:+.2f}, PF={best[1]['profit_factor']:.2f}, "
          f"DD={best[1]['max_drawdown_pct']*100:.1f}%, Sharpe={best[1]['sharpe_ratio']}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
