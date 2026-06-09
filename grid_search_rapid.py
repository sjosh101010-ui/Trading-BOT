"""
Grid search for rapid strategy parameters across symbols.
Two-stage per symbol: Stage 1 = SL x TP, Stage 2 = PA threshold.
"""
import pandas as pd
from pathlib import Path
from backtest import runner as r
from backtest.runner import backtest

SYMBOLS = ["BTCUSD", "EURUSD", "GBPUSD", "AUDUSD"]
SL_MULTS = [0.3, 0.5, 0.7, 1.0, 1.5]
TP_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

def run(sl, tp, pa_thresh, df, symbol):
    r.RAPID_SL_MULT = sl
    r.RAPID_TP_MULT = tp
    r.RAPID_PA_THRESHOLD = pa_thresh
    _, m = backtest(df, symbol=symbol, initial_balance=100.0, strategy="rapid", progress=False)
    return m

for symbol in SYMBOLS:
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH: {symbol}")
    print(f"{'='*60}")
    candidates = list(Path("data/raw").glob(f"{symbol}_H1_*.csv"))
    if not candidates:
        print(f"  No H1 data found for {symbol}, skipping...")
        continue
    df = pd.read_csv(str(candidates[0]), parse_dates=True, index_col=0)

    # Stage 1
    print(f"\n  Stage 1: SL x TP (pa_thresh=0.3)")
    stage1 = []
    for sl in SL_MULTS:
        for tp in TP_MULTS:
            m = run(sl, tp, 0.3, df, symbol)
            stage1.append({"sl": sl, "tp": tp, "trades": m["total_trades"],
                           "pf": m["profit_factor"], "pnl": m["total_pnl"],
                           "wr": m["win_rate"], "dd": m["max_drawdown_pct"],
                           "sharpe": m["sharpe_ratio"]})
            print(f"  SL={sl:3.1f} TP={tp:3.1f} | T={m['total_trades']:4d} PF={m['profit_factor']:.2f} "
                  f"P=+${m['total_pnl']:.0f} WR={m['win_rate']:.1%} DD={m['max_drawdown_pct']:.1%} "
                  f"S={m['sharpe_ratio']:.2f}")

    s1 = pd.DataFrame(stage1)
    s1["score"] = s1["pf"] * s1["sharpe"] * (1 - s1["dd"]) * s1["pnl"].clip(lower=0)
    best = s1.sort_values("score", ascending=False).iloc[0]
    print(f"\n  Best Stage 1: SL={best['sl']} TP={best['tp']} | "
          f"T={best['trades']:.0f} PF={best['pf']:.2f} P=+${best['pnl']:.0f} "
          f"DD={best['dd']:.1%} S={best['sharpe']:.2f}")

    # Stage 2
    best_sl, best_tp = best["sl"], best["tp"]
    print(f"\n  Stage 2: PA threshold (SL={best_sl} TP={best_tp})")
    stage2 = []
    for pa in THRESHOLDS:
        m = run(best_sl, best_tp, pa, df, symbol)
        stage2.append({"pa": pa, "trades": m["total_trades"],
                       "pf": m["profit_factor"], "pnl": m["total_pnl"],
                       "wr": m["win_rate"], "dd": m["max_drawdown_pct"],
                       "sharpe": m["sharpe_ratio"]})
        print(f"  PA={pa:.2f} | T={m['total_trades']:4d} PF={m['profit_factor']:.2f} "
              f"P=+${m['total_pnl']:.0f} WR={m['win_rate']:.1%} DD={m['max_drawdown_pct']:.1%} "
              f"S={m['sharpe_ratio']:.2f}")

    s2 = pd.DataFrame(stage2)
    s2["score"] = s2["pf"] * s2["sharpe"] * (1 - s2["dd"]) * s2["pnl"].clip(lower=0)
    final = s2.sort_values("score", ascending=False).iloc[0]
    print(f"\n  >>> BEST >>> SL={best_sl} TP={best_tp} PA={final['pa']} | "
          f"Trades={final['trades']:.0f} PF={final['pf']:.2f} PNL=+${final['pnl']:.0f} "
          f"WR={final['wr']:.1%} DD={final['dd']:.1%} Sharpe={final['sharpe']:.2f}")

print(f"\n{'='*60}")
print(f"  DONE")
print(f"{'='*60}")
