"""
Combined multi-pair walk-forward.
Fixed TP values per symbol (no per-window grid search).
Uses real position sizing from runner with spread costs.
"""
import pandas as pd
from pathlib import Path
from backtest import runner as r

SYMBOLS = ["BTCUSD", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD"]
FIXED_TPS = {"BTCUSD": 3.0, "EURUSD": 4.0, "GBPUSD": 3.0, "AUDUSD": 1.5, "USDCAD": 4.0}
SL = 0.3
PA = 0.3

WINDOWS = [
    ("2024-06-05", "2024-12-05", "2024-12-05", "2025-03-05"),
    ("2024-09-05", "2025-03-05", "2025-03-05", "2025-06-05"),
    ("2024-12-05", "2025-06-05", "2025-06-05", "2025-09-05"),
    ("2025-03-05", "2025-09-05", "2025-09-05", "2025-12-05"),
    ("2025-06-05", "2025-12-05", "2025-12-05", "2026-03-05"),
    ("2025-09-05", "2026-03-05", "2026-03-05", "2026-06-05"),
]

data_dir = Path("data/raw")
data = {}
for sym in SYMBOLS:
    candidates = list(data_dir.glob(f"{sym}_H1_*.csv"))
    if candidates:
        data[sym] = pd.read_csv(str(candidates[0]), parse_dates=True, index_col=0)

print(f"{'='*90}")
print(f"  COMBINED MULTI-PAIR WALK-FORWARD ({len(SYMBOLS)}-way split, fixed TPs)")
print(f"  TPs: {', '.join(f'{k}={v}' for k,v in FIXED_TPS.items())}")
print(f"  Spread: 0.5 pips")
print(f"{'='*90}")

balance = 100.0
all_trades = []

for wi, (tr_s, tr_e, te_s, te_e) in enumerate(WINDOWS):
    print(f"\n  Window {wi+1}: Test {te_s[:7]}->{te_e[:7]}")

    share = balance / len(SYMBOLS)
    win_trades_list = []
    total_pnl = 0

    for sym in SYMBOLS:
        test = data[sym][te_s:te_e]
        r.RAPID_TP_MULT = FIXED_TPS.get(sym, 2.0)
        r.RAPID_SL_MULT = SL
        r.RAPID_PA_THRESHOLD = PA
        trades_df, m = r.backtest(test, symbol=sym, initial_balance=share, strategy="rapid", progress=False)
        total_pnl += m["total_pnl"]
        if trades_df is not None and not trades_df.empty:
            trades_df["symbol"] = sym
            win_trades_list.append(trades_df)

    balance += total_pnl
    if win_trades_list:
        all_trades.append(pd.concat(win_trades_list, ignore_index=True))

    t = sum(len(t) for t in win_trades_list)
    print(f"    Trades={t}  PNL=${total_pnl:.0f}  Ending bal: ${balance:.0f}")

combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
print(f"\n{'='*90}")
print(f"  FINAL RESULTS")
print(f"{'='*90}")
print(f"  Starting balance: $100")
print(f"  Final balance:    ${balance:.0f}")
print(f"  Total return:     {balance-100:.0f}%")
print(f"  Total trades:     {len(combined)}")

for sym in SYMBOLS:
    mask = combined["symbol"] == sym if "symbol" in combined.columns else pd.Series(False, index=combined.index)
    n = mask.sum()
    pnl = combined[mask]["pnl_usd"].sum() if n > 0 else 0
    wins = (combined[mask]["status"] == "WIN").sum() if n > 0 else 0
    print(f"    {sym}: {n} trades, ${pnl:.0f} P&L, {wins} wins")
print(f"{'='*90}")
