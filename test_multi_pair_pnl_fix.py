"""
Multi-pair backtest: EURUSD, GBPUSD, USDJPY, AUDUSD
Uses current backtest.runner (which has _pnl_to_usd for USDJPY).
Overrides config for fixed SL=1.0 ATR, TP=2.0 ATR, 0.1 lot, session 1-9 UTC.
"""
import sys
import os
import pandas as pd
import numpy as np

# --- Import & patch config BEFORE importing runner ---
import config as cfg

# Override rapid strategy params
cfg.RAPID_SL_MULT = 1.0
cfg.RAPID_TP_MULT = {"EURUSD": 2.0, "GBPUSD": 2.0, "USDJPY": 2.0, "AUDUSD": 2.0}
cfg.RAPID_PA_THRESHOLD = 0.5
cfg.TRADE_SESSION_START = 1
cfg.TRADE_SESSION_END = 9

# Override initial balance
INITIAL_BALANCE = 10_000.0

# Override spread to match Fusion Markets
cfg.SPREAD_PIPS = 0.3  # EURUSD; OK as approximation for all 4

# Monkey-patch compute_lot_size_sim to return fixed 0.1 lot
import risk.lot_sizer as ls
_original_lot_sizer = ls.compute_lot_size_sim
ls.compute_lot_size_sim = lambda *a, **kw: 0.1

# Also patch CONTRACT_SIZE for USDJPY (ensure 100,000)
cfg.CONTRACT_SIZE["USDJPY"] = 100_000
cfg.CONTRACT_SIZE["EURUSD"] = 100_000
cfg.CONTRACT_SIZE["GBPUSD"] = 100_000
cfg.CONTRACT_SIZE["AUDUSD"] = 100_000

# Ensure ATR_MULT exists for all 4 (for fallback sl_tp)
cfg.ATR_MULT["USDJPY"] = {"sl": 1.0, "tp": 2.0}
cfg.ATR_MULT["GBPUSD"] = {"sl": 1.0, "tp": 2.0}
cfg.ATR_MULT["AUDUSD"] = {"sl": 1.0, "tp": 2.0}

# Now safely import runner
from backtest.runner import backtest, _pnl_to_usd

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]

CSV_FILES = {
    "EURUSD": "data/raw/EURUSD_M5_2023-01-01_2026-06-05.csv",
    "GBPUSD": "data/raw/GBPUSD_M5_2023-06-09_2026-06-09.csv",
    "USDJPY": "data/raw/USDJPY_M5_2023-06-09_2026-06-09.csv",
    "AUDUSD": "data/raw/AUDUSD_M5_2023-06-09_2026-06-09.csv",
}


def test_pnl_to_usd():
    """Verify _pnl_to_usd for each pair."""
    print("Verifying _pnl_to_usd conversion...")
    tests = [
        # (symbol, pnl_raw, entry, exit, expected_approx)
        ("EURUSD", 100.0, 1.10, 1.11, 100.0),    # USD-quoted → no conversion
        ("GBPUSD", 100.0, 1.30, 1.31, 100.0),    # USD-quoted → no conversion
        ("AUDUSD", 100.0, 0.70, 0.71, 100.0),    # USD-quoted → no conversion
        ("USDJPY", -5000.0, 150.00, 149.50, -33.33),  # JPY-quoted → divide by mid=149.75
        ("USDJPY", 10000.0, 150.00, 151.00, 66.45),   # JPY-quoted → divide by mid=150.50
    ]
    for sym, pnl, entry, exit, expected in tests:
        result = _pnl_to_usd(pnl, sym, entry, exit)
        ok = abs(result - expected) < 0.1 or (sym.endswith("USD") and result == expected)
        status = "✓" if ok else "✗"
        print(f"  {status} {sym}: {pnl} → {result:.2f} (expected ~{expected:.2f})")
        if not ok:
            print(f"    WARNING: unexpected conversion!")
    print()


def run_symbol(symbol: str, csv_path: str) -> tuple:
    """Run backtest for one symbol, return (trades_df, metrics_dict)."""
    print(f"\n{'='*55}")
    print(f"  Running {symbol}...")
    print(f"  Data: {csv_path}")
    print(f"{'='*55}")

    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index)
    print(f"  Loaded {len(df)} candles ({df.index[0]} to {df.index[-1]})")

    trades, metrics = backtest(
        df=df,
        symbol=symbol,
        initial_balance=INITIAL_BALANCE,
        strategy="rapid",
        progress=True,
    )
    return trades, metrics


def compute_basic_metrics(trades: pd.DataFrame) -> dict:
    """Compute simple metrics from trade log."""
    if trades.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0, "pf": 0,
                "total_pnl": 0, "avg_rr": 0, "max_dd_pct": 0}

    closed = trades[trades["status"].isin(["WIN", "LOSS"])].copy()
    if closed.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "wr": 0, "pf": 0,
                "total_pnl": 0, "avg_rr": 0, "max_dd_pct": 0}

    wins = closed[closed["status"] == "WIN"]
    losses = closed[closed["status"] == "LOSS"]
    n_wins = len(wins)
    n_losses = len(losses)
    n_total = len(closed)
    wr = n_wins / n_total if n_total > 0 else 0

    gross_profit = wins["pnl_usd"].sum() if not wins.empty else 0
    gross_loss = abs(losses["pnl_usd"].sum()) if not losses.empty else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    total_pnl = gross_profit - gross_loss

    avg_win = wins["pnl_usd"].mean() if not wins.empty else 0
    avg_loss = losses["pnl_usd"].mean() if not losses.empty else 0
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Equity curve / drawdown
    balance = INITIAL_BALANCE
    peak = INITIAL_BALANCE
    max_dd = 0
    for _, t in closed.iterrows():
        balance += t["pnl_usd"]
        peak = max(peak, balance)
        dd = peak - balance
        max_dd = max(max_dd, dd)
    max_dd_pct = max_dd / peak if peak > 0 else 0

    # Max losing streak
    streak = 0
    max_streak = 0
    for _, t in closed.iterrows():
        if t["pnl_usd"] < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "trades": n_total,
        "wins": n_wins,
        "losses": n_losses,
        "wr": wr,
        "pf": pf,
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_rr": round(avg_rr, 2),
        "max_dd_pct": round(max_dd_pct, 4),
        "max_losing_streak": max_streak,
    }


def main():
    print("=" * 55)
    print("  MULTI-PAIR BACKTEST (FIXED USDJPY PNL)")
    print(f"  SL=1.0×ATR, TP=2.0×ATR, 0.1 lot, Session 1-9 UTC")
    print(f"  Initial Balance: ${INITIAL_BALANCE:,.0f}")
    print("=" * 55)

    # Step 1: verify PnL conversion
    test_pnl_to_usd()

    # Step 2: run each pair
    all_trades = []
    all_metrics = {}
    for sym in SYMBOLS:
        csv_path = CSV_FILES[sym]
        if not os.path.exists(csv_path):
            print(f"\n  SKIP {sym}: file not found at {csv_path}")
            continue
        trades, metrics = run_symbol(sym, csv_path)
        basic = compute_basic_metrics(trades)
        all_metrics[sym] = basic
        if not trades.empty:
            trades["symbol"] = sym
            all_trades.append(trades)

    # Step 3: print per-pair results
    print(f"\n{'='*55}")
    print(f"  PER-PAIR RESULTS")
    print(f"{'='*55}")
    for sym in SYMBOLS:
        m = all_metrics.get(sym, {})
        print(f"\n  {sym}:")
        print(f"    Trades:       {m.get('trades', 0)}")
        print(f"    Win Rate:     {m.get('wr', 0):.1%}")
        print(f"    Profit Fact:  {m.get('pf', 0):.2f}")
        print(f"    Total P&L:    ${m.get('total_pnl', 0):+.2f}")
        print(f"    Avg Win/Loss: ${m.get('avg_win', 0):+.2f} / ${m.get('avg_loss', 0):+.2f}")
        print(f"    Avg R:R:      {m.get('avg_rr', 0):.2f}")
        print(f"    Max DD:       {m.get('max_dd_pct', 0):.1%}")
        print(f"    Max L Streak: {m.get('max_losing_streak', 0)}")

    # Step 4: combined results
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined = combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        print(f"\n{'='*55}")
        print(f"  COMBINED RESULTS ({len(SYMBOLS)} pairs, {len(combined)} total trades)")
        print(f"{'='*55}")

        closed = combined[combined["status"].isin(["WIN", "LOSS"])]
        wins = closed[closed["status"] == "WIN"]
        losses = closed[closed["status"] == "LOSS"]
        n_wins = len(wins)
        n_losses = len(losses)
        n_closed = len(closed)
        wr = n_wins / n_closed if n_closed > 0 else 0
        gp = wins["pnl_usd"].sum() if not wins.empty else 0
        gl = abs(losses["pnl_usd"].sum()) if not losses.empty else 0
        pf = gp / gl if gl > 0 else 0
        total_pnl = gp - gl
        avg_win = wins["pnl_usd"].mean() if not wins.empty else 0
        avg_loss = losses["pnl_usd"].mean() if not losses.empty else 0
        avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Equity curve
        balance = INITIAL_BALANCE
        peak = INITIAL_BALANCE
        max_dd = 0
        streak = 0
        max_streak = 0
        for _, t in closed.iterrows():
            balance += t["pnl_usd"]
            peak = max(peak, balance)
            dd = peak - balance
            max_dd = max(max_dd, dd)
            if t["pnl_usd"] < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        max_dd_pct = max_dd / peak if peak > 0 else 0

        print(f"  Pairs:          {', '.join(SYMBOLS)}")
        print(f"  Total trades:   {n_closed}")
        print(f"  Win Rate:       {wr:.1%}")
        print(f"  Profit Factor:  {pf:.2f}")
        print(f"  Total P&L:      ${total_pnl:+.2f}")
        print(f"  Final Balance:  ${balance:.2f}")
        print(f"  Avg Win/Loss:   ${avg_win:+.2f} / ${avg_loss:+.2f}")
        print(f"  Avg R:R:        {avg_rr:.2f}")
        print(f"  Max Drawdown:   {max_dd_pct:.1%}")
        print(f"  Max L Streak:   {max_streak}")

        # Per-pair breakdown in combined
        print(f"\n  Per-pair within combined:")
        for sym in SYMBOLS:
            mask = combined["symbol"] == sym
            sym_trades = combined[mask]
            sym_closed = sym_trades[sym_trades["status"].isin(["WIN", "LOSS"])]
            sym_pnl = sym_closed["pnl_usd"].sum()
            print(f"    {sym}: {len(sym_closed)} trades, P&L ${sym_pnl:+.2f}")

    else:
        print("\n  No trades generated for any pair.")

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"{'='*55}")

    # Save combined log
    if all_trades:
        combined.to_csv("logs/backtest_combined_pnl_fix.csv", index=False)
        print(f"  Combined trade log saved: logs/backtest_combined_pnl_fix.csv")

    # Restore lot sizer
    ls.compute_lot_size_sim = _original_lot_sizer


if __name__ == "__main__":
    main()
