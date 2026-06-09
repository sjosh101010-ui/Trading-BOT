#!/usr/bin/env python3
"""
AI Forex Trading System — Daily Use CLI

Usage:
  python trade.py                       Interactive menu
  python trade.py daily                 Full daily routine (download + backtest + status)
  python trade.py status                System status summary
  python trade.py backtest              Run backtest (--from-date YYYY-MM-DD)
  python trade.py train                 Retrain ML model on latest data
  python trade.py download              Download fresh EURUSD M5 data (--days 60)
  python trade.py live                  Start live simulator (yfinance data)
  python trade.py reset --amount 1000   Reset broker balance
  python trade.py balance               Show saved broker balance
  python trade.py walkforward           Run walk-forward analysis
  python trade.py gridsearch            Run grid search optimization
  python trade.py validate              Run H1 validation
  python trade.py paper                 Start paper trader (SimpleFX)
  python trade.py dashboard             Launch Streamlit dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import json
import subprocess
from datetime import datetime, timezone, timedelta
from config import SYMBOLS

BASE = Path(__file__).parent
SYMBOL = SYMBOLS[0]


def banner():
    print("""
  ╔══════════════════════════════════════════════╗
  ║        AI Forex Trading System v3            ║
  ║        EURUSD High Conviction                ║
  ║        24/7 Trading                          ║
  ╚══════════════════════════════════════════════╝
    """)


def run(cmd: str, capture: bool = False) -> str:
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=BASE)
        if result.returncode != 0:
            print(f"[ERROR] {result.stderr.strip()}")
        return result.stdout.strip()
    subprocess.run(cmd, shell=True, cwd=BASE)
    return ""


def cmd_daily():
    banner()
    print(f"  [1/4] Downloading latest {SYMBOL} M5 data...")
    run(f"python3 data/download.py --symbol {SYMBOL} --days 60")

    print("\n  [2/4] Running daily backtest...")
    run(f"python3 run_backtest.py --csv data/raw/{SYMBOL}_M5_60d.csv --symbol {SYMBOL} --strategy hc --balance 1000")

    print("\n  [3/4] System Status:")
    cmd_status(silent=True)

    print("\n  [4/4] Daily Summary")
    print(f"  Run: python trade.py live  (to start live monitor)")
    print(f"  Run: python trade.py train (to retrain ML model)")
    print("  Done.")


def cmd_status(silent: bool = False):
    if not silent:
        banner()

    model_path = BASE / "models" / f"{SYMBOL.lower()}_xgb.pkl"
    data_dir = BASE / "data" / "raw"
    log_dir = BASE / "logs"

    print(f"  System Status — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print("  " + "-" * 46)

    data_files = list(data_dir.glob(f"{SYMBOL}_M5_*.csv"))
    if data_files:
        newest = max(data_files, key=lambda p: p.stat().st_mtime)
        size_kb = newest.stat().st_size / 1024
        mod_time = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%b %d %H:%M")
        print(f"  Data:     {newest.name} ({size_kb:.0f} KB, updated {mod_time})")
    else:
        print(f"  Data:     None — run 'python trade.py download'")

    if model_path.exists():
        mod_time = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%b %d %H:%M")
        size_kb = model_path.stat().st_size / 1024
        print(f"  Model:    {model_path.name} ({size_kb:.0f} KB, saved {mod_time})")
    else:
        print(f"  Model:    None — run 'python trade.py train'")

    backtest_log = log_dir / f"backtest_{SYMBOL.lower()}.csv"
    if backtest_log.exists():
        import pandas as pd
        df = pd.read_csv(backtest_log)
        if not df.empty:
            closed = df[df["status"].isin(["WIN", "LOSS"])]
            wins = len(closed[closed["status"] == "WIN"])
            losses = len(closed[closed["status"] == "LOSS"])
            total = len(closed)
            wr = wins / total * 100 if total > 0 else 0
            pnl = closed["pnl_usd"].sum()
            print(f"  Last BT:  {total} trades, {wins}W/{losses}L, WR {wr:.1f}%, P&L ${pnl:+.2f}")
    else:
        print(f"  Last BT:  None — run 'python trade.py backtest'")

    print("  " + "-" * 46)
    print(f"  Starting: python trade.py live")


def cmd_backtest():
    banner()
    args = sys.argv[2:]
    strat = "hc"
    from_date = None
    for i, a in enumerate(args):
        if a == "--from-date" and i + 1 < len(args):
            from_date = args[i + 1]
        if a == "--strategy" and i + 1 < len(args):
            strat = args[i + 1]
    date_str = f" from {from_date}" if from_date else ""
    print(f"  Running {strat} backtest{date_str}...\n")
    cmd = f"python3 run_backtest.py --csv data/raw/{SYMBOL}_M5_60d.csv --symbol {SYMBOL} --strategy {strat} --balance 1000"
    if from_date:
        cmd += f" --trade-date {from_date}"
    run(cmd)


def cmd_train():
    banner()
    print(f"  Retraining {SYMBOL} XGBoost model...")
    print("  (this may take ~30-60 seconds)")
    run("python3 train_models.py")


def cmd_download():
    banner()
    days = 60
    if len(sys.argv) > 2:
        try:
            days = int(sys.argv[2])
        except:
            pass
    print(f"  Downloading {days} days of {SYMBOL} M5 data...")
    run(f"python3 data/download.py --symbol {SYMBOL} --days {days}")


def cmd_reset():
    banner()
    amount = 100.0
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    for i, a in enumerate(args):
        if a == "--amount" and i + 1 < len(args):
            try:
                amount = float(args[i + 1])
            except ValueError:
                print(f"  Invalid amount: {args[i+1]}")
                return
    from execution.sim_broker import SimBroker
    SimBroker.reset(amount)


def cmd_balance():
    from execution.sim_broker import STATE_FILE
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        print(f"  Broker balance: ${state.get('balance', 0):.2f}")
        print(f"  Equity:         ${state.get('equity', 0):.2f}")
        print(f"  Open positions: {len([p for p in state.get('positions', []) if p.get('status') == 'OPEN'])}")
        print(f"  Updated:        {state.get('updated', 'N/A')}")
    else:
        print("  No saved broker state (run live first)")


def cmd_live():
    banner()
    print("  Starting Live Simulator (yfinance data)")
    print("  Press Ctrl+C to stop\n")
    run("python3 main.py")


def cmd_walkforward():
    banner()
    print("  Running walk-forward analysis...\n")
    run("python3 walk_forward.py")


def cmd_gridsearch():
    banner()
    print("  Running grid search optimization...\n")
    run("python3 grid_search.py")


def cmd_validate():
    banner()
    print("  Running H1 validation...\n")
    run("python3 validate_h1.py")


def cmd_paper():
    banner()
    print("  Starting Paper Trader (SimpleFX)...\n")
    run("python3 paper_trade.py")


def cmd_dashboard():
    banner()
    print("  Launching Streamlit dashboard...\n")
    run("streamlit run dashboard/app.py")


def cmd_menu():
    banner()
    options = {
        "1": ("Daily Routine", "Download, backtest, status", cmd_daily),
        "2": ("Backtest", "Run backtest (--strategy hc)", cmd_backtest),
        "3": ("Train Model", "Retrain XGBoost", cmd_train),
        "4": ("Download Data", "Fetch latest M5 data", cmd_download),
        "5": ("Live Simulator", "Start live monitor", cmd_live),
        "6": ("Reset Balance", "Reset broker (--amount 1000)", cmd_reset),
        "7": ("Balance", "Show broker balance", cmd_balance),
        "8": ("Walk-Forward", "Run walk-forward analysis", cmd_walkforward),
        "9": ("Grid Search", "Run parameter optimization", cmd_gridsearch),
        "10": ("Validate H1", "Run H1 validation", cmd_validate),
        "11": ("Paper Trader", "Start SimpleFX paper trade", cmd_paper),
        "12": ("Dashboard", "Launch Streamlit UI", cmd_dashboard),
        "s": ("Status", "System status summary", cmd_status),
        "q": ("Quit", "", None),
    }
    for k, (label, desc, _) in options.items():
        print(f"  [{k:>2s}] {label:18s} {desc}")
    print()
    choice = input("  Choose: ").strip().lower()
    if choice in options:
        fn = options[choice][2]
        if fn:
            fn()
    elif choice.isdigit() and choice in options:
        fn = options[choice][2]
        if fn:
            fn()
    else:
        print(f"  Unknown option: {choice}")


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ["menu"]
    cmd_map = {
        "menu": cmd_menu,
        "daily": cmd_daily,
        "status": cmd_status,
        "backtest": cmd_backtest,
        "train": cmd_train,
        "download": cmd_download,
        "live": cmd_live,
        "reset": cmd_reset,
        "balance": cmd_balance,
        "walkforward": cmd_walkforward,
        "gridsearch": cmd_gridsearch,
        "validate": cmd_validate,
        "paper": cmd_paper,
        "dashboard": cmd_dashboard,
    }
    cmd = args[0]
    if cmd in cmd_map:
        cmd_map[cmd]()
    else:
        print(f"Usage: python trade.py [{'|'.join(cmd_map.keys())}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
