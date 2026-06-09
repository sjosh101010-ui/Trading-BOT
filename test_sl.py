#!/usr/bin/env python3
"""Run proper backtest with different SL multipliers using the real backtest runner."""
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config
from backtest.runner import run_backtest, backtest

sl_values = [0.5, 1.0, 1.5]
orig_sl = config.RAPID_SL_MULT

for sl in sl_values:
    config.RAPID_SL_MULT = sl
    importlib.reload(config)  # Refresh the imported module
    # Re-import the runner modules that use config values
    importlib.reload(sys.modules.get('risk.lot_sizer'))
    
    print(f"\n{'='*55}")
    print(f"  TESTING SL = {sl}×ATR (TP=4×ATR)")
    print(f"{'='*55}")
    
    from backtest.runner import run_backtest as rb
    trades, metrics = rb(
        csv_path="data/raw/EURUSD_M5_30d.csv",
        symbol="EURUSD", initial_balance=100.0, strategy="rapid"
    )

config.RAPID_SL_MULT = orig_sl
