#!/usr/bin/env python3
"""Quick debug script to check what scores are being generated."""
import sys
sys.path.insert(0, '/Users/sdcc-mis01/Personal/Trade (EURUSD)')

import pandas as pd
import numpy as np
from datetime import datetime, timezone

from run_backtest import generate_synthetic_data
from backtest.runner import backtest
from config import BUY_THRESHOLD, SELL_THRESHOLD, ML_CONFIDENCE_GATE

# Generate small dataset
df = generate_synthetic_data("BTCUSD", 100)
print(f"Generated {len(df)} candles")
print(f"Data range: {df.index[0]} to {df.index[-1]}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nBasic stats:")
print(df.describe())

# Run backtest with debug
print("\n" + "="*60)
print("Running backtest...")
print(f"Config: BUY_THRESHOLD={BUY_THRESHOLD}, SELL_THRESHOLD={SELL_THRESHOLD}")
print(f"ML_CONFIDENCE_GATE={ML_CONFIDENCE_GATE}")
print("="*60)

trades, metrics = backtest(df=df, symbol="BTCUSD", initial_balance=100.0, strategy="m5")

print(f"\nTrades: {len(trades)}")
if trades:
    print(f"First trade: {trades[0]}")
print(f"\nMetrics: {metrics}")