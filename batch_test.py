#!/usr/bin/env python3
"""Batch backtest various configs via subprocess for clean module reload."""
import subprocess, re, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

CONFIG_PATH = "config.py"

def set_config(sl, tp, threshold, adx_threshold, use_ema50_str):
    with open(CONFIG_PATH, "r") as f:
        c = f.read()
    c = re.sub(r'RAPID_SL_MULT = [0-9]+\.[0-9]+', f'RAPID_SL_MULT = {sl}', c)
    c = re.sub(r'"EURUSD": [0-9]+\.[0-9]+', f'"EURUSD": {tp}', c)
    c = re.sub(r'RAPID_PA_THRESHOLD = [0-9]+\.[0-9]+', f'RAPID_PA_THRESHOLD = {threshold}', c)
    with open(CONFIG_PATH, "w") as f:
        f.write(c)
    
    # Also update backtest runner's rapid strategy if needed
    # Need to modify runner.py too for ADX/EMA50 toggle
    with open("backtest/runner.py", "r") as f:
        r = f.read()
    
    if adx_threshold > 0:
        r = re.sub(r'adx_val < [0-9]+', f'adx_val < {adx_threshold}', r)
    else:
        r = re.sub(r'adx_val < [0-9]+\n\s+continue', '', r)
    
    # For EMA50, we need to toggle the guard lines
    if use_ema50_str == "yes":
        pass
    else:
        pass
    
    with open("backtest/runner.py", "w") as f:
        f.write(r)

# Test: higher PA threshold + wider SL
tests = [
    ("PA≥0.3 SL=1.5× TP=4× NoADX NoEMA50", dict(sl=1.5, tp=4.0, pa=0.3)),
    ("PA≥0.5 SL=1.5× TP=4× NoADX NoEMA50", dict(sl=1.5, tp=4.0, pa=0.5)),
    ("PA≥0.3 SL=2.0× TP=4× NoADX NoEMA50", dict(sl=2.0, tp=4.0, pa=0.3)),
    ("PA≥0.3 SL=0.5× TP=4× ADX≥12 EMA50", dict(sl=0.5, tp=4.0, pa=0.3)),
]

# First restore runner back to ADX+EMA50 version
print("Need to fix runner and approach differently")
