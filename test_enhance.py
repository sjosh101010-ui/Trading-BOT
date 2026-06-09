#!/usr/bin/env python3
"""Quick targeted tests via subprocess for clean reloads."""
import subprocess, re

def test(label, sl, tp, pa_thresh):
    with open("config.py") as f:
        c = f.read()
    c = re.sub(r'RAPID_SL_MULT = [0-9]+\.[0-9]+', f'RAPID_SL_MULT = {sl}', c)
    c = re.sub(r'"EURUSD": [0-9]+\.[0-9]+', f'"EURUSD": {tp}', c)
    c = re.sub(r'RAPID_PA_THRESHOLD = [0-9]+\.[0-9]+', f'RAPID_PA_THRESHOLD = {pa_thresh}', c)
    with open("config.py", "w") as f:
        f.write(c)
    
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  SL={sl}×  TP={tp}×  PA≥{pa_thresh}")
    print(f"{'='*55}")
    
    result = subprocess.run(
        ["python3", "run_backtest.py", "--csv", "data/raw/EURUSD_M5_30d.csv",
         "--symbol", "EURUSD", "--strategy", "rapid", "--balance", "100.0"],
        capture_output=True, text=True, timeout=300
    )
    for line in result.stdout.split("\n"):
        if any(kw in line for kw in ["Results:", "Win Rate:", "Profit Factor", 
                                      "Total P&L", "Final Balance", "Avg Win",
                                      "Max Drawdown", "Total Trades"]):
            print(f"  {line.strip()}")

# Restore runner back to current state (ADX+EMA50)
print("Test 1: Current config (baseline)")
test("PA≥0.01 SL=0.3× TP=4× ADX+EMA50", sl=0.3, tp=4.0, pa_thresh=0.01)

print("\nTest 2: Higher PA threshold + wider SL")
test("PA≥0.30 SL=1.5× TP=4× ADX+EMA50", sl=1.5, tp=4.0, pa_thresh=0.30)

print("\nTest 3: Higher PA threshold, moderate SL")
test("PA≥0.20 SL=1.0× TP=4× ADX+EMA50", sl=1.0, tp=4.0, pa_thresh=0.20)

print("\nTest 4: Strong signal, wide SL")
test("PA≥0.40 SL=2.0× TP=4× ADX+EMA50", sl=2.0, tp=4.0, pa_thresh=0.40)
