import pandas as pd
from pathlib import Path
from analysis.ml_model import train_xgb, train_lstm

DATA_DIR = Path("data/raw")
SYMBOLS_M5 = {"BTCUSD"}


def load_m5_csv(symbol: str) -> pd.DataFrame:
    candidates = list(DATA_DIR.glob(f"{symbol}_M5_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No M5 CSV found for {symbol} in {DATA_DIR}")

    best = max(candidates, key=lambda p: p.stat().st_size)
    print(f"  Loading {best}...")
    df = pd.read_csv(best, parse_dates=True, index_col=0)
    print(f"  {len(df)} candles loaded")
    return df


def main():
    print("=" * 50)
    print("  AI Forex — Model Training")
    print("=" * 50)

    for sym in sorted(SYMBOLS_M5):
        print(f"\n[{sym}] Training XGBoost...")
        try:
            df = load_m5_csv(sym)
            train_xgb(sym, df)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\n{'='*50}")
    print("  All models trained. Verify with:")
    print("  ls -la models/")
    print("=" * 50)


if __name__ == "__main__":
    main()
