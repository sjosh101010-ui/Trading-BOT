import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from analysis.feature_engineering import add_indicators
from analysis.ml_model import FEATURE_COLS
from config import SYMBOLS

SYMBOL = SYMBOLS[0]
DATA_DIR = Path("data/raw")


def load_data() -> pd.DataFrame:
    candidates = list(DATA_DIR.glob(f"{SYMBOL}_M5_*.csv"))
    best = max(candidates, key=lambda p: p.stat().st_size)
    print(f"Loading {best}...")
    df = pd.read_csv(best, parse_dates=True, index_col=0)
    print(f"  {len(df)} candles loaded, range {df.index[0]} to {df.index[-1]}")
    df = add_indicators(df)
    return df


def build_dataset(df: pd.DataFrame, lookahead: int, threshold: float):
    df = df.copy()
    df["future_close"] = df["close"].shift(-lookahead)
    df["label"] = np.where(
        df["future_close"] > df["close"] * (1 + threshold), 1,
        np.where(df["future_close"] < df["close"] * (1 - threshold), 0, np.nan)
    )
    df.dropna(inplace=True)
    valid_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[valid_cols].values
    y = df["label"].values.astype(int)
    return X, y, valid_cols


def grid_search_labels(df: pd.DataFrame):
    print("\n=== Phase 1: Label Definition Grid ===")
    lookaheads = [2, 3, 5, 8, 13]
    thresholds = [0.0002, 0.0003, 0.0005, 0.0008, 0.0010]
    best_acc = 0
    best_params = {}
    results = []

    for la in lookaheads:
        for th in thresholds:
            X, y, cols = build_dataset(df, la, th)
            if len(np.unique(y)) < 2:
                continue
            tscv = TimeSeriesSplit(n_splits=3)
            accs = []
            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                model = XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=42, verbosity=0,
                )
                model.fit(X_tr, y_tr, verbose=False)
                acc = accuracy_score(y_te, model.predict(X_te))
                accs.append(acc)
            mean_acc = np.mean(accs)
            n_pos = int(y.sum())
            n_total = len(y)
            results.append((la, th, mean_acc, n_pos, n_total))
            marker = " ***" if mean_acc > best_acc else ""
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = {"lookahead": la, "threshold": th}
            print(f"  lookahead={la:2d}  threshold={th:.4f}  acc={mean_acc:.4f}  pos={n_pos}/{n_total}{marker}")

    print(f"\nBest label: lookahead={best_params['lookahead']}, threshold={best_params['threshold']}, acc={best_acc:.4f}")
    return best_params


def grid_search_hyperparams(df: pd.DataFrame, lookahead: int, threshold: float):
    print("\n=== Phase 2: Hyperparameter Grid ===")
    X, y, cols = build_dataset(df, lookahead, threshold)
    print(f"Data: {len(X)} samples, {len(cols)} features, pos={int(y.sum())}/{len(y)}")

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
    }

    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_acc = 0
    best_combo = {}
    n_combos = np.prod([len(v) for v in values])
    print(f"Total combinations: {n_combos}")
    if n_combos > 500:
        print("Too many combos, using grouped search...")
        tuned_params = {}

        for key in keys:
            best_for_key = 0
            best_val = None
            for val in param_grid[key]:
                params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
                          "subsample": 0.8, "colsample_bytree": 0.8,
                          "min_child_weight": 1, "gamma": 0}
                params.update(tuned_params)
                params[key] = val
                tscv = TimeSeriesSplit(n_splits=3)
                accs = []
                for train_idx, test_idx in tscv.split(X):
                    X_tr, X_te = X[train_idx], X[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)
                    model = XGBClassifier(
                        use_label_encoder=False, eval_metric="logloss",
                        random_state=42, verbosity=0, **params
                    )
                    model.fit(X_tr, y_tr, verbose=False)
                    acc = accuracy_score(y_te, model.predict(X_te))
                    accs.append(acc)
                mean_acc = np.mean(accs)
                marker = " ***" if mean_acc > best_for_key else ""
                print(f"  {key}={str(val):>6s}  acc={mean_acc:.4f}{marker}")
                if mean_acc > best_for_key:
                    best_for_key = mean_acc
                    best_val = val
            tuned_params[key] = best_val
            print(f"  -> Best {key}: {best_val} (acc={best_for_key:.4f})")

        return tuned_params
    else:
        tscv = TimeSeriesSplit(n_splits=3)
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            accs = []
            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                model = XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss",
                    random_state=42, verbosity=0, **params
                )
                model.fit(X_tr, y_tr, verbose=False)
                acc = accuracy_score(y_te, model.predict(X_te))
                accs.append(acc)
            mean_acc = np.mean(accs)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_combo = params
                print(f"  NEW BEST acc={mean_acc:.4f}: {params}")
        print(f"\nBest hyperparams: {best_combo}")
        print(f"Best accuracy: {best_acc:.4f}")
        return best_combo


def feature_importance(df: pd.DataFrame, lookahead: int, threshold: float, best_params: dict):
    print("\n=== Phase 3: Feature Importance ===")
    X, y, cols = build_dataset(df, lookahead, threshold)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0, **best_params
    )
    model.fit(X_scaled, y, verbose=False)
    importances = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])
    print(f"{'Feature':<20s} {'Importance':>10s}")
    print("-" * 30)
    for feat, imp in importances:
        print(f"{feat:<20s} {imp:>10.4f}")
    return importances


def final_train(df: pd.DataFrame, lookahead: int, threshold: float, best_params: dict):
    print("\n=== Phase 4: Final Train ===")
    from analysis.ml_model import train_xgb
    print(f"Training with: lookahead={lookahead}, threshold={threshold}, params={best_params}")
    import analysis.ml_model as ml
    ml.ML_LOOKAHEAD_BARS = lookahead
    ml.ML_LABEL_THRESHOLD = threshold
    ml.XGB_N_ESTIMATORS = best_params.get("n_estimators", 300)
    train_xgb(SYMBOL, df)


def main():
    print("=" * 60)
    print(f"  XGBoost Grid Search — {SYMBOL} M5")
    print("=" * 60)
    df = load_data()
    best_label = grid_search_labels(df)
    best_hp = grid_search_hyperparams(df, best_label["lookahead"], best_label["threshold"])
    feature_importance(df, best_label["lookahead"], best_label["threshold"], best_hp)
    final_train(df, best_label["lookahead"], best_label["threshold"], best_hp)


if __name__ == "__main__":
    main()
