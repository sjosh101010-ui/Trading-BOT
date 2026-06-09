import pytest
import pandas as pd
import numpy as np
from analysis.ml_model import build_dataset, FEATURE_COLS


def test_build_dataset_shape():
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({
        "open": np.random.uniform(1.07, 1.09, 100),
        "high": np.random.uniform(1.07, 1.09, 100),
        "low": np.random.uniform(1.07, 1.09, 100),
        "close": np.random.uniform(1.07, 1.09, 100),
        "volume": np.random.randint(100, 1000, 100),
    }, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) + 0.002
    df["low"] = df[["open", "close"]].min(axis=1) - 0.002

    X, y = build_dataset(df)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(FEATURE_COLS)
    assert y.ndim == 1
