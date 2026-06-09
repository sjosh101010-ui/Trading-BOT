import pytest
import pandas as pd
import numpy as np
from analysis.price_action import detect_patterns, compute_pa_score, detect_sr_zone


def make_candle(open, high, low, close):
    return pd.DataFrame({
        "open": [open], "high": [high], "low": [low], "close": [close],
        "volume": [100]
    })


def test_compute_pa_score_range():
    idx = pd.date_range("2024-01-01", periods=10, freq="5min")
    df = pd.DataFrame({
        "open": np.random.uniform(1.07, 1.09, 10),
        "high": np.random.uniform(1.07, 1.09, 10),
        "low": np.random.uniform(1.07, 1.09, 10),
        "close": np.random.uniform(1.07, 1.09, 10),
        "volume": np.random.randint(100, 1000, 10),
    }, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) + 0.002
    df["low"] = df[["open", "close"]].min(axis=1) - 0.002

    score = compute_pa_score(df)
    assert -1.0 <= score <= 1.0


def test_detect_sr_zone():
    idx = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({
        "open": np.random.uniform(1.07, 1.09, 100),
        "high": np.random.uniform(1.07, 1.09, 100),
        "low": np.random.uniform(1.07, 1.09, 100),
        "close": np.random.uniform(1.07, 1.09, 100),
        "volume": np.random.randint(100, 1000, 100),
    }, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) + 0.005
    df["low"] = df[["open", "close"]].min(axis=1) - 0.005

    zones = detect_sr_zone(df)
    assert "support" in zones
    assert "resistance" in zones
    assert zones["support"] < zones["resistance"]
