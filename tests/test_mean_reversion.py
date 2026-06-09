import pytest
import pandas as pd
import numpy as np
from analysis.mean_reversion import compute_mr_score


def make_mr_df(close_price, rsi_val, bb_upper, bb_lower, bb_mid,
               atr_val=0.002, adx_val=20, vol_ratio=1.0,
               prev_close=None, prev_open=None):
    idx = pd.date_range("2024-01-01", periods=5, freq="5min")
    if prev_close is None:
        prev_close = close_price * 0.999
    if prev_open is None:
        prev_open = close_price * 1.001
    data = {
        "open": [close_price * 0.9995] * 4 + [close_price * 0.9995],
        "high": [close_price * 1.001] * 5,
        "low": [close_price * 0.999] * 5,
        "close": [prev_close] * 4 + [close_price],
        "volume": [500] * 5,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid,
        "rsi": rsi_val, "atr": atr_val, "adx": adx_val, "vol_ratio": vol_ratio,
    }
    df = pd.DataFrame(data, index=idx)
    df.loc[df.index[-2], "close"] = prev_close
    df.loc[df.index[-2], "open"] = prev_open
    return df


def test_oversold_buy_signal():
    df = make_mr_df(
        close_price=1.0760, rsi_val=30, bb_upper=1.0900,
        bb_lower=1.0760, bb_mid=1.0830, atr_val=0.002
    )
    score = compute_mr_score(df)
    assert score > 0, f"Expected positive for oversold, got {score}"
    assert -1.0 <= score <= 1.0


def test_overbought_sell_signal():
    df = make_mr_df(
        close_price=1.0890, rsi_val=70, bb_upper=1.0900,
        bb_lower=1.0760, bb_mid=1.0830, atr_val=0.002
    )
    score = compute_mr_score(df)
    assert score < 0, f"Expected negative for overbought, got {score}"
    assert -1.0 <= score <= 1.0


def test_strong_trend_no_signal():
    df = make_mr_df(
        close_price=1.0760, rsi_val=30, bb_upper=1.0900,
        bb_lower=1.0760, bb_mid=1.0830, atr_val=0.002, adx_val=30
    )
    score = compute_mr_score(df)
    assert score == 0.0, f"Expected 0 for ADX >= 28, got {score}"


def test_mid_range_no_signal():
    df = make_mr_df(
        close_price=1.0830, rsi_val=50, bb_upper=1.0900,
        bb_lower=1.0760, bb_mid=1.0830, atr_val=0.002
    )
    score = compute_mr_score(df)
    assert score == 0.0


def test_insufficient_data():
    df = pd.DataFrame({"close": [1.08] * 3, "high": [1.081] * 3,
                       "low": [1.079] * 3, "volume": [100] * 3,
                       "bb_upper": 1.09, "bb_lower": 1.07, "bb_mid": 1.08,
                       "rsi": 30, "atr": 0.002, "adx": 20, "vol_ratio": 1.0})
    score = compute_mr_score(df)
    assert score == 0.0


def test_atr_zero_returns_zero():
    df = make_mr_df(
        close_price=1.0760, rsi_val=30, bb_upper=1.0900,
        bb_lower=1.0760, bb_mid=1.0830, atr_val=0
    )
    score = compute_mr_score(df)
    assert score == 0.0


def test_reversal_bonus_bull():
    df = make_mr_df(
        close_price=1.0760, rsi_val=30, bb_upper=1.0900,
        bb_lower=1.0760, bb_mid=1.0830, atr_val=0.002,
        prev_close=1.0750, prev_open=1.0770
    )
    score = compute_mr_score(df)
    assert score > 0


def test_reversal_bonus_bear():
    df = make_mr_df(
        close_price=1.0890, rsi_val=70, bb_upper=1.0900,
        bb_lower=1.0760, bb_mid=1.0830, atr_val=0.002,
        prev_close=1.0900, prev_open=1.0880
    )
    score = compute_mr_score(df)
    assert score < 0
