import pytest
import pandas as pd
import numpy as np
from analysis.feature_engineering import add_indicators
from analysis.m5_scalper import compute_m5_score


def make_scalper_df(close_price, rsi_val, bb_upper, bb_lower, bb_mid,
                    ema8, ema21, ema50, atr_val, high=None, low=None):
    idx = pd.date_range("2024-01-01", periods=10, freq="5min")
    if high is None:
        high = close_price * 1.001
    if low is None:
        low = close_price * 0.999
    df = pd.DataFrame({
        "open": close_price * 0.9995, "high": high, "low": low,
        "close": close_price, "volume": 500,
        "ema_8": ema8, "ema_21": ema21, "ema_50": ema50,
        "rsi": rsi_val, "atr": atr_val,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid,
    }, index=idx)
    return df


def test_oversold_bullish():
    df = make_scalper_df(
        close_price=1.0800, rsi_val=28, bb_upper=1.0850, bb_lower=1.0750,
        bb_mid=1.0800, ema8=1.0790, ema21=1.0780, ema50=1.0770, atr_val=0.002
    )
    score = compute_m5_score(df)
    assert score > 0, f"Expected positive score for oversold, got {score}"
    assert -1.0 <= score <= 1.0


def test_overbought_bearish():
    df = make_scalper_df(
        close_price=1.0800, rsi_val=75, bb_upper=1.0850, bb_lower=1.0750,
        bb_mid=1.0800, ema8=1.0810, ema21=1.0820, ema50=1.0830, atr_val=0.002
    )
    score = compute_m5_score(df)
    assert score < 0, f"Expected negative score for overbought, got {score}"
    assert -1.0 <= score <= 1.0


def test_bb_lower_bounce():
    df = make_scalper_df(
        close_price=1.0753, rsi_val=38, bb_upper=1.0850, bb_lower=1.0750,
        bb_mid=1.0800, ema8=1.0790, ema21=1.0780, ema50=1.0770, atr_val=0.002
    )
    score = compute_m5_score(df)
    assert score > 0, f"Expected positive score near BB lower, got {score}"


def test_bb_upper_rejection():
    df = make_scalper_df(
        close_price=1.0847, rsi_val=62, bb_upper=1.0850, bb_lower=1.0750,
        bb_mid=1.0800, ema8=1.0810, ema21=1.0820, ema50=1.0830, atr_val=0.002
    )
    score = compute_m5_score(df)
    assert score < 0, f"Expected negative score near BB upper, got {score}"


def test_trend_amplification_bull():
    high = np.array([1.0810, 1.0815, 1.0820, 1.0825, 1.0830,
                     1.0835, 1.0840, 1.0845, 1.0850, 1.0855])
    low = np.array([1.0790, 1.0795, 1.0800, 1.0805, 1.0810,
                    1.0815, 1.0820, 1.0825, 1.0830, 1.0835])
    df = make_scalper_df(
        close_price=1.0840, rsi_val=35, bb_upper=1.0900, bb_lower=1.0760,
        bb_mid=1.0830, ema8=1.0830, ema21=1.0810, ema50=1.0790,
        atr_val=0.002, high=high[-1], low=low[-1]
    )
    df["high"] = high
    df["low"] = low
    score = compute_m5_score(df)
    assert score > 0, f"Expected positive in uptrend, got {score}"


def test_score_clamped():
    df = make_scalper_df(
        close_price=1.0800, rsi_val=28, bb_upper=1.0850, bb_lower=1.0750,
        bb_mid=1.0800, ema8=1.0790, ema21=1.0780, ema50=1.0770, atr_val=0.002
    )
    score = compute_m5_score(df)
    assert -1.0 <= score <= 1.0


def test_atr_zero_returns_zero():
    df = make_scalper_df(
        close_price=1.0800, rsi_val=50, bb_upper=1.0850, bb_lower=1.0750,
        bb_mid=1.0800, ema8=1.0800, ema21=1.0800, ema50=1.0800, atr_val=0
    )
    score = compute_m5_score(df)
    assert score == 0.0


def test_neutral_market():
    df = make_scalper_df(
        close_price=1.0800, rsi_val=50, bb_upper=1.0850, bb_lower=1.0750,
        bb_mid=1.0800, ema8=1.0800, ema21=1.0800, ema50=1.0800, atr_val=0.002
    )
    score = compute_m5_score(df)
    assert -1.0 <= score <= 1.0
