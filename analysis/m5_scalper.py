import pandas as pd
import numpy as np


def compute_m5_score(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    price = last["close"]
    atr_val = last["atr"]
    if atr_val <= 0:
        return 0.0

    n = len(df)
    rsi = last.get("rsi", 50)

    if n < 3:
        return 0.0

    ret1 = (price - df.iloc[-2]["close"]) / atr_val
    ret2 = (df.iloc[-2]["close"] - df.iloc[-3]["close"]) / atr_val
    extend = ret1 + ret2 * 0.5

    vol_series = df["atr"]
    vol_median = vol_series.rolling(50).median().iloc[-1] if n >= 50 else atr_val
    vol_ratio = atr_val / vol_median if vol_median > 0 else 1.0
    if vol_ratio < 0.6 or vol_ratio > 2.0:
        return 0.0

    max_range = (df["high"].iloc[-20:].max() - df["low"].iloc[-20:].min()) / atr_val if n >= 20 else 5.0
    pos = (price - df["low"].iloc[-20:].min()) / (df["high"].iloc[-20:].max() - df["low"].iloc[-20:].min()) if n >= 20 and max_range > 0 else 0.5

    at_top = pos > 0.85
    at_bottom = pos < 0.15
    overbought_rsi = rsi > 70
    oversold_rsi = rsi < 30
    extended_up = extend > 0.8
    extended_down = extend < -0.8

    strong_buy = oversold_rsi and extended_down
    strong_sell = overbought_rsi and extended_up
    mod_buy = at_bottom and extended_down
    mod_sell = at_top and extended_up

    score = 0.0
    if strong_buy:
        score += 0.6
    elif oversold_rsi:
        score += 0.3
    elif extended_down:
        score += 0.25
    elif mod_buy:
        score += 0.2

    if strong_sell:
        score -= 0.6
    elif overbought_rsi:
        score -= 0.3
    elif extended_up:
        score -= 0.25
    elif mod_sell:
        score -= 0.2

    if n >= 5:
        cls = df["close"].iloc[-6:-1]
        if all(cls.iloc[i] > cls.iloc[i-1] for i in range(1, len(cls))):
            score -= 0.15
        elif all(cls.iloc[i] < cls.iloc[i-1] for i in range(1, len(cls))):
            score += 0.15

    return round(max(-1.0, min(1.0, score)), 4)


def check_price_divergence(
    df: pd.DataFrame, pa_score: float, lookback: int = 8, lookahead: int = 4
) -> bool:
    """
    Check if PA score diverges from short-term price structure.
    Returns True if divergence exists → trade should be skipped.

    Logic:
    - PA says BUY but price broke below recent support
      (last `lookahead` closes are all below the lowest low of the first
      `lookback` bars in the window) → skip
    - PA says SELL but price broke above recent resistance
      (last `lookahead` closes are all above the highest high of the first
      `lookback` bars in the window) → skip
    """
    if abs(pa_score) < 0.01 or len(df) < lookback + lookahead + 1:
        return False

    window = df.tail(lookback + lookahead + 1).iloc[:-1]

    if pa_score > 0:
        support = window.head(lookback)["low"].min()
        if window.tail(lookahead)["close"].max() < support:
            return True
    else:
        resistance = window.head(lookback)["high"].max()
        if window.tail(lookahead)["close"].min() > resistance:
            return True

    return False
