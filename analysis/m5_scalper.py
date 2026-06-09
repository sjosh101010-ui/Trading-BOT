import pandas as pd
import numpy as np


def compute_m5_score(df: pd.DataFrame) -> float:
    last = df.iloc[-1]
    price = last["close"]
    rsi = last["rsi"]
    bb_upper = last["bb_upper"]
    bb_lower = last["bb_lower"]
    bb_mid = last["bb_mid"]
    ema8 = last["ema_8"]
    ema21 = last["ema_21"]
    ema50 = last["ema_50"]
    atr_val = last["atr"]

    if atr_val <= 0 or pd.isna(bb_lower) or pd.isna(bb_upper):
        return 0.0

    score = 0.0

    is_uptrend = price > ema50
    is_downtrend = price < ema50
    no_trend = not is_uptrend and not is_downtrend

    if rsi < 30:
        score += 0.5
    elif rsi < 40:
        score += 0.25
    elif rsi > 70:
        score -= 0.5
    elif rsi > 60:
        score -= 0.25

    bb_upper_dist = (bb_upper - price) / atr_val
    bb_lower_dist = (price - bb_lower) / atr_val

    if bb_lower_dist < 0.3 and rsi < 45 and not is_downtrend:
        score += 0.4
    elif bb_lower_dist < 0.5 and rsi < 40 and not is_downtrend:
        score += 0.25

    if bb_upper_dist < 0.3 and rsi > 55 and not is_uptrend:
        score -= 0.4
    elif bb_upper_dist < 0.5 and rsi > 60 and not is_uptrend:
        score -= 0.25

    ema_dist = (price - ema21) / atr_val
    if is_uptrend and -1.5 < ema_dist < -0.3:
        score += 0.3
    if is_downtrend and 0.3 < ema_dist < 1.5:
        score -= 0.3

    if len(df) > 5:
        range_high = df["high"].iloc[-6:-1].max()
        range_low = df["low"].iloc[-6:-1].min()
        range_width = (range_high - range_low) / atr_val

        if price > range_high and range_width < 1.5 and is_uptrend:
            score += 0.3
        elif price < range_low and range_width < 1.5 and is_downtrend:
            score -= 0.3

    if is_uptrend and score > 0:
        score *= 1.2
    elif is_downtrend and score < 0:
        score *= 1.2

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
