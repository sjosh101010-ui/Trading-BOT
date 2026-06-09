import pandas as pd
import numpy as np


def compute_tf_score(df: pd.DataFrame, threshold: float = 0.25) -> float:
    if len(df) < 50:
        return 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["close"]
    ema50 = last.get("ema_50", price)
    ema21 = last.get("ema_21", price)
    atr_val = last.get("atr", 0.001)
    adx_val = last.get("adx", 20)

    if atr_val <= 0:
        return 0.0

    uptrend = price > ema50
    downtrend = price < ema50

    # Require ADX > 25 (trending market)
    if adx_val <= 25:
        return 0.0

    # Distance from EMA21 in ATR units
    dist_ema21 = (price - ema21) / atr_val

    score = 0.0

    if uptrend and -threshold <= dist_ema21 <= threshold:
        score = 0.6
        if prev["close"] < prev["open"] and last["close"] > last["open"]:
            score += 0.15
    elif downtrend and -threshold <= dist_ema21 <= threshold:
        score = -0.6
        if prev["close"] > prev["open"] and last["close"] < last["open"]:
            score -= 0.15

    return round(max(-1.0, min(1.0, score)), 4)
