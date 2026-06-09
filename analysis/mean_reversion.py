import pandas as pd
import numpy as np


def compute_mr_score(df: pd.DataFrame) -> float:
    if len(df) < 5:
        return 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["close"]
    bb_upper = last["bb_upper"]
    bb_lower = last["bb_lower"]
    bb_mid = last["bb_mid"]
    rsi = last["rsi"]
    atr_val = last.get("atr", 0.001)
    adx = last.get("adx", 20)
    vol_ratio = last.get("vol_ratio", 1.0)

    if atr_val <= 0 or pd.isna(bb_lower) or pd.isna(bb_upper):
        return 0.0

    # Skip in strong trends
    if adx >= 28:
        return 0.0

    score = 0.0

    # How far price is from BB mid (in ATR units)
    dist_from_mid = (price - bb_mid) / atr_val
    bb_width_atr = (bb_upper - bb_lower) / atr_val

    # BUY conditions: oversold + price below mid + returning
    if rsi <= 38 and price <= bb_mid:
        # Distance below BB mid in ATR
        reversion_potential = max(0, (bb_mid - price) / atr_val)
        rsi_extreme = max(0, (38 - rsi) / 38)
        score = 0.5 + min(reversion_potential * 0.2, 0.3) + rsi_extreme * 0.2

        # Bonus: if prev bar was even lower (starting to reverse)
        if prev["close"] < prev["open"] and last["close"] > last["open"]:
            score += 0.1
        # Bonus: near BB lower
        if price <= bb_lower:
            score += 0.1

    # SELL conditions: overbought + price above mid + returning
    elif rsi >= 62 and price >= bb_mid:
        reversion_potential = max(0, (price - bb_mid) / atr_val)
        rsi_extreme = max(0, (rsi - 62) / 38)
        score = -0.5 - min(reversion_potential * 0.2, 0.3) - rsi_extreme * 0.2

        if prev["close"] > prev["open"] and last["close"] < last["open"]:
            score -= 0.1
        if price >= bb_upper:
            score -= 0.1

    return round(max(-1.0, min(1.0, score)), 4)
