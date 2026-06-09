import pandas as pd
import numpy as np
from analysis.feature_engineering import candle_body, candle_range, upper_shadow, lower_shadow

PATTERN_WEIGHTS = {
    "hammer":          0.7,
    "shooting_star":   0.7,
    "engulfing_bull":  0.9,
    "engulfing_bear":  0.9,
    "doji":            0.3,
    "morning_star":    0.8,
    "evening_star":    0.8,
    "pin_bar_bull":    0.75,
    "pin_bar_bear":    0.75,
}

BULLISH_PATTERNS = {"hammer", "engulfing_bull", "morning_star", "pin_bar_bull"}
BEARISH_PATTERNS = {"shooting_star", "engulfing_bear", "evening_star", "pin_bar_bear"}


def detect_patterns(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {}
    c   = df.iloc[-1]
    p1  = df.iloc[-2]
    p2  = df.iloc[-3]

    body    = abs(c.close - c.open)
    rng     = c.high - c.low
    up_shad = c.high - max(c.open, c.close)
    lo_shad = min(c.open, c.close) - c.low
    prev_body = abs(p1.close - p1.open)

    results = {}

    results["hammer"] = (
        lo_shad >= 2 * body and
        up_shad <= 0.1 * rng and
        p1.close < p1.open
    )

    results["shooting_star"] = (
        up_shad >= 2 * body and
        lo_shad <= 0.1 * rng and
        p1.close > p1.open
    )

    results["engulfing_bull"] = (
        c.close > c.open and
        p1.close < p1.open and
        c.open < p1.close and
        c.close > p1.open
    )

    results["engulfing_bear"] = (
        c.close < c.open and
        p1.close > p1.open and
        c.open > p1.close and
        c.close < p1.open
    )

    results["doji"] = (body <= 0.1 * rng) if rng > 0 else False

    results["pin_bar_bull"] = (lo_shad >= 3 * body and up_shad <= 0.5 * body)
    results["pin_bar_bear"] = (up_shad >= 3 * body and lo_shad <= 0.5 * body)

    results["morning_star"] = (
        p2.close < p2.open and
        abs(p1.close - p1.open) < 0.3 * abs(p2.close - p2.open) and
        c.close > c.open and
        c.close > (p2.open + p2.close) / 2
    )

    results["evening_star"] = (
        p2.close > p2.open and
        abs(p1.close - p1.open) < 0.3 * abs(p2.close - p2.open) and
        c.close < c.open and
        c.close < (p2.open + p2.close) / 2
    )

    return results


def compute_pa_score(df: pd.DataFrame) -> float:
    patterns = detect_patterns(df)
    bull_score = sum(PATTERN_WEIGHTS.get(p, 0) for p, v in patterns.items() if v and p in BULLISH_PATTERNS)
    bear_score = sum(PATTERN_WEIGHTS.get(p, 0) for p, v in patterns.items() if v and p in BEARISH_PATTERNS)
    total = bull_score + bear_score
    if total == 0:
        return 0.0
    return round((bull_score - bear_score) / total, 4)


def detect_sr_zone(df: pd.DataFrame, lookback: int = 50) -> dict:
    recent = df.tail(lookback)
    highs = recent["high"].nlargest(5)
    lows  = recent["low"].nsmallest(5)
    current_price = df["close"].iloc[-1]
    resistance = highs[highs > current_price].min() if any(highs > current_price) else highs.max()
    support    = lows[lows < current_price].max()  if any(lows < current_price)  else lows.min()
    return {"support": round(support, 5), "resistance": round(resistance, 5)}
