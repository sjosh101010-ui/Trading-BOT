from datetime import datetime, timezone

from signals.signal_types import TradeSignal
from config import W_PA, W_ML, W_SENT, BUY_THRESHOLD, SELL_THRESHOLD, SENTIMENT_OVERRIDE_THRESH


def fuse(symbol: str,
         pa_score: float,
         ml_prob: float,
         sentiment_score: float,
         trend_score: float = 0.0) -> TradeSignal:
    ml_score = (ml_prob - 0.5) * 2

    w_pa, w_ml, w_sent = W_PA, W_ML, W_SENT

    if abs(ml_score) < 0.01:
        w_ml = 0.0
        trend_influence = 0.25
        remaining = 1.0 - trend_influence
        total = w_pa + w_sent
        if total > 0:
            w_pa = w_pa / total * remaining
            w_sent = w_sent / total * remaining
        final_score = (
            w_pa * pa_score +
            trend_influence * trend_score +
            w_sent * sentiment_score
        )
        notes_hint = "ML redistributed"
    else:
        final_score = (
            w_pa * pa_score +
            w_ml * ml_score +
            w_sent * sentiment_score
        )
        notes_hint = ""

    final_score = round(final_score, 4)

    if final_score > BUY_THRESHOLD and sentiment_score < -SENTIMENT_OVERRIDE_THRESH:
        return TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            direction="SKIP",
            final_score=final_score,
            pa_score=pa_score,
            ml_prob=ml_prob,
            sentiment_score=sentiment_score,
            confidence=0.0,
            notes="Blocked: strong negative sentiment contradicts BUY signal",
        )
    if final_score < SELL_THRESHOLD and sentiment_score > SENTIMENT_OVERRIDE_THRESH:
        return TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            direction="SKIP",
            final_score=final_score,
            pa_score=pa_score,
            ml_prob=ml_prob,
            sentiment_score=sentiment_score,
            confidence=0.0,
            notes="Blocked: strong positive sentiment contradicts SELL signal",
        )

    if final_score >= BUY_THRESHOLD:
        direction = "BUY"
    elif final_score <= SELL_THRESHOLD:
        direction = "SELL"
    else:
        direction = "SKIP"

    confidence = min(abs(final_score) / 1.0, 1.0)
    notes = notes_hint if direction != "SKIP" else ""

    return TradeSignal(
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        direction=direction,
        final_score=final_score,
        pa_score=pa_score,
        ml_prob=ml_prob,
        sentiment_score=sentiment_score,
        confidence=confidence,
        notes=notes,
    )
