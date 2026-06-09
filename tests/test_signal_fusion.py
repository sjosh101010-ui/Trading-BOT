import pytest
from signals.fusion import fuse


def test_buy_signal():
    signal = fuse("EURUSD", pa_score=0.8, ml_prob=0.8, sentiment_score=0.1)
    assert signal.direction == "BUY"
    assert 0 <= signal.confidence <= 1


def test_sell_signal():
    signal = fuse("EURUSD", pa_score=-0.8, ml_prob=0.2, sentiment_score=-0.1)
    assert signal.direction == "SELL"


def test_skip_signal():
    signal = fuse("EURUSD", pa_score=0.1, ml_prob=0.52, sentiment_score=0.0)
    assert signal.direction == "SKIP"


def test_sentiment_override_buy():
    signal = fuse("EURUSD", pa_score=0.8, ml_prob=0.8, sentiment_score=-0.7)
    assert signal.direction == "SKIP"


def test_sentiment_override_sell():
    signal = fuse("EURUSD", pa_score=-0.8, ml_prob=0.2, sentiment_score=0.7)
    assert signal.direction == "SKIP"


def test_confidence_range():
    cases = [
        (0.8, 0.8, 0.1),
        (-0.8, 0.2, -0.1),
        (0.0, 0.5, 0.0),
    ]
    for pa, ml, sent in cases:
        signal = fuse("EURUSD", pa, ml, sent)
        assert 0 <= signal.confidence <= 1
