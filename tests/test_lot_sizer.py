import pytest
from risk.lot_sizer import compute_lot_size_sim
from config import MIN_LOT, MAX_LOT


def test_basic_lot_size():
    lot = compute_lot_size_sim("EURUSD", account_balance=100.0, sl_pips=10, confidence=0.5)
    assert MIN_LOT <= lot <= MAX_LOT


def test_higher_confidence_bigger_lot():
    low_conf = compute_lot_size_sim("EURUSD", 100.0, 10, 0.3)
    high_conf = compute_lot_size_sim("EURUSD", 100.0, 10, 0.9)
    assert high_conf >= low_conf


def test_larger_balance_bigger_lot():
    small = compute_lot_size_sim("EURUSD", 100.0, 10, 0.5)
    large = compute_lot_size_sim("EURUSD", 10000.0, 10, 0.5)
    assert large >= small


def test_min_lot_clamping():
    lot = compute_lot_size_sim("EURUSD", 1.0, 100, 0.1)
    assert lot == MIN_LOT


def test_zero_sl_pips_returns_min_lot():
    lot = compute_lot_size_sim("EURUSD", 100.0, sl_pips=0, confidence=0.5)
    assert lot == MIN_LOT


def test_negative_sl_pips_returns_min_lot():
    lot = compute_lot_size_sim("EURUSD", 100.0, sl_pips=-5, confidence=0.5)
    assert lot == MIN_LOT


def test_unknown_symbol_uses_defaults():
    lot = compute_lot_size_sim("UNKNOWN", 100.0, 10, 0.5)
    assert MIN_LOT <= lot <= MAX_LOT


def test_peak_balance_tracking():
    compute_lot_size_sim("TEST", 100.0, 10, 0.5)
    lot1 = compute_lot_size_sim("TEST", 100.0, 10, 0.5)
    assert MIN_LOT <= lot1 <= MAX_LOT
    lot2 = compute_lot_size_sim("TEST", 200.0, 10, 0.5)
    assert lot2 >= lot1


def test_lot_rounded_to_two_decimals():
    lot = compute_lot_size_sim("EURUSD", 100.0, 10, 0.5)
    assert round(lot, 2) == lot
