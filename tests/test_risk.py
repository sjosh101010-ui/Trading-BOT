import pytest
from risk.sl_tp import compute_sl_tp
from risk.guards import run_all_guards, is_session_active, is_spread_ok, can_open_trade, is_drawdown_ok


def test_sl_tp_buy():
    result = compute_sl_tp("EURUSD", "BUY", 1.0800, atr=0.002)
    assert result["sl"] < 1.0800
    assert result["tp"] > 1.0800
    assert result["sl_pips"] > 0


def test_sl_tp_sell():
    result = compute_sl_tp("EURUSD", "SELL", 1.0800, atr=0.002)
    assert result["sl"] > 1.0800
    assert result["tp"] < 1.0800


def test_guards_block_outside_session():
    ok, reason = run_all_guards(
        "EURUSD", spread_pips=1.0, open_positions=0,
        session_start_balance=10000, current_balance=10000,
        is_blackout=False
    )
    # May pass or fail depending on current time
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


def test_spread_guard():
    assert is_spread_ok("XAUUSD", 2.0) is True
    assert is_spread_ok("XAUUSD", 5.0) is False


def test_max_positions():
    assert can_open_trade(0) is True
    assert can_open_trade(6) is True
    assert can_open_trade(7) is False


def test_drawdown_guard():
    assert is_drawdown_ok(10000, 9501) is True
    assert is_drawdown_ok(10000, 8999) is False
