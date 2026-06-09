import pytest
from datetime import datetime, timezone
from execution.sim_broker import SimBroker
from signals.signal_types import TradeSignal


def make_signal(direction="BUY", score=0.5, ml=0.6, sent=0.0):
    return TradeSignal(
        timestamp=datetime.now(timezone.utc),
        symbol="EURUSD",
        direction=direction,
        final_score=score,
        pa_score=score,
        ml_prob=ml,
        sentiment_score=sent,
        confidence=abs(score),
    )


def test_initial_state():
    broker = SimBroker(initial_balance=1000.0)
    assert broker.balance == 1000.0
    assert broker.equity == 1000.0
    assert broker.positions == []
    assert broker.get_open_positions() == []


def test_open_trade():
    broker = SimBroker(1000.0)
    signal = make_signal("BUY", 0.6)
    result = broker.open_trade(signal, lot=0.1, sl=1.0790, tp=1.0810)
    assert result["success"] is True
    assert result["ticket"] == 1
    assert len(broker.get_open_positions()) == 1
    assert len(broker.get_open_positions("EURUSD")) == 1
    assert len(broker.get_open_positions("GBPUSD")) == 0


def test_open_trade_details():
    broker = SimBroker(1000.0)
    signal = make_signal("SELL", -0.6)
    broker.open_trade(signal, lot=0.2, sl=1.0810, tp=1.0790)
    pos = broker.get_open_positions()[0]
    assert pos["direction"] == "SELL"
    assert pos["lot"] == 0.2
    assert pos["sl"] == 1.0810
    assert pos["tp"] == 1.0790
    assert pos["status"] == "OPEN"
    assert "ticket" in pos
    assert "timestamp" in pos


def test_trade_id_increments():
    broker = SimBroker(1000.0)
    signal = make_signal("BUY")
    r1 = broker.open_trade(signal, 0.1, 1.0790, 1.0810)
    r2 = broker.open_trade(signal, 0.1, 1.0790, 1.0810)
    assert r1["ticket"] == 1
    assert r2["ticket"] == 2


def test_simulate_tick_hit_sl_buy():
    broker = SimBroker(1000.0)
    signal = make_signal("BUY")
    broker.open_trade(signal, lot=0.1, sl=1.0790, tp=1.0810)
    pos = broker.get_open_positions()[0]
    pos["entry_price"] = 1.0800
    pos["sl"] = 1.0795
    pos["tp"] = 1.0820
    start_balance = broker.balance
    for _ in range(1000):
        broker.simulate_tick()
        if broker.get_open_positions() == []:
            break
    assert broker.get_open_positions() == []
    assert broker.balance != start_balance


def test_simulate_tick_hit_tp_buy():
    broker = SimBroker(1000.0)
    signal = make_signal("BUY")
    broker.open_trade(signal, lot=0.1, sl=1.0790, tp=1.0808)
    pos = broker.get_open_positions()[0]
    pos["entry_price"] = 1.0800
    pos["sl"] = 1.0790
    pos["tp"] = 1.0808
    start_balance = broker.balance
    for _ in range(1000):
        broker.simulate_tick()
        if broker.get_open_positions() == []:
            break
    assert broker.get_open_positions() == []


def test_multiple_positions():
    broker = SimBroker(1000.0)
    broker.open_trade(make_signal("BUY"), 0.1, 1.0790, 1.0810)
    broker.open_trade(make_signal("BUY"), 0.1, 1.0790, 1.0810)
    assert len(broker.get_open_positions()) == 2


def test_equity_tracks_balance():
    broker = SimBroker(1000.0)
    signal = make_signal("BUY")
    broker.open_trade(signal, 0.1, 1.0790, 1.0810)
    broker.simulate_tick()
    assert broker.equity == broker.balance
