import pytest
import pandas as pd
import numpy as np
from backtest.metrics import calculate_metrics, print_metrics


def make_trades_df(win_count=5, loss_count=3, win_pnl=10.0, loss_pnl=-5.0):
    rows = []
    ts = pd.date_range("2024-01-01", periods=win_count + loss_count, freq="h")
    for i in range(win_count):
        rows.append({
            "timestamp": ts[i], "exit_time": ts[i] + pd.Timedelta(hours=2),
            "symbol": "EURUSD", "direction": "BUY",
            "lot_size": 0.1, "entry_price": 1.0800, "exit_price": 1.0810,
            "sl_price": 1.0790, "tp_price": 1.0810,
            "pnl_usd": win_pnl, "status": "WIN",
        })
    for i in range(loss_count):
        rows.append({
            "timestamp": ts[win_count + i],
            "exit_time": ts[win_count + i] + pd.Timedelta(hours=1),
            "symbol": "EURUSD", "direction": "SELL",
            "lot_size": 0.1, "entry_price": 1.0800, "exit_price": 1.0805,
            "sl_price": 1.0810, "tp_price": 1.0790,
            "pnl_usd": loss_pnl, "status": "LOSS",
        })
    return pd.DataFrame(rows)


def test_empty_trades():
    df = pd.DataFrame()
    metrics = calculate_metrics(df)
    assert "error" in metrics


def test_no_closed_trades():
    rows = [{
        "timestamp": pd.Timestamp("2024-01-01"),
        "exit_time": pd.Timestamp("2024-01-01"),
        "symbol": "EURUSD", "direction": "BUY",
        "lot_size": 0.1, "entry_price": 1.0800, "exit_price": 1.0810,
        "sl_price": 1.0790, "tp_price": 1.0810,
        "pnl_usd": 0.0, "status": "OPEN",
    }]
    df = pd.DataFrame(rows)
    metrics = calculate_metrics(df)
    assert "error" in metrics


def test_win_rate():
    df = make_trades_df(win_count=7, loss_count=3)
    metrics = calculate_metrics(df)
    assert metrics["wins"] == 7
    assert metrics["losses"] == 3
    assert metrics["win_rate"] == pytest.approx(0.7)


def test_total_pnl():
    df = make_trades_df(win_count=5, loss_count=3, win_pnl=10.0, loss_pnl=-5.0)
    metrics = calculate_metrics(df)
    expected_pnl = 5 * 10.0 + 3 * (-5.0)
    assert metrics["total_pnl"] == pytest.approx(expected_pnl)


def test_profit_factor():
    df = make_trades_df(win_count=5, loss_count=3, win_pnl=10.0, loss_pnl=-5.0)
    metrics = calculate_metrics(df)
    assert metrics["profit_factor"] == 3.33


def test_final_balance():
    df = make_trades_df(win_count=5, loss_count=3, win_pnl=10.0, loss_pnl=-5.0)
    metrics = calculate_metrics(df, initial_balance=1000.0)
    expected = 1000.0 + (5 * 10.0 + 3 * (-5.0))
    assert metrics["final_balance"] == pytest.approx(expected)


def test_max_drawdown():
    df = make_trades_df(win_count=2, loss_count=2, win_pnl=10.0, loss_pnl=-5.0)
    metrics = calculate_metrics(df, initial_balance=100.0)
    assert metrics["max_drawdown"] >= 0


def test_avg_win_and_loss():
    df = make_trades_df(win_count=4, loss_count=2, win_pnl=8.0, loss_pnl=-4.0)
    metrics = calculate_metrics(df)
    assert metrics["avg_win"] == pytest.approx(8.0)
    assert metrics["avg_loss"] == pytest.approx(-4.0)


def test_print_metrics_no_error(capsys):
    df = make_trades_df()
    metrics = calculate_metrics(df)
    print_metrics(metrics)
    captured = capsys.readouterr()
    assert "Win Rate" in captured.out
    assert "Total P&L" in captured.out


def test_print_metrics_with_error(capsys):
    metrics = {"error": "No trades to analyze"}
    print_metrics(metrics)
    captured = capsys.readouterr()
    assert "No trades to analyze" in captured.out


def test_all_expected_keys():
    df = make_trades_df()
    metrics = calculate_metrics(df)
    expected_keys = {
        "total_trades", "wins", "losses", "win_rate", "profit_factor",
        "total_pnl", "final_balance", "avg_win", "avg_loss",
        "max_drawdown", "max_drawdown_pct", "avg_bars_held",
        "sharpe_ratio", "trade_start", "trade_end",
    }
    assert expected_keys.issubset(metrics.keys())
