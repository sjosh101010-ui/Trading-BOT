#!/usr/bin/env python3
"""
M5 Paper Trader — Vantage MT5 Execution
Yahoo M5 data for signals, MetaTrader5 API for execution.
Run inside Windows VM with MT5 terminal.
"""
import sys, time, logging, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import yfinance as yf

from analysis.feature_engineering import add_indicators
from analysis.m5_scalper import compute_m5_score, check_price_divergence
from risk.lot_sizer import compute_lot_size_sim
from execution.mt5_broker import MT5Broker
from config import (
    SYMBOLS, RAPID_SL_MULT, RAPID_TP_MULT,
    SPREAD_PIPS, LEVERAGE, INITIAL_BALANCE, RAPID_BE_ATR_MULT,
)

logging.basicConfig(
    filename="logs/paper_trade_mt5.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

REFRESH_SECONDS = 60

broker = MT5Broker()
trade_log = []
LIVE_STATE_FILE = "logs/live_state.json"
LIVE_TRADE_FILE = "logs/paper_trades_live.csv"
os.makedirs("logs", exist_ok=True)


def pip_size(sym):
    from config import pip_size as _ps
    return _ps(sym)


def write_live_state(balance, positions):
    pos_list = []
    for p in positions.values():
        pos_list.append({
            "id": p.get("id"),
            "symbol": p.get("symbol"),
            "direction": p.get("direction"),
            "volume": p.get("volume"),
            "openPrice": p.get("openPrice"),
            "profit": p.get("profit"),
            "stopLoss": p.get("sl"),
            "takeProfit": p.get("tp"),
        })
    now = datetime.now(timezone.utc) + timedelta(hours=8)
    state = {
        "balance": round(balance, 2),
        "open_positions": len(pos_list),
        "positions": pos_list,
        "manila_time": now.strftime("%H:%M"),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(LIVE_STATE_FILE, "w") as f:
        json.dump(state, f)


def append_trade_csv(trade):
    df = pd.DataFrame([trade])
    df.to_csv(LIVE_TRADE_FILE, mode="a", header=not os.path.exists(LIVE_TRADE_FILE), index=False)


def fetch_m5(symbol, bars=300):
    yahoo_map = {"BTCUSD": "BTC-USD", "EURUSD": "EURUSD=X", "USDZAR": "USDZAR=X"}
    ticker = yahoo_map.get(symbol, f"{symbol}=X")
    df = yf.download(ticker, period="5d", interval="5m", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index.name = "time"
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df["volume"] = df["volume"].clip(lower=1)
    return df.tail(bars)


def compute_rapid_signal(df):
    pa_score = compute_m5_score(df)
    if abs(pa_score) < 0.01:
        return None, pa_score, 0
    if check_price_divergence(df, pa_score):
        return None, pa_score, 0
    return ("BUY" if pa_score > 0 else "SELL"), pa_score, 0


def open_trade(symbol, direction, pa_score, adx_val, df, candle_time):
    price = df.iloc[-1]["close"]
    atr = float(df.iloc[-1].get("atr", 0.001))
    if atr <= 0:
        return

    ps = pip_size(symbol)
    spread_pts = SPREAD_PIPS * ps
    entry_price = price + spread_pts / 2 if direction == "BUY" else price - spread_pts / 2

    tp_mult = RAPID_TP_MULT.get(symbol, 2.0) if isinstance(RAPID_TP_MULT, dict) else RAPID_TP_MULT
    sl_price = entry_price - RAPID_SL_MULT * atr if direction == "BUY" else entry_price + RAPID_SL_MULT * atr
    tp_price = entry_price + tp_mult * atr if direction == "BUY" else entry_price - tp_mult * atr
    sl_pips = abs(entry_price - sl_price) / ps

    info = broker.get_account_info()
    balance = info["balance"] if info else INITIAL_BALANCE
    lot = compute_lot_size_sim(symbol, balance, sl_pips, min(1.0, abs(pa_score)), price=price)
    if lot <= 0:
        return

    digits = broker.get_digits(symbol)
    result = broker.open_market_order(
        symbol, direction, lot,
        sl_price=round(sl_price, digits),
        tp_price=round(tp_price, digits),
    )
    if result.get("success"):
        trade_id = result.get("ticket")
        logging.info(f"[{symbol}] ENTRY: {direction} {lot}L @ via MT5 id={trade_id}")
        print(f"  >>> {symbol} ENTRY: {direction} {lot:.2f}L @ MT5 id={trade_id}")
        if trade_id:
            tracked[trade_id] = {
                "symbol": symbol,
                "direction": direction,
                "lot": lot,
                "entry_price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "entry_time": candle_time,
                "adx_entry": adx_val,
            }
    else:
        err = result.get("error", "unknown")
        logging.warning(f"[{symbol}] ORDER FAILED: {err}")


tracked = {}


def check_closed_positions(active_orders):
    active_ids = set(active_orders.keys())
    closed_ids = set(tracked.keys()) - active_ids
    for tid in closed_ids:
        t = tracked.pop(tid, None)
        if t is None:
            continue
        exit_time = datetime.now(timezone.utc)
        entry_time = t.get("entry_time", exit_time)
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
        duration_h = (exit_time - entry_time).total_seconds() / 3600
        trade_log.append({
            "time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": t["symbol"],
            "direction": t["direction"],
            "lot": t["lot"],
            "entry": t["entry_price"],
            "exit": None,
            "pnl": 0,
            "status": "CLOSED",
            "entry_time": str(entry_time),
            "adx_entry": t.get("adx_entry", 0),
            "duration_hours": round(duration_h, 2),
        })
        append_trade_csv(trade_log[-1])
        logging.info(f"[{t['symbol']}] CLOSED: {t['direction']} {t['lot']}L id={tid}")


_last_signal = {}


def print_status(balance, positions):
    now = datetime.now(timezone.utc) + timedelta(hours=8)
    print(f"\n{'='*55}")
    print(f"  Paper Trader (Vantage MT5)  |  {now.strftime('%H:%M')} Manila  |  Bal: ${balance:.2f}")
    print(f"{'='*55}")
    for p_id, p in positions.items():
        sym = p.get("symbol", "?")
        side = p.get("direction", "?")
        vol = p.get("volume", 0)
        op = p.get("openPrice", 0)
        pf = p.get("profit", 0)
        print(f"  {sym}: {side} {vol}L @ {op:.5f}  PnL=${pf:.2f}")
    if not positions and _last_signal:
        s = _last_signal
        print(f"  Last eval: PA={s['pa']:+.2f}  Price={s['price']:.5f}  \u2192 {'NO TRADE' if not s['direction'] else s['direction']}")
    write_live_state(balance, positions)
    print(f"{'='*55}")


def run():
    print(f"\n{'='*55}")
    print(f"  M5 Paper Trader \u2014 Vantage MT5 Execution")
    print(f"  Pair: {SYMBOLS[0]}  |  Server: {os.getenv('MT5_SERVER', 'VantageInternational-Demo')}")
    print(f"{'='*55}\n")

    last_closed = {}
    while True:
        try:
            now = datetime.now(timezone.utc)
            info = broker.get_account_info()
            balance = info["balance"] if info else INITIAL_BALANCE

            active_orders = broker.get_open_positions()

            check_closed_positions(active_orders)

            dfs = {}
            for sym in SYMBOLS:
                df = fetch_m5(sym)
                if df is None:
                    continue
                df = add_indicators(df)
                dfs[sym] = df

                if len(df) >= 2:
                    closed_time = df.index[-2]
                    last_price = df.iloc[-1]["close"]
                    pa_now = compute_m5_score(df)
                    adx_now = df.iloc[-1].get("adx", 0) or 0
                    new_candle = last_closed.get(sym) != closed_time
                    direction = None
                    if new_candle:
                        last_closed[sym] = closed_time
                        signal_df = df.iloc[:-1]
                        direction, pa_score, adx_val = compute_rapid_signal(signal_df)
                        if direction:
                            has_pos = any(
                                o.get("symbol") == sym for o in active_orders.values()
                            )
                            if not has_pos:
                                open_trade(sym, direction, pa_score, adx_now, signal_df, closed_time)
                    _last_signal.update({
                        "price": last_price, "pa": pa_now,
                        "direction": direction if new_candle else _last_signal.get("direction"),
                    })

            print_status(balance, active_orders)

            for _ in range(REFRESH_SECONDS):
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n  Shutdown.")
            if trade_log:
                pd.DataFrame(trade_log).to_csv("logs/paper_trades_mt5.csv", index=False)
            logging.info("Shutdown.")
            break
        except Exception as ex:
            logging.error(f"Loop error: {ex}", exc_info=True)
            print(f"  Error: {ex}")
            time.sleep(60)


if __name__ == "__main__":
    run()
