#!/usr/bin/env python3
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
from execution.oanda_broker import OANDABroker
from config import (
    SYMBOLS, RAPID_SL_MULT, RAPID_TP_MULT, RAPID_BE_ATR_MULT,
    INITIAL_BALANCE, pip_size,
)

logging.basicConfig(
    filename="logs/paper_trade_oanda.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

REFRESH_SECONDS = 60
broker = OANDABroker()
tracked = {}
LIVE_STATE_FILE = "logs/live_state_oanda.json"
os.makedirs("logs", exist_ok=True)

def write_live_state(balance, positions):
    pos_list = []
    for pid, p in positions.items():
        if isinstance(p, dict):
            pos_list.append({
                "id": pid, "symbol": p.get("symbol"),
                "direction": p.get("direction"), "volume": p.get("volume"),
                "openPrice": p.get("openPrice"), "profit": p.get("profit"),
                "sl": p.get("sl"), "tp": p.get("tp"),
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
    adx_val = df.iloc[-1].get("adx", 0) or 0
    if abs(pa_score) < 0.01:
        return None, pa_score, 0
    if check_price_divergence(df, pa_score):
        return None, pa_score, 0
    return ("BUY" if pa_score > 0 else "SELL"), pa_score, adx_val

def open_trade(symbol, direction, pa_score, df):
    last = df.iloc[-1]
    price = float(last["close"])
    atr = float(last.get("atr", 0.001))
    if atr <= 0:
        return

    info = broker.get_account_info()
    if not info:
        print("  Cannot get account info")
        return
    balance = info["balance"]

    tp_mult = RAPID_TP_MULT.get(symbol, 2.0) if isinstance(RAPID_TP_MULT, dict) else RAPID_TP_MULT
    sl_price = price - RAPID_SL_MULT * atr if direction == "BUY" else price + RAPID_SL_MULT * atr
    tp_price = price + tp_mult * atr if direction == "BUY" else price - tp_mult * atr
    sl_pips = abs(price - sl_price) / pip_size(symbol)

    lot = compute_lot_size_sim(symbol, balance, sl_pips, min(1.0, abs(pa_score)), price=price)
    if lot <= 0:
        return

    result = broker.open_market_order(
        symbol, direction, lot,
        sl_price=round(sl_price, 5),
        tp_price=round(tp_price, 5),
    )
    if result.get("success"):
        trade_id = result.get("ticket")
        logging.info(f"[{symbol}] ENTRY: {direction} {lot}L @ {price:.5f} via OANDA id={trade_id}")
        print(f"  >>> {symbol} ENTRY: {direction} {lot:.2f}L @ OANDA id={trade_id}")
        if trade_id:
            tracked[trade_id] = {
                "symbol": symbol, "direction": direction, "lot": lot,
                "entry_price": price, "sl": sl_price, "tp": tp_price,
            }
    else:
        err = result.get("error", "unknown")
        logging.warning(f"[{symbol}] ORDER FAILED: {err}")
        print(f"  >>> ORDER FAILED: {err}")

def check_closed_positions(open_positions):
    closed_ids = set(tracked.keys()) - set(open_positions.keys())
    for tid in closed_ids:
        t = tracked.pop(tid, None)
        if t is None:
            continue
        logging.info(f"[{t['symbol']}] CLOSED: {t['direction']} {t['lot']}L id={tid}")
        print(f"  >>> CLOSED: {t['direction']} {t['lot']}L id={tid}")

_last_signal = {}

def print_status(balance, positions):
    now = datetime.now(timezone.utc) + timedelta(hours=8)
    print(f"\n{'='*55}")
    print(f"  Paper Trader (OANDA)  |  {now.strftime('%H:%M')} Manila  |  Bal: ${balance:.2f}")
    print(f"{'='*55}")
    for pid, p in positions.items():
        print(f"  {p['symbol']}: {p['direction']} {p['volume']}L @ {p['openPrice']:.5f}  PnL=${p.get('profit',0):.2f}")
    if not positions and _last_signal:
        s = _last_signal
        print(f"  Last eval: PA={s['pa']:+.2f}  ADX={s['adx']:.0f}  Price={s['price']:.5f}  → {'NO TRADE' if not s['direction'] else s['direction']}")
    write_live_state(balance, positions)
    print(f"{'='*55}")

def run():
    print(f"\n{'='*55}")
    print(f"  M5 Paper Trader — OANDA Live Execution")
    print(f"  Pair: {SYMBOLS[0]}  |  Server: OANDA DEMO")
    print(f"{'='*55}\n")

    last_closed = {}
    while True:
        try:
            info = broker.get_account_info()
            balance = info["balance"] if info else INITIAL_BALANCE
            open_positions = broker.get_open_positions() or {}

            check_closed_positions(open_positions)

            for sym in SYMBOLS:
                df = fetch_m5(sym)
                if df is None:
                    continue
                df = add_indicators(df)
                if len(df) < 2:
                    continue

                closed_time = df.index[-2]
                last_price = float(df.iloc[-1]["close"])
                pa_now = compute_m5_score(df)
                adx_now = float(df.iloc[-1].get("adx", 0) or 0)
                new_candle = last_closed.get(sym) != closed_time
                direction = None

                if new_candle:
                    last_closed[sym] = closed_time
                    signal_df = df.iloc[:-1]
                    direction, pa_score, adx_val = compute_rapid_signal(signal_df)
                    if direction:
                        has_pos = any(
                            o.get("symbol") == sym for o in open_positions.values()
                        ) or any(
                            t.get("symbol") == sym for t in tracked.values()
                        )
                        if not has_pos:
                            open_trade(sym, direction, pa_score, signal_df)

                _last_signal.update({
                    "price": last_price, "pa": pa_now, "adx": adx_now,
                    "direction": direction if new_candle else _last_signal.get("direction"),
                })

            print_status(balance, open_positions)

            for _ in range(REFRESH_SECONDS):
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n  Shutdown.")
            logging.info("Shutdown.")
            break
        except Exception as ex:
            logging.error(f"Loop error: {ex}", exc_info=True)
            print(f"  Error: {ex}")
            time.sleep(60)

if __name__ == "__main__":
    run()
