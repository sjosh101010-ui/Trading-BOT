#!/usr/bin/env python3
"""
Live Trading Monitor — BTCUSD Rapid M5 Scalper
Fetches real data from Yahoo Finance, computes Rapid signals, shows dashboard.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

import yfinance as yf
from analysis.feature_engineering import add_indicators
from analysis.m5_scalper import compute_m5_score
from risk.lot_sizer import compute_lot_size_sim
from execution.sim_broker import SimBroker
from config import (
    SYMBOLS, INITIAL_BALANCE, TRADE_SESSION_START, TRADE_SESSION_END,
    MAX_DAILY_TRADES, MAX_DAILY_LOSS_PCT, LOOP_INTERVAL_SECONDS,
    RAPID_SL_MULT, RAPID_TP_MULT, RAPID_PA_THRESHOLD, RAPID_BE_ATR_MULT,
    pip_size, contract_size,
)

SYMBOL = SYMBOLS[0]
PS = pip_size(SYMBOL)

logging.basicConfig(
    filename="logs/system.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

TERM_WIDTH = 58


def fetch_latest_m5(symbol: str, bars: int = 200) -> pd.DataFrame:
    yahoo_map = {"BTCUSD": "BTC-USD", "EURUSD": "EURUSD=X"}
    ticker = yahoo_map.get(symbol, f"{symbol}=X")
    df = yf.download(ticker, period="5d", interval="5m", progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index.name = "time"
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df["volume"] = df["volume"].clip(lower=1)
    return df.tail(bars)


def print_header():
    t = datetime.now(timezone.utc)
    print(f"\n{'='*TERM_WIDTH}")
    tz = "UTC"
    print(f"  BTCUSD Rapid M5  |  {t.strftime('%H:%M')} {tz}  |  24/7")
    print(f"{'='*TERM_WIDTH}")


def print_signal(direction, pa_score, adx_val, lot, sl, tp, entry):
    if direction == "BUY":
        arrow = "▲ BUY"
    elif direction == "SELL":
        arrow = "▼ SELL"
    else:
        arrow = "— SKIP"

    pa_str = f"{pa_score:+.2f}"
    entry_str = f"{entry:.5f}"
    sl_str = f"{sl:.5f}"
    tp_str = f"{tp:.5f}"
    lot_str = f"{lot:.3f}"

    lines = [f"  Signal:  {arrow}  |  PA={pa_str}  ADX={adx_val:.0f}"]
    if direction != "SKIP":
        lines += [
            f"  Entry:   {entry_str}",
            f"  SL/TP:   {sl_str} / {tp_str}",
            f"  Lot:     {lot_str}",
        ]
    for l in lines:
        print(f"  {l}")


def print_positions(broker):
    positions = broker.get_open_positions()
    if not positions:
        print(f"  Positions: None")
        return
    print(f"  Positions: {len(positions)} open")
    for p in positions:
        direction = p["direction"]
        entry = p["entry_price"]
        lot = p["lot"]
        sl = p["sl"]
        tp = p["tp"]
        print(f"    {direction} {lot}L @ {entry:.5f}  SL:{sl:.5f} TP:{tp:.5f}")


def print_daily(broker, daily_trades, daily_start, session_start):
    print(f"  Balance:  ${broker.balance:.2f}")
    print(f"  Equity:   ${broker.equity:.2f}")
    daily_pnl = broker.balance - daily_start
    total_pnl = broker.balance - session_start
    daily_pct = daily_pnl / daily_start * 100 if daily_start > 0 else 0
    total_pct = total_pnl / session_start * 100 if session_start > 0 else 0
    print(f"  Today:    ${daily_pnl:+.2f} ({daily_pct:+.1f}%)  |  Total: ${total_pnl:+.2f} ({total_pct:+.1f}%)")
    print(f"  Trades:   {daily_trades}/{MAX_DAILY_TRADES} today  |  Refresh: {LOOP_INTERVAL_SECONDS}s")


def compute_rapid_signal(df):
    pa_score = compute_m5_score(df)
    adx_val = float(df.iloc[-1].get("adx", 0) or 0)
    if abs(pa_score) < RAPID_PA_THRESHOLD:
        return "SKIP", pa_score, adx_val
    direction = "BUY" if pa_score > 0 else "SELL"
    return direction, pa_score, adx_val


def run():
    broker = SimBroker.load_or_create(initial_balance=INITIAL_BALANCE)
    daily_trades = 0
    daily_start_balance = broker.balance
    session_start_balance = broker.balance
    current_trade_day = None

    last_closed_candle_time = None
    traded_candle_times = set()

    print(f"\n{'='*TERM_WIDTH}")
    print(f"  Live Rapid Monitor Started")
    print(f"  Balance: ${broker.balance:.2f}  |  {SYMBOL} only")
    print(f"  Session: {TRADE_SESSION_START}:00-{TRADE_SESSION_END}:00 UTC")
    print(f"{'='*TERM_WIDTH}")

    while True:
        try:
            current_utc = datetime.now(timezone.utc)
            utc_hour = current_utc.hour
            trade_day = current_utc.date()

            if current_trade_day is None or trade_day != current_trade_day:
                current_trade_day = trade_day
                daily_trades = 0
                daily_start_balance = broker.balance

            if not (TRADE_SESSION_START <= utc_hour < TRADE_SESSION_END):
                print(f"\n  [{current_utc.strftime('%H:%M:%S')}] Outside session hours. Sleeping 60s...", end=" ")
                broker.save()
                for _ in range(60):
                    time.sleep(1)
                continue

            df = fetch_latest_m5(SYMBOL, bars=300)
            df = add_indicators(df)
            if len(df) < 50:
                print(f"\n  Not enough data ({len(df)} candles), sleeping...")
                time.sleep(60)
                continue

            entry_price = df.iloc[-1]["close"]
            if sys.platform == "darwin" and os.isatty(sys.stdout.fileno()):
                os.system("clear")
            print_header()

            closed_candle_time = df.index[-2]
            new_candle_closed = (
                last_closed_candle_time is None
                or closed_candle_time != last_closed_candle_time
            )
            last_closed_candle_time = closed_candle_time

            lot = 0
            sl_price = entry_price
            tp_price = entry_price
            direction = "SKIP"
            pa_score = compute_m5_score(df)
            adx_val = float(df.iloc[-1].get("adx", 0) or 0)
            has_pos = bool(broker.get_open_positions(SYMBOL))

            if not has_pos and new_candle_closed and closed_candle_time not in traded_candle_times:
                signal_window = df.iloc[:-1]
                direction, pa_score, adx_val = compute_rapid_signal(signal_window)

                if direction != "SKIP":
                    atr = float(df.iloc[-2].get("atr", 0.001)) or 0.001
                    if atr <= 0:
                        atr = 0.001

                    tp_mult = RAPID_TP_MULT if isinstance(RAPID_TP_MULT, (int, float)) else RAPID_TP_MULT.get(SYMBOL, 2.0)
                    if direction == "BUY":
                        sl_price = entry_price - RAPID_SL_MULT * atr
                        tp_price = entry_price + tp_mult * atr
                        be_trigger = entry_price + RAPID_BE_ATR_MULT * atr if RAPID_BE_ATR_MULT > 0 else None
                    else:
                        sl_price = entry_price + RAPID_SL_MULT * atr
                        tp_price = entry_price - tp_mult * atr
                        be_trigger = entry_price - RAPID_BE_ATR_MULT * atr if RAPID_BE_ATR_MULT > 0 else None

                    sl_price = round(sl_price, 5)
                    tp_price = round(tp_price, 5)
                    if be_trigger:
                        be_trigger = round(be_trigger, 5)
                    sl_pips = abs(entry_price - sl_price) / PS if PS > 0 else 0

                    daily_loss_pct = (daily_start_balance - broker.balance) / daily_start_balance if daily_start_balance > 0 else 0
                    if daily_trades < MAX_DAILY_TRADES and daily_loss_pct < MAX_DAILY_LOSS_PCT:
                        lot = compute_lot_size_sim(SYMBOL, broker.balance, sl_pips, min(1.0, abs(pa_score)))

            print_signal(direction, pa_score, adx_val, lot, sl_price, tp_price, entry_price)
            print()
            print_positions(broker)
            print()
            print_daily(broker, daily_trades, daily_start_balance, session_start_balance)
            print(f"{'='*TERM_WIDTH}")
            print(f"  {current_utc.strftime('%H:%M:%S')} UTC  |  Price: {entry_price:.5f}")
            print(f"{'='*TERM_WIDTH}")

            if lot > 0 and not broker.get_open_positions(SYMBOL):
                from signals.signal_types import TradeSignal
                ts = TradeSignal(
                    timestamp=datetime.now(timezone.utc),
                    symbol=SYMBOL, direction=direction,
                    final_score=pa_score, pa_score=pa_score,
                    ml_prob=0.5, sentiment_score=0.0,
                    confidence=min(1.0, abs(pa_score)),
                )
                result = broker.open_trade(ts, lot, sl_price, tp_price, entry_price=entry_price, be_trigger=be_trigger)
                if result["success"]:
                    daily_trades += 1
                    traded_candle_times.add(closed_candle_time)
                    logging.info(f"[RAPID] Trade opened: {direction} {lot}L @ {entry_price:.5f}")
                    print(f"  >>> RAPID TRADE EXECUTED: {direction} {lot}L <<<")

            current_high = df.iloc[-1]["high"]
            current_low = df.iloc[-1]["low"]
            closed_trades = broker.simulate_tick(current_high=current_high, current_low=current_low)
            for t in closed_trades:
                pnl_str = f"+${t['pnl_usd']:.2f}" if t['pnl_usd'] >= 0 else f"-${abs(t['pnl_usd']):.2f}"
                label = "SL HIT" if t['status'] == 'LOSS' else "TP HIT"
                print(f"  >>> {label}: {t['direction']} {t['lot']}L  PnL: {pnl_str} <<<")
                logging.info(f"[RAPID] {label}: {t['direction']} {t['lot']}L PnL={pnl_str}")

            for _ in range(LOOP_INTERVAL_SECONDS):
                time.sleep(1)

        except KeyboardInterrupt:
            broker.save()
            print(f"\n{'='*TERM_WIDTH}")
            print(f"  Shutdown. Final balance: ${broker.balance:.2f}")
            print(f"{'='*TERM_WIDTH}")
            break
        except Exception as ex:
            logging.error(f"Loop error: {ex}", exc_info=True)
            print(f"\n  Error: {ex}")
            broker.save()
            time.sleep(30)


if __name__ == "__main__":
    if "--reset" in sys.argv:
        idx = sys.argv.index("--reset")
        if idx + 1 < len(sys.argv):
            try:
                amount = float(sys.argv[idx + 1])
                SimBroker.reset(amount)
            except ValueError:
                print("Usage: python main.py --reset <amount>")
                sys.exit(1)
        else:
            print("Usage: python main.py --reset <amount>")
            sys.exit(1)
    else:
        run()
