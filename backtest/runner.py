import pandas as pd
import numpy as np
from datetime import datetime, timezone
from copy import deepcopy

from analysis.feature_engineering import add_indicators, compute_trend_score
from analysis.price_action import compute_pa_score, detect_sr_zone
from analysis.m5_scalper import compute_m5_score

from analysis.mean_reversion import compute_mr_score
from analysis.ml_model import predict, XGB_SYMBOLS
from signals.fusion import fuse
from risk.sl_tp import compute_sl_tp
from risk.lot_sizer import compute_lot_size_sim, _peak_balance
from backtest.metrics import calculate_metrics, print_metrics
from config import (
    TRADE_SESSION_START, TRADE_SESSION_END, MAX_DAILY_TRADES, MAX_DAILY_LOSS_PCT,
    REQUIRE_ML_PA_CONFLUENCE, ML_CONFIDENCE_GATE,
    HC_PA_SCORE_THRESHOLD, HC_ML_PROB_BUY, HC_ML_PROB_SELL,
    HC_SL_ATR_MULT, HC_TP_ATR_MULT_DEFAULT, HC_TP_ATR_MULT_NEAR_SR,
    RAPID_SL_MULT, RAPID_TP_MULT, RAPID_PA_THRESHOLD,
    SPREAD_PIPS, SLIPPAGE_PIPS, LEVERAGE, pip_size,
)


def backtest(
    df: pd.DataFrame,
    symbol: str = "BTCUSD",
    initial_balance: float = 1000.0,
    lookback: int = 100,
    step: int = 1,
    progress: bool = True,
    strategy: str = "m5",
    trade_date: pd.Timestamp = None,
    trade_start_date: pd.Timestamp = None,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = add_indicators(df)

    trades = []
    lev_balance = initial_balance * LEVERAGE
    balance = lev_balance
    open_positions = {}
    total_candles = len(df)
    start_idx = max(lookback, 50)
    _peak_balance[symbol] = lev_balance

    candles_processed = 0
    daily_trades = 0
    daily_start_balance = lev_balance
    current_trade_date = None

    balance = lev_balance

    pending_entry = None

    for i in range(start_idx, total_candles):
        candle_time = df.index[i]
        current_hour = candle_time.hour
        if not (TRADE_SESSION_START <= current_hour < TRADE_SESSION_END):
            pending_entry = None
            continue
        if trade_date is not None and candle_time.date() != trade_date.date():
            pending_entry = None
            continue
        if trade_start_date is not None and candle_time.date() < trade_start_date.date():
            pending_entry = None
            continue

        current = df.iloc[i]

        candles_processed += 1
        if progress and candles_processed % 500 == 0:
            pct = (i - start_idx) / (total_candles - start_idx) * 100
            print(f"  Progress: {pct:.0f}% ({i}/{total_candles})  Balance: ${balance:.2f}")

        balance += _check_sl_tp(current, open_positions, trades, df.index[i])

        if pending_entry is not None and symbol not in open_positions:
            entry = pending_entry
            pending_entry = None

            trade_day = candle_time.date()
            if current_trade_date is None or trade_day != current_trade_date:
                current_trade_date = trade_day
                daily_trades = 0
                daily_start_balance = balance
            daily_loss_pct = (daily_start_balance - balance) / daily_start_balance if daily_start_balance > 0 else 0
            if daily_trades >= MAX_DAILY_TRADES or daily_loss_pct >= MAX_DAILY_LOSS_PCT:
                continue

            price = current["open"]
            atr = float(current["atr"]) if current["atr"] > 0 else 0.001

            ps = pip_size(symbol)
            spread_pts = SPREAD_PIPS * ps
            entry_price = price + spread_pts / 2 if entry["direction"] == "BUY" else price - spread_pts / 2

            if entry.get("from_mr"):
                sl_mult = 1.5
                tp_mult = entry["tp_mult"]
                sl_price = entry_price - sl_mult * atr if entry["direction"] == "BUY" else entry_price + sl_mult * atr
                tp_price = entry_price + tp_mult * atr if entry["direction"] == "BUY" else entry_price - tp_mult * atr
                sl_pips = abs(entry_price - sl_price) / ps
            elif entry["from_hc"]:
                sl_mult = entry.get("sl_mult", HC_SL_ATR_MULT)
                tp_mult = entry["tp_mult"]
                sl_price = entry_price - sl_mult * atr if entry["direction"] == "BUY" else entry_price + sl_mult * atr
                tp_price = entry_price + tp_mult * atr if entry["direction"] == "BUY" else entry_price - tp_mult * atr
                sl_pips = abs(entry_price - sl_price) / ps
            else:
                sl_tp = compute_sl_tp(symbol, entry["direction"], entry_price, atr)
                sl_price = sl_tp["sl"]
                tp_price = sl_tp["tp"]
                sl_pips = sl_tp["sl_pips"]

            lot = compute_lot_size_sim(symbol, balance, sl_pips, entry["confidence"], price=price)
            if lot <= 0:
                continue

            open_positions[symbol] = {
                "ticket": len(trades) + 1,
                "symbol": symbol,
                "direction": entry["direction"],
                "lot": lot,
                "entry_price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "entry_time": df.index[i],
                "entry_idx": i,
                "adx_entry": entry.get("adx_entry", 0),
            }
            daily_trades += 1
            continue

        if i % step != 0:
            continue

        window = df.iloc[: i + 1]
        current = df.iloc[i]

        if strategy == "pa":
            pa_score = compute_pa_score(window)
        elif strategy == "m5":
            pa_score = compute_m5_score(window)
        elif strategy == "mr":
            pa_score = 0.0
        else:
            pa_score = compute_m5_score(window)
        trend_score = compute_trend_score(window)
        ml_prob = predict(symbol, window) if strategy not in ("mr", "rapid") else 0.5

        if strategy == "hc":
            signal = fuse(symbol, pa_score, ml_prob, 0.0, trend_score)

            if signal.direction == "SKIP":
                continue

            if symbol in open_positions:
                continue

            last = window.iloc[-1]
            price = last["close"]
            ema50 = last.get("ema_50", price)
            ema21 = last.get("ema_21", price)
            atr = float(current["atr"]) if current["atr"] > 0 else 0.001

            is_uptrend = price > ema50
            is_downtrend = price < ema50

            if signal.direction == "BUY" and is_downtrend:
                continue
            if signal.direction == "SELL" and is_uptrend:
                continue

            if abs(pa_score) < HC_PA_SCORE_THRESHOLD:
                continue

            if signal.direction == "BUY" and ml_prob < HC_ML_PROB_BUY:
                continue
            if signal.direction == "SELL" and ml_prob > HC_ML_PROB_SELL:
                continue

            sr = detect_sr_zone(window)
            near_sr = False
            if signal.direction == "BUY" and sr["support"]:
                if abs(price - sr["support"]) / atr < 1.0:
                    near_sr = True
            if signal.direction == "SELL" and sr["resistance"]:
                if abs(price - sr["resistance"]) / atr < 1.0:
                    near_sr = True

            tp_mult = HC_TP_ATR_MULT_NEAR_SR if near_sr else HC_TP_ATR_MULT_DEFAULT
            pending_entry = {
                "direction": signal.direction,
                "confidence": signal.confidence,
                "tp_mult": tp_mult,
                "from_hc": True,
            }
            continue

        if strategy in ("mr", "hybrid"):
            mr_score = compute_mr_score(window)
            if abs(mr_score) < 0.5:
                continue
            if symbol in open_positions:
                continue
            direction = "BUY" if mr_score > 0 else "SELL"

            last = window.iloc[-1]
            price = last["close"]
            ema50 = last.get("ema_50", price)
            ema21 = last.get("ema_21", price)
            atr = float(current["atr"]) if current["atr"] > 0 else 0.001

            is_uptrend = price > ema50
            is_downtrend = price < ema50

            if direction == "BUY" and is_downtrend:
                continue
            if direction == "SELL" and is_uptrend:
                continue

            # ADX filter: skip if strong trend (already baked into mr_score, but reinforce)
            adx_val = last.get("adx", 20)
            if adx_val > 35:
                continue

            sr = detect_sr_zone(window)
            near_sr = False
            if direction == "BUY" and sr["support"]:
                if abs(price - sr["support"]) / atr < 1.0:
                    near_sr = True
            if direction == "SELL" and sr["resistance"]:
                if abs(price - sr["resistance"]) / atr < 1.0:
                    near_sr = True

            tp_mult = 1.5 if near_sr else 1.0
            pending_entry = {
                "direction": direction,
                "confidence": min(1.0, abs(mr_score)),
                "tp_mult": tp_mult,
                "from_hc": True,
            }
            continue

        if strategy == "rapid":
            if symbol in open_positions:
                continue

            last = window.iloc[-1]
            price = last["close"]
            ema50 = last.get("ema_50", price)
            ema21 = last.get("ema_21", price)
            atr = float(current["atr"]) if current["atr"] > 0 else 0.001
            if atr <= 0:
                continue

            is_uptrend = price > ema50
            is_downtrend = price < ema50

            if abs(pa_score) < RAPID_PA_THRESHOLD:
                continue

            adx_val = last.get("adx", 0)
            if adx_val < 12:
                continue

            if pa_score > 0 and not (price > ema50):
                continue
            if pa_score < 0 and not (price < ema50):
                continue

            direction = "BUY" if pa_score > 0 else "SELL"

            tp_mult = RAPID_TP_MULT.get(symbol, 2.0) if isinstance(RAPID_TP_MULT, dict) else RAPID_TP_MULT
            pending_entry = {
                "direction": direction,
                "confidence": min(1.0, abs(pa_score)),
                "tp_mult": tp_mult,
                "sl_mult": RAPID_SL_MULT,
                "from_hc": True,
                "adx_entry": adx_val,
            }
            continue

        if strategy == "hybrid":
            if symbol in open_positions:
                continue

            last = window.iloc[-1]
            price = last["close"]
            ema50 = last.get("ema_50", price)
            ema21 = last.get("ema_21", price)
            atr = float(current["atr"]) if current["atr"] > 0 else 0.001
            if atr <= 0:
                continue

            is_uptrend = price > ema50
            is_downtrend = price < ema50
            adx_val = last.get("adx", 20)

            # Try rapid first
            if abs(pa_score) >= RAPID_PA_THRESHOLD:
                if (pa_score > 0 and is_uptrend) or (pa_score < 0 and is_downtrend):
                    direction = "BUY" if pa_score > 0 else "SELL"
                    tp_mult = RAPID_TP_MULT.get(symbol, 2.0) if isinstance(RAPID_TP_MULT, dict) else RAPID_TP_MULT
                    pending_entry = {
                        "direction": direction,
                        "confidence": min(1.0, abs(pa_score)),
                        "tp_mult": tp_mult,
                        "sl_mult": RAPID_SL_MULT,
                        "from_hc": True,
                    "adx_entry": adx_val,
                    }
                    continue

            # Fallback to MR
            if abs(mr_score) >= 0.5 and adx_val <= 35:
                mr_direction = "BUY" if mr_score > 0 else "SELL"
                if not (mr_direction == "BUY" and is_downtrend) and not (mr_direction == "SELL" and is_uptrend):
                    sr = detect_sr_zone(window)
                    near_sr = False
                    if mr_direction == "BUY" and sr.get("support"):
                        if abs(price - sr["support"]) / atr < 1.0:
                            near_sr = True
                    if mr_direction == "SELL" and sr.get("resistance"):
                        if abs(price - sr["resistance"]) / atr < 1.0:
                            near_sr = True
                    mr_tp = 1.5 if near_sr else 1.0
                    pending_entry = {
                        "direction": mr_direction,
                        "confidence": min(1.0, abs(mr_score)),
                        "tp_mult": mr_tp,
                        "from_hc": True,
                    }
                    continue

        signal = fuse(symbol, pa_score, ml_prob, 0.0, trend_score)

        if symbol in open_positions:
            continue

        if signal.direction == "SKIP":
            continue

        if ML_CONFIDENCE_GATE > 0:
            if signal.direction == "BUY" and ml_prob < 0.5 + ML_CONFIDENCE_GATE:
                continue
            if signal.direction == "SELL" and ml_prob > 0.5 - ML_CONFIDENCE_GATE:
                continue

        last = window.iloc[-1]
        price = last["close"]
        ema50 = last.get("ema_50", price)
        ema21 = last.get("ema_21", price)

        is_uptrend = price > ema50
        is_downtrend = price < ema50

        if signal.direction == "BUY" and is_downtrend:
            continue
        if signal.direction == "SELL" and is_uptrend:
            continue

        atr = float(current["atr"])
        if atr <= 0:
            continue

        pending_entry = {
            "direction": signal.direction,
            "confidence": signal.confidence,
            "from_hc": False,
        }

    for sym, pos in list(open_positions.items()):
        exit_price = pos["entry_price"]
        _close_trade(pos, exit_price, trades, df.index[-1], "OPEN", 0.0)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = calculate_metrics(trades_df, initial_balance)
    metrics["total_pnl"] = sum(t["pnl_usd"] for t in trades if "pnl_usd" in t)

    return trades_df, metrics


CONTRACT_SIZE = {
    "BTCUSD": 1,
    "EURUSD": 100_000,
}

def _pnl_to_usd(pnl: float, symbol: str, entry: float, exit_price: float) -> float:
    if symbol.endswith("USD"):
        return pnl
    mid = (entry + exit_price) / 2
    if mid <= 0:
        return pnl
    return pnl / mid


def _check_sl_tp(current_row, open_positions, trades, current_time) -> float:
    total_pnl = 0.0
    for sym in list(open_positions.keys()):
        pos = open_positions[sym]
        direction = pos["direction"]
        high = current_row["high"]
        low = current_row["low"]
        sl = pos["sl"]
        tp = pos["tp"]
        entry = pos["entry_price"]
        lot = pos["lot"]
        csize = CONTRACT_SIZE.get(sym, 100_000)
        ps = pip_size(sym)
        spread_pts = SPREAD_PIPS * ps
        slippage_pts = SLIPPAGE_PIPS * ps

        if direction == "BUY":
            hit_sl = low <= sl
            hit_tp = high >= tp
        else:
            hit_sl = high >= sl
            hit_tp = low <= tp

        if hit_sl:
            exit_price = sl - spread_pts / 2 - slippage_pts if direction == "BUY" else sl + spread_pts / 2 + slippage_pts
            pnl = (exit_price - entry) * lot * csize
            if direction == "SELL":
                pnl = -pnl
            pnl = _pnl_to_usd(pnl, sym, entry, exit_price)
            _close_trade(pos, exit_price, trades, current_time, "LOSS", pnl)
            total_pnl += pnl
            del open_positions[sym]
        elif hit_tp:
            exit_price = tp - spread_pts / 2 - slippage_pts if direction == "BUY" else tp + spread_pts / 2 + slippage_pts
            pnl = (exit_price - entry) * lot * csize
            if direction == "SELL":
                pnl = -pnl
            pnl = _pnl_to_usd(pnl, sym, entry, exit_price)
            _close_trade(pos, exit_price, trades, current_time, "WIN", pnl)
            total_pnl += pnl
            del open_positions[sym]
    return total_pnl


def _close_trade(pos, exit_price, trades, current_time, status, pnl):
    trade = {
        "timestamp": pos["entry_time"],
        "symbol": pos["symbol"],
        "direction": pos["direction"],
        "lot_size": pos["lot"],
        "entry_price": pos["entry_price"],
        "exit_price": exit_price,
        "sl_price": pos["sl"],
        "tp_price": pos["tp"],
        "pnl_usd": round(pnl, 2),
        "status": status,
        "exit_time": current_time,
        "adx_entry": pos.get("adx_entry", 0),
        "duration_hours": round((current_time - pos["entry_time"]).total_seconds() / 3600, 2),
    }
    trades.append(trade)
    return pnl


def run_backtest(csv_path: str = None, df: pd.DataFrame = None, symbol: str = "BTCUSD",
                 strategy: str = "m5", trade_date: pd.Timestamp = None,
                 trade_start_date: pd.Timestamp = None, **kwargs):
    if csv_path is not None:
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    elif df is None:
        raise ValueError("Provide either csv_path or df")

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df.columns = df.columns.str.lower()

    strategy_names = {"m5": "M5 Scalper", "pa": "Price Action", "hc": "High Conviction", "mr": "Mean Reversion", "rapid": "Rapid M5 Scalper"}
    strategy_name = strategy_names.get(strategy, strategy)
    print(f"\n{'='*50}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Backtesting {symbol}")
    print(f"  Data: {len(df)} candles ({df.index[0]} to {df.index[-1]})")
    print(f"{'='*50}")

    kwargs.setdefault("strategy", strategy)
    trades, metrics = backtest(df, symbol=symbol, trade_date=trade_date,
                                trade_start_date=trade_start_date, **kwargs)

    closed = len(trades[trades['status'].isin(['WIN','LOSS'])]) if not trades.empty else 0
    print(f"\n  Results: {len(trades)} trades ({closed} closed)")
    print_metrics(metrics)

    if not trades.empty:
        trades.to_csv(f"logs/backtest_{symbol.lower()}.csv", index=False)
        print(f"  Trade log saved: logs/backtest_{symbol.lower()}.csv")

    return trades, metrics


def download_sample_data(symbol: str = "BTCUSD", days: int = 90):
    import requests
    from io import StringIO

    print(f"Downloading {days} days of {symbol} M5 data...")
    end = int(datetime.now().timestamp())
    start = end - days * 86400

    url = (
        f"https://www.forexaholic.com/api/historical/{symbol}/M5"
        f"?start={start}&end={end}"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.columns = df.columns.str.lower()
        path = f"data/raw/{symbol}_M5_{days}d.csv"
        df.to_csv(path)
        print(f"Saved to {path} ({len(df)} candles)")
        return df
    except Exception as e:
        print(f"Could not download: {e}")
        print("You can manually download CSV from:")
        print("  - https://www.dukascopy.com (free historical data)")
        print("  - https://www.investing.com")
        print("  - https://www.histdata.com")
        return None
