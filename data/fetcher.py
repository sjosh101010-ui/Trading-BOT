import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

from config import SYMBOLS

load_dotenv()

try:
    import MetaTrader5 as mt5
    TIMEFRAME_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "H1":  mt5.TIMEFRAME_H1,
    }
except ImportError:
    mt5 = None
    TIMEFRAME_MAP = {}

def connect() -> bool:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 not installed — cannot connect")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    ok = mt5.login(
        int(os.getenv("MT5_LOGIN")),
        os.getenv("MT5_PASSWORD"),
        os.getenv("MT5_SERVER"),
    )
    if not ok:
        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    return True

def disconnect():
    if mt5 is not None:
        mt5.shutdown()

def fetch_ohlcv(symbol: str, timeframe: str, n_bars: int = 500) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 not installed — cannot fetch data")
    rates = mt5.copy_rates_from_pos(
        symbol, TIMEFRAME_MAP[timeframe], 0, n_bars
    )
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data for {symbol} {timeframe}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    df.set_index("time", inplace=True)
    return df

def fetch_tick(symbol: str) -> dict:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 not installed — cannot fetch ticks")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise ValueError(f"No tick for {symbol}")
    return {"bid": tick.bid, "ask": tick.ask, "spread": round(tick.ask - tick.bid, 5)}
