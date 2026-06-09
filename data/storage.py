import sqlite3
import pandas as pd
from pathlib import Path

from config import DB_PATH

def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                time      TEXT NOT NULL,
                open      REAL, high REAL, low REAL, close REAL, volume REAL,
                UNIQUE(symbol, timeframe, time)
            );
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                direction   TEXT NOT NULL,
                lot_size    REAL NOT NULL,
                entry_price REAL,
                sl_price    REAL,
                tp_price    REAL,
                exit_price  REAL,
                pnl_pips    REAL,
                pnl_usd     REAL,
                signal_score REAL,
                status      TEXT DEFAULT "OPEN",
                notes       TEXT
            );
            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT NOT NULL,
                symbol        TEXT NOT NULL,
                pa_score      REAL,
                ml_score      REAL,
                sentiment_score REAL,
                final_signal  TEXT,
                final_score   REAL
            );
        """)

def store_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str):
    with get_connection() as conn:
        for idx, row in df.iterrows():
            conn.execute(
                "INSERT OR IGNORE INTO ohlcv VALUES (NULL,?,?,?,?,?,?,?,?)",
                (symbol, timeframe, str(idx),
                 row.open, row.high, row.low, row.close, row.volume)
            )

def load_ohlcv(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT * FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY time DESC LIMIT ?",
            conn, params=(symbol, timeframe, limit)
        )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df.sort_index()

def store_signal(timestamp, symbol, pa_score, ml_score, sentiment_score, final_signal, final_score):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO signals VALUES (NULL,?,?,?,?,?,?,?)",
            (timestamp, symbol, pa_score, ml_score, sentiment_score, final_signal, final_score)
        )

def store_trade(timestamp, symbol, direction, lot_size, entry_price, sl_price, tp_price,
                exit_price=None, pnl_pips=None, pnl_usd=None, signal_score=None,
                status="OPEN", notes=""):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO trades VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?)",
            (timestamp, symbol, direction, lot_size, entry_price, sl_price, tp_price,
             exit_price, pnl_pips, pnl_usd, signal_score, status, notes)
        )
