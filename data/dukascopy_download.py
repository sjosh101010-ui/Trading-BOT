from __future__ import annotations

import argparse
import lzma
import struct
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

BASE_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
CHUNK_SIZE = 20
SESSION_START = 1
SESSION_END = 10
MAX_RETRIES = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    return parser.parse_args()


def download_hour(symbol: str, dt: datetime) -> list[dict] | None:
    url = BASE_URL.format(
        symbol=symbol,
        year=dt.year,
        month=dt.month - 1,
        day=dt.day,
        hour=dt.hour,
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200 or len(resp.content) == 0:
                return None
            break
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                return None
            continue
    else:
        return None

    try:
        raw = lzma.decompress(resp.content)
    except (lzma.LZMAError, EOFError):
        return None

    ticks = []
    for i in range(0, len(raw), CHUNK_SIZE):
        time_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">5i", raw[i : i + CHUNK_SIZE])
        ticks.append({
            "ts": dt + timedelta(milliseconds=time_ms),
            "price": bid_raw / 100000,
            "volume": ask_vol + bid_vol,
        })
    return ticks


def download_day(symbol: str, day: datetime) -> pd.DataFrame:
    hours = [day + timedelta(hours=h) for h in range(SESSION_START, SESSION_END)]
    all_ticks = []
    with ThreadPoolExecutor(max_workers=9) as pool:
        futures = {pool.submit(download_hour, symbol, h): h for h in hours}
        for future in as_completed(futures):
            ticks = future.result()
            if ticks:
                all_ticks.extend(ticks)

    if not all_ticks:
        return pd.DataFrame()

    df = pd.DataFrame(all_ticks)
    df = df.set_index("ts").sort_index()
    ohlcv = pd.DataFrame({
        "open": df["price"].resample("5min").first(),
        "high": df["price"].resample("5min").max(),
        "low": df["price"].resample("5min").min(),
        "close": df["price"].resample("5min").last(),
        "volume": df["volume"].resample("5min").sum(),
    })
    for col in ["open", "high", "low", "close"]:
        ohlcv[col] = ohlcv[col].round(5)
    ohlcv["symbol"] = symbol
    return ohlcv[["symbol", "open", "high", "low", "close", "volume"]].dropna()


def download_day_batch(symbol, days):
    results = []
    for d in days:
        ohlcv = download_day(symbol, d)
        results.append(ohlcv)
    return results


def main():
    args = parse_args()
    symbol = args.symbol
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    all_days = []
    cur = start
    while cur <= end:
        all_days.append(cur.replace(hour=0, minute=0, second=0, microsecond=0))
        cur += timedelta(days=1)

    total_days = len(all_days)
    BATCH_SIZE = 1
    all_candles = []
    done = 0

    for batch_start in range(0, total_days, BATCH_SIZE):
        batch = all_days[batch_start:batch_start + BATCH_SIZE]
        day = batch[0]
        ohlcv = download_day(symbol, day)
        if not ohlcv.empty:
            all_candles.append(ohlcv)
        done += 1
        pct = done / total_days * 100
        print(f"\r  {pct:.0f}% ({done}/{total_days} days, latest: {day.date()})", end="", flush=True)

    print()
    if not all_candles:
        print("No data downloaded.")
        return

    result = pd.concat(all_candles).sort_index()
    result.index.name = "timestamp"

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_M5_{args.start}_{args.end}.csv"
    result.to_csv(out_path)
    print(f"\nSaved {len(result)} candles to {out_path}")
    print(f"Range: {result.index[0]} -> {result.index[-1]}")


if __name__ == "__main__":
    main()
