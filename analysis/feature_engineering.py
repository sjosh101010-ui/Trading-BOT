import pandas as pd
import numpy as np


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=length, adjust=False).mean()
    avg_loss = loss.ewm(span=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({"MACD": macd_line, "MACDs": signal_line})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def bbands(series: pd.Series, length: int, std: int) -> pd.DataFrame:
    mid = series.rolling(length).mean()
    std_dev = series.rolling(length).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    return pd.DataFrame({"BBU": upper, "BBM": mid, "BBL": lower})


def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int, d: int) -> pd.DataFrame:
    low_k = low.rolling(k).min()
    high_k = high.rolling(k).max()
    stoch_k = 100 * (close - low_k) / (high_k - low_k).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return pd.DataFrame({"STOCHk": stoch_k, "STOCHd": stoch_d})


def compute_trend_score(df: pd.DataFrame) -> float:
    if len(df) < 50:
        return 0.0
    last = df.iloc[-1]
    price = last["close"]
    trend_score = 0.0

    if price > last["ema_50"]:
        trend_score += 0.3
    else:
        trend_score -= 0.3

    if price > last["ema_21"]:
        trend_score += 0.2
    else:
        trend_score -= 0.2

    if last["ema_8"] > last["ema_21"] > last["ema_50"]:
        trend_score += 0.3
    elif last["ema_8"] < last["ema_21"] < last["ema_50"]:
        trend_score -= 0.3

    rsi = last["rsi"]
    if rsi > 70:
        trend_score -= 0.2
    elif rsi < 30:
        trend_score += 0.2

    return round(max(-1.0, min(1.0, trend_score)), 4)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=high.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(length).mean()
    plus_di = 100 * plus_dm.rolling(length).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(length).mean() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(length).mean()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    raw = tp * volume
    pos = raw.where(tp > tp.shift(1), 0).rolling(length).sum()
    neg = raw.where(tp < tp.shift(1), 0).rolling(length).sum()
    m = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return m


def ema_slope(series: pd.Series, length: int = 3) -> pd.Series:
    return (series - series.shift(length)) / series.shift(length).replace(0, np.nan)


def autocorr(series: pd.Series, lag: int = 1, window: int = 20) -> pd.Series:
    def _ac(x):
        if len(x) < window:
            return np.nan
        return x.autocorr(lag=lag)
    return series.rolling(window + lag).apply(lambda x: _ac(pd.Series(x)), raw=False)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_8"]   = ema(df["close"], 8)
    df["ema_21"]  = ema(df["close"], 21)
    df["ema_50"]  = ema(df["close"], 50)
    df["rsi"]     = rsi(df["close"], 14)
    m = macd(df["close"], 12, 26, 9)
    df["macd"]    = m["MACD"]
    df["macd_sig"]= m["MACDs"]
    df["atr"]     = atr(df["high"], df["low"], df["close"], 14)
    bb = bbands(df["close"], 20, 2)
    df["bb_upper"]= bb["BBU"]
    df["bb_lower"]= bb["BBL"]
    df["bb_mid"]  = bb["BBM"]
    s = stoch(df["high"], df["low"], df["close"], 14, 3)
    df["stoch_k"] = s["STOCHk"]
    df["stoch_d"] = s["STOCHd"]

    rng = df["high"] - df["low"]
    body_val = (df["close"] - df["open"]).abs()
    df["body_ratio"]  = body_val / rng.replace(0, np.nan)
    df["close_in_rng"] = (df["close"] - df["low"]) / rng.replace(0, np.nan)
    df["shadow_upper"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng.replace(0, np.nan)
    df["shadow_lower"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng.replace(0, np.nan)
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(50).mean().replace(0, np.nan)
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    df["ret_1"]     = df["close"].pct_change(1)
    df["ret_3"]     = df["close"].pct_change(3)
    df["adx"]       = adx(df["high"], df["low"], df["close"], 14)

    df["roc_5"]  = df["close"].pct_change(5)
    df["roc_10"] = df["close"].pct_change(10)
    df["roc_20"] = df["close"].pct_change(20)

    df["ema_slope_8"]  = ema_slope(df["ema_8"], 3)
    df["ema_slope_21"] = ema_slope(df["ema_21"], 3)

    df["dist_ema_8"]  = df["close"] / df["ema_8"].replace(0, np.nan) - 1
    df["dist_ema_21"] = df["close"] / df["ema_21"].replace(0, np.nan) - 1
    df["dist_ema_50"] = df["close"] / df["ema_50"].replace(0, np.nan) - 1

    df["cci"] = cci(df["high"], df["low"], df["close"], 20)
    df["wr"]  = williams_r(df["high"], df["low"], df["close"], 14)
    df["mfi"] = mfi(df["high"], df["low"], df["close"], df["volume"], 14)

    df["hist_vol_10"] = df["ret_1"].rolling(10).std()
    df["hist_vol_20"] = df["ret_1"].rolling(20).std()
    df["atr_pct"]     = df["atr"] / df["close"].replace(0, np.nan)

    df["inside_bar"] = ((df["high"] < df["high"].shift(1)) & (df["low"] > df["low"].shift(1))).astype(float)
    df["outside_bar"] = ((df["high"] > df["high"].shift(1)) & (df["low"] < df["low"].shift(1))).astype(float)
    df["bull_2"] = ((df["close"] > df["open"]) & (df["close"].shift(1) > df["open"].shift(1))).astype(float)
    df["bear_2"] = ((df["close"] < df["open"]) & (df["close"].shift(1) < df["open"].shift(1))).astype(float)

    df["hh_10"] = df["high"].rolling(10).max()
    df["ll_10"] = df["low"].rolling(10).min()
    df["dist_hh_10"] = (df["close"] - df["hh_10"]) / df["hh_10"].replace(0, np.nan)
    df["dist_ll_10"] = (df["close"] - df["ll_10"]) / df["ll_10"].replace(0, np.nan)

    df["donchian_pos"] = (df["close"] - df["ll_10"]) / (df["hh_10"] - df["ll_10"]).replace(0, np.nan)

    df["autocorr_1"] = autocorr(df["close"], lag=1, window=10)
    df["skew_10"]    = df["ret_1"].rolling(10).skew()

    time_cols = {"time", "timestamp"}
    idx = None
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif time_cols & set(df.columns):
        tcol = (time_cols & set(df.columns)).pop()
        idx = pd.to_datetime(df[tcol])
    if idx is not None:
        hour = idx.hour if hasattr(idx, "hour") else pd.to_datetime(idx).hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        if hasattr(idx, "dayofweek"):
            df["dow"] = idx.dayofweek / 7.0

    df.dropna(inplace=True)
    return df

def candle_body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()

def candle_range(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]

def upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)

def lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]
