from config import ATR_MULT, pip_size as _pip_size


def compute_sl_tp(
    symbol: str,
    direction: str,
    entry_price: float,
    atr: float,
) -> dict:
    mult = ATR_MULT.get(symbol, ATR_MULT["EURUSD"])
    sl_dist = atr * mult["sl"]
    tp_dist = atr * mult["tp"]

    if direction == "BUY":
        sl = round(entry_price - sl_dist, 5)
        tp = round(entry_price + tp_dist, 5)
    else:
        sl = round(entry_price + sl_dist, 5)
        tp = round(entry_price - tp_dist, 5)

    ps = _pip_size(symbol)
    pip_multiplier = int(1 / ps)
    sl_pips = round(sl_dist * pip_multiplier, 1)

    return {"sl": sl, "tp": tp, "sl_pips": sl_pips}
