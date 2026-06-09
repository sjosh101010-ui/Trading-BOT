from config import MIN_LOT, MAX_LOT, BASE_RISK_PCT, MAX_RISK_PCT, INITIAL_BALANCE, RISK_SCALE_FACTOR, MAX_RISK_MULTIPLIER, MAX_RISK_PCT_PER_TRADE, CONTRACT_SIZE, TWO_DECIMAL

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


_peak_balance = {}  # symbol -> peak balance

PEAK_DECAY_RATE = 0.5

def _get_risk_mult(symbol: str, balance: float) -> float:
    prev = _peak_balance.get(symbol, INITIAL_BALANCE)
    dd = (prev - balance) / prev if prev > 0 else 0
    if dd > 0:
        decayed = prev * (1 - PEAK_DECAY_RATE * dd)
        _peak_balance[symbol] = max(decayed, balance)
    else:
        _peak_balance[symbol] = max(prev, balance)
    profit_ratio = (_peak_balance[symbol] - INITIAL_BALANCE) / max(INITIAL_BALANCE, 1)
    return min(MAX_RISK_MULTIPLIER, 1.0 + profit_ratio * RISK_SCALE_FACTOR)


def _pip_value_sim(symbol: str, price: float = None) -> float:
    csize = CONTRACT_SIZE.get(symbol, 100_000)
    zero_decimals = {"JPY", "KRW", "TWD", "INR", "IDR", "VND"}
    quote = symbol[-3:]
    if symbol in TWO_DECIMAL:
        ps = 0.01
    elif quote in zero_decimals:
        ps = 0.01
    else:
        ps = 0.0001
    pip_val_quote = csize * ps
    if symbol.endswith("USD"):
        return pip_val_quote
    elif symbol.startswith("USD") and price and price > 0:
        return pip_val_quote / price
    return pip_val_quote


def compute_lot_size(
    symbol: str,
    account_balance: float,
    sl_pips: float,
    confidence: float,
) -> float:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 not available — use compute_lot_size_sim() instead")
    base_risk = BASE_RISK_PCT.get(symbol, 0.01)
    max_risk = MAX_RISK_PCT.get(symbol, 0.02)
    risk_pct = base_risk + confidence * (max_risk - base_risk)
    risk_mult = _get_risk_mult(symbol, account_balance)
    risk_pct = min(risk_pct * risk_mult, MAX_RISK_PCT_PER_TRADE)
    risk_usd = account_balance * risk_pct

    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Cannot get symbol info for {symbol}")

    contract_size = info.trade_contract_size
    tick_size     = info.trade_tick_size
    tick_value    = info.trade_tick_value
    pip_value     = tick_value / tick_size * 0.0001

    if sl_pips <= 0 or pip_value <= 0:
        return MIN_LOT

    raw_lot = risk_usd / (sl_pips * pip_value)
    lot = max(MIN_LOT, min(MAX_LOT, round(raw_lot, 2)))
    return lot


def compute_lot_size_sim(
    symbol: str,
    account_balance: float,
    sl_pips: float,
    confidence: float,
    price: float = None,
) -> float:
    base_risk = BASE_RISK_PCT.get(symbol, 0.01)
    max_risk = MAX_RISK_PCT.get(symbol, 0.02)
    risk_pct = base_risk + confidence * (max_risk - base_risk)
    risk_mult = _get_risk_mult(symbol, account_balance)
    risk_pct = min(risk_pct * risk_mult, MAX_RISK_PCT_PER_TRADE)
    risk_usd = account_balance * risk_pct

    pip_value = _pip_value_sim(symbol, price)

    if sl_pips <= 0 or pip_value <= 0:
        return MIN_LOT

    raw_lot = risk_usd / (sl_pips * pip_value)
    lot = max(MIN_LOT, min(MAX_LOT, round(raw_lot, 2)))
    return lot
