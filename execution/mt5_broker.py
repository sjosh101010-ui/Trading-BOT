import MetaTrader5 as mt5

from signals.signal_types import TradeSignal
from config import MAGIC_NUMBER


def open_trade(
    signal: TradeSignal,
    lot: float,
    sl: float,
    tp: float,
) -> dict:
    tick = mt5.symbol_info_tick(signal.symbol)
    if tick is None:
        return {"success": False, "ticket": None, "error": "No tick data"}

    price = tick.ask if signal.direction == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    signal.symbol,
        "volume":    lot,
        "type":      order_type,
        "price":     price,
        "sl":        sl,
        "tp":        tp,
        "deviation": 10,
        "magic":     MAGIC_NUMBER,
        "comment":   f"AI_bot_{signal.direction}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"success": True, "ticket": result.order, "error": ""}
    else:
        return {"success": False, "ticket": None, "error": str(result.comment)}


def get_open_positions(symbol: str = None) -> list:
    positions = mt5.positions_get(symbol=symbol) or []
    return [p for p in positions if p.magic == MAGIC_NUMBER]
