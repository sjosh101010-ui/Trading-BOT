import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "VantageInternational-Demo")


class MT5Broker:
    def __init__(self):
        self.initialized = False
        self._ensure_init()

    def _ensure_init(self):
        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
        self.initialized = True
        if MT5_LOGIN and MT5_PASSWORD and mt5.account_info() is None:
            if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    def get_account_info(self):
        self._ensure_init()
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance": info.balance,
            "equity": info.equity,
            "currency": info.currency,
        }

    def get_open_positions(self, symbol=None):
        self._ensure_init()
        if symbol:
            pos = mt5.positions_get(symbol=symbol)
        else:
            pos = mt5.positions_get()
        if pos is None:
            return {}
        result = {}
        for p in pos:
            result[p.ticket] = {
                "id": p.ticket,
                "symbol": p.symbol,
                "direction": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "openPrice": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
            }
        return result

    def open_market_order(self, symbol, side, volume, sl_price=None, tp_price=None,
                          sl_distance=None, tp_distance=None):
        self._ensure_init()
        order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "error": "Cannot get tick"}
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        if sl_distance is not None:
            min_sl = 0.00030
            sl_distance = max(sl_distance, min_sl)
            sl_price = price - sl_distance if order_type == mt5.ORDER_TYPE_BUY else price + sl_distance
        if tp_distance is not None:
            tp_price = price + tp_distance if order_type == mt5.ORDER_TYPE_BUY else price - tp_distance

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "rapid_scalper",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl_price is not None:
            request["sl"] = round(sl_price, 5)
        if tp_price is not None:
            request["tp"] = round(tp_price, 5)
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"MT5 error {result.retcode}: {result.comment}"}
        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": volume,
        }

    def close_position(self, symbol, volume, side, close_by_id):
        self._ensure_init()
        mt5.symbol_select(symbol, True)
        position = mt5.positions_get(ticket=close_by_id)
        if not position:
            return False
        pos = position[0]
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": close_by_id,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "rapid_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def get_digits(self, symbol):
        self._ensure_init()
        info = mt5.symbol_info(symbol)
        return info.digits if info else 5

    @staticmethod
    def pip_size(symbol):
        return 0.0001
