import os, requests, json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OANDA_API_KEY")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
DEMO = os.getenv("OANDA_DEMO", "true").lower() == "true"
BASE = "https://api-fxpractice.oanda.com" if DEMO else "https://api-fxtrade.oanda.com"

SYMBOL_MAP = {
    "EURUSD": "EUR_USD",
    "GBPUSD": "GBP_USD",
    "USDJPY": "USD_JPY",
    "USDCAD": "USD_CAD",
    "AUDUSD": "AUD_USD",
    "NZDUSD": "NZD_USD",
    "GBPJPY": "GBP_JPY",
    "EURJPY": "EUR_JPY",
    "BTCUSD": "BTC_USD",
}


class OANDABroker:
    def __init__(self):
        self.api_key = API_KEY
        self.account_id = ACCOUNT_ID
        self.base = BASE
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _s2o(self, symbol):
        return SYMBOL_MAP.get(symbol, symbol.replace("=", "_") if "=" in symbol else symbol)

    def _lot_to_units(self, lot):
        return int(round(lot * 100000))

    def _req(self, method, path, data=None):
        url = f"{self.base}/v3/accounts/{self.account_id}{path}"
        r = requests.request(method, url, headers=self._headers,
                             json=data, timeout=10)
        if r.status_code >= 400:
            return {"success": False, "error": r.text[:500]}
        return {"success": True, "data": r.json()}

    def get_account_info(self):
        r = self._req("GET", "/summary")
        if not r["success"]:
            return None
        a = r["data"].get("account", {})
        return {
            "balance": float(a.get("balance", 0)),
            "equity": float(a.get("NAV", 0)),
            "open_trades": a.get("openTradeCount", 0),
            "currency": a.get("currency", "USD"),
        }

    def get_open_positions(self):
        r = self._req("GET", "/openTrades")
        if not r["success"]:
            return {}
        trades = r["data"].get("trades", [])
        result = {}
        for t in trades:
            units = float(t.get("currentUnits", 0))
            result[t["id"]] = {
                "id": t["id"],
                "symbol": t["instrument"].replace("_", ""),
                "direction": "BUY" if units > 0 else "SELL",
                "volume": abs(units) / 100000,
                "openPrice": float(t["price"]),
                "sl": float(t.get("stopLossOrder", {}).get("price", 0)) if "stopLossOrder" in t else None,
                "tp": float(t.get("takeProfitOrder", {}).get("price", 0)) if "takeProfitOrder" in t else None,
                "profit": float(t.get("unrealizedPL", 0)) + float(t.get("financing", 0)),
            }
        return result

    def open_market_order(self, symbol, side, volume, sl_price=None, tp_price=None):
        units = self._lot_to_units(volume)
        if side.upper() == "SELL":
            units = -units
        order = {
            "order": {
                "type": "MARKET",
                "instrument": self._s2o(symbol),
                "units": str(units),
            }
        }
        if sl_price is not None:
            order["order"]["stopLossOnFill"] = {"price": str(round(sl_price, 5))}
        if tp_price is not None:
            order["order"]["takeProfitOnFill"] = {"price": str(round(tp_price, 5))}
        r = self._req("POST", "/orders", order)
        if not r["success"]:
            return {"success": False, "error": r.get("error", "unknown")}
        fill = r["data"].get("orderFillTransaction", {})
        trade_id = None
        if "tradeOpened" in fill:
            trade_id = fill["tradeOpened"].get("tradeID")
        elif "id" in fill:
            trade_id = fill["id"]
        return {
            "success": True,
            "ticket": trade_id,
            "price": float(fill.get("price", 0)),
            "volume": volume,
        }

    def close_position(self, symbol, volume, side, close_by_id):
        r = self._req("PUT", f"/trades/{close_by_id}/close")
        return r["success"]

    def get_digits(self, symbol):
        return 5

    @staticmethod
    def pip_size(symbol):
        return 0.0001
