import os, requests, json, time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SL_CLIENT_ID", "a6f1fe1ed39d4990b05e5dfe22304b17")
CLIENT_SECRET = os.getenv("SL_CLIENT_SECRET", "0545019f-4aff-45fe-98f8-627f3b6f76e4")
BASE = "https://rest.simplefx.com"
CANDLES = "https://candles-core.simplefx.com"
DEMO_LOGIN = int(os.getenv("SL_DEMO_LOGIN", "3122956"))

class SimpleFXBroker:
    def __init__(self):
        self.token = None
        self.token_expires = 0
        self.account = None
        self.digits_cache = {}
        self.BASE = BASE
        self.DEMO_LOGIN = DEMO_LOGIN

    def _auth(self):
        if self.token and time.time() < self.token_expires - 60:
            return
        r = requests.post(f"{BASE}/api/v3/auth/key", json={
            "clientId": CLIENT_ID, "clientSecret": CLIENT_SECRET
        }, headers={"Content-Type": "application/json"}, timeout=10)
        data = r.json()
        self.token = data['data']['token']
        self.token_expires = time.time() + 3600

    def _headers(self):
        self._auth()
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def get_account_info(self):
        self._auth()
        r = requests.get(f"{BASE}/api/v3/accounts", headers=self._headers())
        for a in r.json()['data']:
            if a['login'] == DEMO_LOGIN:
                self.account = a
                return a
        return None

    def get_digits(self, symbol):
        if symbol in self.digits_cache:
            return self.digits_cache[symbol]
        r = requests.get("https://simplefx.com/utils/instruments.json", timeout=10)
        data = r.json()
        for key, inst in data.items():
            if isinstance(inst, dict) and inst.get('symbol') == symbol:
                d = inst.get('digits', 5)
                self.digits_cache[symbol] = d
                return d
        self.digits_cache[symbol] = 5
        return 5

    def get_candles(self, symbol, count=200, period=3600):
        now = int(datetime.now(timezone.utc).timestamp())
        fr = now - (count + 10) * period
        r = requests.get(f"{CANDLES}/api/v3/candles", params={
            "symbol": symbol, "cPeriod": period,
            "timeFrom": fr, "timeTo": now
        }, headers=self._headers(), timeout=10)
        if r.status_code != 200:
            return None
        data = r.json().get('data', [])
        return data[-count:] if len(data) > count else data

    def get_active_orders(self):
        self._auth()
        r = requests.post(f"{BASE}/api/v3/trading/orders/active", json={
            "login": DEMO_LOGIN, "reality": "DEMO"
        }, headers=self._headers(), timeout=10)
        if r.status_code != 200:
            return {}, []
        data = r.json().get('data', {})
        market = {}
        for o in data.get('marketOrders', []):
            market[o['id']] = o
        return market, data.get('pendingOrders', [])

    def open_market_order(self, symbol, side, volume, sl_price=None, tp_price=None):
        self._auth()
        digits = self.get_digits(symbol)
        order = {
            "Reality": "DEMO",
            "Login": DEMO_LOGIN,
            "Symbol": symbol,
            "Side": side.upper(),
            "Volume": volume,
            "IsFIFO": True,
        }
        if sl_price is not None:
            order["StopLoss"] = round(sl_price, digits)
        if tp_price is not None:
            order["TakeProfit"] = round(tp_price, digits)
        r = requests.post(f"{BASE}/api/v3/trading/orders/market",
                          json=order, headers=self._headers(), timeout=10)
        if r.status_code in (200, 201):
            resp = r.json()
            orders = resp.get('data', {}).get('marketOrders', [])
            if orders:
                o = orders[0].get('order', {})
                return {
                    "success": True,
                    "ticket": o.get('id'),
                    "price": o.get('openPrice'),
                    "volume": o.get('volume'),
                    "side": o.get('side'),
                }
            return {"success": True, "ticket": None, "price": None, "volume": volume}
        return {"success": False, "error": r.text[:300]}

    def close_position(self, symbol, volume, side, close_by_id):
        self._auth()
        opp = "SELL" if side.upper() == "BUY" else "BUY"
        order = {
            "Reality": "DEMO",
            "Login": DEMO_LOGIN,
            "Symbol": symbol,
            "Side": opp,
            "Volume": volume,
            "CloseBy": close_by_id,
            "IsFIFO": True,
        }
        r = requests.post(f"{BASE}/api/v3/trading/orders/market",
                          json=order, headers=self._headers(), timeout=10)
        return r.status_code in (200, 201)
