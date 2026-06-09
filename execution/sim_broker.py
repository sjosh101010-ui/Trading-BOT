import json, os
import random
from datetime import datetime, timezone
from pathlib import Path

from signals.signal_types import TradeSignal
from config import contract_size

STATE_FILE = Path(__file__).parent.parent / "logs" / "broker_state.json"


class SimBroker:
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions = []
        self.trade_id = 0
        self._dirty = False

    def save(self):
        state = {
            "balance": self.balance,
            "equity": self.equity,
            "trade_id": self.trade_id,
            "positions": self.positions,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        self._dirty = False

    @classmethod
    def load_or_create(cls, initial_balance: float) -> "SimBroker":
        broker = cls(initial_balance)
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                broker.balance = state.get("balance", initial_balance)
                broker.equity = state.get("equity", broker.balance)
                broker.trade_id = state.get("trade_id", 0)
                broker.positions = state.get("positions", [])
            except Exception:
                pass
        return broker

    @staticmethod
    def reset(amount: float):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "balance": amount,
            "equity": amount,
            "trade_id": 0,
            "positions": [],
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        print(f"  Broker state reset to ${amount:.2f}")

    def open_trade(self, signal: TradeSignal, lot: float, sl: float, tp: float, entry_price: float = None, be_trigger: float = None) -> dict:
        self.trade_id += 1
        if entry_price is None:
            entry_price = sl + abs(tp - sl) * 0.4
        trade = {
            "ticket": self.trade_id,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "lot": lot,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "status": "OPEN",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "be_trigger": be_trigger,
            "be_activated": False,
        }
        self.positions.append(trade)
        self._dirty = True
        self.save()
        return {"success": True, "ticket": self.trade_id, "error": ""}

    def get_open_positions(self, symbol: str = None):
        if symbol:
            return [p for p in self.positions if p["symbol"] == symbol and p["status"] == "OPEN"]
        return [p for p in self.positions if p["status"] == "OPEN"]

    def simulate_tick(self, current_high=None, current_low=None):
        closed = []
        for pos in self.positions:
            if pos["status"] != "OPEN":
                continue
            csize = contract_size(pos.get("symbol", "BTCUSD"))

            if current_high is not None and current_low is not None:
                price_high, price_low = current_high, current_low
            else:
                price_level = abs(pos["entry_price"]) if pos["entry_price"] != 0 else 1
                noise = random.uniform(-price_level * 0.001, price_level * 0.001)
                price_high = pos["entry_price"] + abs(noise)
                price_low = pos["entry_price"] - abs(noise)

            # Breakeven trail: if price hits be_trigger, move SL to breakeven
            if not pos.get("be_activated") and pos.get("be_trigger") is not None:
                be_trig = pos["be_trigger"]
                if pos["direction"] == "BUY" and price_high >= be_trig:
                    pos["sl"] = pos["entry_price"] + 0.0001
                    pos["be_activated"] = True
                elif pos["direction"] == "SELL" and price_low <= be_trig:
                    pos["sl"] = pos["entry_price"] - 0.0001
                    pos["be_activated"] = True

            if pos["direction"] == "BUY":
                if price_low <= pos["sl"]:
                    pos["status"] = "LOSS"
                    pos["exit_price"] = pos["sl"]
                    pos["pnl_usd"] = round((pos["sl"] - pos["entry_price"]) * pos["lot"] * csize, 2)
                    self.balance += pos["pnl_usd"]
                    closed.append({**pos})
                elif price_high >= pos["tp"]:
                    pos["status"] = "WIN"
                    pos["exit_price"] = pos["tp"]
                    pos["pnl_usd"] = round((pos["tp"] - pos["entry_price"]) * pos["lot"] * csize, 2)
                    self.balance += pos["pnl_usd"]
                    closed.append({**pos})
            else:
                if price_high >= pos["sl"]:
                    pos["status"] = "LOSS"
                    pos["exit_price"] = pos["sl"]
                    pos["pnl_usd"] = round((pos["entry_price"] - pos["sl"]) * pos["lot"] * csize, 2)
                    self.balance += pos["pnl_usd"]
                    closed.append({**pos})
                elif price_low <= pos["tp"]:
                    pos["status"] = "WIN"
                    pos["exit_price"] = pos["tp"]
                    pos["pnl_usd"] = round((pos["entry_price"] - pos["tp"]) * pos["lot"] * csize, 2)
                    self.balance += pos["pnl_usd"]
                    closed.append({**pos})
        self.equity = self.balance
        self.save()
        return closed
