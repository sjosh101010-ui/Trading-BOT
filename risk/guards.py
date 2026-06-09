from datetime import datetime, timezone

from config import MAX_OPEN_POSITIONS, MAX_DRAWDOWN_PCT, MAX_SPREAD, TRADE_SESSION_START, TRADE_SESSION_END


def is_session_active() -> bool:
    now = datetime.now(timezone.utc)
    hour = now.hour
    return TRADE_SESSION_START <= hour < TRADE_SESSION_END


def is_spread_ok(symbol: str, current_spread_pips: float) -> bool:
    max_spread = MAX_SPREAD.get(symbol, 2.0)
    return current_spread_pips <= max_spread


def is_drawdown_ok(session_start_balance: float, current_balance: float) -> bool:
    if session_start_balance == 0:
        return True
    drawdown = (session_start_balance - current_balance) / session_start_balance
    return drawdown < MAX_DRAWDOWN_PCT


def can_open_trade(open_positions: int) -> bool:
    return open_positions < MAX_OPEN_POSITIONS


def run_all_guards(
    symbol: str,
    spread_pips: float,
    open_positions: int,
    session_start_balance: float,
    current_balance: float,
    is_blackout: bool,
) -> tuple[bool, str]:
    if not is_session_active():
        return False, "Outside trading session (London/NY only)"
    if is_blackout:
        return False, "News blackout window active"
    if not is_spread_ok(symbol, spread_pips):
        return False, f"Spread too wide: {spread_pips} pips"
    if not can_open_trade(open_positions):
        return False, f"Max positions reached: {open_positions}"
    if not is_drawdown_ok(session_start_balance, current_balance):
        return False, "Max daily drawdown reached — trading paused"
    return True, "OK"
