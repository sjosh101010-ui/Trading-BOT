import requests
from datetime import datetime, timezone, timedelta

from config import NEWS_BLACKOUT_MINUTES

PAIR_TO_CURRENCIES = {
    "BTCUSD": ["USD"],
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "XAUUSD": ["USD"],
}

def fetch_events_today() -> list[dict]:
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        events = resp.json()
        high_impact = [
            {
                "time": datetime.fromisoformat(e["date"]).replace(tzinfo=timezone.utc),
                "currency": e["country"],
                "impact": e["impact"],
                "title": e["title"],
            }
            for e in events if e.get("impact") == "High"
        ]
        return high_impact
    except Exception as ex:
        print(f"[calendar] Warning: could not fetch events — {ex}")
        return []

def is_news_blackout(symbol: str, events: list[dict], window_minutes: int = None) -> bool:
    if window_minutes is None:
        window_minutes = NEWS_BLACKOUT_MINUTES
    now = datetime.now(timezone.utc)
    relevant_currencies = PAIR_TO_CURRENCIES.get(symbol, [])
    for ev in events:
        if ev["currency"] not in relevant_currencies:
            continue
        delta = abs((ev["time"] - now).total_seconds() / 60)
        if delta <= window_minutes:
            return True
    return False
