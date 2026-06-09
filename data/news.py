import feedparser
from datetime import datetime, timezone, timedelta

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.fxstreet.com/rss/news",
]

def fetch_recent_headlines(lookback_hours: int = 2) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    headlines = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    if pub >= cutoff:
                        headlines.append(entry.title)
        except Exception as ex:
            print(f"[news] Warning: feed error {url} — {ex}")
    return headlines
