"""Recent news headlines + sentiment for a ticker, via yfinance.

This is explicitly a different kind of input than the rest of the app:
everywhere else, a signal only earns credit if it showed a measured edge
in backtest/engine.py's out-of-sample tests (see README's "signal-edge
adjustment"). There's no honest way to backtest news sentiment with free,
non-point-in-time data, so it can't be validated the same way. It's kept
separate and clearly labeled as unvalidated everywhere it's surfaced,
rather than blended into the backtested technical score.

Sentiment is scored with VADER (see requirements.txt), a lexicon-based
tool tuned for short, informal text. Its stock lexicon under-reacts to
finance-specific vocabulary ("surges", "downgrade", "layoffs" all score
near-neutral out of the box), so FINANCE_LEXICON below extends it with
common market-headline terms. This is still a blunt instrument - a
sarcastic or complex headline can score wrong - which is exactly why it
only ever nudges guidance, never triggers a SELL by itself (see
screener/exit.py).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .universe import to_yahoo_symbol

NEWS_LOOKBACK_DAYS = 3
NEWS_MAX_HEADLINES = 8
NEWS_CACHE_TTL_MINUTES = 30

FINANCE_LEXICON = {
    "surge": 2.5, "surges": 2.5, "surged": 2.5, "surging": 2.5,
    "soar": 3.0, "soars": 3.0, "soared": 3.0, "soaring": 3.0,
    "rally": 2.0, "rallies": 2.0, "rallied": 2.0, "rallying": 2.0,
    "beat": 1.8, "beats": 1.8, "beating": 1.8,
    "plunge": -2.8, "plunges": -2.8, "plunged": -2.8, "plunging": -2.8,
    "tumble": -2.2, "tumbles": -2.2, "tumbled": -2.2, "tumbling": -2.2,
    "crash": -3.0, "crashes": -3.0, "crashed": -3.0, "crashing": -3.0,
    "slump": -2.0, "slumps": -2.0, "slumped": -2.0, "slumping": -2.0,
    "downgrade": -1.8, "downgrades": -1.8, "downgraded": -1.8,
    "upgrade": 1.8, "upgrades": 1.8, "upgraded": 1.8,
    "layoffs": -2.0, "layoff": -2.0,
    "bankruptcy": -3.0, "fraud": -3.0, "lawsuit": -1.5,
    "investigation": -1.5, "recall": -1.8, "breach": -2.0,
    "miss": -1.5, "misses": -1.5, "missed": -1.5,
}

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(FINANCE_LEXICON)


@dataclass
class NewsItem:
    title: str
    publisher: str
    url: str | None
    published_at: datetime
    sentiment: float  # VADER compound score, -1 (very negative) to +1 (very positive)


@dataclass
class NewsSummary:
    items: list[NewsItem]
    avg_sentiment: float | None  # None if there are no recent headlines to score

    @property
    def headline_count(self) -> int:
        return len(self.items)


_cache: dict[str, tuple[datetime, NewsSummary]] = {}


def _parse_published_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fetch_news_summary(symbol: str) -> NewsSummary:
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)
    raw_items = yf.Ticker(to_yahoo_symbol(symbol)).news or []

    items: list[NewsItem] = []
    for raw in raw_items:
        content = raw.get("content", raw)  # tolerate older yfinance's flatter shape
        title = content.get("title")
        published_at = _parse_published_at(content.get("pubDate") or content.get("providerPublishTime"))
        if not title or published_at is None or published_at < cutoff:
            continue
        url = (content.get("canonicalUrl") or {}).get("url") or content.get("link")
        publisher = (content.get("provider") or {}).get("displayName", "Unknown source")
        sentiment = _analyzer.polarity_scores(title)["compound"]
        items.append(NewsItem(title=title, publisher=publisher, url=url, published_at=published_at, sentiment=sentiment))

    items.sort(key=lambda item: item.published_at, reverse=True)
    items = items[:NEWS_MAX_HEADLINES]
    avg_sentiment = sum(item.sentiment for item in items) / len(items) if items else None
    return NewsSummary(items=items, avg_sentiment=avg_sentiment)


def get_news_summary(symbol: str) -> NewsSummary:
    """Recent (last NEWS_LOOKBACK_DAYS) headlines + sentiment for `symbol`,
    cached for NEWS_CACHE_TTL_MINUTES to avoid re-fetching on every request."""
    now = datetime.now(timezone.utc)
    cached = _cache.get(symbol)
    if cached and now - cached[0] < timedelta(minutes=NEWS_CACHE_TTL_MINUTES):
        return cached[1]

    try:
        summary = _fetch_news_summary(symbol)
    except Exception:
        summary = NewsSummary(items=[], avg_sentiment=None)

    _cache[symbol] = (now, summary)
    return summary
