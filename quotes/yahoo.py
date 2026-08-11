"""Live-ish stock quotes via Yahoo Finance (yfinance).

Yahoo Finance isn't a true real-time feed for free use - expect prices
to lag the market by a few seconds to a couple of minutes.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import yfinance as yf


@dataclass
class Quote:
    symbol: str
    price: float
    currency: str
    previous_close: float

    @property
    def change(self) -> float:
        return self.price - self.previous_close

    @property
    def change_pct(self) -> float:
        return self.change / self.previous_close * 100 if self.previous_close else 0.0


def get_quote(symbol: str) -> Quote:
    """Fetch the current quote for a single ticker symbol, e.g. "AAPL"."""
    info = yf.Ticker(symbol).fast_info
    return Quote(
        symbol=symbol.upper(),
        price=info["lastPrice"],
        currency=info["currency"],
        previous_close=info["previousClose"],
    )


def get_quotes(symbols: list[str], max_workers: int = 10) -> dict[str, Quote]:
    """Fetch current quotes for multiple ticker symbols concurrently.

    Symbols that fail to fetch (delisted, rate-limited, etc.) are skipped
    rather than failing the whole batch.
    """
    def safe_get(symbol: str) -> Quote | None:
        try:
            return get_quote(symbol)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        quotes = pool.map(safe_get, symbols)
    return {quote.symbol: quote for quote in quotes if quote is not None}


@dataclass
class ExtendedQuote:
    symbol: str
    market_state: str | None
    regular_price: float | None
    pre_price: float | None
    pre_change_pct: float | None
    post_price: float | None
    post_change_pct: float | None

    @property
    def label(self) -> str | None:
        """Which extended-hours print (if any) is the relevant one to show."""
        if self.pre_price is not None:
            return "Pre-market"
        if self.post_price is not None:
            return "After-hours"
        return None

    @property
    def price(self) -> float | None:
        return self.pre_price if self.pre_price is not None else self.post_price

    @property
    def change_pct(self) -> float | None:
        return self.pre_change_pct if self.pre_price is not None else self.post_change_pct


def get_extended_quote(symbol: str) -> ExtendedQuote | None:
    """Pre-market/after-hours price, via yfinance's fuller (slower) `.info`
    endpoint - `fast_info` doesn't carry extended-hours fields at all."""
    info = yf.Ticker(symbol).info
    if not info.get("hasPrePostMarketData"):
        return None
    return ExtendedQuote(
        symbol=symbol.upper(),
        market_state=info.get("marketState"),
        regular_price=info.get("regularMarketPrice"),
        pre_price=info.get("preMarketPrice"),
        pre_change_pct=info.get("preMarketChangePercent"),
        post_price=info.get("postMarketPrice"),
        post_change_pct=info.get("postMarketChangePercent"),
    )


def get_extended_quotes(symbols: list[str], max_workers: int = 10) -> dict[str, ExtendedQuote]:
    def safe_get(symbol: str) -> ExtendedQuote | None:
        try:
            return get_extended_quote(symbol)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        quotes = pool.map(safe_get, symbols)
    return {q.symbol: q for q in quotes if q is not None}
