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
