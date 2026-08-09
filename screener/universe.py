"""S&P 500 ticker universe, cached locally so we don't re-scrape every run."""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "sp500.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Trader.ic-screener/1.0)"}


def get_sp500_tickers(refresh: bool = False) -> list[str]:
    """Return S&P 500 ticker symbols (Wikipedia's list), using a local cache.

    Falls back to the cache if a live refresh is requested but the network
    fetch fails, so the screener still works offline after the first run.
    """
    if refresh or not CACHE_PATH.exists():
        try:
            html = requests.get(WIKI_URL, headers=HEADERS, timeout=15).text
            table = pd.read_html(StringIO(html))[0]
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(CACHE_PATH, index=False)
        except Exception:
            if not CACHE_PATH.exists():
                raise

    return pd.read_csv(CACHE_PATH)["Symbol"].tolist()


def to_yahoo_symbol(symbol: str) -> str:
    """Wikipedia uses '.' for share classes (BRK.B); Yahoo Finance uses '-'."""
    return symbol.replace(".", "-")
