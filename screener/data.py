"""Long-range historical price data with on-disk caching.

Free Yahoo Finance access rate-limits aggressively under bulk load (we hit
this earlier just downloading 6 months for the S&P 500). Backtesting needs
years of history per ticker, so every symbol's daily series is cached to
disk and only re-fetched once it goes stale - repeated backtest runs during
development don't re-hit the network at all.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .universe import to_yahoo_symbol

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "history"
CACHE_MAX_AGE = timedelta(hours=20)
RETRY_DELAYS = (5, 15, 40)  # seconds; Yahoo's free endpoint rate-limits hard under bulk load


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.upper().replace('^', '_')}.csv"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(1)
    return df


def get_long_history(symbol: str, period: str = "10y", force_refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV for one ticker or index (e.g. '^VIX', '^TNX'), disk-cached."""
    path = _cache_path(symbol)
    if not force_refresh and path.exists():
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        if age < CACHE_MAX_AGE:
            try:
                return pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception:
                pass  # fall through and re-fetch on a corrupt cache file

    yahoo_symbol = to_yahoo_symbol(symbol)
    df = pd.DataFrame()
    for attempt, delay in enumerate((0, *RETRY_DELAYS)):
        if delay:
            time.sleep(delay)
        df = yf.download(yahoo_symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        df = _flatten(df)
        if not df.empty:
            break

    if not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    elif path.exists():
        # Rate-limited (or otherwise failed) with a stale cache available - stale data beats none.
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            pass
    return df


def get_long_history_bulk(symbols: list[str], period: str = "10y", max_workers: int = 4) -> dict[str, pd.DataFrame]:
    """Fetch+cache many tickers. Sequential-ish (low concurrency) by design -
    this is for backtesting, not the live UI, so it favors not getting
    rate-limited over raw speed."""

    def fetch(sym: str):
        try:
            return sym, get_long_history(sym, period=period)
        except Exception:
            return sym, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return dict(pool.map(fetch, symbols))
