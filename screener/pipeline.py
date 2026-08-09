"""Shared pipeline: fetch S&P 500 price history and score every ticker.

Used by both the CLI (screener/__main__.py) and the local web app (app.py)
so the scanning logic lives in exactly one place.
"""

import yfinance as yf

from .scorer import Signal, score_ticker
from .universe import get_sp500_tickers, to_yahoo_symbol


def run_screen(limit: int | None = None, refresh: bool = False) -> list[tuple[str, Signal]]:
    """Return (symbol, Signal) pairs sorted by score, highest first."""
    tickers = get_sp500_tickers(refresh=refresh)
    if limit:
        tickers = tickers[:limit]
    yahoo_tickers = [to_yahoo_symbol(t) for t in tickers]

    data = yf.download(
        yahoo_tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=True,
    )

    results = []
    for orig, yt in zip(tickers, yahoo_tickers):
        try:
            df = data[yt] if len(yahoo_tickers) > 1 else data
        except (KeyError, TypeError):
            continue
        signal = score_ticker(df)
        if signal:
            results.append((orig, signal))

    results.sort(key=lambda item: item[1].score, reverse=True)
    return results


def _flatten(df):
    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(1)
    return df


def fetch_daily_history(symbol: str, period: str = "6mo"):
    """Daily OHLCV for one ticker, used for both scoring and charting."""
    df = yf.download(
        to_yahoo_symbol(symbol), period=period, interval="1d", progress=False, auto_adjust=True
    )
    return _flatten(df)


def fetch_intraday_history(symbol: str):
    """Minute-level price path for the current (or most recent) session."""
    yahoo_symbol = to_yahoo_symbol(symbol)
    df = yf.download(yahoo_symbol, period="1d", interval="1m", progress=False, auto_adjust=True)
    if df.empty:
        df = yf.download(yahoo_symbol, period="5d", interval="5m", progress=False, auto_adjust=True)
    return _flatten(df)
