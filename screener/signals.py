"""Named, discrete technical signal events.

Each function returns a boolean pd.Series aligned to the input dataframe's
index: True on the exact day the event fires. Built as true crossover/event
detectors (comparing yesterday vs today) rather than fuzzy "was this
condition true sometime in the last N days" checks - that distinction
matters because the latter both misses real events outside the window and
false-fires on stale conditions. This directly fixes an earlier version of
this screener, which approximated "RSI crossed back above 30" by comparing
today's RSI to RSI exactly 4 days ago - a coincidence-prone shortcut, not a
real crossover check.

These are the building blocks for both the live score (screener/scoring.py)
and the backtester (backtest/engine.py) - the exact same function is used
in both places, so "what the score saw" and "what the backtest tested" are
guaranteed to be the same definition.
"""

import pandas as pd

from .indicators import macd, rsi, sma

BULLISH_SIGNALS = (
    "rsi_oversold_reversal",
    "macd_bullish_crossover",
    "golden_cross",
    "reclaim_50dma",
    "breakout_52w_high",
)

BEARISH_SIGNALS = (
    "rsi_overbought_reversal",
    "macd_bearish_crossover",
    "death_cross",
    "lose_50dma",
    "breakdown_52w_low",
)


def _crossover_up(a: pd.Series, b: pd.Series) -> pd.Series:
    """True where series `a` crosses from <= b to > b."""
    return (a.shift(1) <= b.shift(1)) & (a > b)


def _crossover_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)


def rsi_oversold_reversal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI crosses back above 30 having just been below it - the classic
    oversold-reversal buy signal (as described by e.g. Schwab's investor
    education material)."""
    r = rsi(df["Close"], period)
    return _crossover_up(r, pd.Series(30.0, index=df.index))


def rsi_overbought_reversal(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI crosses back below 70 having just been above it - the mirror
    overbought-reversal sell signal."""
    r = rsi(df["Close"], period)
    return _crossover_down(r, pd.Series(70.0, index=df.index))


def macd_bullish_crossover(df: pd.DataFrame) -> pd.Series:
    macd_line, signal_line = macd(df["Close"])
    return _crossover_up(macd_line, signal_line)


def macd_bearish_crossover(df: pd.DataFrame) -> pd.Series:
    macd_line, signal_line = macd(df["Close"])
    return _crossover_down(macd_line, signal_line)


def golden_cross(df: pd.DataFrame) -> pd.Series:
    sma50, sma200 = sma(df["Close"], 50), sma(df["Close"], 200)
    return _crossover_up(sma50, sma200)


def death_cross(df: pd.DataFrame) -> pd.Series:
    sma50, sma200 = sma(df["Close"], 50), sma(df["Close"], 200)
    return _crossover_down(sma50, sma200)


def reclaim_50dma(df: pd.DataFrame) -> pd.Series:
    close, sma50 = df["Close"], sma(df["Close"], 50)
    return _crossover_up(close, sma50)


def lose_50dma(df: pd.DataFrame) -> pd.Series:
    close, sma50 = df["Close"], sma(df["Close"], 50)
    return _crossover_down(close, sma50)


def breakout_52w_high(df: pd.DataFrame) -> pd.Series:
    """Today's close is a new 252-trading-day high."""
    close = df["Close"]
    rolling_high = close.rolling(252).max()
    return (close >= rolling_high) & rolling_high.notna()


def breakdown_52w_low(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    rolling_low = close.rolling(252).min()
    return (close <= rolling_low) & rolling_low.notna()


_SIGNAL_FUNCS = {
    "rsi_oversold_reversal": rsi_oversold_reversal,
    "rsi_overbought_reversal": rsi_overbought_reversal,
    "macd_bullish_crossover": macd_bullish_crossover,
    "macd_bearish_crossover": macd_bearish_crossover,
    "golden_cross": golden_cross,
    "death_cross": death_cross,
    "reclaim_50dma": reclaim_50dma,
    "lose_50dma": lose_50dma,
    "breakout_52w_high": breakout_52w_high,
    "breakdown_52w_low": breakdown_52w_low,
}


def compute_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """All named signals as boolean columns, aligned to df's index."""
    return pd.DataFrame({name: fn(df).fillna(False) for name, fn in _SIGNAL_FUNCS.items()})


def bullish_confluence_count(signals: pd.DataFrame, window: int = 3) -> pd.Series:
    """How many distinct bullish signals fired within the trailing `window`
    days - the 'confluence' concept: several independent signals agreeing
    is a materially stronger setup than any single one alone."""
    bullish = signals[list(BULLISH_SIGNALS)].rolling(window, min_periods=1).max()
    return bullish.sum(axis=1)


def bearish_confluence_count(signals: pd.DataFrame, window: int = 3) -> pd.Series:
    bearish = signals[list(BEARISH_SIGNALS)].rolling(window, min_periods=1).max()
    return bearish.sum(axis=1)
