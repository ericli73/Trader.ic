"""Combine technical indicators into a swing-trade score with human-readable reasoning.

This is a heuristic screener, not a predictive model: each signal is a common
technical rule of thumb, weighted by how strong a setup it typically indicates.
It surfaces candidates worth a closer look, not guaranteed winners.
"""

from dataclasses import dataclass

import pandas as pd

from .indicators import macd, rsi, sma

MIN_HISTORY_DAYS = 60


@dataclass
class Signal:
    score: int
    reasons: list[str]


def score_ticker(df: pd.DataFrame) -> Signal | None:
    """df must have 'Close' and 'Volume' columns, oldest row first."""
    if df is None or "Close" not in df or "Volume" not in df:
        return None

    close = df["Close"].dropna()
    volume = df["Volume"].dropna()
    if len(close) < MIN_HISTORY_DAYS or len(volume) < MIN_HISTORY_DAYS:
        return None

    score = 0
    reasons: list[str] = []

    rsi14 = rsi(close, 14)
    latest_rsi, prev_rsi = rsi14.iloc[-1], rsi14.iloc[-4]
    if pd.notna(latest_rsi) and pd.notna(prev_rsi):
        if prev_rsi < 30 <= latest_rsi:
            score += 2
            reasons.append(f"RSI {latest_rsi:.0f} turning up from oversold")
        elif 40 <= latest_rsi <= 60 and latest_rsi > prev_rsi:
            score += 1
            reasons.append(f"RSI {latest_rsi:.0f} rising from neutral")
        elif latest_rsi > 70:
            score -= 1
            reasons.append(f"RSI {latest_rsi:.0f} overbought")

    macd_line, signal_line = macd(close)
    if pd.notna(macd_line.iloc[-4]) and pd.notna(signal_line.iloc[-4]):
        crossed_up = (
            macd_line.iloc[-4] < signal_line.iloc[-4]
            and macd_line.iloc[-1] > signal_line.iloc[-1]
        )
        if crossed_up:
            score += 2
            reasons.append("MACD bullish crossover in the last 3 sessions")

    sma20, sma50 = sma(close, 20), sma(close, 50)
    if pd.notna(sma20.iloc[-1]) and pd.notna(sma50.iloc[-1]):
        if close.iloc[-1] > sma20.iloc[-1] > sma50.iloc[-1]:
            score += 1
            reasons.append("Price above rising SMA20/50 uptrend")
        if pd.notna(sma20.iloc[-6]) and pd.notna(sma50.iloc[-6]):
            if sma20.iloc[-6] < sma50.iloc[-6] and sma20.iloc[-1] > sma50.iloc[-1]:
                score += 2
                reasons.append("SMA20 golden-crossed SMA50 recently")

    avg_vol20 = volume.rolling(20).mean().iloc[-1]
    latest_vol = volume.iloc[-1]
    if pd.notna(avg_vol20) and avg_vol20 > 0 and latest_vol > 1.5 * avg_vol20:
        score += 1
        reasons.append(f"Volume {latest_vol / avg_vol20:.1f}x the 20-day average")

    if len(close) > 11:
        ret10 = close.iloc[-1] / close.iloc[-11] - 1
        if ret10 > 0:
            score += 1
            reasons.append(f"10-day return {ret10 * 100:+.1f}%")

    if not reasons:
        reasons.append("no strong signals fired")

    return Signal(score=score, reasons=reasons)


def recommend(score: int) -> tuple[str, str]:
    """Map a score to a coarse call. A heuristic opinion, not a guarantee."""
    if score >= 5:
        return "BUY", "Multiple bullish technical signals are aligned right now."
    if score >= 2:
        return "WATCH", "Some bullish signals present, but the setup isn't strongly confirmed yet."
    if score >= -1:
        return "HOLD", "No clear technical edge in either direction at the moment."
    return "AVOID", "Bearish or overbought signals currently outweigh bullish ones."
