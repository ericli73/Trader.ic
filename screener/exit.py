"""Sell/hold guidance for a tracked position.

Combines three things that are honestly different in kind, and says so:
  1. Standard, mechanical risk-management rules (fixed stop-loss, ATR
     trailing stop) - these aren't predictions, they're pre-committed
     limits on how much you're willing to lose.
  2. The same technical score/signal engine used everywhere else in this
     app (screener/scoring.py) - so "the model turned bearish" uses the
     identical, backtest-informed logic as the rest of the screener, not
     a separately invented rule.
  3. Recent news sentiment (screener/news.py) - explicitly unvalidated,
     since there's no honest way to backtest it with free data. It can
     only ever nudge HOLD to TRIM, never trigger a SELL by itself, and
     every reason it adds is labeled UNVALIDATED so it's never confused
     with the backtested signals above.
"""

from dataclasses import dataclass

import pandas as pd

from .config import (
    ATR_TRAILING_STOP_MULT,
    NEWS_SENTIMENT_MIN_HEADLINES,
    NEWS_SENTIMENT_TRIM_THRESHOLD,
    STOP_LOSS_PCT,
)
from .indicators import atr
from .news import NewsSummary, get_news_summary
from .positions import Position
from .scoring import TechnicalScore


@dataclass
class ExitGuidance:
    action: str  # "SELL", "TRIM", "HOLD"
    pnl_pct: float
    pnl_dollars: float | None
    current_price: float
    stop_loss_price: float
    trailing_stop_price: float | None
    reasons: list[str]
    news: NewsSummary


def compute_exit_guidance(position: Position, df: pd.DataFrame, current_price: float, ts: TechnicalScore) -> ExitGuidance:
    pnl_pct = current_price / position.buy_price - 1
    pnl_dollars = (current_price - position.buy_price) * position.shares if position.shares else None
    stop_loss_price = position.buy_price * (1 - STOP_LOSS_PCT)

    # "Highest close since purchase" is only meaningful if we actually know
    # when the purchase happened - without buy_date, falling back to the
    # entire available history's high would silently reference a peak that
    # may have nothing to do with this position (e.g. a multi-year-old high
    # long before you bought), producing a trailing stop that looks precise
    # but isn't. So: no buy_date, no trailing stop, not a guessed one.
    trailing_stop_price = None
    close = df["Close"]
    if position.buy_date:
        try:
            since_purchase = close[close.index >= pd.Timestamp(position.buy_date)]
        except (TypeError, ValueError):
            since_purchase = close.iloc[0:0]
        if len(since_purchase) >= 5 and len(df) > 20:
            highest_since_buy = since_purchase.max()
            atr_val = atr(df["High"], df["Low"], df["Close"]).iloc[-1]
            if pd.notna(atr_val):
                trailing_stop_price = float(highest_since_buy - ATR_TRAILING_STOP_MULT * atr_val)

    reasons: list[str] = []
    action = "HOLD"

    if current_price <= stop_loss_price:
        action = "SELL"
        reasons.append(
            f"Price is at/below your {STOP_LOSS_PCT * 100:.0f}% stop-loss (${stop_loss_price:.2f}) - "
            "a standard risk-management limit, not a prediction."
        )

    if trailing_stop_price is not None and current_price <= trailing_stop_price:
        action = "SELL"
        reasons.append(
            f"Price broke below its ATR trailing stop (${trailing_stop_price:.2f} = highest close since "
            f"purchase minus {ATR_TRAILING_STOP_MULT:.0f}x ATR)."
        )

    if ts.tier in ("SELL", "STRONG SELL"):
        action = "SELL"
        reasons.append(f"Current technical signal is {ts.tier} (score {ts.score:.0f}/100) - see the analysis above.")
    elif ts.tier == "WEAK SELL" and action != "SELL":
        action = "TRIM"
        reasons.append(f"Technical signal has weakened to {ts.tier} (score {ts.score:.0f}/100).")

    if action == "HOLD":
        direction = "Up" if pnl_pct >= 0 else "Down"
        reasons.append(f"{direction} {pnl_pct * 100:+.1f}% from your cost basis - above your stop-loss, no sell signal triggered.")

    if position.buy_date is None:
        reasons.append("No purchase date on file, so no ATR trailing stop is shown - add one on the position form for a tighter, more relevant exit level.")

    news = get_news_summary(position.symbol)
    if (
        action == "HOLD"
        and news.avg_sentiment is not None
        and news.headline_count >= NEWS_SENTIMENT_MIN_HEADLINES
        and news.avg_sentiment <= NEWS_SENTIMENT_TRIM_THRESHOLD
    ):
        action = "TRIM"
        reasons.append(
            f"UNVALIDATED: {news.headline_count} recent headlines skew negative "
            f"(avg sentiment {news.avg_sentiment:+.2f}) - not backtested, worth reading before deciding."
        )

    return ExitGuidance(
        action=action,
        pnl_pct=pnl_pct,
        pnl_dollars=pnl_dollars,
        current_price=current_price,
        stop_loss_price=stop_loss_price,
        trailing_stop_price=trailing_stop_price,
        reasons=reasons,
        news=news,
    )
