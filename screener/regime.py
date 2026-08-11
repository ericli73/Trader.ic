"""Market regime classification.

The same technical signal doesn't mean the same thing in every market
environment - a "buy the dip" RSI reversal behaves differently in a
strong uptrend than in a confirmed downtrend. This computes a coarse,
transparent regime score from three well-known, freely-available inputs:

  - SPY trend (price vs 200SMA, 50/200 golden-cross state, 3-month momentum)
  - VIX level (fear/complacency gauge)
  - Treasury yield curve slope (10y - 13w, via ^TNX/^IRX)

Each is a simple point-in-time function of price/yield data, so - unlike
fundamentals - this can be computed for any historical date too, which is
what lets the backtester segment results by regime instead of only
reporting one blended number.
"""

from dataclasses import dataclass

import pandas as pd

from .data import get_long_history
from .indicators import sma

REGIME_LABELS = ["Strong Bear", "Bear", "Neutral", "Bull", "Strong Bull"]


@dataclass
class Regime:
    label: str
    score: float
    spy_above_200sma: bool | None
    golden_cross: bool | None
    spy_3m_momentum: float | None
    vix: float | None
    yield_curve_slope: float | None


def _trend_score(spy_close: pd.Series) -> pd.Series:
    sma200 = sma(spy_close, 200)
    sma50 = sma(spy_close, 50)
    mom3m = spy_close.pct_change(63)

    above_200 = (spy_close > sma200).astype(float) * 2 - 1
    golden = (sma50 > sma200).astype(float) * 2 - 1
    momentum = pd.Series(0.0, index=spy_close.index)
    momentum[mom3m > 0.05] = 1.0
    momentum[mom3m < -0.05] = -1.0

    return above_200 + golden + momentum


def _vix_score(vix_close: pd.Series) -> pd.Series:
    score = pd.Series(0.0, index=vix_close.index)
    score[vix_close < 15] = 1.0
    score[(vix_close >= 20) & (vix_close < 30)] = -1.0
    score[vix_close >= 30] = -2.0
    return score


def _yield_curve_score(slope: pd.Series) -> pd.Series:
    score = pd.Series(0.0, index=slope.index)
    score[slope > 0.5] = 1.0
    score[slope < -0.5] = -1.0
    return score


def label_for_score(score: float) -> str:
    if score >= 3:
        return "Strong Bull"
    if score >= 1:
        return "Bull"
    if score > -1:
        return "Neutral"
    if score > -3:
        return "Bear"
    return "Strong Bear"


def compute_regime_series(spy: pd.DataFrame, vix: pd.DataFrame, tnx: pd.DataFrame, irx: pd.DataFrame) -> pd.Series:
    """Daily regime score aligned to SPY's trading calendar, for backtesting."""
    spy_close = spy["Close"]
    trend = _trend_score(spy_close)

    vix_aligned = vix["Close"].reindex(spy_close.index).ffill()
    vix_pts = _vix_score(vix_aligned)

    slope = (tnx["Close"] - irx["Close"]).reindex(spy_close.index).ffill()
    yc_pts = _yield_curve_score(slope)

    return trend + vix_pts + yc_pts


def get_current_regime() -> Regime:
    """Snapshot regime classification as of the latest cached trading day."""
    spy = get_long_history("SPY", period="2y")
    vix = get_long_history("^VIX", period="2y")
    tnx = get_long_history("^TNX", period="2y")
    irx = get_long_history("^IRX", period="2y")

    if spy.empty:
        return Regime("Neutral", 0.0, None, None, None, None, None)

    spy_close = spy["Close"]
    sma200 = sma(spy_close, 200)
    sma50 = sma(spy_close, 50)
    mom3m = spy_close.pct_change(63).iloc[-1] if len(spy_close) > 63 else None

    above_200 = bool(spy_close.iloc[-1] > sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None
    golden = bool(sma50.iloc[-1] > sma200.iloc[-1]) if pd.notna(sma50.iloc[-1]) and pd.notna(sma200.iloc[-1]) else None
    vix_last = float(vix["Close"].iloc[-1]) if not vix.empty else None
    slope_last = float(tnx["Close"].iloc[-1] - irx["Close"].iloc[-1]) if not tnx.empty and not irx.empty else None

    if not (vix.empty or tnx.empty or irx.empty):
        series = compute_regime_series(spy, vix, tnx, irx)
        score = float(series.iloc[-1])
    else:
        # Degrade gracefully - trend-only score if VIX/yield data is unavailable.
        score = float(_trend_score(spy_close).iloc[-1])

    return Regime(
        label=label_for_score(score),
        score=score,
        spy_above_200sma=above_200,
        golden_cross=golden,
        spy_3m_momentum=float(mom3m) if mom3m is not None and pd.notna(mom3m) else None,
        vix=vix_last,
        yield_curve_slope=slope_last,
    )
