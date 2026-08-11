"""Composite technical score: 0-100, with a transparent sub-factor
breakdown, a confidence score, and a signal tier.

Scope, stated plainly: this is a **technical/momentum/risk model only**.
Fundamental quality, valuation, growth, sentiment, and insider/institutional
activity are not included - see README for why (mainly: no free source of
point-in-time historical fundamentals, so those factors can't be
backtested honestly, only scored as a current snapshot). Adding them as
lower-confidence-weighted factors is the planned next phase, not done here.

The score is built from:
  - sub-factors computed purely from price/volume (momentum, trend, risk,
    volume/confluence, relative strength vs SPY) - see screener/indicators.py
  - a market regime tilt (screener/regime.py)
  - a data-driven "signal edge" adjustment: if a named technical signal
    (screener/signals.py) fired recently, its adjustment is sized by what
    that signal *actually did*, out-of-sample, in backtest/run_backtest.py -
    not by how textbook-bullish it sounds. A signal with no measured edge
    gets no credit, even if it fired.

Every number here is a `None` (not a fabricated 50) when the underlying
data couldn't be computed, and `missing_data` lists exactly what was
skipped and why.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import (
    FACTOR_WEIGHTS,
    MIN_CONFIDENCE_FOR_SIGNAL,
    MIN_DATA_COMPLETENESS,
    PRIMARY_HORIZON,
    RECENT_SIGNAL_WINDOW,
    REGIME_ADJUSTMENT_MAX_POINTS,
    SCORE_TIERS,
    SIGNAL_EDGE_MAX_POINTS,
)
from .indicators import adx, atr, momentum_return, relative_strength, sma
from .regime import Regime
from .signals import BEARISH_SIGNALS, BULLISH_SIGNALS, bearish_confluence_count, bullish_confluence_count, compute_all_signals

BACKTEST_PATH = Path(__file__).resolve().parent.parent / "data" / "backtest_results.json"

SIGNAL_DESCRIPTIONS = {
    "rsi_oversold_reversal": "RSI crossed back above 30 after being oversold",
    "rsi_overbought_reversal": "RSI crossed back below 70 after being overbought",
    "macd_bullish_crossover": "MACD crossed above its signal line",
    "macd_bearish_crossover": "MACD crossed below its signal line",
    "golden_cross": "50-day average crossed above the 200-day average (golden cross)",
    "death_cross": "50-day average crossed below the 200-day average (death cross)",
    "reclaim_50dma": "Price reclaimed the 50-day moving average",
    "lose_50dma": "Price broke below the 50-day moving average",
    "breakout_52w_high": "Price hit a new 52-week high",
    "breakdown_52w_low": "Price hit a new 52-week low",
}

_backtest_cache = None


def load_backtest_results() -> dict:
    global _backtest_cache
    if _backtest_cache is None:
        _backtest_cache = json.loads(BACKTEST_PATH.read_text()) if BACKTEST_PATH.exists() else {}
    return _backtest_cache


def format_evidence(evidence: dict | None) -> dict | None:
    """Pre-formatted, None/NaN-safe display strings for the UI - keeps the
    template free of arithmetic and NaN edge cases."""
    if not evidence:
        return None

    def fmt(key: str, spec: str, scale: float = 1) -> str:
        value = evidence.get(key)
        if value is None or (isinstance(value, float) and value != value):
            return "N/A"
        return spec.format(value * scale)

    return {
        "n": evidence["n"],
        "win_rate": fmt("win_rate", "{:.0f}%", 100),
        "mean_return": fmt("mean_return", "{:+.1f}%", 100),
        "excess_mean_return": fmt("excess_mean_return", "{:+.1f}%", 100),
        "sharpe": fmt("sharpe", "{:.2f}"),
        "sortino": fmt("sortino", "{:.2f}"),
        "profit_factor": fmt("profit_factor", "{:.2f}"),
        "max_drawdown": fmt("max_drawdown", "{:.1f}%", 100),
        "ci_low": fmt("ci_low", "{:+.1f}%", 100),
        "ci_high": fmt("ci_high", "{:+.1f}%", 100),
        "p_value": "<0.0001" if evidence["p_value"] < 0.0001 else f"{evidence['p_value']:.4f}",
        "significant": evidence["p_value"] < 0.05,
    }


def get_evidence(signal_name: str, horizon: int = PRIMARY_HORIZON) -> dict | None:
    """Out-of-sample (test-period) backtest stats for a named signal, or
    None if there's no backtest for it or the sample was too small."""
    horizon_data = load_backtest_results().get("signals", {}).get(signal_name, {}).get(str(horizon))
    if not horizon_data:
        return None
    test_stats = horizon_data.get("test")
    return test_stats if test_stats and test_stats.get("sufficient") else None


def _scale(value: float | None, lo: float, hi: float) -> float | None:
    if value is None or pd.isna(value):
        return None
    pct = (value - lo) / (hi - lo) if hi != lo else 0.5
    return float(max(0.0, min(100.0, pct * 100)))


def compute_raw_factors(df: pd.DataFrame, spy_close: pd.Series | None) -> tuple[dict, list[str]]:
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(close)
    factors: dict = {}
    missing: list[str] = []

    factors["mom_12_1"] = float(close.iloc[-22] / close.iloc[-253] - 1) if n > 253 else None
    if factors["mom_12_1"] is None:
        missing.append("12-month momentum (needs 1y+ history)")
    factors["mom_20d"] = momentum_return(close, 20)

    sma50, sma100, sma200 = sma(close, 50), sma(close, 100), sma(close, 200)
    factors["price_above_sma50"] = bool(close.iloc[-1] > sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None
    factors["price_above_sma200"] = bool(close.iloc[-1] > sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else None
    factors["golden_cross_state"] = (
        bool(sma50.iloc[-1] > sma200.iloc[-1]) if pd.notna(sma50.iloc[-1]) and pd.notna(sma200.iloc[-1]) else None
    )
    if factors["price_above_sma200"] is None:
        missing.append("200-day moving average (needs ~200d history)")

    adx_series = adx(high, low, close)
    factors["adx"] = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else None
    if factors["adx"] is None:
        missing.append("ADX trend strength")

    atr_series = atr(high, low, close)
    factors["atr_pct"] = float(atr_series.iloc[-1] / close.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else None

    if n > 252:
        window = close.iloc[-252:]
        drawdown = window / window.cummax() - 1
        factors["max_drawdown_1y"] = float(drawdown.min())
    else:
        factors["max_drawdown_1y"] = None
        missing.append("1-year max drawdown (needs 1y+ history)")

    factors["beta"] = None
    if spy_close is not None and n > 252 and len(spy_close) > 252:
        aligned = pd.concat([close.pct_change(), spy_close.pct_change()], axis=1, join="inner").dropna().tail(252)
        aligned.columns = ["stock", "spy"]
        if len(aligned) > 50 and aligned["spy"].var() > 0:
            factors["beta"] = float(aligned["stock"].cov(aligned["spy"]) / aligned["spy"].var())
    if factors["beta"] is None:
        missing.append("beta vs S&P 500 (needs 1y+ history)")

    if n > 20:
        avg20 = volume.rolling(20).mean().iloc[-1]
        factors["rel_volume"] = float(volume.iloc[-1] / avg20) if avg20 else None
    else:
        factors["rel_volume"] = None
        missing.append("relative volume")

    for months, days in ((3, 63), (6, 126), (12, 252)):
        key = f"rs_{months}m"
        factors[key] = relative_strength(close, spy_close, days) if spy_close is not None else None
        if factors[key] is None:
            missing.append(f"{months}-month relative strength vs S&P 500")

    return factors, missing


def momentum_subscore(f: dict) -> float | None:
    primary = _scale(f.get("mom_12_1"), -0.30, 0.60)
    short = _scale(f.get("mom_20d"), -0.15, 0.15)
    if primary is None:
        return short
    if short is None:
        return primary
    return 0.75 * primary + 0.25 * short


def trend_subscore(f: dict) -> float | None:
    parts = []
    if f.get("price_above_sma50") is not None:
        parts.append(70 if f["price_above_sma50"] else 30)
    if f.get("price_above_sma200") is not None:
        parts.append(70 if f["price_above_sma200"] else 30)
    if f.get("golden_cross_state") is not None:
        parts.append(65 if f["golden_cross_state"] else 35)
    if not parts:
        return None
    base = sum(parts) / len(parts)
    if f.get("adx") is not None:
        strength = min(f["adx"], 40) / 40
        direction = 1 if base >= 50 else -1
        base += direction * strength * 15
    return float(max(0.0, min(100.0, base)))


def risk_subscore(f: dict) -> float | None:
    parts = []
    atr_score = _scale(f.get("atr_pct"), 0.01, 0.06)
    if atr_score is not None:
        parts.append(100 - atr_score)
    beta_score = _scale(f.get("beta"), 0.5, 2.0)
    if beta_score is not None:
        parts.append(100 - beta_score)
    if f.get("max_drawdown_1y") is not None:
        parts.append(_scale(f["max_drawdown_1y"], -0.5, 0.0))
    if not parts:
        return None
    return float(sum(parts) / len(parts))


_BULL_CONFLUENCE_MAP = {0: 20, 1: 35, 2: 45, 3: 75, 4: 90, 5: 100}
_BEAR_CONFLUENCE_MAP = {0: 80, 1: 65, 2: 55, 3: 25, 4: 10, 5: 0}


def volume_confluence_subscore(f: dict, bull_count: int, bear_count: int) -> float:
    parts = []
    rel_vol_score = _scale(f.get("rel_volume"), 0.5, 2.5)
    if rel_vol_score is not None:
        parts.append(rel_vol_score)
    parts.append(_BULL_CONFLUENCE_MAP.get(int(bull_count), 50))
    parts.append(_BEAR_CONFLUENCE_MAP.get(int(bear_count), 50))
    return float(sum(parts) / len(parts))


def relative_strength_subscore(f: dict) -> float | None:
    scaled = [_scale(f.get(k), -0.20, 0.20) for k in ("rs_3m", "rs_6m", "rs_12m")]
    scaled = [s for s in scaled if s is not None]
    return float(sum(scaled) / len(scaled)) if scaled else None


@dataclass
class TechnicalScore:
    score: float
    confidence: float
    tier: str
    data_completeness: float
    subscores: dict
    active_bullish_signals: list
    active_bearish_signals: list
    evidence: dict | None
    evidence_signal_name: str | None
    reasons: list
    risks: list
    missing_data: list
    regime: Regime


def _pick_evidence(bull_count: int, active_bullish: list, active_bearish: list):
    if bull_count >= 3 and (ev := get_evidence("bullish_confluence_3plus")):
        return "bullish_confluence_3plus", ev
    if bull_count >= 2 and (ev := get_evidence("bullish_confluence_2plus")):
        return "bullish_confluence_2plus", ev

    candidates = []
    for name in active_bullish + active_bearish:
        ev = get_evidence(name)
        if ev:
            candidates.append((name, ev))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: abs(item[1]["excess_mean_return"]), reverse=True)
    return candidates[0]


def _build_reasoning(factors, active_bullish, active_bearish, evidence, evidence_name, regime, bull_count):
    reasons, risks = [], []

    if bull_count >= 3:
        reasons.append(f"{bull_count} independent bullish signals aligned within the last 3 sessions (confluence)")
    for name in active_bullish:
        reasons.append(SIGNAL_DESCRIPTIONS[name])
    for name in active_bearish:
        risks.append(SIGNAL_DESCRIPTIONS[name])

    if evidence and evidence_name:
        label = evidence_name.replace("_", " ")
        significance = "statistically significant" if evidence["p_value"] < 0.05 else "not statistically significant"
        excess = evidence["excess_mean_return"]
        note = (
            f"Backtest evidence for '{label}' (out-of-sample, n={evidence['n']}): "
            f"{evidence['win_rate'] * 100:.0f}% win rate, {evidence['mean_return'] * 100:+.1f}% avg return, "
            f"{excess * 100:+.1f}% vs. baseline drift ({significance})"
        )
        if excess > 0:
            reasons.append(note)
        else:
            risks.append(note + " - this signal has not beaten just holding, historically")
    elif active_bullish or active_bearish:
        reasons.append(
            "Active signal(s) have no sufficient/significant backtested edge on the validation universe - "
            "treated as weak evidence, not a strong basis for the call"
        )

    mom = factors.get("mom_12_1")
    if mom is not None:
        (reasons if mom > 0 else risks).append(f"12-month momentum (excl. last month): {mom * 100:+.1f}%")

    rs12 = factors.get("rs_12m")
    if rs12 is not None:
        (reasons if rs12 > 0 else risks).append(f"12-month relative strength vs. S&P 500: {rs12 * 100:+.1f}%")

    if factors.get("beta") is not None and factors["beta"] > 1.5:
        risks.append(f"High beta ({factors['beta']:.2f}) - amplified moves vs. the market")
    if factors.get("max_drawdown_1y") is not None and factors["max_drawdown_1y"] < -0.30:
        risks.append(f"Large trailing 1-year drawdown ({factors['max_drawdown_1y'] * 100:.0f}%)")

    reasons.append(f"Market regime: {regime.label} (regime score {regime.score:+.0f}, range -6 to +6)")

    if not reasons:
        reasons.append("No strong signals in either direction")

    return reasons, risks


def compute_technical_score(df: pd.DataFrame, spy_close: pd.Series | None, regime: Regime) -> TechnicalScore:
    if df is None or df.empty or len(df) < 30:
        return TechnicalScore(
            score=50.0, confidence=0.0, tier="INSUFFICIENT EVIDENCE", data_completeness=0.0,
            subscores={}, active_bullish_signals=[], active_bearish_signals=[],
            evidence=None, evidence_signal_name=None, reasons=[], risks=[],
            missing_data=["not enough price history to compute any technical factors"], regime=regime,
        )

    factors, missing = compute_raw_factors(df, spy_close)
    signals_df = compute_all_signals(df)

    bull_count_today = int(bullish_confluence_count(signals_df).iloc[-1])
    bear_count_today = int(bearish_confluence_count(signals_df).iloc[-1])

    recent = signals_df.tail(RECENT_SIGNAL_WINDOW)
    active_bullish = [name for name in BULLISH_SIGNALS if recent[name].any()]
    active_bearish = [name for name in BEARISH_SIGNALS if recent[name].any()]

    subscores = {
        "momentum": momentum_subscore(factors),
        "trend": trend_subscore(factors),
        "risk": risk_subscore(factors),
        "volume_confluence": volume_confluence_subscore(factors, bull_count_today, bear_count_today),
        "relative_strength": relative_strength_subscore(factors),
    }
    computed = {k: v for k, v in subscores.items() if v is not None}
    # Completeness reflects the underlying raw factors, not just whether each
    # subscore function managed to produce *some* number from partial data -
    # a subscore can still compute with half its inputs missing, which would
    # otherwise silently report 100% completeness on a thinly-covered stock.
    data_completeness = sum(1 for v in factors.values() if v is not None) / len(factors)
    base_score = (
        sum(FACTOR_WEIGHTS[k] * v for k, v in computed.items()) / sum(FACTOR_WEIGHTS[k] for k in computed)
        if computed else 50.0
    )

    regime_adj = max(-6, min(6, regime.score)) / 6 * REGIME_ADJUSTMENT_MAX_POINTS

    evidence_name, evidence = _pick_evidence(bull_count_today, active_bullish, active_bearish)
    edge_adj = 0.0
    if evidence:
        significance_factor = 1.0 if evidence["p_value"] < 0.05 else 0.4
        edge_adj = max(-1.0, min(1.0, evidence["excess_mean_return"] / 0.03)) * SIGNAL_EDGE_MAX_POINTS * significance_factor

    final_score = max(0.0, min(100.0, base_score + regime_adj + edge_adj))

    if computed:
        vals = list(computed.values())
        mean_v = sum(vals) / len(vals)
        std_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
        agreement = max(0.0, 100 - std_v)
    else:
        agreement = 30.0
    evidence_strength = 85 if evidence and evidence["p_value"] < 0.05 else (55 if evidence else 45)
    regime_extremity_penalty = 10 if abs(regime.score) >= 5 else 0

    confidence = max(0.0, min(100.0,
        0.4 * data_completeness * 100 + 0.35 * agreement + 0.25 * evidence_strength - regime_extremity_penalty
    ))

    if data_completeness < MIN_DATA_COMPLETENESS:
        tier = "INSUFFICIENT EVIDENCE"
    elif confidence < MIN_CONFIDENCE_FOR_SIGNAL:
        tier = "NO SIGNAL"
    else:
        tier = next(label for threshold, label in SCORE_TIERS if final_score >= threshold)

    reasons, risks = _build_reasoning(factors, active_bullish, active_bearish, evidence, evidence_name, regime, bull_count_today)

    return TechnicalScore(
        score=final_score,
        confidence=confidence,
        tier=tier,
        data_completeness=data_completeness,
        subscores=subscores,
        active_bullish_signals=active_bullish,
        active_bearish_signals=active_bearish,
        evidence=evidence,
        evidence_signal_name=evidence_name,
        reasons=reasons,
        risks=risks,
        missing_data=missing,
        regime=regime,
    )
