"""Statistical backtesting of named technical signals.

Design choices, stated up front because they materially affect what the
numbers mean:

- Point-in-time correctness: every signal (screener/signals.py) is a pure
  function of past price/volume, so there's no look-ahead bias in *when*
  a signal fires. Forward returns always come from strictly later bars.
- Survivorship bias: the universe is today's S&P 500 constituents. There's
  no free source of historical index membership that includes companies
  later delisted, acquired, or bankrupted, so this cannot avoid
  survivorship bias - the worst-case outcomes are structurally
  underrepresented. Every result carries that caveat.
- Train/test split: a single chronological split, not full walk-forward
  re-fitting. Simpler than true walk-forward validation, but it still
  prevents the single biggest error - reporting in-sample performance as
  if it were predictive. The "test" numbers are the ones to trust; "train"
  is shown for comparison, not as a claim.
- Minimum sample size: below MIN_SAMPLE for a bucket, no statistics are
  reported for it - "insufficient evidence" beats a fabricated Sharpe
  ratio from four data points.
- Baseline comparison: a signal's raw average forward return means little
  on its own - stocks drift upward on average anyway. Every result is
  reported alongside the *unconditional* forward return for the same
  universe/horizon (every valid day, not just signal days), so "excess
  return" reflects what the signal actually adds over just holding.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from screener.regime import label_for_score

TRADING_DAYS_PER_YEAR = 252
MIN_SAMPLE = 30
DEFAULT_HORIZONS = (5, 20, 60, 120)


@dataclass
class HorizonStats:
    horizon: int
    n: int
    sufficient: bool
    win_rate: float = float("nan")
    mean_return: float = float("nan")
    median_return: float = float("nan")
    std_return: float = float("nan")
    baseline_mean_return: float = float("nan")
    excess_mean_return: float = float("nan")
    sharpe: float = float("nan")
    sortino: float = float("nan")
    profit_factor: float = float("nan")
    worst_return: float = float("nan")
    best_return: float = float("nan")
    max_drawdown: float = float("nan")
    t_stat: float = float("nan")
    p_value: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")


def _sequential_max_drawdown(returns_by_date: pd.Series) -> float:
    """Synthetic equal-weighted sequential equity curve built from trade
    returns ordered by signal date. Overlapping trades and position sizing
    aren't modeled - this is a tail-risk sanity check, not a portfolio
    simulation."""
    if returns_by_date.empty:
        return float("nan")
    ordered = returns_by_date.sort_index()
    equity = (1 + ordered).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def _horizon_stats(sample: pd.DataFrame, baseline_returns: pd.Series, horizon: int) -> HorizonStats:
    n = len(sample)
    if n < MIN_SAMPLE:
        return HorizonStats(horizon=horizon, n=n, sufficient=False)

    values = sample["ret"].to_numpy()
    mean, std, median = float(np.mean(values)), float(np.std(values, ddof=1)), float(np.median(values))
    win_rate = float((values > 0).mean())

    downside = values[values < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float("nan")

    periods_per_year = TRADING_DAYS_PER_YEAR / horizon
    sharpe = mean / std * np.sqrt(periods_per_year) if std > 0 else float("nan")
    sortino = mean / downside_std * np.sqrt(periods_per_year) if downside_std and downside_std > 0 else float("nan")

    gains, losses = values[values > 0].sum(), values[values < 0].sum()
    profit_factor = float(gains / abs(losses)) if losses != 0 else float("nan")

    t_stat, p_value = stats.ttest_1samp(values, 0.0)
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=stats.sem(values))

    baseline_mean = float(baseline_returns.mean()) if len(baseline_returns) else float("nan")

    return HorizonStats(
        horizon=horizon,
        n=n,
        sufficient=True,
        win_rate=win_rate,
        mean_return=mean,
        median_return=median,
        std_return=std,
        baseline_mean_return=baseline_mean,
        excess_mean_return=(mean - baseline_mean) if pd.notna(baseline_mean) else float("nan"),
        sharpe=float(sharpe),
        sortino=float(sortino),
        profit_factor=profit_factor,
        worst_return=float(values.min()),
        best_return=float(values.max()),
        max_drawdown=_sequential_max_drawdown(pd.Series(values, index=sample["date"])),
        t_stat=float(t_stat),
        p_value=float(p_value),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


def evaluate_signal(
    name: str,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    universe: dict[str, pd.DataFrame],
    regime_series: pd.Series | None = None,
    horizons=DEFAULT_HORIZONS,
    train_frac: float = 0.7,
    min_history: int = 300,
) -> dict:
    """Backtest one signal across a universe of {symbol: OHLCV df}.

    Returns overall/train/test stats per horizon, plus a per-regime
    breakdown for the "overall" period (regime buckets are usually too
    sparse to also split by train/test on top).
    """
    per_horizon_signal: dict[int, list[pd.DataFrame]] = {h: [] for h in horizons}
    per_horizon_baseline: dict[int, list[pd.DataFrame]] = {h: [] for h in horizons}
    date_bounds = []

    for symbol, df in universe.items():
        if df is None or df.empty or len(df) < min_history:
            continue
        close = df["Close"]
        fire_mask = signal_fn(df).reindex(df.index).fillna(False).to_numpy()
        date_bounds.append(close.index.min())
        date_bounds.append(close.index.max())

        for h in horizons:
            fwd = (close.shift(-h) / close - 1).to_numpy()
            valid = ~pd.isna(fwd)

            base_df = pd.DataFrame({"date": close.index[valid], "ret": fwd[valid], "symbol": symbol})
            per_horizon_baseline[h].append(base_df)

            fire_valid = fire_mask & valid
            sig_df = pd.DataFrame({"date": close.index[fire_valid], "ret": fwd[fire_valid], "symbol": symbol})
            per_horizon_signal[h].append(sig_df)

    if not date_bounds:
        return {"signal": name, "insufficient_data": True}

    split_date = min(date_bounds) + (max(date_bounds) - min(date_bounds)) * train_frac
    result = {"signal": name, "split_date": split_date, "horizons": {}}

    for h in horizons:
        sig = pd.concat(per_horizon_signal[h], ignore_index=True) if per_horizon_signal[h] else pd.DataFrame(columns=["date", "ret", "symbol"])
        base = pd.concat(per_horizon_baseline[h], ignore_index=True) if per_horizon_baseline[h] else pd.DataFrame(columns=["date", "ret", "symbol"])

        if regime_series is not None and not sig.empty:
            sig = sig.copy()
            sig["regime"] = [
                label_for_score(regime_series.asof(d)) if pd.notna(regime_series.asof(d)) else "Unknown"
                for d in sig["date"]
            ]
        else:
            sig["regime"] = "Unknown"

        train_sig, test_sig = sig[sig["date"] < split_date], sig[sig["date"] >= split_date]
        train_base, test_base = base[base["date"] < split_date], base[base["date"] >= split_date]

        by_regime = {
            regime: _horizon_stats(sig[sig["regime"] == regime], base["ret"], h)
            for regime in sorted(sig["regime"].unique())
        }

        result["horizons"][h] = {
            "overall": _horizon_stats(sig, base["ret"], h),
            "train": _horizon_stats(train_sig, train_base["ret"], h),
            "test": _horizon_stats(test_sig, test_base["ret"], h),
            "by_regime": by_regime,
        }

    return result
