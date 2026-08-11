"""Run the full signal backtest suite once and cache results to disk.

This is deliberately NOT run on every request - it's a ~40-stock, 10-year
validation run, expensive enough that it belongs offline. The live app
(screener/scoring.py) loads the resulting JSON to answer "does this signal
that just fired historically show a real, statistically significant edge?"

Usage: python -m backtest.run_backtest
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from screener.data import get_long_history, get_long_history_bulk
from screener.regime import compute_regime_series
from screener.signals import BEARISH_SIGNALS, BULLISH_SIGNALS, _SIGNAL_FUNCS, compute_all_signals, bullish_confluence_count
from backtest.engine import evaluate_signal

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "backtest_results.json"

# A fixed, diverse-by-sector set of large, long-listed liquid stocks - not
# the full S&P 500. This keeps the (expensive, 10-year) backtest safely
# clear of Yahoo Finance's free-tier rate limits and finishes in minutes
# rather than hours. It is NOT survivorship-bias-free (see engine.py
# docstring) and is a validation sample, not the full universe the live
# screener ranks day to day.
VALIDATION_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC",
    "JNJ", "PFE", "UNH", "XOM", "CVX", "PG", "KO", "PEP", "WMT", "HD",
    "DIS", "NFLX", "INTC", "CSCO", "ORCL", "IBM", "GE", "CAT", "BA", "MMM",
    "V", "MA", "COST", "MCD", "NKE", "ADBE", "CRM", "QCOM", "TXN", "VZ",
]


def _confluence_signal(min_count: int):
    def fn(df):
        signals = compute_all_signals(df)
        return bullish_confluence_count(signals) >= min_count
    return fn


def _stats_to_dict(horizon_result: dict) -> dict:
    return {
        "overall": asdict(horizon_result["overall"]),
        "train": asdict(horizon_result["train"]),
        "test": asdict(horizon_result["test"]),
        "by_regime": {k: asdict(v) for k, v in horizon_result["by_regime"].items()},
    }


def main():
    print(f"Fetching {len(VALIDATION_UNIVERSE)} tickers x 10y...")
    universe = get_long_history_bulk(VALIDATION_UNIVERSE, period="10y", max_workers=3)

    print("Fetching regime inputs (SPY, VIX, TNX, IRX)...")
    spy = get_long_history("SPY", period="10y")
    vix = get_long_history("^VIX", period="10y")
    tnx = get_long_history("^TNX", period="10y")
    irx = get_long_history("^IRX", period="10y")
    regime_series = compute_regime_series(spy, vix, tnx, irx)

    all_signal_names = list(BULLISH_SIGNALS) + list(BEARISH_SIGNALS)
    results = {}

    for name in all_signal_names:
        print(f"Backtesting {name}...")
        res = evaluate_signal(name, _SIGNAL_FUNCS[name], universe, regime_series)
        if res.get("insufficient_data"):
            continue
        results[name] = {str(h): _stats_to_dict(v) for h, v in res["horizons"].items()}

    for min_count in (2, 3):
        name = f"bullish_confluence_{min_count}plus"
        print(f"Backtesting {name}...")
        res = evaluate_signal(name, _confluence_signal(min_count), universe, regime_series)
        if not res.get("insufficient_data"):
            results[name] = {str(h): _stats_to_dict(v) for h, v in res["horizons"].items()}

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": VALIDATION_UNIVERSE,
        "universe_size": len(VALIDATION_UNIVERSE),
        "period": "10y",
        "caveats": [
            "Survivorship bias: only companies that still exist and are still S&P 500-eligible today are included. Delisted, acquired, or bankrupt companies are not represented.",
            "Validation universe only: 40 large, long-listed liquid stocks, not the full S&P 500.",
            "Single chronological train/test split, not full walk-forward re-optimization.",
            "Point-in-time correct for price/volume signals only - no fundamental, sentiment, or insider data is included in this backtest.",
        ],
        "signals": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
