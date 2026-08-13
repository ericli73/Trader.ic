"""Configurable weights and thresholds for the composite technical score.

Kept in one place, not scattered through scoring logic, so they can be
revisited as the backtest evidence in data/backtest_results.json improves
or expands - these are starting points informed by that evidence, not
values hand-tuned to make any particular backtest look good.
"""

# Sub-factor weights for the composite 0-100 technical score. Must sum to 1.0.
FACTOR_WEIGHTS = {
    "momentum": 0.25,
    "trend": 0.20,
    "risk": 0.15,
    "volume_confluence": 0.20,
    "relative_strength": 0.20,
}

# Points added/subtracted to the composite score based on market regime
# (-6..+6 raw regime score, scaled to this many points). A blunt, additive
# tilt - not a full regime-conditional re-weighting of every factor, which
# would need per-regime backtest calibration beyond the current validation
# sample size.
REGIME_ADJUSTMENT_MAX_POINTS = 8

# Points added/subtracted based on the backtested edge of whatever named
# signal(s) recently fired, scaled by how large and significant that edge
# was in the out-of-sample test period.
SIGNAL_EDGE_MAX_POINTS = 15

# Signal tiers, keyed by the minimum composite score for that tier.
SCORE_TIERS = [
    (80, "STRONG BUY"),
    (65, "BUY"),
    (55, "WEAK BUY"),
    (45, "HOLD"),
    (35, "WEAK SELL"),
    (20, "SELL"),
    (0, "STRONG SELL"),
]

MIN_DATA_COMPLETENESS = 0.6   # below this -> "INSUFFICIENT EVIDENCE" regardless of score
MIN_CONFIDENCE_FOR_SIGNAL = 30  # below this -> "NO SIGNAL" regardless of score

# How many trading days back a named signal still counts as "recently fired".
RECENT_SIGNAL_WINDOW = 5

# Primary horizon (trading days) used to pull backtest evidence for the
# expected-return display and the signal-edge score adjustment.
PRIMARY_HORIZON = 20

# --- Position / exit guidance -------------------------------------------
# These are standard, widely-used risk-management heuristics (not
# backtested signals): a fixed stop-loss below cost basis (the "sell if
# you're down more than X%" rule popularized by William O'Neil's CANSLIM),
# and an ATR-based trailing stop (the "Chandelier Exit" - highest close
# since purchase, minus a multiple of ATR). They're mechanical risk limits,
# not predictions, and are shown as clearly labeled as such.
STOP_LOSS_PCT = 0.08
ATR_TRAILING_STOP_MULT = 3.0

# News sentiment (screener/news.py) is explicitly unvalidated - there's no
# honest way to backtest it with free, non-point-in-time data. So it never
# triggers a SELL by itself: at most it nudges HOLD -> TRIM, and only when
# there's enough recent negative headline volume to not just be noise.
NEWS_SENTIMENT_TRIM_THRESHOLD = -0.35  # avg VADER compound score at/below this counts as negative
NEWS_SENTIMENT_MIN_HEADLINES = 3       # need at least this many recent headlines to act on sentiment
