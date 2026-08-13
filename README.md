# Trader.ic

A quantitative technical screener for the S&P 500: live prices, a
multi-factor 0-100 score per stock, and BUY/HOLD/SELL-style signals backed
by out-of-sample backtest evidence rather than textbook assumptions about
what "should" work.

**Not financial advice.** This is a decision-support tool built on free,
near-real-time data (Yahoo Finance) - see [Limitations](#limitations)
before trusting any number it shows you.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. First load runs a full S&P 500 scan
(~15-30s); after that it auto-rescans every 20 minutes in the background.

There's also a CLI:

```bash
python -m screener --top 15          # print the top 15 to the terminal
python -m screener --limit 20        # quick test on the first 20 tickers
python -m quotes AAPL MSFT GOOGL     # one-off live quotes
```

## How it works

### The score

Every stock gets a **0-100 technical score**, built from five sub-factors,
each shown separately so you can see *why*:

| Factor | What it measures |
|---|---|
| Momentum | 12-month return (excluding the most recent month, the classic academic momentum window) blended with 20-day momentum |
| Trend | Price vs. 50/200-day moving averages, golden/death cross state, ADX trend strength |
| Risk | ATR%, beta vs. S&P 500, trailing 1-year max drawdown (higher sub-score = *lower* risk) |
| Volume & confluence | Relative volume plus how many named technical signals recently fired together |
| Relative strength | 3/6/12-month excess return vs. the S&P 500 |

These combine (configurable weights in `screener/config.py`) into a base
score, then two adjustments are applied:

- **Market regime tilt** - a small nudge based on SPY trend, VIX, and the
  Treasury yield curve (see `screener/regime.py`).
- **Signal-edge adjustment** - if a named technical signal fired recently
  (RSI reversal, MACD crossover, golden cross, etc. - see
  `screener/signals.py`), the score moves by an amount sized to what that
  *specific signal has actually done historically*, out-of-sample, not by
  how bullish it sounds. A signal with no measured edge gets no credit.

A **confidence score** (0-100%) is reported alongside the score, driven by
data completeness, agreement across sub-factors, and whether the backtest
evidence behind the call is statistically significant. Below a minimum
data-completeness or confidence threshold, the tier becomes `INSUFFICIENT
EVIDENCE` or `NO SIGNAL` instead of forcing a call - see
`screener/config.py` for the thresholds.

Signal tiers: `STRONG BUY`, `BUY`, `WEAK BUY`, `HOLD`, `WEAK SELL`, `SELL`,
`STRONG SELL`, `NO SIGNAL`, `INSUFFICIENT EVIDENCE`.

### The backtest

`backtest/engine.py` answers, for any named signal: *when this has fired
historically, what actually happened to forward returns at 5/20/60/120
trading days?* - win rate, mean/median return, Sharpe, Sortino, profit
factor, a t-test p-value, and a 95% confidence interval, always compared
against the **unconditional baseline** return for the same universe and
horizon (a signal's raw return means little if it's just market drift).

Run it yourself with:

```bash
python -m backtest.run_backtest
```

This writes `data/backtest_results.json`, which `screener/scoring.py`
reads to decide whether a fired signal deserves any credit.

**What it found**, on the validation universe (see below): RSI
oversold-reversal shows a real, significant edge at 20-120 day horizons;
MACD crossovers and 52-week breakouts are statistically significant but
show near-zero or *negative* excess return over baseline (they mostly ride
general drift); 2-signal confluence doesn't beat baseline either; but
**3+ signals firing together shows a real edge** (+2.9% excess return,
70% win rate, Sharpe 1.7, p=0.002). The scoring model weights these
findings accordingly rather than treating every textbook signal as equally
useful.

## Limitations

Read this before trusting any score or expected-return number.

- **Technical/momentum/risk only.** No fundamentals, valuation, growth,
  sentiment, or insider/institutional data. Not an oversight - there's no
  free source of *point-in-time* historical fundamentals, so those factors
  can't be backtested honestly with the data this project has access to.
  They'd have to be scored as a current snapshot only, clearly labeled as
  such, which is the natural next phase.
- **Survivorship bias.** The backtest validation universe (`AAPL`, `MSFT`,
  `GOOGL`, ... 40 large, long-listed liquid stocks - see
  `backtest/run_backtest.py`) is today's still-listed large caps. There's
  no free source of historical index membership including companies that
  were later delisted, acquired, or went bankrupt, so worst-case outcomes
  are structurally underrepresented in every backtest number.
  Full S&P 500 backtesting for this reason is not run by default - see [Full-universe backtesting](#full-universe-backtesting) if you want to change this.
- **Single train/test split**, not full walk-forward re-optimization.
  Chronological 70/30 split; "test" numbers are the ones to trust, "train"
  is shown for comparison only.
- **Not a licensed real-time feed.** Prices are Yahoo Finance's free,
  unofficial endpoint - typically accurate to within seconds to low
  minutes, and it rate-limits aggressively under sustained bulk use (you
  may see a scan complete with fewer than 503 stocks scored if this
  happens; it self-heals on the next auto-rescan).
- **A backtested edge is not a guarantee.** Sample sizes, p-values, and
  confidence intervals are reported so you can judge evidence strength
  yourself - "statistically significant" describes the historical sample,
  not a promise about tomorrow.

### Full-universe backtesting

`backtest/run_backtest.py` runs against a fixed 40-stock sample by design
- large enough to get real statistics, small enough to avoid Yahoo's rate
limits and finish in minutes. To backtest a larger or different universe,
edit `VALIDATION_UNIVERSE` in that file; expect a much longer run and a
higher chance of hitting rate limits on the initial (uncached) fetch.

## Project structure

```
app.py                    Flask dashboard (live scan, stock detail pages, JSON APIs)
quotes/                   Live quote fetching (yfinance fast_info)
screener/
  universe.py             S&P 500 ticker list (scraped from Wikipedia, cached)
  data.py                 Long-history fetch with disk caching + retry/backoff
  indicators.py           RSI, MACD, SMA, ATR, ADX, momentum, relative strength
  regime.py               Market regime classification (SPY/VIX/yield curve)
  signals.py              Named technical signal event detectors
  scoring.py              Composite 0-100 score, confidence, signal tiers, reasoning
  config.py               Configurable weights and thresholds
  pipeline.py             Orchestrates a full scan (used by both the CLI and the web app)
  __main__.py             CLI entry point (`python -m screener`)
backtest/
  engine.py               Forward-return statistics engine (win rate, Sharpe, p-values, train/test, regime segmentation)
  run_backtest.py         Script that runs the full signal suite and caches results to data/backtest_results.json
static/, templates/       Web UI (vanilla HTML/CSS/JS + Chart.js)
data/                     Generated/cached (S&P 500 list, price history cache, backtest results) - not committed
```
