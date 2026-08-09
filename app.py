"""Local web dashboard for the S&P 500 technical screener.

Run with: python app.py, then open http://127.0.0.1:5000
"""

import os
import threading
import time
from datetime import datetime

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

from quotes.yahoo import get_quote, get_quotes
from screener.indicators import sma
from screener.pipeline import fetch_daily_history, fetch_intraday_history, run_screen
from screener.scorer import recommend, score_ticker
from screener.universe import to_yahoo_symbol

app = Flask(__name__)

SCAN_INTERVAL_SECONDS = 20 * 60  # auto-rescan the full S&P 500 every 20 minutes

_cache = {"results": None, "timestamp": None}
_cache_lock = threading.Lock()


def _scan_and_cache(limit=None):
    results = run_screen(limit=limit)
    with _cache_lock:
        _cache["results"] = results
        _cache["timestamp"] = datetime.now()


def _background_scanner():
    while True:
        time.sleep(SCAN_INTERVAL_SECONDS)
        try:
            _scan_and_cache()
            print(f"[auto-rescan] rescanned at {_cache['timestamp']:%H:%M:%S}")
        except Exception as exc:
            print(f"[auto-rescan] scan failed: {exc}")


@app.route("/")
def index():
    top = request.args.get("top", default=25, type=int)
    limit = request.args.get("limit", default=None, type=int)
    refresh = request.args.get("refresh") == "1"

    if refresh or _cache["results"] is None:
        _scan_and_cache(limit=limit)

    with _cache_lock:
        results, timestamp = _cache["results"], _cache["timestamp"]

    shown = results[:top]
    quotes = get_quotes([symbol for symbol, _ in shown])
    rows = [(symbol, signal, recommend(signal.score)[0]) for symbol, signal in shown]

    return render_template(
        "index.html",
        rows=rows,
        quotes=quotes,
        total_scored=len(results),
        timestamp=timestamp,
        top=top,
        scan_interval_minutes=SCAN_INTERVAL_SECONDS // 60,
    )


@app.route("/api/scan-meta")
def api_scan_meta():
    with _cache_lock:
        timestamp = _cache["timestamp"]
    return jsonify({"timestamp": timestamp.isoformat() if timestamp else None})


@app.route("/api/quotes")
def api_quotes():
    symbols = [s for s in request.args.get("symbols", "").upper().split(",") if s]
    quotes = get_quotes(symbols)
    return jsonify(
        {
            symbol: {
                "price": q.price,
                "change": q.change,
                "change_pct": q.change_pct,
            }
            for symbol, q in quotes.items()
        }
    )


def _series_to_list(series, ndigits=2):
    return [None if pd.isna(v) else round(float(v), ndigits) for v in series]


@app.route("/stock/<symbol>")
def stock_detail(symbol):
    symbol = symbol.upper()
    daily = fetch_daily_history(symbol)
    if daily.empty or "Close" not in daily:
        abort(404, f"No price history found for {symbol}")

    signal = score_ticker(daily)
    label, blurb = recommend(signal.score) if signal else ("N/A", "Not enough price history to score this stock yet.")
    quote = get_quote(to_yahoo_symbol(symbol))

    close = daily["Close"]
    chart = {
        "dates": [d.strftime("%Y-%m-%d") for d in daily.index],
        "close": _series_to_list(close),
        "sma20": _series_to_list(sma(close, 20)),
        "sma50": _series_to_list(sma(close, 50)),
    }

    return render_template(
        "stock.html",
        symbol=symbol,
        quote=quote,
        signal=signal,
        label=label,
        blurb=blurb,
        chart=chart,
    )


@app.route("/api/intraday/<symbol>")
def api_intraday(symbol):
    df = fetch_intraday_history(symbol.upper())
    if df.empty or "Close" not in df:
        return jsonify({"times": [], "prices": []})
    return jsonify(
        {
            "times": [t.strftime("%H:%M") for t in df.index],
            "prices": _series_to_list(df["Close"]),
        }
    )


DEBUG = True

if __name__ == "__main__":
    # With the Werkzeug reloader active, this script runs once in a parent
    # watcher process and again in the actual serving subprocess (which has
    # WERKZEUG_RUN_MAIN=true set). Only the latter should do the real init,
    # or the initial scan runs twice and doubles the load on Yahoo Finance.
    is_reloader_subprocess = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if not DEBUG or is_reloader_subprocess:
        print("Running initial S&P 500 scan (~15-30s)...")
        _scan_and_cache()
        threading.Thread(target=_background_scanner, daemon=True).start()
    app.run(debug=DEBUG, port=5000)
