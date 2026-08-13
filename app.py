"""Local web dashboard for the S&P 500 technical screener.

Run with: python app.py, then open http://127.0.0.1:5000
"""

import os
import threading
import time
from datetime import datetime

import pandas as pd
from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from quotes.yahoo import get_extended_quote, get_extended_quotes, get_quote, get_quotes
from screener.data import get_long_history
from screener.exit import compute_exit_guidance
from screener.indicators import sma
from screener.pipeline import fetch_daily_history, fetch_intraday_history, run_screen, score_symbol
from screener.positions import delete_position, get_all_positions, get_position, save_position
from screener.scoring import format_evidence
from screener.universe import get_sp500_tickers, to_yahoo_symbol

app = Flask(__name__)

SCAN_INTERVAL_SECONDS = 20 * 60  # auto-rescan the full S&P 500 every 20 minutes


@app.context_processor
def inject_ticker_list():
    """S&P 500 symbols for the nav search bar's autocomplete, on every page."""
    try:
        return {"all_tickers": get_sp500_tickers()}
    except Exception:
        return {"all_tickers": []}


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
    symbols = [symbol for symbol, _ in shown]
    quotes = get_quotes(symbols)
    extended_quotes = get_extended_quotes(symbols)

    return render_template(
        "index.html",
        rows=shown,
        quotes=quotes,
        extended_quotes=extended_quotes,
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
    daily = fetch_daily_history(symbol)  # 6mo, for the price/SMA chart only
    if daily.empty or "Close" not in daily:
        abort(404, f"No price history found for {symbol}")

    ts = score_symbol(symbol)  # separate, deeper (5y) history for the actual score
    evidence_display = format_evidence(ts.evidence) if ts else None
    quote = get_quote(to_yahoo_symbol(symbol))
    try:
        extended_quote = get_extended_quote(to_yahoo_symbol(symbol))
    except Exception:
        extended_quote = None

    close = daily["Close"]
    chart = {
        "dates": [d.strftime("%Y-%m-%d") for d in daily.index],
        "close": _series_to_list(close),
        "sma20": _series_to_list(sma(close, 20)),
        "sma50": _series_to_list(sma(close, 50)),
    }

    position = get_position(symbol)
    exit_guidance = None
    if position and ts:
        long_history = get_long_history(symbol, period="5y")
        if not long_history.empty:
            exit_guidance = compute_exit_guidance(position, long_history, quote.price, ts)

    return render_template(
        "stock.html",
        symbol=symbol,
        quote=quote,
        extended_quote=extended_quote,
        ts=ts,
        evidence_display=evidence_display,
        chart=chart,
        position=position,
        exit_guidance=exit_guidance,
    )


@app.route("/stock/<symbol>/position", methods=["POST"])
def save_position_route(symbol):
    symbol = symbol.upper()
    buy_price = request.form.get("buy_price", type=float)
    shares = request.form.get("shares", type=float) or None
    buy_date = request.form.get("buy_date") or None
    if buy_price and buy_price > 0:
        save_position(symbol, buy_price, shares, buy_date)
    return redirect(url_for("stock_detail", symbol=symbol))


@app.route("/stock/<symbol>/position/delete", methods=["POST"])
def delete_position_route(symbol):
    delete_position(symbol.upper())
    return redirect(request.referrer or url_for("stock_detail", symbol=symbol.upper()))


@app.route("/positions")
def positions_page():
    rows = []
    for position in get_all_positions():
        ts = score_symbol(position.symbol)
        quote = get_quote(to_yahoo_symbol(position.symbol))
        guidance = None
        if ts:
            long_history = get_long_history(position.symbol, period="5y")
            if not long_history.empty:
                guidance = compute_exit_guidance(position, long_history, quote.price, ts)
        rows.append((position, quote, guidance))
    return render_template("positions.html", rows=rows)


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
