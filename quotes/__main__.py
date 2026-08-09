import sys

from .yahoo import get_quotes


def main() -> None:
    symbols = sys.argv[1:] or ["AAPL"]
    for symbol, quote in get_quotes(symbols).items():
        change = quote.price - quote.previous_close
        pct = change / quote.previous_close * 100
        print(f"{symbol}: {quote.price:.2f} {quote.currency} ({change:+.2f}, {pct:+.2f}%)")


if __name__ == "__main__":
    main()
