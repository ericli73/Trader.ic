import argparse

import pandas as pd

from .pipeline import run_screen


def parse_args():
    parser = argparse.ArgumentParser(description="Technical swing-trade screener over the S&P 500.")
    parser.add_argument("--top", type=int, default=15, help="How many top candidates to print.")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N tickers (for quick testing).")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch the S&P 500 ticker list instead of using the cache.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Downloading 2y daily history and scoring S&P 500 stocks...")
    results = run_screen(limit=args.limit, refresh=args.refresh)

    print(f"\nTop {args.top} candidates as of {pd.Timestamp.now():%Y-%m-%d %H:%M} "
          f"({len(results)} scored):\n")
    for rank, (symbol, ts) in enumerate(results[: args.top], start=1):
        print(f"{rank:>2}. {symbol:<6} {ts.tier:<12} score={ts.score:5.1f}  confidence={ts.confidence:3.0f}%  "
              + "; ".join(ts.reasons[:3]))

    print("\nTechnical screener output only - not financial advice. Verify independently before trading.")


if __name__ == "__main__":
    main()
