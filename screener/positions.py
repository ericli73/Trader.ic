"""Local storage for the positions you actually hold (cost basis).

Lets the screener give sell/hold guidance relative to what you paid,
not just an abstract score. Single-user, local file storage - this is a
personal tool, not a multi-user app, so a JSON file is enough.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

POSITIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "positions.json"


@dataclass
class Position:
    symbol: str
    buy_price: float
    shares: float | None = None
    buy_date: str | None = None


def _load() -> dict:
    if not POSITIONS_PATH.exists():
        return {}
    try:
        return json.loads(POSITIONS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_PATH.write_text(json.dumps(data, indent=2))


def get_position(symbol: str) -> Position | None:
    raw = _load().get(symbol.upper())
    return Position(**raw) if raw else None


def save_position(symbol: str, buy_price: float, shares: float | None = None, buy_date: str | None = None) -> Position:
    data = _load()
    position = Position(symbol=symbol.upper(), buy_price=buy_price, shares=shares, buy_date=buy_date)
    data[symbol.upper()] = asdict(position)
    _save(data)
    return position


def delete_position(symbol: str) -> None:
    data = _load()
    data.pop(symbol.upper(), None)
    _save(data)


def get_all_positions() -> list[Position]:
    return [Position(**raw) for raw in _load().values()]
