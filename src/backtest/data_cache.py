"""SQLite-backed persistent OHLCV cache.

The cache stores historical bars keyed by (symbol, timeframe, ts_ms) so that
backtest runs and re-runs do not re-fetch from the exchange. SQLite is the
storage layer because it is already a project dependency, handles concurrent
access via locks, and lets us serve range queries cheaply.

Bars are validated before insertion. A bar with NaN, OHLC-ordering violations,
non-positive prices, negative volume, or a negative timestamp is rejected with
``ValidationError`` — silent ingestion of corrupt bars is the most expensive
class of bug for a backtest harness.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class ValidationError(ValueError):
    """Raised when an OHLCV bar fails sanity checks."""


@dataclass(frozen=True)
class OHLCVBar:
    """A single OHLCV bar at a millisecond timestamp."""

    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def validate(self) -> None:
        for name, val in (
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
        ):
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"{name} is NaN or inf")
        if self.ts_ms < 0:
            raise ValidationError("timestamp must be non-negative")
        if self.volume < 0:
            raise ValidationError("volume must be non-negative")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValidationError("price must be strictly positive")
        if self.high < self.low:
            raise ValidationError("high must be >= low")
        if self.open < self.low or self.open > self.high:
            raise ValidationError("open must be within [low, high]")
        if self.close < self.low or self.close > self.high:
            raise ValidationError("close must be within [low, high]")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol     TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,
    ts_ms      INTEGER NOT NULL,
    open       REAL    NOT NULL,
    high       REAL    NOT NULL,
    low        REAL    NOT NULL,
    close      REAL    NOT NULL,
    volume     REAL    NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts_ms)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup ON ohlcv(symbol, timeframe, ts_ms);
"""


class OHLCVCache:
    """Persistent OHLCV cache backed by SQLite."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def store(self, symbol: str, timeframe: str, bars: list[OHLCVBar]) -> None:
        if not bars:
            return
        for b in bars:
            b.validate()
        rows = [
            (symbol, timeframe, b.ts_ms, b.open, b.high, b.low, b.close, b.volume)
            for b in bars
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO ohlcv "
            "(symbol, timeframe, ts_ms, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    def load_range(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int
    ) -> pd.DataFrame:
        if start_ms > end_ms:
            return _empty_df()
        cur = self._conn.execute(
            "SELECT ts_ms, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = ? AND timeframe = ? AND ts_ms >= ? AND ts_ms <= ? "
            "ORDER BY ts_ms ASC",
            (symbol, timeframe, start_ms, end_ms),
        )
        rows = cur.fetchall()
        if not rows:
            return _empty_df()
        df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
        df = df.astype({
            "ts_ms": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        })
        return df

    def range(self, symbol: str, timeframe: str) -> tuple[int, int] | None:
        cur = self._conn.execute(
            "SELECT MIN(ts_ms), MAX(ts_ms) FROM ohlcv "
            "WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0]), int(row[1])

    def detect_gaps(
        self, symbol: str, timeframe: str, expected_interval_ms: int
    ) -> list[tuple[int, int]]:
        cur = self._conn.execute(
            "SELECT ts_ms FROM ohlcv "
            "WHERE symbol = ? AND timeframe = ? "
            "ORDER BY ts_ms ASC",
            (symbol, timeframe),
        )
        timestamps = [r[0] for r in cur.fetchall()]
        gaps: list[tuple[int, int]] = []
        for prev, curr in zip(timestamps, timestamps[1:]):
            if curr - prev > expected_interval_ms:
                gaps.append((int(prev), int(curr)))
        return gaps


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": pd.Series([], dtype="int64"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="float64"),
        }
    )
