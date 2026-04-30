"""Clock abstraction for the backtest harness.

Production code calls ``clock.now()`` / ``clock.unix()`` instead of
``datetime.now()`` / ``time.time()`` so the backtest can advance simulated
time deterministically. ``WallClock`` is used in production; ``MockClock`` is
injected by the backtest harness.

All clocks return UTC-aware datetimes. Naive datetimes are rejected at
construction to prevent the entire timezone-bug class.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Anything that can report current time as both datetime and unix seconds."""

    def now(self) -> datetime:
        """Current UTC-aware datetime."""
        ...

    def unix(self) -> float:
        """Current unix timestamp in seconds (with fractional precision)."""
        ...


class WallClock:
    """Real-time clock backed by ``time.time()``. Always UTC-aware."""

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def unix(self) -> float:
        return time.time()

    def unix_ms(self) -> int:
        return int(time.time() * 1000)


class MockClock:
    """Deterministic clock for backtests. Thread-safe.

    Use ``advance()`` for monotonic forward steps and ``set()`` for explicit
    time-travel (e.g., resetting between CPCV paths).
    """

    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("MockClock requires a timezone-aware datetime")
        self._now = start.astimezone(timezone.utc)
        self._lock = threading.Lock()

    @classmethod
    def from_unix(cls, ts: float) -> MockClock:
        return cls(datetime.fromtimestamp(ts, tz=timezone.utc))

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def unix(self) -> float:
        with self._lock:
            return self._now.timestamp()

    def unix_ms(self) -> int:
        with self._lock:
            return int(self._now.timestamp() * 1000)

    def advance(self, amount: timedelta | float | int) -> None:
        delta = amount if isinstance(amount, timedelta) else timedelta(seconds=float(amount))
        if delta < timedelta(0):
            raise ValueError("advance() requires a non-negative duration")
        with self._lock:
            self._now = self._now + delta

    def set(self, when: datetime) -> None:
        if when.tzinfo is None:
            raise ValueError("set() requires a timezone-aware datetime")
        with self._lock:
            self._now = when.astimezone(timezone.utc)
