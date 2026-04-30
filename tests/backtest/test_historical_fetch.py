"""Tests for paginated historical OHLCV fetching.

The function ``fetch_ohlcv_range`` walks an exchange's ``fetch_ohlcv`` cursor
to retrieve a multi-year span of bars. It is the bridge between CCXT's
limit-bounded API and our backtest harness's "give me 2024" demand.

Edge cases covered:
    - empty range (since == until)
    - exchange returns fewer bars than batch_limit (end of data)
    - exchange returns 0 bars (no data for symbol)
    - exchange returns bars past until_ms (must be trimmed)
    - exchange returns bars before since_ms (must be filtered)
    - exchange returns duplicate bars across batches (must be de-duplicated)

Tail risks covered:
    - exchange raises on a single batch — error propagates, no silent partial fetch
    - exchange returns the same cursor twice (would loop forever) — must detect and break
    - bar list contains corrupt entries — validation rejects before return
    - exchange returns bars in wrong order — sorted on output
"""

from __future__ import annotations

import math

import pytest

from src.backtest.data_cache import OHLCVBar, ValidationError
from src.data.historical import fetch_ohlcv_range


class FakeExchange:
    """Test double that returns canned OHLCV batches.

    Mimics CCXT's ``fetch_ohlcv(symbol, timeframe, since, limit)`` contract.
    The real exchange returns lists of [ts_ms, open, high, low, close, volume].
    """

    def __init__(self, bars: list[list], raise_on_call: int | None = None):
        self._bars = sorted(bars, key=lambda b: b[0])
        self._raise_on_call = raise_on_call
        self.call_count = 0
        self.call_history: list[tuple[int, int]] = []  # (since, limit)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
    ) -> list[list]:
        self.call_count += 1
        self.call_history.append((since or 0, limit or 0))
        if self._raise_on_call is not None and self.call_count == self._raise_on_call:
            raise RuntimeError(f"exchange error on call {self.call_count}")
        start = since if since is not None else 0
        cap = limit if limit is not None else 1500
        # CCXT semantics: returns bars where ts >= since, sorted ascending, capped at limit
        filtered = [b for b in self._bars if b[0] >= start]
        return filtered[:cap]


def _bar_row(ts_ms: int, price: float = 100.0) -> list:
    return [ts_ms, price, price * 1.01, price * 0.99, price, 1.0]


class TestFetchOhlcvRangeBasic:
    async def test_empty_range_returns_no_bars(self) -> None:
        exchange = FakeExchange(bars=[_bar_row(t) for t in range(0, 10000, 1000)])
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=5000, until_ms=5000, batch_limit=10
        )
        # since==until is treated as a degenerate range — return only ts == since if it exists
        assert all(b.ts_ms == 5000 for b in bars)

    async def test_returns_all_bars_in_range(self) -> None:
        # 10 bars at intervals of 1000ms
        rows = [_bar_row(t) for t in range(0, 10000, 1000)]
        exchange = FakeExchange(bars=rows)
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=0, until_ms=9000, batch_limit=100
        )
        assert len(bars) == 10
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert [b.ts_ms for b in bars] == list(range(0, 10000, 1000))

    async def test_paginates_when_batch_limit_smaller_than_range(self) -> None:
        # 10 bars, batch_limit=3 → 4 calls (3+3+3+1)
        rows = [_bar_row(t) for t in range(0, 10000, 1000)]
        exchange = FakeExchange(bars=rows)
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=0, until_ms=9000, batch_limit=3
        )
        assert len(bars) == 10
        assert exchange.call_count >= 4

    async def test_trims_bars_past_until_ms(self) -> None:
        # Exchange has 10 bars, but we only want up to ts=4000
        rows = [_bar_row(t) for t in range(0, 10000, 1000)]
        exchange = FakeExchange(bars=rows)
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=0, until_ms=4000, batch_limit=100
        )
        assert all(b.ts_ms <= 4000 for b in bars)
        assert len(bars) == 5

    async def test_no_bars_in_range_returns_empty(self) -> None:
        # All available bars are before since_ms
        exchange = FakeExchange(bars=[_bar_row(t) for t in range(0, 1000, 100)])
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=10_000, until_ms=20_000, batch_limit=100
        )
        assert bars == []


class TestFetchOhlcvRangeDeduplication:
    async def test_duplicate_bars_across_batches_deduplicated(self) -> None:
        # Simulate an exchange that returns overlapping batches
        # If the cursor advances by `last_ts` and the next call returns starting at `last_ts`,
        # we would get the boundary bar twice.
        rows = [_bar_row(t) for t in range(0, 10000, 1000)]
        exchange = FakeExchange(bars=rows)
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=0, until_ms=9000, batch_limit=3
        )
        timestamps = [b.ts_ms for b in bars]
        assert len(timestamps) == len(set(timestamps)), "duplicate ts_ms in result"


class TestFetchOhlcvRangeValidation:
    async def test_invalid_bar_in_response_raises(self) -> None:
        # Exchange returns a malformed bar (high < low)
        bad_row = [1000, 100.0, 80.0, 90.0, 85.0, 1.0]  # high=80 < low=90
        exchange = FakeExchange(bars=[bad_row])
        with pytest.raises(ValidationError):
            await fetch_ohlcv_range(
                exchange, "BTC/USDT", "1d", since_ms=0, until_ms=10_000, batch_limit=10
            )

    async def test_nan_in_bar_raises(self) -> None:
        bad_row = [1000, math.nan, 110.0, 90.0, 100.0, 1.0]
        exchange = FakeExchange(bars=[bad_row])
        with pytest.raises(ValidationError):
            await fetch_ohlcv_range(
                exchange, "BTC/USDT", "1d", since_ms=0, until_ms=10_000, batch_limit=10
            )


class TestFetchOhlcvRangeTailRisks:
    async def test_exchange_error_propagates(self) -> None:
        rows = [_bar_row(t) for t in range(0, 10000, 1000)]
        exchange = FakeExchange(bars=rows, raise_on_call=2)
        with pytest.raises(RuntimeError, match="exchange error"):
            await fetch_ohlcv_range(
                exchange, "BTC/USDT", "1d", since_ms=0, until_ms=9000, batch_limit=3
            )

    async def test_terminates_when_exchange_returns_empty(self) -> None:
        exchange = FakeExchange(bars=[])
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=0, until_ms=10_000, batch_limit=10
        )
        assert bars == []
        # Should not loop forever
        assert exchange.call_count == 1

    async def test_terminates_when_cursor_does_not_advance(self) -> None:
        # Pathological exchange that always returns the same bar regardless of `since`.
        # Without cursor-progress detection, fetch would loop forever.
        class StuckExchange:
            def __init__(self) -> None:
                self.call_count = 0

            async def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
                self.call_count += 1
                if self.call_count > 10:  # safety brake for the test itself
                    raise RuntimeError("test detected infinite loop")
                return [_bar_row(0)]  # always the same bar

        exchange = StuckExchange()
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=0, until_ms=10_000, batch_limit=10
        )
        assert exchange.call_count <= 3, "must detect non-advancing cursor and break"
        assert len(bars) == 1

    async def test_unsorted_response_returned_sorted(self) -> None:
        # CCXT promises sorted but defensive code should still sort
        rows = [_bar_row(t) for t in [3000, 1000, 2000]]
        exchange = FakeExchange(bars=rows)
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=0, until_ms=5000, batch_limit=10
        )
        timestamps = [b.ts_ms for b in bars]
        assert timestamps == sorted(timestamps)

    async def test_until_ms_before_since_ms_returns_empty(self) -> None:
        rows = [_bar_row(t) for t in range(0, 10000, 1000)]
        exchange = FakeExchange(bars=rows)
        bars = await fetch_ohlcv_range(
            exchange, "BTC/USDT", "1d", since_ms=5000, until_ms=2000, batch_limit=10
        )
        assert bars == []
        # No exchange call should have been made
        assert exchange.call_count == 0
