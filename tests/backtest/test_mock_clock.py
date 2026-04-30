"""Tests for the Clock abstraction used by the backtest harness.

The Clock decouples wall time from simulated time. Production code calls
``clock.now()`` / ``clock.unix()`` instead of ``datetime.now()`` /
``time.time()``; the backtest harness injects a ``MockClock`` that advances
deterministically.

Edge cases covered:
    - timezone-naive inputs are rejected (we standardize on UTC)
    - monotonicity (advancing time can never go backwards)
    - microsecond precision preserved
    - ``WallClock`` returns UTC
    - ``MockClock.advance`` accepts both ``timedelta`` and seconds
    - negative advances are rejected
    - DST boundaries are irrelevant because we work in UTC

Tail risks:
    - thread-safety of MockClock (two threads advancing concurrently)
    - long-duration overflow (clock at year 9999)
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from src.backtest.mock_clock import Clock, MockClock, WallClock


class TestWallClock:
    def test_now_returns_utc_aware_datetime(self) -> None:
        c = WallClock()
        t = c.now()
        assert t.tzinfo is not None
        assert t.tzinfo.utcoffset(t) == timedelta(0)

    def test_unix_returns_float(self) -> None:
        c = WallClock()
        u = c.unix()
        assert isinstance(u, float)
        assert u > 0

    def test_unix_matches_now_within_one_second(self) -> None:
        c = WallClock()
        n = c.now()
        u = c.unix()
        # the two calls happen back-to-back; allow 1s slop
        assert abs(n.timestamp() - u) < 1.0


class TestMockClockConstruction:
    def test_accepts_timezone_aware_datetime(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        c = MockClock(start)
        assert c.now() == start

    def test_rejects_timezone_naive_datetime(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            MockClock(datetime(2024, 1, 1))

    def test_accepts_unix_timestamp(self) -> None:
        c = MockClock.from_unix(1_704_067_200.0)  # 2024-01-01 00:00:00 UTC
        assert c.now().year == 2024
        assert c.now().month == 1
        assert c.now().day == 1

    def test_preserves_microseconds(self) -> None:
        start = datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        c = MockClock(start)
        assert c.now().microsecond == 123456


class TestMockClockAdvance:
    def test_advance_with_timedelta(self) -> None:
        c = MockClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        c.advance(timedelta(hours=1))
        assert c.now() == datetime(2024, 1, 1, 1, tzinfo=timezone.utc)

    def test_advance_with_seconds(self) -> None:
        c = MockClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        c.advance(3600)
        assert c.now() == datetime(2024, 1, 1, 1, tzinfo=timezone.utc)

    def test_advance_zero_is_valid(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        c = MockClock(start)
        c.advance(0)
        assert c.now() == start

    def test_negative_advance_rejected(self) -> None:
        c = MockClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        with pytest.raises(ValueError, match="non-negative"):
            c.advance(-1)
        with pytest.raises(ValueError, match="non-negative"):
            c.advance(timedelta(seconds=-1))

    def test_advance_preserves_utc_timezone(self) -> None:
        c = MockClock(datetime(2024, 6, 1, tzinfo=timezone.utc))
        c.advance(timedelta(days=180))  # crosses DST boundary in many zones
        assert c.now().tzinfo is not None
        assert c.now().tzinfo.utcoffset(c.now()) == timedelta(0)

    def test_set_jumps_to_arbitrary_time(self) -> None:
        c = MockClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        future = datetime(2030, 6, 15, tzinfo=timezone.utc)
        c.set(future)
        assert c.now() == future

    def test_set_can_go_backward(self) -> None:
        # set() is for explicit time travel (e.g., resetting between CPCV paths)
        # advance() is monotonic; set() is not
        c = MockClock(datetime(2024, 6, 1, tzinfo=timezone.utc))
        past = datetime(2024, 1, 1, tzinfo=timezone.utc)
        c.set(past)
        assert c.now() == past

    def test_set_rejects_naive_datetime(self) -> None:
        c = MockClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        with pytest.raises(ValueError, match="timezone-aware"):
            c.set(datetime(2024, 6, 1))


class TestMockClockUnix:
    def test_unix_matches_now_timestamp(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        c = MockClock(start)
        assert c.unix() == start.timestamp()

    def test_unix_advances_with_advance(self) -> None:
        c = MockClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        before = c.unix()
        c.advance(timedelta(hours=1))
        after = c.unix()
        assert after - before == 3600.0

    def test_unix_ms_returns_int_milliseconds(self) -> None:
        c = MockClock(datetime(2024, 1, 1, 0, 0, 0, 500_000, tzinfo=timezone.utc))
        ms = c.unix_ms()
        assert isinstance(ms, int)
        # 2024-01-01 00:00:00.500 UTC == 1704067200500 ms
        assert ms == 1_704_067_200_500


class TestMockClockProtocol:
    def test_satisfies_clock_protocol(self) -> None:
        # MockClock must be substitutable for Clock anywhere in the codebase
        c: Clock = MockClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert hasattr(c, "now")
        assert hasattr(c, "unix")

    def test_wallclock_satisfies_clock_protocol(self) -> None:
        c: Clock = WallClock()
        assert hasattr(c, "now")
        assert hasattr(c, "unix")


class TestMockClockTailRisks:
    def test_far_future_does_not_overflow(self) -> None:
        # year 9999 is the upper bound of datetime
        far_future = datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        c = MockClock(far_future)
        assert c.now() == far_future
        # unix() may return a very large number but should not raise
        u = c.unix()
        assert u > 0

    def test_advance_past_year_max_raises(self) -> None:
        c = MockClock(datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc))
        with pytest.raises((OverflowError, ValueError)):
            c.advance(timedelta(days=365))

    def test_thread_safe_concurrent_advances(self) -> None:
        # Multiple threads advancing simultaneously must not lose updates
        c = MockClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        n_threads = 10
        advances_per_thread = 100

        def worker() -> None:
            for _ in range(advances_per_thread):
                c.advance(1)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_seconds = n_threads * advances_per_thread
        actual_delta = c.now() - datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert actual_delta == timedelta(seconds=expected_seconds)
