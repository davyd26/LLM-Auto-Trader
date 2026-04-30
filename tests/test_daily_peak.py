"""Tests for daily peak equity anchoring (Phase 3 §3B.12)."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.risk.daily_peak import DailyPeakTracker


def _ts(date_str: str, hour: int = 12) -> float:
    dt = datetime.fromisoformat(date_str + f"T{hour:02d}:00:00+00:00")
    return dt.timestamp()


class TestDailyPeakTracker:
    def test_session_peak_tracks_highest(self) -> None:
        t = DailyPeakTracker()
        t.update(equity=1000.0, now=_ts("2026-04-23"))
        t.update(equity=1200.0, now=_ts("2026-04-23", hour=13))
        t.update(equity=1100.0, now=_ts("2026-04-23", hour=14))
        assert t.session_peak == 1200.0

    def test_daily_peak_resets_on_new_day(self) -> None:
        t = DailyPeakTracker()
        t.update(equity=1200.0, now=_ts("2026-04-23"))
        t.update(equity=1100.0, now=_ts("2026-04-24"))  # new day
        assert t.daily_peak == 1100.0
        assert t.session_peak == 1200.0  # session unchanged

    def test_session_drawdown_vs_session_peak(self) -> None:
        t = DailyPeakTracker()
        t.update(equity=1200.0, now=_ts("2026-04-23"))
        t.update(equity=1050.0, now=_ts("2026-04-23", hour=15))
        # Drawdown: (1200 - 1050) / 1200 = 12.5%
        assert t.session_drawdown(1050.0) == pytest.approx(0.125)

    def test_daily_drawdown_independent_from_session(self) -> None:
        t = DailyPeakTracker()
        t.update(equity=1200.0, now=_ts("2026-04-23"))
        # New day starts at 1050 (previous day peak was 1200)
        t.update(equity=1050.0, now=_ts("2026-04-24"))
        # Loss during day — daily peak is today's 1050
        t.update(equity=1020.0, now=_ts("2026-04-24", hour=13))
        # Daily DD: (1050 - 1020) / 1050 ≈ 2.86%
        assert t.daily_drawdown(1020.0) == pytest.approx(30 / 1050, rel=1e-6)
        # Session DD: (1200 - 1020) / 1200 = 15%
        assert t.session_drawdown(1020.0) == pytest.approx(0.15)

    def test_empty_state_returns_zero_drawdown(self) -> None:
        t = DailyPeakTracker()
        assert t.session_drawdown(1000.0) == 0.0
        assert t.daily_drawdown(1000.0) == 0.0
