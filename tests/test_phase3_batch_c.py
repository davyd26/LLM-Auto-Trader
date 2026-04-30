"""Tests for Phase 3 Batch C: USDT monitor + event calendar."""

from __future__ import annotations

from src.data.events import Event, EventCalendar
from src.data.stablecoin import USDTMonitor


class TestUSDTMonitor:
    def test_on_peg_no_stress(self) -> None:
        m = USDTMonitor(threshold=0.005, sustain_count=3)
        status = m.record({"coinbase": 1.0001, "kraken": 0.9999})
        assert status.off_peg is False
        assert status.sustained is False

    def test_single_off_peg_not_sustained(self) -> None:
        m = USDTMonitor(threshold=0.005, sustain_count=3)
        m.record({"coinbase": 1.0, "kraken": 1.0})
        status = m.record({"coinbase": 0.98, "kraken": 0.98})
        assert status.off_peg is True
        assert status.sustained is False

    def test_three_consecutive_off_peg_is_sustained(self) -> None:
        m = USDTMonitor(threshold=0.005, sustain_count=3)
        for _ in range(3):
            status = m.record({"coinbase": 0.98, "kraken": 0.98})
        assert status.sustained is True

    def test_median_across_sources(self) -> None:
        m = USDTMonitor(threshold=0.005, sustain_count=3)
        status = m.record({"a": 1.0, "b": 0.5, "c": 1.0})  # median = 1.0
        assert status.price == 1.0
        assert status.off_peg is False

    def test_no_sources_returns_default(self) -> None:
        m = USDTMonitor()
        status = m.record({})
        assert status.off_peg is False


class TestEventCalendar:
    def test_empty_calendar_returns_default_state(self) -> None:
        cal = EventCalendar()
        st = cal.state(now=1000)
        assert st.next_event is None
        assert st.pre_halt_active is False

    def test_high_impact_within_30min_halts(self) -> None:
        cal = EventCalendar(pre_halt_minutes=30)
        cal.set_events([Event(name="FOMC", ts=1500, impact="HIGH", category="macro")])
        st = cal.state(now=1500 - 10 * 60)  # 10 min before
        assert st.pre_halt_active is True

    def test_medium_impact_within_30min_does_not_halt(self) -> None:
        cal = EventCalendar(pre_halt_minutes=30)
        cal.set_events([Event(name="X", ts=1500, impact="MEDIUM", category="macro")])
        st = cal.state(now=1500 - 10 * 60)
        assert st.pre_halt_active is False
        assert st.warning_active is True  # within 4h window

    def test_symbol_filter(self) -> None:
        cal = EventCalendar()
        cal.set_events([Event(
            name="ABC unlock", ts=1500, impact="HIGH",
            category="token_unlock", symbols=["ABC/USDT:USDT"],
        )])
        st_btc = cal.state(now=1000, symbol="BTC/USDT:USDT")
        assert st_btc.pre_halt_active is False
        st_abc = cal.state(now=1000, symbol="ABC/USDT:USDT")
        assert st_abc.next_event is not None

    def test_purge_expired(self) -> None:
        cal = EventCalendar()
        cal.set_events([
            Event(name="old", ts=100, impact="HIGH", category="macro"),
            Event(name="new", ts=2000, impact="HIGH", category="macro"),
        ])
        removed = cal.purge_expired(now=1000)
        assert removed == 1
        assert len(cal.upcoming(now=1000)) == 1
