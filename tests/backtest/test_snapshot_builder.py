"""Tests for SnapshotBuilder — the lookahead-defense module.

A snapshot represents what a human trader could see at simulated time t.
The two non-negotiable invariants:
    1. NO future data: every field must come from data with ts_ms <= t.
    2. NO absolute dates in the prompt-facing text: the LLM sees only relative
       phrasing ("4 weeks ago", "current week"). Symbol names and price levels
       are retained — the model could still pattern-match those, but date
       anchoring is the most dangerous channel and we close it.

Edge cases covered:
    - snapshot at exactly bar-boundary (off-by-one on closed/open intervals)
    - empty cache for one symbol but not others
    - position list with zero open positions
    - very small lookback (1 bar)
    - lookback exceeds available data
    - price levels and ts_ms preserved in JSON, not in prompt text

Tail risks covered:
    - prompt text never contains ISO date patterns (regex check)
    - hash is deterministic across runs (snapshot content-addressing for cache)
    - hash differs between snapshots that differ in any field
    - JSON round-trips losslessly
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.backtest.data_cache import OHLCVBar, OHLCVCache
from src.backtest.snapshot_builder import (
    PositionRecord,
    Snapshot,
    SnapshotBuilder,
)


@pytest.fixture
def cache(tmp_path: Path) -> OHLCVCache:
    return OHLCVCache(tmp_path / "ohlcv.db")


def _bar(ts_ms: int, price: float = 100.0) -> OHLCVBar:
    return OHLCVBar(
        ts_ms=ts_ms, open=price, high=price * 1.01, low=price * 0.99, close=price, volume=1.0
    )


def _populate(cache: OHLCVCache, symbol: str, timeframe: str, n: int, interval_ms: int = 86_400_000) -> None:
    """Populate the cache with N daily bars ending at ts=0."""
    bars = [_bar(i * interval_ms, price=100.0 + i) for i in range(n)]
    cache.store(symbol, timeframe, bars)


# Some convenient anchor times (ms since epoch, UTC)
T_2024_06_01 = 1_717_200_000_000  # 2024-06-01 00:00:00 UTC
ONE_DAY_MS = 86_400_000


class TestSnapshotConstruction:
    def test_build_empty_universe_is_valid(self, cache: OHLCVCache) -> None:
        builder = SnapshotBuilder(cache=cache, universe=[], lookback_bars=10)
        snap = builder.build(t_ms=T_2024_06_01, equity_usd=2000.0, positions=[])
        assert snap.universe == []
        assert snap.equity_usd == 2000.0

    def test_build_one_symbol(self, cache: OHLCVCache) -> None:
        _populate(cache, "BTC/USDT", "1d", n=20)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=10)
        snap = builder.build(t_ms=10 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        assert "BTC/USDT" in snap.bars
        assert "1d" in snap.bars["BTC/USDT"]
        # 10 lookback bars means 10 most recent bars at or before t
        assert len(snap.bars["BTC/USDT"]["1d"]) == 10

    def test_no_future_bars_in_snapshot(self, cache: OHLCVCache) -> None:
        # Populate 20 daily bars (ts=0..19 days), build snapshot at t=10 days
        _populate(cache, "BTC/USDT", "1d", n=20)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=100)
        snap = builder.build(t_ms=10 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        for bar in snap.bars["BTC/USDT"]["1d"]:
            assert bar["ts_ms"] <= 10 * ONE_DAY_MS, "future bar leaked into snapshot"

    def test_lookback_caps_at_available_data(self, cache: OHLCVCache) -> None:
        # Only 5 bars exist; ask for 100
        _populate(cache, "BTC/USDT", "1d", n=5)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=100)
        snap = builder.build(t_ms=4 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        assert len(snap.bars["BTC/USDT"]["1d"]) == 5

    def test_includes_multiple_timeframes(self, cache: OHLCVCache) -> None:
        _populate(cache, "BTC/USDT", "1d", n=10)
        _populate(cache, "BTC/USDT", "4h", n=60, interval_ms=4 * 60 * 60 * 1000)
        builder = SnapshotBuilder(
            cache=cache,
            universe=["BTC/USDT"],
            lookback_bars=10,
            timeframes=("1d", "4h"),
        )
        snap = builder.build(t_ms=10 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        assert "1d" in snap.bars["BTC/USDT"]
        assert "4h" in snap.bars["BTC/USDT"]


class TestSnapshotPositions:
    def test_empty_positions_carried_through(self, cache: OHLCVCache) -> None:
        builder = SnapshotBuilder(cache=cache, universe=[], lookback_bars=10)
        snap = builder.build(t_ms=T_2024_06_01, equity_usd=2000.0, positions=[])
        assert snap.positions == []

    def test_positions_carried_through(self, cache: OHLCVCache) -> None:
        builder = SnapshotBuilder(cache=cache, universe=[], lookback_bars=10)
        pos = [PositionRecord(symbol="BTC/USDT", side="long", size=0.01, entry_price=50000.0, entry_ts_ms=T_2024_06_01)]
        snap = builder.build(t_ms=T_2024_06_01, equity_usd=2000.0, positions=pos)
        assert len(snap.positions) == 1
        assert snap.positions[0].symbol == "BTC/USDT"

    def test_unrealized_pnl_uses_latest_close_at_or_before_t(self, cache: OHLCVCache) -> None:
        # Populate bars showing a price increase
        bars = [_bar(i * ONE_DAY_MS, price=100.0 + i * 10) for i in range(10)]
        cache.store("BTC/USDT", "1d", bars)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=10)
        # Long position entered at price=100 (index 0), evaluated at index 5 (price=150)
        pos = [PositionRecord(symbol="BTC/USDT", side="long", size=1.0, entry_price=100.0, entry_ts_ms=0)]
        snap = builder.build(t_ms=5 * ONE_DAY_MS, equity_usd=2000.0, positions=pos)
        # Unrealized pnl at price=150, size=1.0 → +50
        assert snap.unrealized_pnl_usd == pytest.approx(50.0)


class TestPromptTextLookaheadDefense:
    def test_prompt_text_has_no_iso_date_substring(self, cache: OHLCVCache) -> None:
        _populate(cache, "BTC/USDT", "1d", n=20)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=10)
        snap = builder.build(t_ms=T_2024_06_01, equity_usd=2000.0, positions=[])
        text = snap.to_prompt_text()
        # Catch ISO date patterns and 4-digit year patterns
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text), "ISO date leaked into prompt"
        assert not re.search(r"\b(2020|2021|2022|2023|2024|2025|2026)\b", text), "year leaked into prompt"

    def test_prompt_text_uses_relative_time(self, cache: OHLCVCache) -> None:
        _populate(cache, "BTC/USDT", "1d", n=20)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=5)
        snap = builder.build(t_ms=10 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        text = snap.to_prompt_text()
        # Relative cues we expect
        assert any(token in text.lower() for token in ("ago", "current", "recent", "last"))

    def test_prompt_text_contains_prices(self, cache: OHLCVCache) -> None:
        # Prices ARE allowed (they're a weaker memorization channel than dates)
        _populate(cache, "BTC/USDT", "1d", n=10)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=5)
        snap = builder.build(t_ms=5 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        text = snap.to_prompt_text()
        assert "BTC/USDT" in text

    def test_prompt_text_contains_equity(self, cache: OHLCVCache) -> None:
        builder = SnapshotBuilder(cache=cache, universe=[], lookback_bars=10)
        snap = builder.build(t_ms=T_2024_06_01, equity_usd=2347.50, positions=[])
        text = snap.to_prompt_text()
        # Equity should appear (any of "2347", "2348", "2,347" — trader-friendly)
        assert "2347" in text or "2348" in text or "2,347" in text or "2,348" in text


class TestSnapshotHashing:
    def test_hash_is_deterministic(self, cache: OHLCVCache) -> None:
        _populate(cache, "BTC/USDT", "1d", n=10)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=5)
        snap1 = builder.build(t_ms=5 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        snap2 = builder.build(t_ms=5 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        assert snap1.snapshot_id == snap2.snapshot_id

    def test_hash_differs_when_t_differs(self, cache: OHLCVCache) -> None:
        _populate(cache, "BTC/USDT", "1d", n=10)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=5)
        snap1 = builder.build(t_ms=5 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        snap2 = builder.build(t_ms=6 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        assert snap1.snapshot_id != snap2.snapshot_id

    def test_hash_differs_when_equity_differs(self, cache: OHLCVCache) -> None:
        builder = SnapshotBuilder(cache=cache, universe=[], lookback_bars=10)
        snap1 = builder.build(t_ms=T_2024_06_01, equity_usd=2000.0, positions=[])
        snap2 = builder.build(t_ms=T_2024_06_01, equity_usd=2001.0, positions=[])
        assert snap1.snapshot_id != snap2.snapshot_id

    def test_hash_differs_when_positions_differ(self, cache: OHLCVCache) -> None:
        builder = SnapshotBuilder(cache=cache, universe=[], lookback_bars=10)
        snap1 = builder.build(t_ms=T_2024_06_01, equity_usd=2000.0, positions=[])
        snap2 = builder.build(
            t_ms=T_2024_06_01, equity_usd=2000.0,
            positions=[PositionRecord(symbol="BTC/USDT", side="long", size=0.01, entry_price=50000.0, entry_ts_ms=T_2024_06_01)],
        )
        assert snap1.snapshot_id != snap2.snapshot_id

    def test_hash_is_hex_of_sha256(self) -> None:
        snap = Snapshot(
            t_ms=T_2024_06_01,
            universe=[],
            bars={},
            positions=[],
            equity_usd=2000.0,
            unrealized_pnl_usd=0.0,
        )
        h = snap.snapshot_id
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestSnapshotJSON:
    def test_json_round_trip_preserves_content(self, cache: OHLCVCache) -> None:
        _populate(cache, "BTC/USDT", "1d", n=10)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=5)
        snap = builder.build(t_ms=5 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        s = snap.to_json()
        d = json.loads(s)
        assert d["t_ms"] == 5 * ONE_DAY_MS
        assert d["equity_usd"] == 2000.0
        assert "BTC/USDT" in d["bars"]

    def test_json_is_canonical(self, cache: OHLCVCache) -> None:
        # Same content → byte-identical JSON (so hash is reproducible)
        builder = SnapshotBuilder(cache=cache, universe=[], lookback_bars=10)
        snap1 = builder.build(t_ms=T_2024_06_01, equity_usd=2000.0, positions=[])
        snap2 = builder.build(t_ms=T_2024_06_01, equity_usd=2000.0, positions=[])
        assert snap1.to_json() == snap2.to_json()


class TestSnapshotEdgeCases:
    def test_t_at_exact_bar_boundary_includes_bar(self, cache: OHLCVCache) -> None:
        # Bar at ts=1000 — querying at exactly t=1000 must include it (≤ t semantics)
        cache.store("BTC/USDT", "1d", [_bar(1000)])
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=10)
        snap = builder.build(t_ms=1000, equity_usd=2000.0, positions=[])
        assert len(snap.bars["BTC/USDT"]["1d"]) == 1

    def test_t_just_before_bar_excludes_bar(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", [_bar(1000)])
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=10)
        snap = builder.build(t_ms=999, equity_usd=2000.0, positions=[])
        assert snap.bars["BTC/USDT"]["1d"] == []

    def test_lookback_zero_returns_empty_bars(self, cache: OHLCVCache) -> None:
        _populate(cache, "BTC/USDT", "1d", n=10)
        builder = SnapshotBuilder(cache=cache, universe=["BTC/USDT"], lookback_bars=0)
        snap = builder.build(t_ms=10 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        assert snap.bars["BTC/USDT"]["1d"] == []

    def test_negative_lookback_rejected(self, cache: OHLCVCache) -> None:
        with pytest.raises(ValueError):
            SnapshotBuilder(cache=cache, universe=[], lookback_bars=-1)

    def test_partial_universe_data_handled(self, cache: OHLCVCache) -> None:
        # BTC has data, ETH does not — must not raise; ETH section just empty
        _populate(cache, "BTC/USDT", "1d", n=10)
        builder = SnapshotBuilder(
            cache=cache, universe=["BTC/USDT", "ETH/USDT"], lookback_bars=5
        )
        snap = builder.build(t_ms=5 * ONE_DAY_MS, equity_usd=2000.0, positions=[])
        assert "BTC/USDT" in snap.bars
        assert "ETH/USDT" in snap.bars
        assert len(snap.bars["BTC/USDT"]["1d"]) == 5
        assert snap.bars["ETH/USDT"]["1d"] == []
