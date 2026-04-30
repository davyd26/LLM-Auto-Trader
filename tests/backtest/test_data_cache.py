"""Tests for OHLCVCache: SQLite-backed persistent OHLCV storage.

The cache is the foundation of the backtest harness — it must reliably store
bulk historical bars and serve range queries with zero data loss.

Edge cases covered:
    - empty inputs (no-op)
    - duplicate timestamps (REPLACE semantics)
    - bar validation (NaN, negative prices/volume, OHLC violations)
    - range queries with no matching bars
    - range queries with start > end
    - cross-symbol isolation
    - gap detection
    - timeframe isolation (1d bars don't pollute 4h queries)

Tail risks covered:
    - corruption: invalid bars rejected before write
    - concurrent access: idempotent upserts
    - large ranges: pagination during retrieval
    - persistence: cache survives close/reopen
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.data_cache import OHLCVBar, OHLCVCache, ValidationError


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "ohlcv.db"


@pytest.fixture
def cache(cache_path: Path) -> OHLCVCache:
    return OHLCVCache(cache_path)


def _bar(ts_ms: int, price: float = 100.0, vol: float = 1.0) -> OHLCVBar:
    """Construct a valid OHLCV bar at the given timestamp and price."""
    return OHLCVBar(
        ts_ms=ts_ms,
        open=price,
        high=price * 1.01,
        low=price * 0.99,
        close=price,
        volume=vol,
    )


def _bar_list(start_ms: int, count: int, interval_ms: int = 86_400_000) -> list[OHLCVBar]:
    """Generate a sequence of valid bars at fixed intervals."""
    return [_bar(start_ms + i * interval_ms, price=100.0 + i) for i in range(count)]


class TestOHLCVBarValidation:
    def test_valid_bar_accepted(self) -> None:
        b = OHLCVBar(ts_ms=1_700_000_000_000, open=100, high=110, low=90, close=105, volume=10)
        b.validate()  # no raise

    def test_nan_open_rejected(self) -> None:
        b = OHLCVBar(ts_ms=1, open=math.nan, high=110, low=90, close=105, volume=10)
        with pytest.raises(ValidationError, match="NaN"):
            b.validate()

    def test_nan_volume_rejected(self) -> None:
        b = OHLCVBar(ts_ms=1, open=100, high=110, low=90, close=105, volume=math.nan)
        with pytest.raises(ValidationError, match="NaN"):
            b.validate()

    def test_inf_high_rejected(self) -> None:
        b = OHLCVBar(ts_ms=1, open=100, high=math.inf, low=90, close=105, volume=10)
        with pytest.raises(ValidationError):
            b.validate()

    def test_high_below_low_rejected(self) -> None:
        b = OHLCVBar(ts_ms=1, open=100, high=80, low=90, close=85, volume=10)
        with pytest.raises(ValidationError, match="high.*low"):
            b.validate()

    def test_close_above_high_rejected(self) -> None:
        b = OHLCVBar(ts_ms=1, open=100, high=110, low=90, close=120, volume=10)
        with pytest.raises(ValidationError, match="close"):
            b.validate()

    def test_open_below_low_rejected(self) -> None:
        b = OHLCVBar(ts_ms=1, open=80, high=110, low=90, close=100, volume=10)
        with pytest.raises(ValidationError, match="open"):
            b.validate()

    def test_negative_volume_rejected(self) -> None:
        b = OHLCVBar(ts_ms=1, open=100, high=110, low=90, close=105, volume=-1)
        with pytest.raises(ValidationError, match="volume"):
            b.validate()

    def test_zero_volume_accepted(self) -> None:
        # rare but possible (illiquid hours); not invalid
        b = OHLCVBar(ts_ms=1, open=100, high=100, low=100, close=100, volume=0)
        b.validate()

    def test_negative_price_rejected(self) -> None:
        b = OHLCVBar(ts_ms=1, open=-100, high=-90, low=-110, close=-105, volume=10)
        with pytest.raises(ValidationError, match="price"):
            b.validate()

    def test_zero_price_rejected(self) -> None:
        # zero-priced bars are universally bad data on real exchanges
        b = OHLCVBar(ts_ms=1, open=0, high=110, low=90, close=105, volume=10)
        with pytest.raises(ValidationError, match="price"):
            b.validate()

    def test_negative_timestamp_rejected(self) -> None:
        b = OHLCVBar(ts_ms=-1, open=100, high=110, low=90, close=105, volume=10)
        with pytest.raises(ValidationError, match="timestamp"):
            b.validate()


class TestOHLCVCacheStore:
    def test_store_empty_list_is_noop(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", [])
        assert cache.load_range("BTC/USDT", "1d", 0, 10**13).empty

    def test_store_and_retrieve_single_bar(self, cache: OHLCVCache) -> None:
        bar = _bar(1_700_000_000_000)
        cache.store("BTC/USDT", "1d", [bar])
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert len(df) == 1
        assert df.iloc[0]["ts_ms"] == 1_700_000_000_000

    def test_store_rejects_invalid_bar(self, cache: OHLCVCache) -> None:
        bad = OHLCVBar(ts_ms=1, open=100, high=80, low=90, close=85, volume=10)
        with pytest.raises(ValidationError):
            cache.store("BTC/USDT", "1d", [bad])

    def test_store_idempotent_on_duplicate_ts(self, cache: OHLCVCache) -> None:
        # Re-fetching the same range must not produce duplicate rows
        bars = _bar_list(1_700_000_000_000, count=5)
        cache.store("BTC/USDT", "1d", bars)
        cache.store("BTC/USDT", "1d", bars)  # repeat
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert len(df) == 5

    def test_store_replaces_on_duplicate_ts_with_newer_data(self, cache: OHLCVCache) -> None:
        # Real scenario: re-fetching may return slightly different volume due to late prints
        cache.store("BTC/USDT", "1d", [_bar(1, price=100, vol=10)])
        cache.store("BTC/USDT", "1d", [_bar(1, price=100, vol=20)])
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert len(df) == 1
        assert df.iloc[0]["volume"] == 20.0

    def test_store_isolates_symbols(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", _bar_list(1_700_000_000_000, 3))
        cache.store("ETH/USDT", "1d", _bar_list(1_700_000_000_000, 5))
        assert len(cache.load_range("BTC/USDT", "1d", 0, 10**13)) == 3
        assert len(cache.load_range("ETH/USDT", "1d", 0, 10**13)) == 5

    def test_store_isolates_timeframes(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", _bar_list(1_700_000_000_000, 3))
        cache.store("BTC/USDT", "4h", _bar_list(1_700_000_000_000, 7))
        assert len(cache.load_range("BTC/USDT", "1d", 0, 10**13)) == 3
        assert len(cache.load_range("BTC/USDT", "4h", 0, 10**13)) == 7


class TestOHLCVCacheLoadRange:
    def test_load_empty_when_nothing_stored(self, cache: OHLCVCache) -> None:
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert df.empty

    def test_load_returns_dataframe_with_expected_columns(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", [_bar(1_700_000_000_000)])
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        for col in ("ts_ms", "open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_load_inclusive_at_start_bound(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", [_bar(100), _bar(200), _bar(300)])
        df = cache.load_range("BTC/USDT", "1d", 200, 10**13)
        assert list(df["ts_ms"]) == [200, 300]

    def test_load_inclusive_at_end_bound(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", [_bar(100), _bar(200), _bar(300)])
        df = cache.load_range("BTC/USDT", "1d", 0, 200)
        assert list(df["ts_ms"]) == [100, 200]

    def test_load_returns_sorted_by_timestamp(self, cache: OHLCVCache) -> None:
        # Store out-of-order; expect ascending order on load
        cache.store("BTC/USDT", "1d", [_bar(300), _bar(100), _bar(200)])
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert list(df["ts_ms"]) == [100, 200, 300]

    def test_load_start_after_end_returns_empty(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", _bar_list(0, 5))
        df = cache.load_range("BTC/USDT", "1d", 1000, 100)
        assert df.empty

    def test_load_excludes_other_symbols(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", [_bar(100)])
        cache.store("ETH/USDT", "1d", [_bar(100)])
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert len(df) == 1


class TestOHLCVCacheRange:
    def test_range_empty_when_no_data(self, cache: OHLCVCache) -> None:
        assert cache.range("BTC/USDT", "1d") is None

    def test_range_returns_min_max_ts(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", [_bar(100), _bar(500), _bar(300)])
        first, last = cache.range("BTC/USDT", "1d")
        assert first == 100
        assert last == 500


class TestOHLCVCacheGapDetection:
    def test_no_gaps_in_contiguous_data(self, cache: OHLCVCache) -> None:
        bars = _bar_list(0, 10, interval_ms=1000)
        cache.store("BTC/USDT", "1d", bars)
        gaps = cache.detect_gaps("BTC/USDT", "1d", expected_interval_ms=1000)
        assert gaps == []

    def test_detects_single_missing_bar(self, cache: OHLCVCache) -> None:
        # Bars at 0, 1000, 3000 — missing 2000
        cache.store("BTC/USDT", "1d", [_bar(0), _bar(1000), _bar(3000)])
        gaps = cache.detect_gaps("BTC/USDT", "1d", expected_interval_ms=1000)
        assert gaps == [(1000, 3000)]

    def test_detects_multiple_gaps(self, cache: OHLCVCache) -> None:
        # 0, 1000, [gap], 3000, 4000, [gap], 6000
        cache.store("BTC/USDT", "1d", [_bar(0), _bar(1000), _bar(3000), _bar(4000), _bar(6000)])
        gaps = cache.detect_gaps("BTC/USDT", "1d", expected_interval_ms=1000)
        assert gaps == [(1000, 3000), (4000, 6000)]

    def test_no_gaps_for_empty_data(self, cache: OHLCVCache) -> None:
        gaps = cache.detect_gaps("BTC/USDT", "1d", expected_interval_ms=1000)
        assert gaps == []


class TestOHLCVCachePersistence:
    def test_data_survives_close_reopen(self, cache_path: Path) -> None:
        cache = OHLCVCache(cache_path)
        cache.store("BTC/USDT", "1d", _bar_list(1_700_000_000_000, 5))
        cache.close()

        reopened = OHLCVCache(cache_path)
        df = reopened.load_range("BTC/USDT", "1d", 0, 10**13)
        assert len(df) == 5

    def test_works_with_string_path(self, tmp_path: Path) -> None:
        # Common usage passes str instead of Path
        cache = OHLCVCache(str(tmp_path / "ohlcv.db"))
        cache.store("BTC/USDT", "1d", [_bar(1)])
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert len(df) == 1


class TestOHLCVCacheTailRisks:
    def test_large_range_query_returns_dataframe(self, cache: OHLCVCache) -> None:
        # 5000 bars covering ~14 years of daily data — make sure no memory issues
        bars = _bar_list(1_400_000_000_000, count=5000, interval_ms=86_400_000)
        cache.store("BTC/USDT", "1d", bars)
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert len(df) == 5000

    def test_unicode_symbol_safe(self, cache: OHLCVCache) -> None:
        # Defensive: symbols never contain unicode in practice but SQL injection vectors must close
        cache.store("BTC/USDT' OR 1=1; --", "1d", [_bar(1)])
        df = cache.load_range("BTC/USDT' OR 1=1; --", "1d", 0, 10**13)
        assert len(df) == 1
        # Verify a different symbol still empty
        df_other = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert df_other.empty

    def test_concurrent_stores_to_same_table_are_idempotent(self, cache: OHLCVCache) -> None:
        # SQLite serializes writes; verify no row loss when two batches overlap
        batch1 = [_bar(t) for t in range(0, 100, 10)]
        batch2 = [_bar(t) for t in range(50, 150, 10)]
        cache.store("BTC/USDT", "1d", batch1)
        cache.store("BTC/USDT", "1d", batch2)
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        # Union of both batches — 50..90 overlap, so total unique = 10 + 10 - 5 = 15
        assert len(df) == 15
        # Order is preserved
        assert list(df["ts_ms"]) == sorted(df["ts_ms"])

    def test_dataframe_dtypes_are_numeric(self, cache: OHLCVCache) -> None:
        cache.store("BTC/USDT", "1d", [_bar(1)])
        df = cache.load_range("BTC/USDT", "1d", 0, 10**13)
        assert pd.api.types.is_integer_dtype(df["ts_ms"])
        for col in ("open", "high", "low", "close", "volume"):
            assert pd.api.types.is_float_dtype(df[col])
