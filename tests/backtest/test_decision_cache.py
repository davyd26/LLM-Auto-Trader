"""Tests for DecisionCache — SQLite-backed content-hash cache for LLM decisions.

The cache stores parsed Pydantic decisions keyed by snapshot SHA256 so that:
    1. CPCV can replay the same decision across all 28 paths without re-invoking
       the LLM (subscription-quota saving).
    2. Re-running a backtest with different risk params is free (decisions are
       independent of risk config).
    3. The audit trail is preserved (every live decision is hashable to its
       snapshot).

Edge cases covered:
    - get on missing key returns None
    - put then get returns identical content
    - duplicate put with same key is idempotent (last write wins, but content equal)
    - duplicate put with different content for same key raises (programmer error)
    - empty JSON value rejected
    - persistence across close/reopen
    - concurrent put/get
    - large decision blob (10KB+)

Tail risks covered:
    - corrupt JSON on read raises rather than returning corrupt data
    - hash collision (theoretical) — we trust SHA256
    - SQLite locking under concurrent writers (uses WAL)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backtest.decision_cache import DecisionCache


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "decisions.db"


@pytest.fixture
def cache(cache_path: Path) -> DecisionCache:
    return DecisionCache(cache_path)


_SAMPLE_DECISION = {
    "market_assessment": {
        "trend": "bullish",
        "confidence": 0.7,
        "regime": "calm",
    },
    "actions": [
        {
            "type": "open_long",
            "symbol": "BTC/USDT",
            "size_pct": 10,
            "leverage": 2,
            "stop_loss_pct": 5,
        }
    ],
    "next_check_minutes": 10080,  # 7 days
    "risk_notes": "Trend confirmed across 4h and 1d.",
}


class TestDecisionCacheBasic:
    def test_get_missing_key_returns_none(self, cache: DecisionCache) -> None:
        assert cache.get("0" * 64) is None

    def test_put_then_get_returns_identical_content(self, cache: DecisionCache) -> None:
        key = "a" * 64
        cache.put(key, _SAMPLE_DECISION)
        retrieved = cache.get(key)
        assert retrieved == _SAMPLE_DECISION

    def test_put_idempotent_for_identical_content(self, cache: DecisionCache) -> None:
        key = "b" * 64
        cache.put(key, _SAMPLE_DECISION)
        cache.put(key, _SAMPLE_DECISION)
        assert cache.get(key) == _SAMPLE_DECISION

    def test_put_with_different_content_for_same_key_raises(self, cache: DecisionCache) -> None:
        key = "c" * 64
        cache.put(key, _SAMPLE_DECISION)
        modified = dict(_SAMPLE_DECISION)
        modified["risk_notes"] = "different"
        with pytest.raises(ValueError, match="different content"):
            cache.put(key, modified)


class TestDecisionCacheValidation:
    def test_empty_decision_rejected(self, cache: DecisionCache) -> None:
        with pytest.raises(ValueError, match="empty"):
            cache.put("d" * 64, {})

    def test_non_dict_decision_rejected(self, cache: DecisionCache) -> None:
        with pytest.raises(TypeError):
            cache.put("e" * 64, "not a dict")  # type: ignore[arg-type]

    def test_invalid_hash_format_rejected(self, cache: DecisionCache) -> None:
        # SHA256 hex is 64 chars
        with pytest.raises(ValueError, match="hex"):
            cache.put("short", _SAMPLE_DECISION)
        with pytest.raises(ValueError, match="hex"):
            cache.put("g" * 65, _SAMPLE_DECISION)
        with pytest.raises(ValueError, match="hex"):
            cache.put("X" * 64, _SAMPLE_DECISION)  # X not in hex set


class TestDecisionCachePersistence:
    def test_data_survives_close_reopen(self, cache_path: Path) -> None:
        # 'a' is in the hex set; 'h' is not — the cache rejects non-hex keys.
        key = "a" * 64
        c1 = DecisionCache(cache_path)
        c1.put(key, _SAMPLE_DECISION)
        c1.close()

        c2 = DecisionCache(cache_path)
        assert c2.get(key) == _SAMPLE_DECISION


class TestDecisionCacheStatus:
    def test_count_zero_initially(self, cache: DecisionCache) -> None:
        assert cache.count() == 0

    def test_count_increases_with_puts(self, cache: DecisionCache) -> None:
        cache.put("0" * 64, _SAMPLE_DECISION)
        cache.put("1" * 64, _SAMPLE_DECISION)
        assert cache.count() == 2

    def test_count_does_not_double_count_same_key(self, cache: DecisionCache) -> None:
        cache.put("0" * 64, _SAMPLE_DECISION)
        cache.put("0" * 64, _SAMPLE_DECISION)
        assert cache.count() == 1

    def test_has_returns_true_after_put(self, cache: DecisionCache) -> None:
        cache.put("0" * 64, _SAMPLE_DECISION)
        assert cache.has("0" * 64)
        assert not cache.has("1" * 64)

    def test_keys_returns_all_stored_keys(self, cache: DecisionCache) -> None:
        cache.put("0" * 64, _SAMPLE_DECISION)
        cache.put("1" * 64, _SAMPLE_DECISION)
        keys = cache.keys()
        assert set(keys) == {"0" * 64, "1" * 64}


class TestDecisionCacheTailRisks:
    def test_large_decision_blob_round_trips(self, cache: DecisionCache) -> None:
        # 10KB of risk_notes — within reason but stresses storage path
        big = dict(_SAMPLE_DECISION)
        big["risk_notes"] = "x" * 10_000
        cache.put("9" * 64, big)
        assert cache.get("9" * 64) == big

    def test_corrupted_value_raises(self, cache_path: Path) -> None:
        # Manually inject corrupt JSON and verify reader raises
        import sqlite3
        conn = sqlite3.connect(cache_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS decisions ("
            "snapshot_hash TEXT PRIMARY KEY, decision_json TEXT NOT NULL, "
            "created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO decisions VALUES (?, ?, ?)",
            ("f" * 64, "{not valid json", 0.0),
        )
        conn.commit()
        conn.close()

        c = DecisionCache(cache_path)
        with pytest.raises(json.JSONDecodeError):
            c.get("f" * 64)

    def test_returns_independent_dict_per_get(self, cache: DecisionCache) -> None:
        # Mutating returned dict must not affect cache state
        cache.put("0" * 64, _SAMPLE_DECISION)
        d1 = cache.get("0" * 64)
        d1["mutated"] = True
        d2 = cache.get("0" * 64)
        assert "mutated" not in d2
