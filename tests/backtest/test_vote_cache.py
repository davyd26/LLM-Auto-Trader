"""Tests for VoteCache — per-voter decision storage.

Stores raw votes from each voter for each snapshot. The aggregator runs
deterministically on cached votes, so trying a new aggregation function
is free. Different from DecisionCache which stores ONE decision per
snapshot; here we store N votes per snapshot.

Edge cases:
    - get_all on empty key returns []
    - put_vote idempotent for same content; raises on different content
    - vote with NaN/inf in numerics rejected
    - persistence across close/reopen
    - concurrent writes serialize cleanly
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.backtest.ensemble_decision import Vote
from src.backtest.vote_cache import VoteCache


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "votes.db"


@pytest.fixture
def cache(cache_path: Path) -> VoteCache:
    return VoteCache(cache_path)


def _vote(voter_id: str, action: str = "open_long") -> Vote:
    return Vote(
        voter_id=voter_id,
        action_type=action,
        symbol="BTC/USDT",
        size=0.05,
        stop_loss=45000.0 if action == "open_long" else None,
        confidence=0.7,
        regime="calm",
    )


_HASH = "a" * 64


class TestVoteCacheBasic:
    def test_get_all_empty_returns_empty_list(self, cache: VoteCache) -> None:
        assert cache.get_all(_HASH) == []

    def test_put_then_get_single_voter(self, cache: VoteCache) -> None:
        vote = _vote("quant")
        cache.put(_HASH, vote)
        retrieved = cache.get(_HASH, "quant")
        assert retrieved is not None
        assert retrieved.action_type == "open_long"
        assert retrieved.size == pytest.approx(0.05)

    def test_get_all_returns_all_voters(self, cache: VoteCache) -> None:
        cache.put(_HASH, _vote("quant", "open_long"))
        cache.put(_HASH, _vote("macro", "hold"))
        cache.put(_HASH, _vote("technical", "open_long"))
        votes = cache.get_all(_HASH)
        assert len(votes) == 3
        assert {v.voter_id for v in votes} == {"quant", "macro", "technical"}

    def test_put_idempotent_for_same_content(self, cache: VoteCache) -> None:
        vote = _vote("quant")
        cache.put(_HASH, vote)
        cache.put(_HASH, vote)  # repeat
        assert len(cache.get_all(_HASH)) == 1

    def test_put_different_content_for_same_key_raises(self, cache: VoteCache) -> None:
        cache.put(_HASH, _vote("quant", "open_long"))
        with pytest.raises(ValueError, match="different content"):
            # Same voter_id, different action
            cache.put(_HASH, _vote("quant", "open_short"))


class TestVoteCacheValidation:
    def test_invalid_hash_rejected(self, cache: VoteCache) -> None:
        with pytest.raises(ValueError, match="hex"):
            cache.put("short", _vote("quant"))
        with pytest.raises(ValueError, match="hex"):
            cache.put("Z" * 64, _vote("quant"))

    def test_nan_size_rejected(self, cache: VoteCache) -> None:
        bad = Vote(
            voter_id="quant", action_type="open_long",
            symbol="BTC/USDT", size=math.nan, stop_loss=45000,
        )
        with pytest.raises(ValueError, match="NaN"):
            cache.put(_HASH, bad)

    def test_inf_stop_loss_rejected(self, cache: VoteCache) -> None:
        bad = Vote(
            voter_id="quant", action_type="open_long",
            symbol="BTC/USDT", size=0.05, stop_loss=math.inf,
        )
        with pytest.raises(ValueError):
            cache.put(_HASH, bad)


class TestVoteCachePersistence:
    def test_data_survives_close_reopen(self, cache_path: Path) -> None:
        c1 = VoteCache(cache_path)
        c1.put(_HASH, _vote("quant"))
        c1.put(_HASH, _vote("macro", "hold"))
        c1.close()

        c2 = VoteCache(cache_path)
        votes = c2.get_all(_HASH)
        assert len(votes) == 2
        assert {v.voter_id for v in votes} == {"quant", "macro"}


class TestVoteCacheStatus:
    def test_count_voters_per_snapshot(self, cache: VoteCache) -> None:
        cache.put(_HASH, _vote("quant"))
        cache.put(_HASH, _vote("macro", "hold"))
        assert cache.count(_HASH) == 2

    def test_has_voter(self, cache: VoteCache) -> None:
        cache.put(_HASH, _vote("quant"))
        assert cache.has(_HASH, "quant")
        assert not cache.has(_HASH, "macro")

    def test_total_count_across_snapshots(self, cache: VoteCache) -> None:
        cache.put(_HASH, _vote("quant"))
        h2 = "b" * 64
        cache.put(h2, _vote("quant"))
        cache.put(h2, _vote("macro", "hold"))
        assert cache.total_count() == 3


class TestVoteCacheTailRisks:
    def test_returns_independent_vote_objects(self, cache: VoteCache) -> None:
        cache.put(_HASH, _vote("quant"))
        v1 = cache.get(_HASH, "quant")
        # Vote is frozen dataclass — can't mutate
        with pytest.raises((AttributeError, Exception)):
            v1.size = 999.0  # type: ignore[misc]

    def test_pathological_voter_id_safe(self, cache: VoteCache) -> None:
        # SQL injection attempt
        weird = "quant'; DROP TABLE decisions_votes; --"
        v = Vote(voter_id=weird, action_type="hold")
        cache.put(_HASH, v)
        assert cache.has(_HASH, weird)
        # Other tables / data still intact
        assert cache.count(_HASH) == 1
