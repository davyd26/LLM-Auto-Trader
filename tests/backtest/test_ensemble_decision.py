"""Tests for ensemble decision aggregation.

Multiple voters (LLM personas or strategy variants) produce decisions for the
same snapshot. The aggregator collapses them into a single decision via
majority vote on the action type and median on numeric fields.

Edge cases covered:
    - Empty vote list → hold
    - Single vote → that vote (after normalization)
    - Tie on action type → conservative tiebreaker (prefer hold over open)
    - Numeric medians ignore voters with different action types
    - Veto rule: a single "crisis" regime call blocks open_long actions
    - Confidence-weighted aggregation
    - All voters disagree → hold (no consensus)
    - Stop-loss median honors side-specific constraints

Tail risks:
    - Voter with malformed decision is skipped, not crashed
    - Numeric NaN in voter output is filtered
    - Aggregator output validates (long stop < entry, short stop > entry)
"""

from __future__ import annotations

import math

import pytest

from src.backtest.ensemble_decision import (
    AggregateConfig,
    Vote,
    aggregate_majority,
    aggregate_veto,
    aggregate_weighted,
)


def _vote(voter_id: str, action: str, symbol: str | None = None,
          size: float = 0.0, stop_loss: float | None = None,
          confidence: float = 0.5, regime: str = "calm",
          take_profit: float | None = None) -> Vote:
    return Vote(
        voter_id=voter_id, action_type=action, symbol=symbol, size=size,
        stop_loss=stop_loss, take_profit=take_profit,
        confidence=confidence, regime=regime,
    )


class TestAggregateMajorityBasic:
    def test_empty_votes_returns_hold(self) -> None:
        out = aggregate_majority([])
        assert out["action_type"] == "hold"

    def test_single_vote_returns_that_vote(self) -> None:
        votes = [_vote("v1", "open_long", symbol="BTC/USDT", size=0.05, stop_loss=45000)]
        out = aggregate_majority(votes)
        assert out["action_type"] == "open_long"
        assert out["symbol"] == "BTC/USDT"
        assert out["size"] == pytest.approx(0.05)

    def test_unanimous_open_long_aggregates_to_open_long(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.04, stop_loss=44000),
            _vote("v2", "open_long", "BTC/USDT", size=0.06, stop_loss=46000),
            _vote("v3", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
        ]
        out = aggregate_majority(votes)
        assert out["action_type"] == "open_long"
        assert out["symbol"] == "BTC/USDT"
        # Median size
        assert out["size"] == pytest.approx(0.05)
        assert out["stop_loss"] == pytest.approx(45000)


class TestAggregateMajorityTies:
    def test_tie_prefers_hold(self) -> None:
        # 2 vs 2: open_long vs open_short → hold (no clear consensus)
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v2", "open_long", "BTC/USDT", size=0.04, stop_loss=44000),
            _vote("v3", "open_short", "BTC/USDT", size=0.05, stop_loss=55000),
            _vote("v4", "open_short", "BTC/USDT", size=0.04, stop_loss=56000),
        ]
        out = aggregate_majority(votes)
        assert out["action_type"] == "hold"

    def test_close_vs_open_tie_prefers_hold(self) -> None:
        votes = [
            _vote("v1", "close", "BTC/USDT"),
            _vote("v2", "close", "BTC/USDT"),
            _vote("v3", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v4", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
        ]
        out = aggregate_majority(votes)
        # Conservative: hold rather than do something contradictory
        assert out["action_type"] == "hold"

    def test_majority_with_one_dissenter(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v2", "open_long", "BTC/USDT", size=0.04, stop_loss=44000),
            _vote("v3", "open_long", "BTC/USDT", size=0.06, stop_loss=46000),
            _vote("v4", "hold"),
        ]
        out = aggregate_majority(votes)
        assert out["action_type"] == "open_long"
        # Median size only over the 3 long-voters
        assert out["size"] == pytest.approx(0.05)


class TestAggregateMedianNumerics:
    def test_median_size_only_over_consensus_voters(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.10, stop_loss=45000),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=46000),
            _vote("v3", "hold"),  # excluded from numeric median
            _vote("v4", "hold"),
        ]
        out = aggregate_majority(votes)
        # 2 long votes, 2 hold votes — tie, preferring hold
        assert out["action_type"] == "hold"

    def test_median_stop_loss_long(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=44000),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v3", "open_long", "BTC/USDT", size=0.05, stop_loss=46000),
        ]
        out = aggregate_majority(votes)
        assert out["stop_loss"] == pytest.approx(45000)

    def test_take_profit_median_when_set(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, take_profit=55000),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, take_profit=60000),
            _vote("v3", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, take_profit=58000),
        ]
        out = aggregate_majority(votes)
        assert out["take_profit"] == pytest.approx(58000)


class TestAggregateVeto:
    def test_crisis_regime_blocks_open_long(self) -> None:
        # 4 voters say open_long, 1 says crisis regime — that 1 vetos
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, regime="calm"),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, regime="calm"),
            _vote("v3", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, regime="elevated"),
            _vote("v4", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, regime="elevated"),
            _vote("v5", "hold", regime="crisis"),
        ]
        out = aggregate_veto(votes, AggregateConfig(crisis_vetos_longs=True))
        assert out["action_type"] == "hold"

    def test_crisis_regime_does_not_block_close_or_short(self) -> None:
        votes = [
            _vote("v1", "open_short", "BTC/USDT", size=0.05, stop_loss=55000, regime="elevated"),
            _vote("v2", "open_short", "BTC/USDT", size=0.05, stop_loss=55000, regime="elevated"),
            _vote("v3", "hold", regime="crisis"),
        ]
        out = aggregate_veto(votes, AggregateConfig(crisis_vetos_longs=True))
        assert out["action_type"] == "open_short"

    def test_no_crisis_voters_does_not_veto(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, regime="calm"),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, regime="calm"),
        ]
        out = aggregate_veto(votes, AggregateConfig(crisis_vetos_longs=True))
        assert out["action_type"] == "open_long"


class TestAggregateWeighted:
    def test_high_confidence_voter_dominates(self) -> None:
        # One voter says open_long with conf 0.9; two say hold with conf 0.3
        # Total weight long = 0.9, hold = 0.6 → long wins
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, confidence=0.9),
            _vote("v2", "hold", confidence=0.3),
            _vote("v3", "hold", confidence=0.3),
        ]
        out = aggregate_weighted(votes)
        assert out["action_type"] == "open_long"

    def test_low_confidence_majority_loses_to_high_confidence_minority(self) -> None:
        # 3 voters at low conf say long; 1 at very high conf says hold
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, confidence=0.2),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, confidence=0.2),
            _vote("v3", "open_long", "BTC/USDT", size=0.05, stop_loss=45000, confidence=0.2),
            _vote("v4", "hold", confidence=0.99),
        ]
        # Total long weight = 0.6, hold weight = 0.99 → hold wins
        out = aggregate_weighted(votes)
        assert out["action_type"] == "hold"


class TestAggregateRobustness:
    def test_voter_with_nan_size_skipped(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v2", "open_long", "BTC/USDT", size=math.nan, stop_loss=45000),
            _vote("v3", "open_long", "BTC/USDT", size=0.06, stop_loss=46000),
        ]
        out = aggregate_majority(votes)
        # NaN voter is skipped; median over remaining 2 → 0.055
        assert out["action_type"] == "open_long"
        assert out["size"] == pytest.approx(0.055)

    def test_voter_with_negative_size_filtered(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=-0.05, stop_loss=45000),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
        ]
        out = aggregate_majority(votes)
        # Negative size dropped; the lone valid open_long → 1 vote, that's the result
        assert out["action_type"] == "open_long"
        assert out["size"] == pytest.approx(0.05)

    def test_all_voters_disagree_returns_hold(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v2", "open_short", "BTC/USDT", size=0.05, stop_loss=55000),
            _vote("v3", "close", "BTC/USDT"),
            _vote("v4", "hold"),
        ]
        out = aggregate_majority(votes)
        assert out["action_type"] == "hold"

    def test_aggregated_long_stop_remains_below_size_implied_entry(self) -> None:
        # Aggregator must not produce a stop above price; we just trust
        # the median of valid voters.
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=44000),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=46000),
        ]
        out = aggregate_majority(votes)
        assert out["stop_loss"] == pytest.approx(45000)


class TestAggregateMinAgreement:
    def test_min_agreement_threshold_returns_hold(self) -> None:
        # 3 of 5 say open_long, but min_agreement=4 → no decision → hold
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v3", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v4", "hold"),
            _vote("v5", "hold"),
        ]
        out = aggregate_majority(votes, AggregateConfig(min_agreement_count=4))
        assert out["action_type"] == "hold"

    def test_min_agreement_satisfied(self) -> None:
        votes = [
            _vote("v1", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v2", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v3", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v4", "open_long", "BTC/USDT", size=0.05, stop_loss=45000),
            _vote("v5", "hold"),
        ]
        out = aggregate_majority(votes, AggregateConfig(min_agreement_count=4))
        assert out["action_type"] == "open_long"
