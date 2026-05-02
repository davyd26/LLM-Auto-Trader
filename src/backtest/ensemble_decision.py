"""Aggregate multiple voters' decisions into one consensus decision.

Each voter (an LLM persona, or a deterministic strategy variant) emits a
``Vote`` for a snapshot. The aggregator collapses the bag of votes into a
single decision by:

    1. **Majority vote** on the action type. Ties prefer ``hold``.
    2. **Median** on numeric fields (size, stop_loss, take_profit) over
       the voters that voted for the winning action type.
    3. **Veto rules** (optional): a single ``crisis`` regime call from any
       voter blocks bullish action types.
    4. **Confidence-weighted** alternative: votes weighted by
       self-reported confidence rather than equal weight.
    5. **Min-agreement gate**: if fewer than N voters agree on the winning
       action, fall back to ``hold`` (no clear consensus).

The output decision matches the existing schema (action_type, symbol,
size, stop_loss, take_profit). Plug into the existing decision_cache.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Optional


@dataclass(frozen=True)
class Vote:
    voter_id: str
    action_type: str  # hold | open_long | open_short | close
    symbol: str | None = None
    size: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float = 0.5  # in [0, 1]
    regime: str = "calm"     # calm | elevated | crisis


@dataclass(frozen=True)
class AggregateConfig:
    crisis_vetos_longs: bool = False
    min_agreement_count: int = 0


def _hold_decision() -> dict:
    return {
        "action_type": "hold",
        "symbol": None,
        "size": 0.0,
        "stop_loss": None,
        "take_profit": None,
    }


def _is_finite(x: float | None) -> bool:
    return x is not None and not math.isnan(x) and not math.isinf(x)


def _sanitize_votes(votes: list[Vote]) -> list[Vote]:
    """Drop votes with malformed numeric fields (NaN, negative size)."""
    cleaned = []
    for v in votes:
        if v.action_type in ("open_long", "open_short"):
            if v.size <= 0 or not _is_finite(v.size):
                continue
            if v.stop_loss is not None and not _is_finite(v.stop_loss):
                continue
            if v.take_profit is not None and not _is_finite(v.take_profit):
                continue
        cleaned.append(v)
    return cleaned


def _build_decision_from(consensus_voters: list[Vote], action_type: str) -> dict:
    """Build a decision from voters that agreed on ``action_type``."""
    if action_type in ("hold", "close"):
        sym = consensus_voters[0].symbol if consensus_voters else None
        return {
            "action_type": action_type,
            "symbol": sym,
            "size": 0.0,
            "stop_loss": None,
            "take_profit": None,
        }

    # open_long / open_short: take median numerics
    sym = consensus_voters[0].symbol
    sizes = [v.size for v in consensus_voters if _is_finite(v.size) and v.size > 0]
    stops = [v.stop_loss for v in consensus_voters if _is_finite(v.stop_loss)]
    tps = [v.take_profit for v in consensus_voters if _is_finite(v.take_profit)]

    return {
        "action_type": action_type,
        "symbol": sym,
        "size": float(median(sizes)) if sizes else 0.0,
        "stop_loss": float(median(stops)) if stops else None,
        "take_profit": float(median(tps)) if tps else None,
    }


def aggregate_majority(
    votes: list[Vote], config: Optional[AggregateConfig] = None,
) -> dict:
    """Plain majority vote with median numeric aggregation.

    Ties prefer ``hold`` (conservative). If a min-agreement count is set
    and not met, falls back to ``hold``.
    """
    if config is None:
        config = AggregateConfig()
    cleaned = _sanitize_votes(votes)
    if not cleaned:
        return _hold_decision()

    counts = Counter(v.action_type for v in cleaned)
    most_common = counts.most_common()
    top_count = most_common[0][1]
    leaders = [a for a, c in most_common if c == top_count]
    # Tiebreak: more than one leader → hold
    if len(leaders) > 1:
        return _hold_decision()
    winner = leaders[0]

    if config.min_agreement_count > 0 and top_count < config.min_agreement_count:
        return _hold_decision()

    consensus = [v for v in cleaned if v.action_type == winner]
    return _build_decision_from(consensus, winner)


def aggregate_weighted(
    votes: list[Vote], config: Optional[AggregateConfig] = None,
) -> dict:
    """Confidence-weighted aggregation. Each vote contributes its confidence
    score to its action type's total; highest-weight action wins."""
    if config is None:
        config = AggregateConfig()
    cleaned = _sanitize_votes(votes)
    if not cleaned:
        return _hold_decision()

    weights: dict[str, float] = {}
    for v in cleaned:
        weights[v.action_type] = weights.get(v.action_type, 0.0) + max(0.0, v.confidence)

    if not weights:
        return _hold_decision()
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top_weight = sorted_w[0][1]
    leaders = [a for a, w in sorted_w if w == top_weight]
    if len(leaders) > 1:
        return _hold_decision()
    winner = leaders[0]

    consensus = [v for v in cleaned if v.action_type == winner]
    return _build_decision_from(consensus, winner)


def aggregate_veto(
    votes: list[Vote], config: Optional[AggregateConfig] = None,
) -> dict:
    """Majority vote with veto rules layered on top.

    If ``crisis_vetos_longs`` and ANY voter reports regime=crisis, then
    open_long actions are blocked (decision falls to hold). open_short and
    close pass through.
    """
    if config is None:
        config = AggregateConfig(crisis_vetos_longs=True)
    decision = aggregate_majority(votes, config)
    if config.crisis_vetos_longs and decision["action_type"] == "open_long":
        if any(v.regime == "crisis" for v in votes):
            return _hold_decision()
    return decision
