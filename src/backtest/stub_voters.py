"""Deterministic stub voter set: 5 personas built from baseline-strategy variants.

Used for ensemble backtesting without real LLM calls. Each persona has a
slightly different worldview encoded as a different ``BaselineConfig``,
producing diverse decisions on the same snapshot. The aggregator then
collapses them into one consensus decision.

Personas (intentionally heterogeneous):

    quant_short  — fast trend (4-week momentum)
    quant_long   — slow trend (12-week momentum)
    macro_trend  — 8-week momentum, tight hedge (H=0.10), conservative size
    technical    — 4-week momentum + vol-targeting (vt=0.04)
    contrarian   — 16-week momentum, larger size when in (treats moves as durable)

Each persona's decision is wrapped in a ``Vote`` for the aggregator.

This module is the deterministic counterpart of dispatching N LLM personas
via Claude Code subagents. Same architecture: many voters → one aggregated
decision. Real LLM dispatch swaps these stub functions for ``claude -p``
calls and the rest of the pipeline (vote_cache, aggregator) is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from src.backtest.baseline_strategy import BaselineConfig, decide_baseline
from src.backtest.ensemble_decision import Vote


@dataclass(frozen=True)
class VoterSpec:
    voter_id: str
    config: BaselineConfig


def _baseline_voter_factory(spec: VoterSpec) -> Callable:
    """Return a function that produces a Vote from a snapshot for this voter."""
    def voter_fn(symbol_bars, current_positions, equity_usd) -> Vote:
        decision = decide_baseline(
            symbol_bars=symbol_bars,
            current_positions=current_positions,
            equity_usd=equity_usd,
            config=spec.config,
        )
        # Extract first action from the decision
        actions = decision.get("actions", [{"type": "hold"}])
        first = actions[0]
        ma = decision.get("market_assessment", {})
        return Vote(
            voter_id=spec.voter_id,
            action_type=first.get("type", "hold"),
            symbol=first.get("symbol"),
            size=float(first.get("size", 0.0) or 0.0),
            stop_loss=first.get("stop_loss") if first.get("stop_loss") is not None else None,
            take_profit=first.get("take_profit") if first.get("take_profit") is not None else None,
            confidence=float(ma.get("confidence", 0.5)),
            regime=str(ma.get("regime", "calm")),
        )
    return voter_fn


def default_weekly_voter_specs(
    primary_symbol: str = "BTC/USDT:USDT",
    hedge_symbol: str = "PAXG/USDT",
) -> list[VoterSpec]:
    """Five-persona ensemble for weekly cadence."""
    base = dict(primary_symbol=primary_symbol, hedge_symbol=hedge_symbol)
    return [
        VoterSpec(
            voter_id="quant_short",
            config=BaselineConfig(
                momentum_lookback_bars=4, drawdown_hedge_threshold=0.20,
                long_size_usd_pct=1.0, **base,
            ),
        ),
        VoterSpec(
            voter_id="quant_long",
            config=BaselineConfig(
                momentum_lookback_bars=12, drawdown_hedge_threshold=0.20,
                long_size_usd_pct=1.0, **base,
            ),
        ),
        VoterSpec(
            voter_id="macro_trend",
            config=BaselineConfig(
                momentum_lookback_bars=8, drawdown_hedge_threshold=0.10,
                long_size_usd_pct=0.5, **base,
            ),
        ),
        VoterSpec(
            voter_id="technical",
            config=BaselineConfig(
                momentum_lookback_bars=4, drawdown_hedge_threshold=0.20,
                long_size_usd_pct=1.0,
                vol_target_pct=0.04, vol_lookback_bars=12, vol_size_max_mult=1.5,
                **base,
            ),
        ),
        VoterSpec(
            voter_id="contrarian",
            config=BaselineConfig(
                momentum_lookback_bars=16, drawdown_hedge_threshold=0.20,
                long_size_usd_pct=1.0, **base,
            ),
        ),
    ]


def champion_tilted_voter_specs(
    primary_symbol: str = "BTC/USDT:USDT",
    hedge_symbol: str = "PAXG/USDT",
) -> list[VoterSpec]:
    """Five-voter set where every voter shares the champion's vol-targeted
    sizing (so the median size is ~champion-equivalent) but disagrees on
    momentum lookback. Tests whether vote-medianed lookback beats any
    single-lookback variant.
    """
    base = dict(
        primary_symbol=primary_symbol,
        hedge_symbol=hedge_symbol,
        drawdown_hedge_threshold=0.20,
        long_size_usd_pct=3.0,
        vol_target_pct=0.015,
        vol_lookback_bars=12,
        vol_size_max_mult=1.67,
    )
    return [
        VoterSpec("trend_4w", BaselineConfig(momentum_lookback_bars=4, **base)),
        VoterSpec("trend_6w", BaselineConfig(momentum_lookback_bars=6, **base)),
        VoterSpec("trend_8w", BaselineConfig(momentum_lookback_bars=8, **base)),
        VoterSpec("trend_12w", BaselineConfig(momentum_lookback_bars=12, **base)),
        VoterSpec("trend_16w", BaselineConfig(momentum_lookback_bars=16, **base)),
    ]


def collect_stub_votes(
    voter_specs: list[VoterSpec],
    symbol_bars: dict[str, list[dict]],
    current_positions: list[dict],
    equity_usd: float,
) -> list[Vote]:
    """Run each voter over the snapshot and return their votes."""
    votes = []
    for spec in voter_specs:
        try:
            v = _baseline_voter_factory(spec)(symbol_bars, current_positions, equity_usd)
            votes.append(v)
        except Exception:
            # Skip a malformed voter rather than break the ensemble
            continue
    return votes


def _is_finite(x):
    return x is not None and not math.isnan(x) and not math.isinf(x)
