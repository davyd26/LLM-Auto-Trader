"""Bet sizing from probability (AFML Ch. 10).

Deterministic monotone transform from LLM probability estimate to a position-size
multiplier. No learning, no tuning. Used as a multiplier on the existing Kelly /
ATR / cap size — the addition is strictly size-reducing (multiplier ∈ [0, 1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _phi(z: float) -> float:
    """Standard normal CDF via erf identity."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bet_size_from_probability(
    p: float,
    num_classes: int = 2,
    min_p: float = 0.55,
    max_scale: float = 1.0,
) -> float:
    """AFML Ch. 10.3 Eq. 10.2 — size multiplier from probability.

    Formula: `m = 2 * Phi(z) - 1` where `z = (p - 1/k) / sqrt(p*(1-p))`,
    k = num_classes. Returns 0 when `p <= min_p` (no-trade cutoff); caps at
    `max_scale`.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if not (0.0 <= min_p <= 1.0):
        raise ValueError(f"min_p must be in [0, 1], got {min_p}")
    if max_scale <= 0:
        raise ValueError(f"max_scale must be > 0, got {max_scale}")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")

    if p <= min_p:
        return 0.0
    if p >= 1.0:
        return float(max_scale)

    denom = p * (1.0 - p)
    if denom <= 0:
        return float(max_scale)

    z = (p - 1.0 / num_classes) / math.sqrt(denom)
    m = 2.0 * _phi(z) - 1.0
    m = max(0.0, min(float(max_scale), m))
    return m


@dataclass
class ScenarioKelly:
    """3-outcome scenario Kelly result (Phase 2 §3A.5; landed early here)."""

    p_base: float
    p_moderate: float
    p_crisis: float
    kelly_size_usdt: float


def scenario_kelly(
    p_profit: float,
    equity: float,
    avg_win_usdt: float,
    avg_loss_moderate_usdt: float,
    avg_loss_crisis_usdt: float,
    kelly_fraction: float = 0.25,
) -> ScenarioKelly:
    """Kelly size across a 3-outcome tree (base / moderate / crisis).

    Probabilities: P(base)=p_profit, P(moderate)=(1-p_profit)*(1-p_profit/2),
                   P(crisis)=(1-p_profit) * (p_profit/2).
    (de Prado's jaam uses a simpler tree; we use a proper well-defined one.)

    Returns a fractional-Kelly USDT size. If expected value is non-positive,
    returns 0.
    """
    if not (0.0 <= p_profit <= 1.0):
        raise ValueError(f"p_profit must be in [0, 1], got {p_profit}")
    if equity <= 0 or avg_win_usdt <= 0 or kelly_fraction <= 0:
        return ScenarioKelly(p_base=p_profit, p_moderate=0.0, p_crisis=0.0, kelly_size_usdt=0.0)

    avg_loss_moderate = max(1e-9, abs(avg_loss_moderate_usdt))
    avg_loss_crisis = max(avg_loss_moderate, abs(avg_loss_crisis_usdt))

    q = 1.0 - p_profit
    p_moderate = q * (1.0 - p_profit / 2.0)
    p_crisis = q * (p_profit / 2.0)
    # Ensure probabilities sum to 1 (numerically); renormalize if needed.
    total = p_profit + p_moderate + p_crisis
    if total <= 0:
        return ScenarioKelly(p_base=p_profit, p_moderate=0.0, p_crisis=0.0, kelly_size_usdt=0.0)
    p_b = p_profit / total
    p_m = p_moderate / total
    p_c = p_crisis / total

    # Expected return per unit bet: weighted by scenario outcomes.
    # Unit bet: win avg_win_usdt at prob p_b; lose avg_loss_moderate at p_m;
    # lose avg_loss_crisis at p_c.
    expected_return = p_b * avg_win_usdt - p_m * avg_loss_moderate - p_c * avg_loss_crisis
    if expected_return <= 0:
        return ScenarioKelly(p_base=p_b, p_moderate=p_m, p_crisis=p_c, kelly_size_usdt=0.0)

    # Effective odds (average loss weighted by loss-probability mass).
    loss_prob = p_m + p_c
    if loss_prob <= 0:
        avg_loss = avg_loss_moderate
    else:
        avg_loss = (p_m * avg_loss_moderate + p_c * avg_loss_crisis) / loss_prob
    if avg_loss <= 0:
        return ScenarioKelly(p_base=p_b, p_moderate=p_m, p_crisis=p_c, kelly_size_usdt=0.0)

    b = avg_win_usdt / avg_loss  # payoff ratio
    p_eff = p_b  # probability of win
    q_eff = 1.0 - p_eff
    full_kelly = (p_eff * b - q_eff) / b
    if full_kelly <= 0:
        return ScenarioKelly(p_base=p_b, p_moderate=p_m, p_crisis=p_c, kelly_size_usdt=0.0)

    size = equity * full_kelly * kelly_fraction
    return ScenarioKelly(
        p_base=p_b, p_moderate=p_m, p_crisis=p_c, kelly_size_usdt=size
    )
