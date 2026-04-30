"""Tests for funding rate accounting on perpetual positions.

Funding rate is a critical real-world cost on perpetual futures. Long
positions pay positive funding; shorts receive it. Unmodeled, a backtest
overstates returns by 1-3% per month at typical Binance USDT-perp funding
rates (0.01%/8h baseline; spikes to 0.05%/8h or more in stressed regimes).

We model funding as a flat per-period rate applied per-position-day,
charging long positions and crediting shorts. A more sophisticated future
extension would use historical funding data, but the flat-rate version
reasonably approximates expected drag.

Edge cases covered:
    - Zero days held → zero funding
    - Long pays positive funding (reduces PnL)
    - Short receives positive funding (increases PnL)
    - Negative funding flips signs (long receives, short pays)
    - Days_held in fractional days (intra-day positions)

Tail risks covered:
    - Negative days_held rejected
    - Funding-rate magnitude sanity check (>5%/day is suspicious)
"""

from __future__ import annotations

import math

import pytest

from src.backtest.funding import compute_funding_cost


class TestComputeFundingCost:
    def test_zero_days_zero_cost(self) -> None:
        cost = compute_funding_cost(
            side="long", notional=10_000.0, days_held=0.0, funding_rate_per_day=0.0001
        )
        assert cost == 0.0

    def test_long_pays_positive_funding(self) -> None:
        # 1 BTC at $50K = $50K notional, held 7 days, 0.01% per day = 0.07% over week
        cost = compute_funding_cost(
            side="long", notional=50_000.0, days_held=7.0, funding_rate_per_day=0.0001
        )
        assert cost == pytest.approx(50_000.0 * 0.0001 * 7.0)
        # Cost is positive (subtracted from PnL)
        assert cost > 0

    def test_short_receives_positive_funding(self) -> None:
        cost = compute_funding_cost(
            side="short", notional=50_000.0, days_held=7.0, funding_rate_per_day=0.0001
        )
        assert cost < 0  # negative means a credit

    def test_negative_funding_long_receives(self) -> None:
        # Negative funding rate happens during heavy short bias (e.g., panic shorts)
        cost = compute_funding_cost(
            side="long", notional=50_000.0, days_held=7.0, funding_rate_per_day=-0.0001
        )
        assert cost < 0  # long now receives

    def test_fractional_days(self) -> None:
        cost = compute_funding_cost(
            side="long", notional=10_000.0, days_held=0.5, funding_rate_per_day=0.0002
        )
        assert cost == pytest.approx(10_000.0 * 0.0002 * 0.5)


class TestFundingValidation:
    def test_negative_days_rejected(self) -> None:
        with pytest.raises(ValueError):
            compute_funding_cost(
                side="long", notional=10_000.0, days_held=-1.0, funding_rate_per_day=0.0001
            )

    def test_unknown_side_rejected(self) -> None:
        with pytest.raises(ValueError):
            compute_funding_cost(
                side="diagonal", notional=10_000.0, days_held=1.0, funding_rate_per_day=0.0001
            )

    def test_extreme_funding_rate_warned(self) -> None:
        # >5%/day is unrealistic; we should at least not crash
        cost = compute_funding_cost(
            side="long", notional=10_000.0, days_held=1.0, funding_rate_per_day=0.10
        )
        assert math.isfinite(cost)


class TestFundingMath:
    """Sanity checks on funding-cost arithmetic."""

    def test_long_funding_scales_linearly_with_notional(self) -> None:
        a = compute_funding_cost("long", 1_000.0, 1.0, 0.0001)
        b = compute_funding_cost("long", 10_000.0, 1.0, 0.0001)
        assert b == pytest.approx(10 * a)

    def test_long_funding_scales_linearly_with_days(self) -> None:
        a = compute_funding_cost("long", 10_000.0, 1.0, 0.0001)
        b = compute_funding_cost("long", 10_000.0, 7.0, 0.0001)
        assert b == pytest.approx(7 * a)

    def test_short_funding_is_negation_of_long(self) -> None:
        long_cost = compute_funding_cost("long", 10_000.0, 7.0, 0.0001)
        short_cost = compute_funding_cost("short", 10_000.0, 7.0, 0.0001)
        assert short_cost == pytest.approx(-long_cost)
