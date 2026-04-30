"""Tests for unified cost + funding + slippage model (Phase 2 §3A.3)."""

from __future__ import annotations

import pytest

from src.analysis.cost_model import (
    TradeCosts,
    estimate_trade_cost,
    net_pnl,
    required_edge_pct,
)


class TestEstimateTradeCost:
    def test_zero_holding_zero_funding(self) -> None:
        c = estimate_trade_cost(
            notional=1000.0, leverage=2,
            holding_seconds=0.0, funding_rate_per_8h=0.0001,
            spread_pct=0.0001, depth=1_000_000.0, taker_fee=0.0001,
        )
        assert c.funding_paid == 0.0
        assert c.fee_entry > 0
        assert c.fee_exit > 0

    def test_funding_scales_linearly_with_time(self) -> None:
        c1 = estimate_trade_cost(
            notional=1000.0, leverage=2,
            holding_seconds=8 * 3600.0, funding_rate_per_8h=0.0001,
            spread_pct=0.0001, depth=1_000_000.0, taker_fee=0.0001,
        )
        c2 = estimate_trade_cost(
            notional=1000.0, leverage=2,
            holding_seconds=16 * 3600.0, funding_rate_per_8h=0.0001,
            spread_pct=0.0001, depth=1_000_000.0, taker_fee=0.0001,
        )
        assert c2.funding_paid == pytest.approx(2 * c1.funding_paid, rel=1e-6)

    def test_total_cost_pct_sum_of_components(self) -> None:
        c = estimate_trade_cost(
            notional=10_000.0, leverage=3,
            holding_seconds=16 * 3600.0, funding_rate_per_8h=0.0002,
            spread_pct=0.001, depth=50_000.0, taker_fee=0.0001,
        )
        expected = (
            (c.fee_entry + c.fee_exit + c.funding_paid
             + c.slippage_entry_est + c.slippage_exit_est)
            / 10_000.0
        )
        assert c.total_cost_pct == pytest.approx(expected, rel=1e-6)

    def test_thin_depth_increases_slippage(self) -> None:
        deep = estimate_trade_cost(
            notional=10_000.0, leverage=2, holding_seconds=3600.0,
            funding_rate_per_8h=0.0, spread_pct=0.0005,
            depth=1_000_000.0, taker_fee=0.0001,
        )
        thin = estimate_trade_cost(
            notional=10_000.0, leverage=2, holding_seconds=3600.0,
            funding_rate_per_8h=0.0, spread_pct=0.0005,
            depth=5_000.0, taker_fee=0.0001,
        )
        assert thin.slippage_entry_est > deep.slippage_entry_est


class TestNetPnl:
    def test_subtracts_total_cost(self) -> None:
        c = TradeCosts(
            fee_entry=0.5, fee_exit=0.5, funding_paid=2.0,
            slippage_entry_est=1.0, slippage_exit_est=1.0, total_cost_pct=0.0005,
        )
        assert net_pnl(gross_pnl=100.0, costs=c) == pytest.approx(95.0)


class TestRequiredEdgePct:
    def test_monotone_in_leverage(self) -> None:
        e1 = required_edge_pct(leverage=1, holding_seconds=8 * 3600,
                                funding_rate_per_8h=0.0001, spread_pct=0.0001,
                                depth=1e6, taker_fee=0.0001, notional=10_000)
        e5 = required_edge_pct(leverage=5, holding_seconds=8 * 3600,
                                funding_rate_per_8h=0.0001, spread_pct=0.0001,
                                depth=1e6, taker_fee=0.0001, notional=10_000)
        assert e5 >= e1  # more leverage → same cost absolute → relative to margin, higher edge required
