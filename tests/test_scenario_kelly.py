"""Tests for scenario-Kelly 3-outcome tree (Phase 2 §3A.5)."""

from __future__ import annotations

import pytest

from src.risk.bet_sizing import ScenarioKelly, scenario_kelly


class TestScenarioKelly:
    def test_high_p_profit_larger_size(self) -> None:
        high = scenario_kelly(
            p_profit=0.75, equity=10_000, avg_win_usdt=200,
            avg_loss_moderate_usdt=100, avg_loss_crisis_usdt=500,
            kelly_fraction=0.25,
        )
        low = scenario_kelly(
            p_profit=0.55, equity=10_000, avg_win_usdt=200,
            avg_loss_moderate_usdt=100, avg_loss_crisis_usdt=500,
            kelly_fraction=0.25,
        )
        assert high.kelly_size_usdt > low.kelly_size_usdt

    def test_probabilities_sum_to_one(self) -> None:
        s = scenario_kelly(
            p_profit=0.65, equity=10_000, avg_win_usdt=150,
            avg_loss_moderate_usdt=100, avg_loss_crisis_usdt=400,
        )
        assert s.p_base + s.p_moderate + s.p_crisis == pytest.approx(1.0, abs=1e-9)

    def test_non_positive_ev_returns_zero(self) -> None:
        # p_profit low with huge crisis loss → EV negative
        s = scenario_kelly(
            p_profit=0.45, equity=10_000, avg_win_usdt=50,
            avg_loss_moderate_usdt=100, avg_loss_crisis_usdt=1000,
            kelly_fraction=0.25,
        )
        assert s.kelly_size_usdt == 0.0

    def test_rejects_invalid_p(self) -> None:
        with pytest.raises(ValueError):
            scenario_kelly(
                p_profit=1.5, equity=10_000, avg_win_usdt=100,
                avg_loss_moderate_usdt=50, avg_loss_crisis_usdt=300,
            )

    def test_zero_equity_returns_zero(self) -> None:
        s = scenario_kelly(
            p_profit=0.7, equity=0, avg_win_usdt=100,
            avg_loss_moderate_usdt=50, avg_loss_crisis_usdt=300,
        )
        assert s.kelly_size_usdt == 0.0

    def test_larger_kelly_fraction_bigger_size(self) -> None:
        small = scenario_kelly(
            p_profit=0.7, equity=10_000, avg_win_usdt=150,
            avg_loss_moderate_usdt=100, avg_loss_crisis_usdt=400,
            kelly_fraction=0.125,
        )
        big = scenario_kelly(
            p_profit=0.7, equity=10_000, avg_win_usdt=150,
            avg_loss_moderate_usdt=100, avg_loss_crisis_usdt=400,
            kelly_fraction=0.33,
        )
        assert big.kelly_size_usdt > small.kelly_size_usdt
