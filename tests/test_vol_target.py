"""Tests for portfolio vol targeting (Phase 3 §3B.11)."""

from __future__ import annotations

import numpy as np
import pytest

from src.risk.vol_target import compute_vol_target_multiplier, realized_portfolio_vol


class TestVolTarget:
    def test_vol_above_target_reduces_exposure(self) -> None:
        m = compute_vol_target_multiplier(
            current_portfolio_vol=0.03, target_portfolio_vol=0.015,
        )
        assert 0.25 <= m < 1.0

    def test_vol_below_target_can_grow(self) -> None:
        m = compute_vol_target_multiplier(
            current_portfolio_vol=0.005, target_portfolio_vol=0.015,
        )
        assert m > 1.0

    def test_respects_bounds(self) -> None:
        m_low = compute_vol_target_multiplier(
            current_portfolio_vol=1.0, target_portfolio_vol=0.015, min_mult=0.25,
        )
        assert m_low == pytest.approx(0.25)
        m_high = compute_vol_target_multiplier(
            current_portfolio_vol=0.001, target_portfolio_vol=0.015, max_mult=1.5,
        )
        assert m_high == pytest.approx(1.5)

    def test_zero_input_returns_one(self) -> None:
        assert compute_vol_target_multiplier(0, 0.015) == 1.0

    def test_realized_portfolio_vol(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 100)
        v = realized_portfolio_vol(returns)
        assert v > 0

    def test_realized_vol_zero_on_constant(self) -> None:
        returns = np.full(50, 0.001)
        v = realized_portfolio_vol(returns)
        assert v == 0.0 or v < 1e-10
