"""Tests for parametric VaR / CVaR (Phase 3 §3B.10)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.tail_risk import (
    cvar_parametric,
    portfolio_cvar_usdt,
    portfolio_var_usdt,
    var_parametric,
)


class TestVaR:
    def test_zero_vol_returns_zero(self) -> None:
        r = np.zeros(100)
        assert var_parametric(r) == 0.0

    def test_cvar_larger_than_var(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 1000)
        v = var_parametric(r, confidence=0.95)
        c = cvar_parametric(r, confidence=0.95)
        assert c > v

    def test_var_scales_with_horizon(self) -> None:
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, 500)
        v1 = var_parametric(r, confidence=0.95, horizon_days=1)
        v10 = var_parametric(r, confidence=0.95, horizon_days=10)
        assert v10 > v1 * 2.5  # roughly sqrt(10) ≈ 3.16x

    def test_higher_confidence_larger_var(self) -> None:
        rng = np.random.default_rng(2)
        r = rng.normal(0, 0.02, 500)
        v_95 = var_parametric(r, confidence=0.95)
        v_99 = var_parametric(r, confidence=0.99)
        assert v_99 > v_95

    def test_portfolio_var_scales_with_equity(self) -> None:
        rng = np.random.default_rng(3)
        r = rng.normal(0, 0.02, 500)
        v10k = portfolio_var_usdt(equity=10_000, daily_returns=r)
        v1k = portfolio_var_usdt(equity=1_000, daily_returns=r)
        assert v10k == pytest.approx(v1k * 10, rel=1e-9)

    def test_portfolio_cvar_scales_with_equity(self) -> None:
        rng = np.random.default_rng(4)
        r = rng.normal(0, 0.02, 500)
        v = portfolio_cvar_usdt(equity=10_000, daily_returns=r)
        assert v > 0
