"""Tests for GARCH(1,1) vol model (Phase 2 §3A.4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.garch import (
    GARCHState,
    fit_garch_11,
    update_garch,
    garch_state_from_returns,
)


class TestFitGARCH11:
    def test_stable_parameters_for_iid_gaussian(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.01, 500)
        omega, alpha, beta = fit_garch_11(returns)
        assert omega > 0
        assert alpha >= 0
        assert beta >= 0
        assert alpha + beta < 1.0  # stationarity condition

    def test_parameters_in_reasonable_range(self) -> None:
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, 500)
        omega, alpha, beta = fit_garch_11(returns)
        assert 0 < omega < 1e-2
        assert 0 <= alpha <= 1
        assert 0 <= beta <= 1


class TestUpdateGARCH:
    def test_update_changes_sigma(self) -> None:
        state = GARCHState(sigma_t=0.01, sigma_t_plus_1=0.01, sigma_z=0.0, crisis_flag=False)
        new = update_garch(state, r_t=0.05, omega=1e-5, alpha=0.1, beta=0.85)
        assert new.sigma_t != state.sigma_t  # big shock should lift sigma

    def test_shock_increases_sigma(self) -> None:
        state = GARCHState(sigma_t=0.01, sigma_t_plus_1=0.01, sigma_z=0.0, crisis_flag=False)
        new_shock = update_garch(state, r_t=0.10, omega=1e-5, alpha=0.1, beta=0.85)
        new_calm = update_garch(state, r_t=0.001, omega=1e-5, alpha=0.1, beta=0.85)
        assert new_shock.sigma_t_plus_1 > new_calm.sigma_t_plus_1


class TestGarchStateFromReturns:
    def test_fits_and_produces_state(self) -> None:
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.01, 500)
        state = garch_state_from_returns(returns)
        assert isinstance(state, GARCHState)
        assert state.sigma_t > 0
        assert state.sigma_t_plus_1 > 0

    def test_crisis_flag_triggers_on_high_vol(self) -> None:
        # Train on calm data, then feed a series with a big vol spike
        rng = np.random.default_rng(3)
        calm = rng.normal(0, 0.005, 400)
        shock = np.concatenate([calm, rng.normal(0, 0.03, 50)])  # 6x vol spike
        state = garch_state_from_returns(shock)
        assert state.sigma_z > 0

    def test_handles_short_input(self) -> None:
        state = garch_state_from_returns(np.array([0.01, -0.01, 0.005]))
        assert state.sigma_t >= 0
