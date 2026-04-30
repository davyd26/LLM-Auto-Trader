"""Tests for CSCV / PBO (AFML Ch. 11.4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.cscv import cscv_pbo, variants_from_realized_trades


class TestCSCVPBO:
    def test_iid_random_pbo_averages_near_half(self) -> None:
        # Single-seed PBO is noisy at small sample sizes; average across seeds.
        pbos = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            returns = rng.normal(0, 0.01, (10, 6))
            pbos.append(cscv_pbo(returns)["pbo"])
        mean_pbo = float(np.mean(pbos))
        assert 0.35 <= mean_pbo <= 0.65  # averaged, should be near null 0.5

    def test_monotone_improving_strategies_low_pbo(self) -> None:
        # If strategies are systematically better along an axis, IS winner is
        # usually the OOS winner → low PBO.
        rng = np.random.default_rng(0)
        n_periods, n_strats = 10, 6
        base = rng.normal(0, 0.01, (n_periods, n_strats))
        boost = np.linspace(0.0, 0.01, n_strats)  # monotone mean advantage
        returns = base + boost[np.newaxis, :]
        result = cscv_pbo(returns)
        assert result["pbo"] < 0.3

    def test_returns_num_trials(self) -> None:
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, (10, 6))
        result = cscv_pbo(returns)
        assert result["num_trials"] == 6

    def test_pbo_in_range(self) -> None:
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.01, (10, 8))
        result = cscv_pbo(returns)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_requires_even_number_of_periods(self) -> None:
        rng = np.random.default_rng(3)
        returns = rng.normal(0, 0.01, (9, 4))  # 9 trimmed to 8
        result = cscv_pbo(returns)
        assert "pbo" in result

    def test_handles_empty(self) -> None:
        result = cscv_pbo(np.zeros((0, 0)))
        assert result["pbo"] == 0.5  # no information → null
        assert result["num_trials"] == 0


class TestVariantsFromRealizedTrades:
    def test_builds_matrix_with_expected_shape(self) -> None:
        trades = [
            {"timestamp_close": 1000, "pnl_pct": 0.01, "llm_confidence": 0.60},
            {"timestamp_close": 2000, "pnl_pct": -0.005, "llm_confidence": 0.55},
            {"timestamp_close": 3000, "pnl_pct": 0.02, "llm_confidence": 0.75},
            {"timestamp_close": 4000, "pnl_pct": 0.008, "llm_confidence": 0.80},
        ]
        grid = [0.50, 0.60, 0.70]
        m = variants_from_realized_trades(trades, min_p_grid=grid)
        assert m.shape[1] == len(grid)
        assert m.shape[0] > 0

    def test_lower_cutoff_includes_more_trades(self) -> None:
        trades = [
            {"timestamp_close": i * 100, "pnl_pct": 0.01 * ((-1) ** i),
             "llm_confidence": 0.5 + 0.05 * i}
            for i in range(10)
        ]
        grid = [0.50, 0.80]
        m = variants_from_realized_trades(trades, min_p_grid=grid)
        # variant 0 (cutoff 0.50): takes all 10 trades
        # variant 1 (cutoff 0.80): takes few or none
        non_zero_0 = int(np.count_nonzero(m[:, 0]))
        non_zero_1 = int(np.count_nonzero(m[:, 1]))
        assert non_zero_0 >= non_zero_1

    def test_empty_trades_returns_zero_matrix(self) -> None:
        m = variants_from_realized_trades([], min_p_grid=[0.55, 0.60])
        assert m.shape[0] == 0 or np.all(m == 0)
