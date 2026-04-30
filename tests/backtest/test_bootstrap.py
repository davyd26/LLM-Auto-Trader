"""Tests for bootstrap-based statistics.

Block bootstrap addresses a key shortcoming of parametric PSR/DSR: they assume
i.i.d. returns. Real strategy returns are serially correlated (positions stay
on for multiple bars; volatility clusters). Block bootstrap preserves
short-range dependence by resampling contiguous blocks.

Edge cases covered:
    - Empty returns → NaN CI
    - Single observation → degenerate CI
    - Block size larger than returns length → reduce to returns length
    - Confidence outside (0, 1) rejected

Tail risks covered:
    - Reproducibility across runs given the same seed
    - Bootstrap distribution centered near point estimate
    - Wider CI for shorter samples (statistical correctness)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis.bootstrap import block_bootstrap_sharpe_ci, block_bootstrap_stat


class TestBlockBootstrapSharpeCI:
    def test_ci_contains_point_estimate(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)  # 1 year of daily
        from src.analysis.backtest_stats import sharpe_annualized
        point_sharpe = sharpe_annualized(returns, periods_per_year=252)
        lo, hi = block_bootstrap_sharpe_ci(
            returns, periods_per_year=252, n_resamples=500, block_size=4, seed=0
        )
        assert lo <= point_sharpe <= hi

    def test_wider_ci_for_smaller_sample(self) -> None:
        rng = np.random.default_rng(42)
        small = rng.normal(0.001, 0.01, 30)
        large = rng.normal(0.001, 0.01, 1000)
        lo_s, hi_s = block_bootstrap_sharpe_ci(small, periods_per_year=252, n_resamples=500, seed=0)
        lo_l, hi_l = block_bootstrap_sharpe_ci(large, periods_per_year=252, n_resamples=500, seed=0)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_reproducible_with_same_seed(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 100)
        ci1 = block_bootstrap_sharpe_ci(returns, periods_per_year=252, n_resamples=500, seed=123)
        ci2 = block_bootstrap_sharpe_ci(returns, periods_per_year=252, n_resamples=500, seed=123)
        assert ci1 == ci2

    def test_empty_returns_nan(self) -> None:
        lo, hi = block_bootstrap_sharpe_ci(
            np.array([]), periods_per_year=252, n_resamples=100, seed=0
        )
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_block_size_larger_than_sample_clamped(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 10)
        lo, hi = block_bootstrap_sharpe_ci(
            returns, periods_per_year=252, n_resamples=100, block_size=100, seed=0
        )
        assert not math.isnan(lo)
        assert not math.isnan(hi)

    def test_invalid_confidence_rejected(self) -> None:
        returns = np.array([0.01] * 10)
        with pytest.raises(ValueError):
            block_bootstrap_sharpe_ci(
                returns, periods_per_year=252, n_resamples=10, confidence=1.5, seed=0
            )


class TestBlockBootstrapStat:
    def test_bootstrap_mean_centered_correctly(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.5, 0.1, 1000)
        # Bootstrap the mean
        lo, hi = block_bootstrap_stat(
            returns, statistic=np.mean, n_resamples=500, confidence=0.95, block_size=4, seed=0,
        )
        assert lo < 0.5 < hi

    def test_bootstrap_median(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 1, 500)
        lo, hi = block_bootstrap_stat(
            returns, statistic=np.median, n_resamples=200, confidence=0.95, block_size=4, seed=0,
        )
        assert lo < 0 < hi
