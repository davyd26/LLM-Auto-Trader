"""Tests for Probabilistic / Deflated Sharpe (AFML Ch. 14)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis.backtest_stats import (
    deflated_sharpe_ratio,
    pnl_moments,
    probabilistic_sharpe_ratio,
    sharpe_annualized,
)


class TestSharpeAnnualized:
    def test_zero_returns_zero_sharpe(self) -> None:
        r = np.zeros(100)
        assert sharpe_annualized(r, periods_per_year=252) == 0.0

    def test_positive_constant_means_high_sharpe_but_zero_std_returns_zero(self) -> None:
        r = np.full(100, 0.001)
        # std is 0 → formula returns 0 (undefined)
        assert sharpe_annualized(r, periods_per_year=252) == 0.0

    def test_scales_with_sqrt_periods(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, 10_000)
        s_daily = sharpe_annualized(r, periods_per_year=1)
        s_ann = sharpe_annualized(r, periods_per_year=252)
        assert s_ann == pytest.approx(s_daily * math.sqrt(252), rel=1e-6)


class TestPnLMoments:
    def test_shape_matches(self) -> None:
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, 500)
        m = pnl_moments(r)
        assert "mean" in m
        assert "std" in m
        assert "skew" in m
        assert "kurtosis" in m

    def test_negative_skew_on_left_tail(self) -> None:
        rng = np.random.default_rng(2)
        r = rng.normal(0, 0.01, 500)
        r[0] = -0.10  # big single loss → negative skew
        m = pnl_moments(r)
        assert m["skew"] < 0


class TestProbabilisticSharpeRatio:
    def test_psr_increases_with_sample_size(self) -> None:
        psr_small = probabilistic_sharpe_ratio(sr_hat=1.0, n=30, skew=0.0, kurt=3.0)
        psr_large = probabilistic_sharpe_ratio(sr_hat=1.0, n=300, skew=0.0, kurt=3.0)
        assert psr_large > psr_small

    def test_psr_bounded_0_1(self) -> None:
        for n in [10, 100, 1000]:
            for sr in [-0.5, 0, 1, 2]:
                psr = probabilistic_sharpe_ratio(sr_hat=sr, n=n, skew=0.0, kurt=3.0)
                assert 0.0 <= psr <= 1.0

    def test_psr_decreases_with_negative_skew_and_high_kurt(self) -> None:
        psr_normal = probabilistic_sharpe_ratio(sr_hat=1.0, n=100, skew=0.0, kurt=3.0)
        psr_badtail = probabilistic_sharpe_ratio(sr_hat=1.0, n=100, skew=-2.0, kurt=10.0)
        assert psr_badtail < psr_normal


class TestDeflatedSharpeRatio:
    def test_dsr_lte_psr(self) -> None:
        # For any num_trials > 1, DSR should be <= PSR against zero benchmark
        psr = probabilistic_sharpe_ratio(sr_hat=1.5, n=100, skew=0.0, kurt=3.0)
        dsr = deflated_sharpe_ratio(
            sr_hat=1.5, n=100, skew=0.0, kurt=3.0,
            num_trials=50, sr_trials_var=0.25,
        )
        assert dsr <= psr

    def test_dsr_decreases_with_more_trials(self) -> None:
        dsr_10 = deflated_sharpe_ratio(
            sr_hat=1.5, n=100, skew=0.0, kurt=3.0,
            num_trials=10, sr_trials_var=0.25,
        )
        dsr_100 = deflated_sharpe_ratio(
            sr_hat=1.5, n=100, skew=0.0, kurt=3.0,
            num_trials=100, sr_trials_var=0.25,
        )
        assert dsr_100 < dsr_10

    def test_dsr_bounded_0_1(self) -> None:
        dsr = deflated_sharpe_ratio(
            sr_hat=1.0, n=100, skew=0.0, kurt=3.0,
            num_trials=50, sr_trials_var=0.5,
        )
        assert 0.0 <= dsr <= 1.0

    def test_dsr_with_one_trial_matches_psr_against_zero(self) -> None:
        # num_trials=1 → expected max SR term vanishes, DSR ≈ PSR vs 0
        psr = probabilistic_sharpe_ratio(sr_hat=1.0, n=100, skew=0.0, kurt=3.0)
        dsr = deflated_sharpe_ratio(
            sr_hat=1.0, n=100, skew=0.0, kurt=3.0,
            num_trials=1, sr_trials_var=0.0,
        )
        assert dsr == pytest.approx(psr, abs=0.02)
