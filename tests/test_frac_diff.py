"""Tests for fractional differentiation (AFML Ch. 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.frac_diff import (
    frac_diff_ffd,
    get_ffd_weights,
    find_min_d,
)


class TestFFDWeights:
    def test_d_zero_returns_identity(self) -> None:
        w = get_ffd_weights(d=0.0, threshold=1e-4)
        assert w.shape[0] == 1
        assert w[0] == pytest.approx(1.0)

    def test_d_one_returns_first_diff_weights(self) -> None:
        w = get_ffd_weights(d=1.0, threshold=1e-4)
        assert w[0] == pytest.approx(1.0)
        assert w[1] == pytest.approx(-1.0)
        assert w.shape[0] == 2

    def test_weights_decay_monotonically(self) -> None:
        w = get_ffd_weights(d=0.5, threshold=1e-5)
        assert w[0] == pytest.approx(1.0)
        abs_w = np.abs(w)
        assert all(abs_w[i] >= abs_w[i + 1] for i in range(len(abs_w) - 1))

    def test_threshold_truncates_tail(self) -> None:
        w_tight = get_ffd_weights(d=0.5, threshold=1e-2)
        w_loose = get_ffd_weights(d=0.5, threshold=1e-6)
        assert len(w_tight) < len(w_loose)
        assert np.abs(w_tight[-1]) >= 1e-2


class TestFracDiffFFD:
    def test_d_zero_returns_original(self) -> None:
        s = pd.Series(np.arange(100.0, 200.0))
        out = frac_diff_ffd(s, d=0.0, threshold=1e-4)
        assert out.iloc[-5:].equals(s.iloc[-5:])

    def test_d_one_approximates_first_diff(self) -> None:
        s = pd.Series(np.cumsum(np.random.default_rng(42).standard_normal(200)))
        out = frac_diff_ffd(s, d=1.0, threshold=1e-4)
        expected = s.diff()
        valid = out.dropna()
        exp_valid = expected.loc[valid.index]
        assert np.allclose(valid.to_numpy(), exp_valid.to_numpy(), atol=1e-9)

    def test_output_has_same_index_tail(self) -> None:
        s = pd.Series(np.arange(300.0), index=pd.date_range("2024-01-01", periods=300, freq="h"))
        out = frac_diff_ffd(s, d=0.5, threshold=1e-4)
        last_valid_idx = out.dropna().index[-1]
        assert last_valid_idx == s.index[-1]

    def test_no_lookahead(self) -> None:
        rng = np.random.default_rng(0)
        s = pd.Series(np.cumsum(rng.standard_normal(400)))
        out_full = frac_diff_ffd(s, d=0.4, threshold=1e-4)
        out_prefix = frac_diff_ffd(s.iloc[:300], d=0.4, threshold=1e-4)
        valid_prefix = out_prefix.dropna()
        same_region = out_full.loc[valid_prefix.index]
        assert np.allclose(same_region.to_numpy(), valid_prefix.to_numpy(), atol=1e-12)

    def test_fractional_d_makes_random_walk_more_stationary(self) -> None:
        from statsmodels.tsa.stattools import adfuller

        rng = np.random.default_rng(123)
        walk = pd.Series(np.cumsum(rng.standard_normal(1000)) + 100)
        frac = frac_diff_ffd(walk, d=0.4, threshold=1e-4).dropna()
        p_raw = adfuller(walk, autolag="AIC")[1]
        p_frac = adfuller(frac, autolag="AIC")[1]
        assert p_frac < p_raw

    def test_handles_short_series(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        out = frac_diff_ffd(s, d=0.5, threshold=1e-4)
        assert len(out) == 3


class TestFindMinD:
    def test_random_walk_needs_d_greater_than_zero(self) -> None:
        rng = np.random.default_rng(7)
        walk = pd.Series(np.cumsum(rng.standard_normal(800)) + 50)
        d_min = find_min_d(walk, max_d=1.0, step=0.1, adf_pvalue=0.05)
        assert d_min > 0.0

    def test_already_stationary_returns_zero(self) -> None:
        rng = np.random.default_rng(8)
        s = pd.Series(rng.standard_normal(800))
        d_min = find_min_d(s, max_d=1.0, step=0.1, adf_pvalue=0.05)
        assert d_min == 0.0

    def test_never_exceeds_max_d(self) -> None:
        s = pd.Series([1.0] * 500)
        d_min = find_min_d(s, max_d=0.5, step=0.1, adf_pvalue=0.05)
        assert d_min <= 0.5
