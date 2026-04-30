"""Tests for log-relative dynamic baselines (Phase 2 §3A.2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.baselines import Baselines, compute_baselines


class TestComputeBaselines:
    def test_price_at_mean_log_rel_near_zero(self) -> None:
        daily = pd.Series([100.0] * 400)
        b = compute_baselines(daily, current_price=100.0)
        assert b.log_rel_1y == pytest.approx(0.0, abs=1e-9)
        assert b.percentile_1y == pytest.approx(0.5, abs=0.1)

    def test_price_above_baseline(self) -> None:
        daily = pd.Series([100.0 + i * 0.0 for i in range(400)])
        b = compute_baselines(daily, current_price=120.0)
        assert b.log_rel_1y == pytest.approx(math.log(1.2), abs=1e-3)
        assert b.percentile_1y > 0.9  # above full distribution

    def test_price_below_baseline(self) -> None:
        daily = pd.Series([100.0] * 400)
        b = compute_baselines(daily, current_price=80.0)
        assert b.log_rel_1y < 0
        assert b.percentile_1y < 0.1

    def test_insufficient_data_returns_zero_baselines(self) -> None:
        daily = pd.Series([100.0, 101.0])
        b = compute_baselines(daily, current_price=100.0)
        assert b.ma_365d == pytest.approx(100.5)  # short-window fallback
        assert b.log_rel_1y == pytest.approx(0.0, abs=0.01)

    def test_percentile_in_range(self) -> None:
        rng = np.random.default_rng(0)
        daily = pd.Series(100 + rng.normal(0, 5, 400))
        b = compute_baselines(daily, current_price=105.0)
        assert 0.0 <= b.percentile_1y <= 1.0

    def test_zero_price_doesnt_crash(self) -> None:
        daily = pd.Series([100.0] * 400)
        b = compute_baselines(daily, current_price=0.0)
        # log(0/ma) is -inf; expect a clipped or zero result
        assert b.log_rel_1y <= 0
