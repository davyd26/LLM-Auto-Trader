"""Tests for CUSUM event filter (AFML Ch. 2.5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis.cusum import CUSUMFilter


class TestCUSUMFilter:
    def test_constant_threshold_no_fire_on_flat(self) -> None:
        f = CUSUMFilter(h=0.02)
        for _ in range(50):
            assert f.update(100.0) == 0

    def test_constant_threshold_fires_on_positive_jump(self) -> None:
        f = CUSUMFilter(h=0.01)
        f.update(100.0)
        f.update(100.1)
        # Single 2% jump — log-return ~0.0198, exceeds 0.01 threshold
        event = f.update(102.1)
        assert event == 1

    def test_constant_threshold_fires_on_negative_jump(self) -> None:
        f = CUSUMFilter(h=0.01)
        f.update(100.0)
        f.update(100.1)
        event = f.update(97.9)  # ~-2.2% log return
        assert event == -1

    def test_reset_zeros_state_after_fire(self) -> None:
        f = CUSUMFilter(h=0.01)
        f.update(100.0)
        f.update(102.5)  # fires
        assert f.s_pos == 0.0
        assert f.s_neg == 0.0

    def test_accumulating_small_moves_fires(self) -> None:
        f = CUSUMFilter(h=0.015)
        f.update(100.0)
        # 5 small 0.4% moves accumulate to 2%, should fire
        prices = [100.4, 100.8, 101.2, 101.6, 102.0]
        fired = any(f.update(p) != 0 for p in prices)
        assert fired

    def test_adaptive_threshold_scales_with_vol(self) -> None:
        # Low-vol regime: small h after warmup
        rng = np.random.default_rng(42)
        f = CUSUMFilter(h=None, k=2.5, ewma_halflife=20, min_periods=20)
        price = 100.0
        for _ in range(50):
            price *= math.exp(rng.normal(0, 0.001))  # 0.1% vol
            f.update(price)
        low_vol_h = f.current_h
        assert low_vol_h is not None
        assert low_vol_h > 0

        # High-vol regime
        f_hi = CUSUMFilter(h=None, k=2.5, ewma_halflife=20, min_periods=20)
        price = 100.0
        for _ in range(50):
            price *= math.exp(rng.normal(0, 0.01))  # 1% vol
            f_hi.update(price)
        hi_vol_h = f_hi.current_h
        assert hi_vol_h is not None
        assert hi_vol_h > low_vol_h * 3  # ~10x difference in σ

    def test_adaptive_returns_none_before_warmup(self) -> None:
        f = CUSUMFilter(h=None, k=2.5, ewma_halflife=20, min_periods=30)
        for _ in range(5):
            event = f.update(100.0)
            assert event == 0
        assert f.current_h is None

    def test_first_update_cannot_fire(self) -> None:
        f = CUSUMFilter(h=0.0001)
        assert f.update(100.0) == 0

    def test_no_lookahead_causal(self) -> None:
        """Repeated replay produces identical events."""
        f1 = CUSUMFilter(h=0.005)
        prices = [100, 100.1, 100.3, 100.5, 101, 99.5, 99, 98]
        events1 = [f1.update(p) for p in prices]
        f2 = CUSUMFilter(h=0.005)
        events2 = [f2.update(p) for p in prices]
        assert events1 == events2

    def test_reset_clears_all_state(self) -> None:
        f = CUSUMFilter(h=0.001)
        for p in [100, 101, 102, 103]:
            f.update(p)
        f.reset()
        assert f.s_pos == 0.0
        assert f.s_neg == 0.0
        assert f.update(100.0) == 0

    def test_h_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            CUSUMFilter(h=-0.01)

    def test_k_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            CUSUMFilter(h=None, k=-1.0)
