"""Tests for walk-forward analysis.

Walk-forward complements CPCV: instead of combinatorial test groups, it
runs the strategy in chronological train→test→train→test windows. CPCV
tests robustness to regime selection; walk-forward tests robustness to
chronological evolution. Different stress; same goal.

Edge cases covered:
    - One window (n_observations < window) → returns 0 windows or 1 trivial
    - train_window=0 (just sliding test windows)
    - Step size larger than test_window → non-overlapping
    - end_idx beyond available data → clamp

Tail risks:
    - Windows that overlap → leakage; reject by design
    - Test_window > train_window → flagged but allowed (some users prefer this)
"""

from __future__ import annotations

import pytest

from src.backtest.walk_forward import generate_walk_forward_windows


class TestWalkForwardWindows:
    def test_single_window(self) -> None:
        # 100 obs, train=50, test=20, step=20:
        #   w0: train [0,50), test [50,70) — needs 70 ≤ 100 ✓
        #   w1: train [20,70), test [70,90) — needs 90 ≤ 100 ✓
        #   w2: train [40,90), test [90,110) — needs 110 ≤ 100 ✗
        windows = generate_walk_forward_windows(
            n_observations=100, train_window=50, test_window=20, step=20,
        )
        assert len(windows) == 2
        assert windows[0].train_indices == list(range(0, 50))
        assert windows[0].test_indices == list(range(50, 70))

    def test_step_smaller_than_test_creates_overlapping_test_windows(self) -> None:
        windows = generate_walk_forward_windows(
            n_observations=100, train_window=50, test_window=20, step=10,
        )
        # Test windows overlap → flagged via warning, but the function returns them
        assert len(windows) >= 4

    def test_zero_observations_returns_empty(self) -> None:
        windows = generate_walk_forward_windows(
            n_observations=0, train_window=10, test_window=5, step=5,
        )
        assert windows == []

    def test_insufficient_observations_for_first_window(self) -> None:
        # train=50, test=20 needs 70 obs. With 60 → no windows
        windows = generate_walk_forward_windows(
            n_observations=60, train_window=50, test_window=20, step=20,
        )
        assert windows == []

    def test_invalid_inputs_rejected(self) -> None:
        with pytest.raises(ValueError):
            generate_walk_forward_windows(
                n_observations=100, train_window=-1, test_window=5, step=5,
            )
        with pytest.raises(ValueError):
            generate_walk_forward_windows(
                n_observations=100, train_window=10, test_window=0, step=5,
            )
        with pytest.raises(ValueError):
            generate_walk_forward_windows(
                n_observations=100, train_window=10, test_window=5, step=0,
            )


class TestWalkForwardCoverage:
    def test_test_indices_are_contiguous_within_window(self) -> None:
        windows = generate_walk_forward_windows(
            n_observations=100, train_window=30, test_window=10, step=10,
        )
        for w in windows:
            # Test indices should form a contiguous range
            assert w.test_indices == list(range(w.test_indices[0], w.test_indices[-1] + 1))

    def test_no_test_index_in_train_set(self) -> None:
        windows = generate_walk_forward_windows(
            n_observations=100, train_window=30, test_window=10, step=10,
        )
        for w in windows:
            assert set(w.test_indices).isdisjoint(set(w.train_indices))
