"""Tests for CPCV — Combinatorial Purged Cross-Validation (López de Prado AFML Ch. 12).

The harness partitions a sequence of T observations into N contiguous groups,
then for every C(N, K) selection of K test groups it constructs a "path" —
the indices the strategy would trade during those test groups. Train indices
are the complement, with observations near test boundaries purged
(``purge``) and observations immediately following test groups embargoed
(``embargo``) to prevent serial-correlation leakage.

The output is a list of paths. Aggregate Sharpe / drawdown / PBO across paths
gives a *distribution* of strategy performance instead of a single optimistic
in-sample number.

Edge cases covered:
    - C(N, K) combinatorics
    - K = N (single path, train empty)
    - K = 1 (N paths, each one group)
    - purge / embargo at sequence boundaries (don't underflow / overflow)
    - purge / embargo larger than group size
    - all observations covered exactly once across all paths' test_indices
    - reproducibility (same inputs → same path list)

Tail risks covered:
    - K > N rejected
    - K = 0 rejected
    - n_observations < n_groups rejected
    - purge or embargo eating entire train set still produces a valid path
    - PBO computation on degenerate input (1 path, 1 config)
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from src.backtest.cpcv import (
    CPCVPath,
    compute_pbo,
    generate_paths,
    summarize_paths,
)


class TestPathGenerationBasic:
    def test_single_path_when_k_equals_n(self) -> None:
        paths = generate_paths(n_observations=10, n_groups=2, k_test=2, purge=0, embargo=0)
        assert len(paths) == 1
        # All observations are test indices
        assert sorted(paths[0].test_indices) == list(range(10))
        assert paths[0].train_indices == []

    def test_n_paths_when_k_equals_one(self) -> None:
        # N=4, K=1 → 4 paths, each containing 1 group's worth of observations
        paths = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)
        assert len(paths) == 4
        for p in paths:
            assert len(p.test_indices) == 3

    def test_combinatorial_count(self) -> None:
        # N=8, K=2 → C(8,2) = 28 paths (the user's chosen config)
        paths = generate_paths(n_observations=104, n_groups=8, k_test=2, purge=0, embargo=0)
        assert len(paths) == math.comb(8, 2)

    def test_test_groups_attribute_records_choice(self) -> None:
        paths = generate_paths(n_observations=12, n_groups=4, k_test=2, purge=0, embargo=0)
        # Test groups should be all 2-combinations of {0,1,2,3}
        actual = sorted(p.test_groups for p in paths)
        expected = sorted(tuple(c) for c in combinations(range(4), 2))
        assert actual == expected


class TestPathCoverage:
    def test_every_observation_appears_in_some_test_path(self) -> None:
        # Without purge/embargo, every observation should be a test index in exactly C(N-1, K-1) paths
        paths = generate_paths(n_observations=12, n_groups=4, k_test=2, purge=0, embargo=0)
        counter = {i: 0 for i in range(12)}
        for p in paths:
            for idx in p.test_indices:
                counter[idx] += 1
        # Each obs should appear in C(3,1) = 3 paths
        assert all(v == 3 for v in counter.values())

    def test_train_and_test_partition_observations_when_no_purge_embargo(self) -> None:
        paths = generate_paths(n_observations=12, n_groups=4, k_test=2, purge=0, embargo=0)
        for p in paths:
            assert set(p.test_indices).isdisjoint(set(p.train_indices))
            assert set(p.test_indices) | set(p.train_indices) == set(range(12))


class TestPurgeAndEmbargo:
    def test_purge_removes_train_observations_before_test_groups(self) -> None:
        # 12 obs, 4 groups (3 each), K=1, purge=1
        # Test group = 1 → indices [3,4,5]; purge=1 → drop train index 2 (right before test)
        paths = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=1, embargo=0)
        # Find the path where test group is {1}
        path = next(p for p in paths if p.test_groups == (1,))
        assert 2 not in path.train_indices
        # Observations farther from the boundary remain in train
        assert 1 in path.train_indices
        assert 0 in path.train_indices

    def test_embargo_removes_train_observations_after_test_groups(self) -> None:
        # Test group = 1 → indices [3,4,5]; embargo=1 → drop train index 6 (right after)
        paths = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=1)
        path = next(p for p in paths if p.test_groups == (1,))
        assert 6 not in path.train_indices
        assert 7 in path.train_indices

    def test_purge_and_embargo_combined(self) -> None:
        paths = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=2, embargo=2)
        path = next(p for p in paths if p.test_groups == (1,))
        # Test indices [3,4,5]; purge=2 → drop 1,2; embargo=2 → drop 6,7
        for idx in (1, 2, 6, 7):
            assert idx not in path.train_indices
        assert 0 in path.train_indices
        assert 8 in path.train_indices

    def test_purge_at_sequence_start_does_not_underflow(self) -> None:
        # Test group = 0 (at start); purge would extend before index 0 — must clamp
        paths = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=5, embargo=0)
        path = next(p for p in paths if p.test_groups == (0,))
        # No train indices should be negative
        assert all(i >= 0 for i in path.train_indices)

    def test_embargo_at_sequence_end_does_not_overflow(self) -> None:
        paths = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=5)
        path = next(p for p in paths if p.test_groups == (3,))
        assert all(i < 12 for i in path.train_indices)

    def test_purge_and_embargo_can_eat_entire_train_set(self) -> None:
        # Aggressive purge/embargo: train set may be empty but path is still valid
        paths = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=10, embargo=10)
        for p in paths:
            assert len(p.test_indices) == 3  # test data preserved
            # Train may be empty
            assert all(0 <= i < 12 for i in p.train_indices)

    def test_purge_does_not_remove_test_indices(self) -> None:
        # Test indices are sacrosanct; purge only affects train
        paths = generate_paths(n_observations=12, n_groups=4, k_test=2, purge=10, embargo=10)
        for p in paths:
            # Each path's test indices are the union of K group ranges
            assert len(p.test_indices) == 6


class TestPathGenerationValidation:
    def test_k_greater_than_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="k_test"):
            generate_paths(n_observations=10, n_groups=2, k_test=3, purge=0, embargo=0)

    def test_k_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="k_test"):
            generate_paths(n_observations=10, n_groups=2, k_test=0, purge=0, embargo=0)

    def test_n_observations_less_than_n_groups_rejected(self) -> None:
        with pytest.raises(ValueError, match="observations"):
            generate_paths(n_observations=3, n_groups=4, k_test=1, purge=0, embargo=0)

    def test_negative_purge_rejected(self) -> None:
        with pytest.raises(ValueError):
            generate_paths(n_observations=10, n_groups=2, k_test=1, purge=-1, embargo=0)

    def test_negative_embargo_rejected(self) -> None:
        with pytest.raises(ValueError):
            generate_paths(n_observations=10, n_groups=2, k_test=1, purge=0, embargo=-1)

    def test_n_groups_one_rejected(self) -> None:
        # With only one group, CV is meaningless
        with pytest.raises(ValueError, match="n_groups"):
            generate_paths(n_observations=10, n_groups=1, k_test=1, purge=0, embargo=0)


class TestPathReproducibility:
    def test_same_inputs_produce_same_paths(self) -> None:
        a = generate_paths(n_observations=12, n_groups=4, k_test=2, purge=1, embargo=1)
        b = generate_paths(n_observations=12, n_groups=4, k_test=2, purge=1, embargo=1)
        assert [(p.test_groups, p.test_indices, p.train_indices) for p in a] == \
               [(p.test_groups, p.test_indices, p.train_indices) for p in b]


class TestUnevenGroups:
    def test_observations_not_divisible_by_n_groups(self) -> None:
        # 10 obs, 3 groups → group sizes 4, 3, 3 (or 3, 3, 4 depending on convention)
        paths = generate_paths(n_observations=10, n_groups=3, k_test=1, purge=0, embargo=0)
        all_obs: set[int] = set()
        for p in paths:
            all_obs.update(p.test_indices)
        assert all_obs == set(range(10)), "every observation must be a test idx in some path"


class TestSummarizePaths:
    def test_summarize_returns_distribution_stats(self) -> None:
        # Synthetic: 28 paths with normally-distributed Sharpes
        rng = np.random.default_rng(42)
        sharpes = rng.normal(loc=1.0, scale=0.5, size=28)
        s = summarize_paths(sharpes)
        assert "median" in s
        assert "p5" in s
        assert "p95" in s
        assert s["p5"] < s["median"] < s["p95"]
        assert s["median"] == pytest.approx(np.median(sharpes))

    def test_summarize_handles_constant_sharpes(self) -> None:
        sharpes = np.ones(10)
        s = summarize_paths(sharpes)
        assert s["median"] == 1.0
        assert s["p5"] == 1.0
        assert s["p95"] == 1.0


class TestPBO:
    def test_pbo_with_one_config_is_undefined_returns_nan(self) -> None:
        # PBO requires multiple configs to evaluate rank inversion; one config → NaN
        sharpes = np.array([[1.0], [0.5], [1.5]])  # 3 paths × 1 config
        pbo = compute_pbo(sharpes)
        assert math.isnan(pbo)

    def test_pbo_zero_when_best_config_consistent_across_paths(self) -> None:
        # Config 0 is always best across all paths → no overfit
        sharpes = np.array([
            [2.0, 1.0, 0.5],  # path 1: c0 > c1 > c2
            [3.0, 1.5, 0.8],  # path 2: c0 > c1 > c2
            [2.5, 1.2, 0.7],  # path 3: c0 > c1 > c2
            [2.2, 1.1, 0.6],  # path 4: c0 > c1 > c2
        ])
        pbo = compute_pbo(sharpes)
        assert pbo == 0.0  # in-sample best is always OOS best — no overfit

    def test_pbo_high_when_random_configs(self) -> None:
        # Random configs → in-sample best should sometimes be OOS poor
        rng = np.random.default_rng(0)
        sharpes = rng.normal(0, 1, size=(20, 30))  # 20 paths, 30 random configs
        pbo = compute_pbo(sharpes)
        # With pure noise, PBO should be near 0.5 (50% chance the IS best is below median OOS)
        assert 0.2 <= pbo <= 0.8

    def test_pbo_bounded_0_1(self) -> None:
        rng = np.random.default_rng(1)
        for shape in [(5, 3), (10, 5), (20, 10)]:
            sharpes = rng.normal(0, 1, size=shape)
            pbo = compute_pbo(sharpes)
            if not math.isnan(pbo):
                assert 0.0 <= pbo <= 1.0


class TestCPCVPath:
    def test_path_is_immutable_dataclass(self) -> None:
        # Frozen dataclass — can't mutate test_groups
        path = CPCVPath(test_groups=(0, 1), test_indices=[0, 1], train_indices=[2, 3])
        with pytest.raises((AttributeError, Exception)):
            path.test_groups = (1, 2)  # type: ignore[misc]
