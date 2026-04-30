"""Combinatorial Purged Cross-Validation (CPCV).

Reference: López de Prado, *Advances in Financial Machine Learning*, Ch. 12.
Bailey & López de Prado (2014), *The Probability of Backtest Overfitting*.

CPCV partitions T observations into N contiguous groups, then for each
combination of K test groups builds a backtest path using only those groups.
Train observations near test boundaries are *purged* (removed before each test
group) and observations immediately after each test group are *embargoed*
(removed after each test group), preventing serial-correlation leakage.

The output is a list of paths. Aggregating Sharpe / drawdown / PBO across
paths produces a *distribution* of strategy performance — far more honest than
a single in-sample number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class CPCVPath:
    """One CPCV path: which groups are test, and the resulting indices."""

    test_groups: tuple[int, ...]
    test_indices: list[int]
    train_indices: list[int]


def generate_paths(
    n_observations: int,
    n_groups: int,
    k_test: int,
    purge: int = 0,
    embargo: int = 0,
) -> list[CPCVPath]:
    """Build C(N, K) CPCV paths.

    The N groups are contiguous and as equal-sized as possible. ``purge``
    excludes the ``purge`` observations immediately *before* each test group
    from the train set; ``embargo`` excludes the ``embargo`` observations
    immediately *after*. Test indices are never purged.
    """
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2 for cross-validation")
    if not 1 <= k_test <= n_groups:
        raise ValueError(f"k_test must be in [1, {n_groups}], got {k_test}")
    if n_observations < n_groups:
        raise ValueError("n_observations must be >= n_groups (no empty groups)")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")

    group_ranges = _split_groups(n_observations, n_groups)
    paths: list[CPCVPath] = []
    for combo in combinations(range(n_groups), k_test):
        test_idx: list[int] = []
        forbidden: set[int] = set()
        for g in combo:
            start, end = group_ranges[g]  # end exclusive
            test_idx.extend(range(start, end))
            for j in range(max(0, start - purge), start):
                forbidden.add(j)
            for j in range(end, min(n_observations, end + embargo)):
                forbidden.add(j)
        forbidden.update(test_idx)
        train_idx = [i for i in range(n_observations) if i not in forbidden]
        paths.append(
            CPCVPath(
                test_groups=combo,
                test_indices=test_idx,
                train_indices=train_idx,
            )
        )
    return paths


def _split_groups(n_obs: int, n_groups: int) -> list[tuple[int, int]]:
    """Return ``n_groups`` contiguous (start, end_exclusive) ranges covering [0, n_obs)."""
    base = n_obs // n_groups
    remainder = n_obs % n_groups
    ranges = []
    cursor = 0
    for g in range(n_groups):
        size = base + (1 if g < remainder else 0)
        ranges.append((cursor, cursor + size))
        cursor += size
    return ranges


def summarize_paths(sharpes: np.ndarray) -> dict[str, float]:
    """Return median / p5 / p95 of a per-path Sharpe distribution."""
    arr = np.asarray(sharpes, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def compute_pbo(sharpes: np.ndarray) -> float:
    """Probability of Backtest Overfitting (Bailey & López de Prado, 2014).

    Input: ``sharpes`` of shape (n_paths, n_configs). Returns the probability
    that the in-sample best config has below-median out-of-sample rank.

    With 1 config the metric is undefined → returns NaN.
    With ≥2 configs: for each path, the IS best config is determined by its
    Sharpe within the path; OOS rank is computed across all *other* paths'
    Sharpes for that config. PBO = fraction of paths where IS best < median
    OOS rank.
    """
    sharpes = np.asarray(sharpes, dtype=float)
    if sharpes.ndim != 2:
        raise ValueError("sharpes must be 2D: (n_paths, n_configs)")

    n_paths, n_configs = sharpes.shape
    if n_configs < 2 or n_paths < 2:
        return math.nan

    bad = 0
    for i in range(n_paths):
        is_row = sharpes[i, :]
        oos_rows = np.delete(sharpes, i, axis=0)
        is_best = int(np.argmax(is_row))
        # Median Sharpe of the IS-best config across OOS paths
        oos_med_for_best = float(np.median(oos_rows[:, is_best]))
        # Median across all configs in OOS
        global_oos_median = float(np.median(oos_rows))
        if oos_med_for_best < global_oos_median:
            bad += 1
    return bad / n_paths
