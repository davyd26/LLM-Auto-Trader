"""Block-bootstrap confidence intervals for backtest statistics.

Parametric PSR/DSR assume i.i.d. returns. Real returns have serial
correlation (positions persist; volatility clusters). Block bootstrap
preserves short-range dependence by resampling contiguous blocks, giving
honest standard errors and CIs that reflect the actual data-generating
process.

References:
    Politis & Romano (1994), "The Stationary Bootstrap"
    Lahiri (2003), "Resampling Methods for Dependent Data"
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from src.analysis.backtest_stats import sharpe_annualized


def block_bootstrap_sample(
    returns: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """One bootstrap resample of length len(returns) using contiguous blocks."""
    n = len(returns)
    if n == 0:
        return np.array([], dtype=float)
    block_size = min(max(1, block_size), n)
    n_blocks = math.ceil(n / block_size)
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    parts = [returns[s : s + block_size] for s in starts]
    sample = np.concatenate(parts)[:n]
    return sample


def block_bootstrap_stat(
    returns: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    block_size: int = 4,
    seed: int | None = None,
) -> tuple[float, float]:
    """Generic block-bootstrap (lo, hi) percentile CI for any 1-D statistic."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return math.nan, math.nan

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resampled = block_bootstrap_sample(arr, block_size, rng)
        samples[i] = float(statistic(resampled))

    alpha = 1.0 - confidence
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1.0 - alpha / 2))
    return lo, hi


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    periods_per_year: int,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    block_size: int = 4,
    seed: int | None = None,
) -> tuple[float, float]:
    """Block-bootstrap CI for the annualized Sharpe ratio."""
    def _sharpe(x: np.ndarray) -> float:
        return float(sharpe_annualized(x, periods_per_year=periods_per_year))

    return block_bootstrap_stat(
        returns,
        statistic=_sharpe,
        n_resamples=n_resamples,
        confidence=confidence,
        block_size=block_size,
        seed=seed,
    )
