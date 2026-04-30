"""Backtest statistics: Probabilistic & Deflated Sharpe (AFML Ch. 14).

- **PSR** (Probabilistic Sharpe Ratio): P(true SR > benchmark | observed SR, n,
  skew, kurtosis). Corrects for finite-sample + non-normal return distributions.
- **DSR** (Deflated Sharpe Ratio): PSR adjusted for the number of strategy
  trials run — defends against cherry-picking the best Sharpe from many tries.

Formulas follow Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

_EULER_MASCHERONI = 0.5772156649015329


def sharpe_annualized(returns: np.ndarray, periods_per_year: int) -> float:
    """Naive Sharpe = mean/std, scaled by sqrt(periods_per_year)."""
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    # Guard against floating-point artifacts on effectively-constant series.
    scale = max(abs(mu), 1.0) * 1e-12
    if sigma <= scale:
        return 0.0
    return (mu / sigma) * math.sqrt(max(1, periods_per_year))


def pnl_moments(returns: np.ndarray) -> dict[str, float]:
    """Mean, std, skewness, kurtosis (non-excess)."""
    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    if n < 2:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 3.0}
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0:
        return {"mean": mu, "std": 0.0, "skew": 0.0, "kurtosis": 3.0}
    z = (arr - mu) / sigma
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # non-excess
    return {"mean": mu, "std": sigma, "skew": skew, "kurtosis": kurt}


def probabilistic_sharpe_ratio(
    sr_hat: float,
    n: int,
    skew: float,
    kurt: float,
    sr_benchmark: float = 0.0,
) -> float:
    """PSR = P(true SR > benchmark) given observed sample statistics.

    Formula (Bailey & López de Prado 2014):
        PSR = Phi((sr_hat - sr_benchmark) * sqrt(n - 1) /
                  sqrt(1 - skew*sr_hat + ((kurt - 1)/4) * sr_hat^2))

    `kurt` is non-excess (kurtosis of normal = 3).
    """
    if n < 2:
        return 0.0
    denom_sq = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    if denom_sq <= 0:
        return 0.0
    denom = math.sqrt(denom_sq)
    z = (sr_hat - sr_benchmark) * math.sqrt(max(0, n - 1)) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(num_trials: int, sr_trials_var: float) -> float:
    """Expected maximum Sharpe across `num_trials` independent strategies.

    From Bailey & López de Prado (2014): E[max SR] ≈ sqrt(V) * ((1-γ)*Φ^{-1}(1 - 1/N)
    + γ * Φ^{-1}(1 - 1/(N*e))), where γ is the Euler-Mascheroni constant and V is
    the variance of Sharpe ratios across trials.
    """
    if num_trials <= 1:
        return 0.0
    v = max(0.0, float(sr_trials_var))
    if v == 0.0:
        return 0.0
    gamma = _EULER_MASCHERONI
    q1 = norm.ppf(1.0 - 1.0 / num_trials)
    q2 = norm.ppf(1.0 - 1.0 / (num_trials * math.e))
    return math.sqrt(v) * ((1.0 - gamma) * q1 + gamma * q2)


def deflated_sharpe_ratio(
    sr_hat: float,
    n: int,
    skew: float,
    kurt: float,
    num_trials: int,
    sr_trials_var: float,
) -> float:
    """DSR = PSR measured against the expected max SR over `num_trials`."""
    bench = expected_max_sharpe(num_trials, sr_trials_var)
    return probabilistic_sharpe_ratio(
        sr_hat=sr_hat, n=n, skew=skew, kurt=kurt, sr_benchmark=bench
    )
