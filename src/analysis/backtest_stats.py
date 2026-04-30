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


# ----------------------------------------------------------------------------
# Drawdown and ratio statistics
# ----------------------------------------------------------------------------

def max_drawdown(equity_curve: np.ndarray) -> tuple[float, int, int]:
    """Maximum peak-to-trough drawdown as a positive fraction of peak.

    Returns ``(drawdown, peak_index, trough_index)``. Equity must be
    non-negative; passing negative values raises since that signals an
    instrumentation bug rather than a legitimate state.
    """
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0 or arr.size == 1:
        return 0.0, 0, 0
    if np.any(arr < 0):
        raise ValueError("equity_curve must be non-negative")

    running_max = np.maximum.accumulate(arr)
    # Avoid zero-division by clamping running_max where it is zero.
    safe_peak = np.where(running_max > 0, running_max, 1.0)
    drawdown = (running_max - arr) / safe_peak
    drawdown = np.where(running_max > 0, drawdown, 0.0)

    trough_idx = int(np.argmax(drawdown))
    peak_idx = int(np.argmax(arr[: trough_idx + 1])) if trough_idx > 0 else 0
    return float(drawdown[trough_idx]), peak_idx, trough_idx


def sortino_ratio(returns: np.ndarray, periods_per_year: int) -> float:
    """Sortino = mean / downside_dev, scaled by sqrt(periods_per_year)."""
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    downside = arr[arr < 0]
    if len(downside) == 0:
        return 0.0
    # Downside deviation: RMS of negative returns (López de Prado convention)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd <= 0:
        return 0.0
    mu = float(np.mean(arr))
    return (mu / dd) * math.sqrt(max(1, periods_per_year))


def calmar_ratio(equity_curve: np.ndarray, periods_per_year: int) -> float:
    """Calmar = annualized return / |max drawdown|."""
    arr = np.asarray(equity_curve, dtype=float)
    if len(arr) < 2:
        return 0.0
    total_return = (arr[-1] / arr[0]) - 1.0 if arr[0] > 0 else 0.0
    n_periods = len(arr) - 1
    if n_periods <= 0 or periods_per_year <= 0:
        return 0.0
    annual_return = (1.0 + total_return) ** (periods_per_year / n_periods) - 1.0 \
        if total_return > -1.0 else -1.0
    dd, _, _ = max_drawdown(arr)
    if dd <= 0:
        # No drawdown — Calmar undefined (return convention)
        return 0.0
    sign = 1.0 if total_return >= 0 else -1.0
    return sign * abs(annual_return) / dd


# ----------------------------------------------------------------------------
# Trade-level statistics
# ----------------------------------------------------------------------------

def hit_rate(trades: list[dict]) -> float:
    """Fraction of trades with strictly positive P&L (out of all trades).

    A zero-PnL trade counts as "not a win" — it sits in the denominator but
    not the numerator. This matches the trader-intuitive read of hit-rate as
    "how often did I make money out of all the times I traded".
    """
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
    return wins / len(trades)


def profit_factor(trades: list[dict]) -> float:
    """Gross profit / |gross loss|. Inf if no losses; NaN if no trades."""
    if not trades:
        return math.nan
    gross_profit = sum(t.get("pnl", 0.0) for t in trades if t.get("pnl", 0.0) > 0)
    gross_loss = sum(-t.get("pnl", 0.0) for t in trades if t.get("pnl", 0.0) < 0)
    if gross_loss <= 0:
        if gross_profit > 0:
            return math.inf
        return 0.0
    return gross_profit / gross_loss


def expectancy(trades: list[dict]) -> float:
    """Average per-trade P&L."""
    if not trades:
        return 0.0
    return sum(t.get("pnl", 0.0) for t in trades) / len(trades)


def mae_mfe(trade: dict) -> tuple[float, float]:
    """Maximum Adverse / Favorable Excursion as fractions of entry price.

    Required keys: ``side``, ``entry_price``, ``high``, ``low``.
    """
    side = trade["side"]
    entry = trade["entry_price"]
    high = trade["high"]
    low = trade["low"]
    if entry <= 0:
        raise ValueError("entry_price must be positive")
    if side == "long":
        mae = (entry - low) / entry
        mfe = (high - entry) / entry
    elif side == "short":
        mae = (high - entry) / entry
        mfe = (entry - low) / entry
    else:
        raise ValueError(f"unknown side {side!r}")
    return float(mae), float(mfe)


# ----------------------------------------------------------------------------
# VaR / CVaR
# ----------------------------------------------------------------------------

def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR at the given confidence (0.95 → 5% tail)."""
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return math.nan
    tail_quantile = 1.0 - confidence
    return float(np.quantile(arr, tail_quantile))


def cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall) — mean of returns at or below VaR."""
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return math.nan
    var = var_historical(arr, confidence=confidence)
    tail = arr[arr <= var]
    if tail.size == 0:
        return float(var)
    return float(np.mean(tail))


# ----------------------------------------------------------------------------
# Regime-conditional slicing
# ----------------------------------------------------------------------------

def regime_conditional_stats(
    returns: np.ndarray, regime_labels: np.ndarray
) -> dict[str, dict[str, float]]:
    """Slice returns by regime label and report basic moments per regime."""
    arr = np.asarray(returns, dtype=float)
    labels = np.asarray(regime_labels)
    if arr.shape != labels.shape:
        raise ValueError("returns and regime_labels must have identical shape")
    out: dict[str, dict[str, float]] = {}
    for label in np.unique(labels):
        mask = labels == label
        slice_ = arr[mask]
        if slice_.size == 0:
            continue
        out[str(label)] = {
            "count": int(slice_.size),
            "mean": float(np.mean(slice_)),
            "std": float(np.std(slice_, ddof=1)) if slice_.size > 1 else 0.0,
        }
    return out
