"""Parametric VaR / CVaR / expected shortfall (Phase 3 §3B.10).

Inputs are realized returns; outputs are USDT-denominated worst-case losses
at chosen confidence levels. Used purely as evaluation / LLM context — no
model fitting, just sample statistics.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def var_parametric(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """Parametric (Gaussian) VaR as a POSITIVE number representing a loss.

    Returns the `1 - confidence` quantile of a Gaussian fit to `returns`,
    projected over `horizon_days` via sqrt-time scaling. Output is the
    magnitude of the loss (positive).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0:
        return 0.0
    z = norm.ppf(1.0 - confidence)
    sqrt_h = math.sqrt(max(1, horizon_days))
    var_return = mu * horizon_days + z * sigma * sqrt_h
    return float(max(0.0, -var_return))


def cvar_parametric(
    returns: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """Gaussian Expected Shortfall (CVaR): E[loss | loss > VaR]."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma <= 0:
        return 0.0
    alpha = 1.0 - confidence
    z = norm.ppf(alpha)
    sqrt_h = math.sqrt(max(1, horizon_days))
    # CVaR formula for Gaussian: CVaR = -(mu - sigma * phi(z)/alpha)
    cvar_return = mu * horizon_days - sigma * sqrt_h * norm.pdf(z) / alpha
    return float(max(0.0, -cvar_return))


def portfolio_var_usdt(
    equity: float,
    daily_returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Portfolio-level VaR in USDT from a history of daily returns."""
    var_frac = var_parametric(daily_returns, confidence=confidence, horizon_days=1)
    return float(equity * var_frac)


def portfolio_cvar_usdt(
    equity: float,
    daily_returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    cvar_frac = cvar_parametric(daily_returns, confidence=confidence, horizon_days=1)
    return float(equity * cvar_frac)
