"""Portfolio-level volatility targeting (Phase 3 §3B.11).

Scales total exposure so the portfolio's realized vol tracks a target. When
realized vol > target, exposure is reduced; when realized < target, exposure
can grow back toward the base cap.
"""

from __future__ import annotations

import math

import numpy as np


def realized_portfolio_vol(
    daily_returns: np.ndarray,
    annualize: bool = False,
) -> float:
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    vol = float(np.std(arr, ddof=1))
    if annualize:
        vol *= math.sqrt(252)
    return vol


def compute_vol_target_multiplier(
    current_portfolio_vol: float,
    target_portfolio_vol: float,
    min_mult: float = 0.25,
    max_mult: float = 1.50,
) -> float:
    """Return an exposure multiplier so `current_vol * mult ≈ target_vol`.

    When current_vol < target_vol, multiplier up to `max_mult`.
    When current_vol > target_vol, multiplier down to `min_mult`.
    """
    if current_portfolio_vol <= 0 or target_portfolio_vol <= 0:
        return 1.0
    raw = target_portfolio_vol / current_portfolio_vol
    return float(max(min_mult, min(max_mult, raw)))
