"""Fractional differentiation (AFML Ch. 5).

Fixed-width-window variant: produces a series that is stationary (ADF-passing)
but retains long memory of the original. No look-ahead — the weight vector only
applies to past values. `d` is meant to be chosen ONCE at startup from a warm-up
window via `find_min_d`, then held constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_ffd_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """Fixed-width-window FFD weights (AFML Ch. 5 §5.4).

    Recurrence: w[k] = -w[k-1] * (d - k + 1) / k, starting at w[0]=1.
    Truncate once |w[k]| < threshold.
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights, dtype=float)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Fractionally-differentiated series with fixed-width window.

    Values before the window is fully populated are NaN — this is by design so
    downstream consumers cannot accidentally use partially-weighted outputs.
    """
    if len(series) == 0:
        return series.copy()

    w = get_ffd_weights(d, threshold)
    width = len(w)
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)

    if len(values) >= width:
        # Reverse weights so dot-product aligns with the trailing window.
        w_rev = w[::-1]
        for i in range(width - 1, len(values)):
            window = values[i - width + 1 : i + 1]
            if np.any(np.isnan(window)):
                continue
            out[i] = float(np.dot(w_rev, window))

    return pd.Series(out, index=series.index, name=series.name)


def find_min_d(
    series: pd.Series,
    max_d: float = 1.0,
    step: float = 0.1,
    adf_pvalue: float = 0.05,
    threshold: float = 1e-4,
) -> float:
    """Smallest `d ∈ [0, max_d]` whose FFD output passes ADF at `adf_pvalue`.

    Walks a coarse grid from 0 upward; returns the first `d` whose series is
    stationary. Returns `max_d` if none passes — caller can fall back to `d=1`.
    """
    from statsmodels.tsa.stattools import adfuller

    if len(series) < 50:
        return 0.0

    clean = series.dropna()
    if clean.empty:
        return 0.0

    # Early-out: if raw series is already stationary, d=0.
    try:
        p_raw = adfuller(clean, autolag="AIC")[1]
    except Exception:
        p_raw = 1.0
    if p_raw < adf_pvalue:
        return 0.0

    d_values = np.arange(step, max_d + step / 2, step)
    for d in d_values:
        diffed = frac_diff_ffd(series, d=float(d), threshold=threshold).dropna()
        if len(diffed) < 30:
            continue
        try:
            p = adfuller(diffed, autolag="AIC")[1]
        except Exception:
            continue
        if p < adf_pvalue:
            return float(d)

    return float(max_d)
