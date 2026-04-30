"""Volatility/trend regime tagger for backtest report slicing.

A simpler, closes-only classifier than ``src.analysis.regime`` (which uses
full OHLCV + ADX/ATR/EMA). This module tags each observation as ``calm``,
``elevated``, ``crisis``, or ``unknown`` for regime-conditional analysis in
backtest reports.

Signals:
    - Trend: price relative to long SMA
    - Vol ratio: short-window realized vol / long-window realized vol
    - Drawdown from running peak

Rules:
    - crisis:   drawdown ≥ 15% (or drawdown ≥ 10% AND vol_ratio ≥ 2.0)
    - elevated: drawdown ≥ 5% OR vol_ratio ≥ 1.5
    - elevated: price below SMA AND drawdown ≥ 3%
    - calm:     otherwise

Drawdown alone is sufficient for crisis in crypto context — the vol_ratio
signal is unreliable when the long window already contains the stress event.
"""

from __future__ import annotations

import math

import numpy as np


def _running_drawdown(closes: np.ndarray) -> float:
    if closes.size == 0:
        return 0.0
    peak = np.maximum.accumulate(closes)
    if peak[-1] <= 0:
        return 0.0
    return float((peak[-1] - closes[-1]) / peak[-1])


def _realized_vol(closes: np.ndarray, window: int) -> float:
    if len(closes) < window + 1:
        return math.nan
    log_rets = np.diff(np.log(closes[-(window + 1):]))
    if len(log_rets) < 2:
        return math.nan
    return float(np.std(log_rets, ddof=1))


def classify_regime(
    closes: np.ndarray,
    sma_window: int = 28,
    vol_short: int = 4,
    vol_long: int = 28,
) -> str:
    """Classify the current regime from a closes-only series."""
    if sma_window <= 0 or vol_short <= 0 or vol_long <= 0:
        raise ValueError("windows must be positive")
    if vol_short >= vol_long:
        raise ValueError("vol_short must be < vol_long")

    arr = np.asarray(closes, dtype=float)
    if arr.size < sma_window or np.any(np.isnan(arr)):
        return "unknown"

    sma = float(np.mean(arr[-sma_window:]))
    last = float(arr[-1])
    above_sma = last >= sma

    vol_s = _realized_vol(arr, vol_short)
    vol_l = _realized_vol(arr, vol_long)
    vol_ratio = vol_s / vol_l if (vol_l and not math.isnan(vol_l) and vol_l > 0) else 1.0
    if math.isnan(vol_ratio):
        vol_ratio = 1.0

    dd = _running_drawdown(arr)

    if dd >= 0.15 or (dd >= 0.10 and vol_ratio >= 2.0):
        return "crisis"
    if dd >= 0.05 or vol_ratio >= 1.5:
        return "elevated"
    if not above_sma and dd >= 0.03:
        return "elevated"
    return "calm"


def classify_regime_series(
    closes: np.ndarray,
    sma_window: int = 28,
    vol_short: int = 4,
    vol_long: int = 28,
) -> list[str]:
    """Classify the regime at every observation along the closes array."""
    arr = np.asarray(closes, dtype=float)
    out: list[str] = []
    for i in range(len(arr)):
        if i + 1 < sma_window:
            out.append("unknown")
            continue
        out.append(
            classify_regime(
                arr[: i + 1],
                sma_window=sma_window,
                vol_short=vol_short,
                vol_long=vol_long,
            )
        )
    return out
