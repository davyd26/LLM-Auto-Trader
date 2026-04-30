"""Log-relative dynamic baselines (Phase 2 §3A.2).

Expresses price features as `ln(price / MA)` so LLM reasoning is invariant to
absolute price levels. Computed from daily closes; intended to be cached for
the day rather than re-computed per tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Baselines:
    ma_365d: float
    ma_90d: float
    log_rel_1y: float
    log_rel_90d: float
    percentile_1y: float
    staleness_days: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def compute_baselines(daily_close: pd.Series, current_price: float) -> Baselines:
    """Compute rolling baselines from a daily-close series.

    If fewer than ~30 days are available, falls back gracefully using the
    available window (log_rel collapses toward zero).
    """
    if daily_close is None or daily_close.empty:
        return Baselines(
            ma_365d=current_price, ma_90d=current_price,
            log_rel_1y=0.0, log_rel_90d=0.0,
            percentile_1y=0.5, staleness_days=0.0,
        )

    arr = daily_close.astype(float).to_numpy()
    window_1y = arr[-365:] if len(arr) >= 365 else arr
    window_90d = arr[-90:] if len(arr) >= 90 else arr

    ma_365 = float(np.mean(window_1y)) if len(window_1y) else current_price
    ma_90 = float(np.mean(window_90d)) if len(window_90d) else current_price

    def _log_rel(price: float, ma: float) -> float:
        if ma <= 0 or price <= 0:
            return 0.0
        return math.log(price / ma)

    log_rel_1y = _log_rel(current_price, ma_365)
    log_rel_90d = _log_rel(current_price, ma_90)

    # Percentile of current_price in the 1-year window (tie-aware)
    if len(window_1y) > 1 and current_price > 0:
        less = float(np.mean(window_1y < current_price))
        eq = float(np.mean(window_1y == current_price))
        percentile = less + 0.5 * eq
    else:
        percentile = 0.5

    staleness = 0.0
    if isinstance(daily_close.index, pd.DatetimeIndex) and len(daily_close.index) > 0:
        last = daily_close.index[-1]
        # Use today's date in UTC for staleness; tz-agnostic.
        now = pd.Timestamp.utcnow().tz_localize(None) if last.tz is None else pd.Timestamp.utcnow()
        try:
            staleness = max(0.0, float((now - last).total_seconds() / 86400.0))
        except Exception:
            staleness = 0.0

    return Baselines(
        ma_365d=round(ma_365, 4),
        ma_90d=round(ma_90, 4),
        log_rel_1y=round(log_rel_1y, 4),
        log_rel_90d=round(log_rel_90d, 4),
        percentile_1y=round(percentile, 3),
        staleness_days=round(staleness, 2),
    )
