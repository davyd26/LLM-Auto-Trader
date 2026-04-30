"""Statistical features for market analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StatisticalFeatures:
    """Statistical features for LLM context."""

    # Returns distribution
    returns_skew: float      # positive = right tail, negative = left tail
    returns_kurtosis: float  # >3 = fat tails
    returns_mean_pct: float

    # Volatility
    realized_vol_20: float      # 20-period realized vol (annualized)
    realized_vol_vs_avg: float  # ratio vs 60-period average vol
    vol_regime: str             # low / normal / high / extreme

    # Trend
    hurst_exponent: float       # >0.5 = trending, <0.5 = mean-reverting
    trend_classification: str   # trending / mean_reverting / random_walk

    # Price positioning
    price_z_score: float        # Z-score relative to 50-period mean

    # Phase 1 — AFML FFD (optional; only set when FFD enabled)
    ffd_d: float | None = None              # fractional differentiation order in use
    ffd_last: float | None = None           # latest FFD value (stationary)
    ffd_hurst: float | None = None          # Hurst on FFD series
    ffd_z_score: float | None = None        # z-score of latest FFD value

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if v is not None:
                d[k] = v
        return d


def compute_statistics(
    df: pd.DataFrame,
    ffd_close: pd.Series | None = None,
    ffd_d: float | None = None,
) -> StatisticalFeatures | None:
    """Compute statistical features from OHLCV data.

    Args:
        df: DataFrame with 'close' column, minimum 60 rows recommended.
        ffd_close: Optional fractionally-differentiated close series (Phase 1 §3.1).
                   When provided, Hurst + z-score are computed on the stationary
                   FFD series alongside the raw-price versions.
        ffd_d: The `d` in use (purely for surfacing to LLM context).
    """
    if df is None or len(df) < 30:
        return None

    close = df["close"].astype(float)
    returns = close.pct_change().dropna()

    if len(returns) < 20:
        return None

    # --- Returns distribution ---
    skew = float(returns.tail(20).skew())
    kurtosis = float(returns.tail(20).kurtosis())
    mean_ret = float(returns.tail(20).mean() * 100)

    # --- Realized volatility ---
    vol_20 = float(returns.tail(20).std() * np.sqrt(365 * 24 * 4) * 100)  # annualized, assuming 15m candles
    vol_60 = float(returns.tail(60).std() * np.sqrt(365 * 24 * 4) * 100) if len(returns) >= 60 else vol_20
    vol_ratio = vol_20 / vol_60 if vol_60 > 0 else 1.0

    if vol_ratio < 0.7:
        vol_regime = "low"
    elif vol_ratio < 1.3:
        vol_regime = "normal"
    elif vol_ratio < 2.0:
        vol_regime = "high"
    else:
        vol_regime = "extreme"

    # --- Hurst exponent (simplified R/S method) ---
    hurst = _hurst_exponent(close.values)
    if hurst > 0.55:
        trend_class = "trending"
    elif hurst < 0.45:
        trend_class = "mean_reverting"
    else:
        trend_class = "random_walk"

    # --- Z-score ---
    mean_50 = float(close.tail(50).mean())
    std_50 = float(close.tail(50).std())
    z_score = (float(close.iloc[-1]) - mean_50) / std_50 if std_50 > 0 else 0

    # --- Optional FFD features (Phase 1 §3.1) ---
    ffd_last_val: float | None = None
    ffd_hurst_val: float | None = None
    ffd_z: float | None = None
    if ffd_close is not None:
        ffd_valid = ffd_close.dropna()
        if len(ffd_valid) >= 30:
            ffd_last_val = float(ffd_valid.iloc[-1])
            ffd_arr = ffd_valid.to_numpy(dtype=float)
            ffd_hurst_val = _hurst_exponent(ffd_arr)
            mean_ffd = float(ffd_valid.tail(50).mean())
            std_ffd = float(ffd_valid.tail(50).std())
            ffd_z = (ffd_last_val - mean_ffd) / std_ffd if std_ffd > 0 else 0.0

    return StatisticalFeatures(
        returns_skew=round(skew, 3),
        returns_kurtosis=round(kurtosis, 3),
        returns_mean_pct=round(mean_ret, 4),
        realized_vol_20=round(vol_20, 1),
        realized_vol_vs_avg=round(vol_ratio, 2),
        vol_regime=vol_regime,
        hurst_exponent=round(hurst, 3),
        trend_classification=trend_class,
        price_z_score=round(z_score, 2),
        ffd_d=ffd_d if ffd_d is not None else None,
        ffd_last=round(ffd_last_val, 6) if ffd_last_val is not None else None,
        ffd_hurst=round(ffd_hurst_val, 3) if ffd_hurst_val is not None else None,
        ffd_z_score=round(ffd_z, 2) if ffd_z is not None else None,
    )


def compute_correlation_matrix(
    price_series: dict[str, pd.Series], window: int = 20
) -> dict[str, dict[str, float]]:
    """Compute rolling correlation matrix between assets.

    Args:
        price_series: {symbol: close_price_series}
        window: rolling window period

    Returns:
        Nested dict: {sym1: {sym2: correlation}}
    """
    if len(price_series) < 2:
        return {}

    returns_df = pd.DataFrame({
        sym: series.pct_change().dropna()
        for sym, series in price_series.items()
    })

    if len(returns_df) < window:
        return {}

    corr = returns_df.tail(window).corr()
    result = {}
    for sym1 in corr.index:
        result[sym1] = {}
        for sym2 in corr.columns:
            if sym1 != sym2:
                result[sym1][sym2] = round(float(corr.loc[sym1, sym2]), 3)
    return result


def _hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Estimate Hurst exponent using R/S analysis.

    H > 0.5: trending (persistent)
    H = 0.5: random walk
    H < 0.5: mean-reverting (anti-persistent)
    """
    n = len(series)
    if n < max_lag * 2:
        return 0.5  # insufficient data

    lags = range(2, min(max_lag + 1, n // 2))
    rs_values = []

    for lag in lags:
        subseries = series[:lag * (n // lag)]
        subseries = subseries.reshape(n // lag, lag)

        means = subseries.mean(axis=1, keepdims=True)
        deviations = subseries - means
        cumulative = np.cumsum(deviations, axis=1)

        r = cumulative.max(axis=1) - cumulative.min(axis=1)
        s = subseries.std(axis=1, ddof=1)

        valid = s > 0
        if not valid.any():
            continue

        rs = (r[valid] / s[valid]).mean()
        rs_values.append((np.log(lag), np.log(rs)))

    if len(rs_values) < 3:
        return 0.5

    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])

    # Linear regression: H = slope
    slope = float(np.polyfit(x, y, 1)[0])
    return max(0.0, min(1.0, slope))
