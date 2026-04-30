"""Microstructure features (AFML Ch. 19).

Bar-level proxies for order-flow toxicity and liquidity — VPIN-lite via tick-rule
classification, Amihud illiquidity, and a Kyle's-lambda OLS proxy. Full VPIN and
Kyle's λ require tick-level trade data; these bar-level versions are documented
approximations that trade precision for portability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MicrostructureFeatures:
    vpin_lite: float          # [0, 1], >0.6 = buy imbalance, <0.4 = sell imbalance
    amihud_20: float          # rolling-20 Amihud illiquidity (abs_ret / dollar_vol)
    amihud_z: float           # z-score of amihud_20 vs its own rolling mean/std
    kyle_lambda_20: float     # abs_return ~ signed_volume OLS slope
    kyle_lambda_z: float
    buy_pressure: str         # "heavy_buy" | "balanced" | "heavy_sell"


def _signed_ret(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].to_numpy(dtype=float)
    return np.sign(np.diff(close, prepend=close[0]))


def _vpin_lite(df: pd.DataFrame, bucket: int = 20) -> float:
    """Buy-volume / total-volume over the most recent `bucket` bars.

    Uses the tick rule (sign of close change) to classify each bar as buy or
    sell. This is the AFML bar-level proxy — a true VPIN would need signed
    trade-level volume buckets.
    """
    if len(df) < bucket + 1:
        return 0.5
    recent = df.iloc[-bucket:]
    closes = recent["close"].to_numpy(dtype=float)
    prev_close = float(df["close"].iloc[-bucket - 1])
    vols = recent["volume"].to_numpy(dtype=float)
    all_closes = np.concatenate([[prev_close], closes])
    signs = np.sign(np.diff(all_closes))
    buy_vol = float(np.sum(vols[signs > 0]))
    sell_vol = float(np.sum(vols[signs < 0]))
    total = buy_vol + sell_vol
    if total <= 0:
        return 0.5
    return buy_vol / total


def _amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> float:
    if len(df) < window + 1:
        return float("nan")
    recent = df.iloc[-window - 1 :].copy()
    closes = recent["close"].to_numpy(dtype=float)
    vols = recent["volume"].to_numpy(dtype=float)
    rets = np.diff(closes) / closes[:-1]
    dollar_vol = closes[1:] * vols[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dollar_vol > 0, np.abs(rets) / np.where(dollar_vol > 0, dollar_vol, 1.0), np.nan)
    illiq = ratio
    if np.all(np.isnan(illiq)):
        return float("nan")
    return float(np.nanmean(illiq))


def _kyle_lambda(df: pd.DataFrame, window: int = 20) -> float:
    """OLS slope of |log_ret_t| ~ signed_volume_t using the last `window` bars."""
    if len(df) < window + 1:
        return 0.0
    recent = df.iloc[-window - 1 :]
    closes = recent["close"].to_numpy(dtype=float)
    vols = recent["volume"].to_numpy(dtype=float)
    rets = np.diff(np.log(np.maximum(closes, 1e-12)))
    signed_vol = np.sign(rets) * vols[1:]
    y = np.abs(rets)
    x = signed_vol
    if np.all(x == 0) or np.std(x) < 1e-12:
        return 0.0
    denom = float(np.sum(x * x))
    if denom <= 0:
        return 0.0
    return float(np.sum(x * y) / denom)


def _rolling_z(series: np.ndarray, window: int) -> float:
    if len(series) < window:
        return 0.0
    recent = series[-window:]
    m = np.nanmean(recent)
    s = np.nanstd(recent)
    if s <= 0 or np.isnan(s):
        return 0.0
    return float((series[-1] - m) / s)


def compute_microstructure(
    df: pd.DataFrame,
    vpin_bucket: int = 20,
    amihud_window: int = 20,
    kyle_window: int = 20,
    z_window: int = 50,
) -> MicrostructureFeatures | None:
    """Compute microstructure feature bundle from an OHLCV DataFrame.

    Returns None if there are too few bars for a stable estimate.
    """
    if df is None or len(df) < max(amihud_window, kyle_window, vpin_bucket) + 5:
        return None

    vpin = _vpin_lite(df, bucket=vpin_bucket)
    amihud = _amihud_illiquidity(df, window=amihud_window)
    kyle = _kyle_lambda(df, window=kyle_window)

    # Build rolling time series of amihud / kyle to compute z-scores.
    amihud_series = []
    kyle_series = []
    start = max(amihud_window, kyle_window) + 1
    for i in range(start, len(df) + 1):
        window_df = df.iloc[:i]
        amihud_series.append(_amihud_illiquidity(window_df, window=amihud_window))
        kyle_series.append(_kyle_lambda(window_df, window=kyle_window))
    amihud_arr = np.array(amihud_series, dtype=float)
    kyle_arr = np.array(kyle_series, dtype=float)
    amihud_z = _rolling_z(amihud_arr, window=z_window) if len(amihud_arr) else 0.0
    kyle_z = _rolling_z(kyle_arr, window=z_window) if len(kyle_arr) else 0.0

    if vpin >= 0.60:
        pressure = "heavy_buy"
    elif vpin <= 0.40:
        pressure = "heavy_sell"
    else:
        pressure = "balanced"

    return MicrostructureFeatures(
        vpin_lite=round(vpin, 3),
        amihud_20=float(amihud) if np.isfinite(amihud) else 0.0,
        amihud_z=round(amihud_z, 3),
        kyle_lambda_20=round(kyle, 9),
        kyle_lambda_z=round(kyle_z, 3),
        buy_pressure=pressure,
    )
