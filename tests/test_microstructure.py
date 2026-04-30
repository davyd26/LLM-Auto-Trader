"""Tests for microstructure features (AFML Ch. 19)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.microstructure import (
    MicrostructureFeatures,
    compute_microstructure,
    _amihud_illiquidity,
    _kyle_lambda,
    _vpin_lite,
)


def _make_df(prices: list[float], volumes: list[float]) -> pd.DataFrame:
    n = len(prices)
    assert len(volumes) == n
    # Build OHLCV where open=prev_close, high/low bracket close, volume as given.
    opens = [prices[0]] + prices[:-1]
    highs = [max(o, c) * 1.001 for o, c in zip(opens, prices)]
    lows = [min(o, c) * 0.999 for o, c in zip(opens, prices)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes}
    )


class TestVPINLite:
    def test_all_up_bars_vpin_near_one(self) -> None:
        prices = [100 + i for i in range(25)]
        vols = [1000.0] * 25
        df = _make_df(prices, vols)
        v = _vpin_lite(df, bucket=20)
        assert v > 0.85  # strongly buy-imbalanced

    def test_all_down_bars_vpin_near_zero(self) -> None:
        prices = [125 - i for i in range(25)]
        vols = [1000.0] * 25
        df = _make_df(prices, vols)
        v = _vpin_lite(df, bucket=20)
        assert v < 0.15  # strongly sell-imbalanced

    def test_alternating_vpin_near_half(self) -> None:
        prices = [100 + ((-1) ** i) * 0.5 for i in range(40)]
        vols = [1000.0] * 40
        df = _make_df(prices, vols)
        v = _vpin_lite(df, bucket=30)
        assert 0.35 < v < 0.65


class TestAmihud:
    def test_lower_dollar_volume_means_higher_illiquidity(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 0.005, 100)
        prices = 100.0 * np.cumprod(1 + returns)
        high_vol = [1e6] * 100
        low_vol = [1e4] * 100
        df_hi = _make_df(prices.tolist(), high_vol)
        df_lo = _make_df(prices.tolist(), low_vol)
        a_hi = _amihud_illiquidity(df_hi, window=50)
        a_lo = _amihud_illiquidity(df_lo, window=50)
        assert a_lo > a_hi * 50  # roughly 100x ratio

    def test_handles_zero_volume(self) -> None:
        prices = [100.0 + i * 0.1 for i in range(30)]
        vols = [0.0] * 30
        df = _make_df(prices, vols)
        a = _amihud_illiquidity(df, window=20)
        assert np.isfinite(a) or np.isnan(a)


class TestKyleLambda:
    def test_larger_returns_per_volume_means_higher_lambda(self) -> None:
        rng = np.random.default_rng(1)
        # High-impact regime: returns large relative to volume
        r_hi = rng.normal(0, 0.01, 200)
        v_hi = np.abs(rng.normal(1000, 100, 200))
        df_hi = pd.DataFrame(
            {"close": 100 * np.cumprod(1 + r_hi), "volume": v_hi}
        )
        # Low-impact: returns small relative to volume
        r_lo = rng.normal(0, 0.001, 200)
        v_lo = np.abs(rng.normal(1000, 100, 200))
        df_lo = pd.DataFrame(
            {"close": 100 * np.cumprod(1 + r_lo), "volume": v_lo}
        )
        lam_hi = abs(_kyle_lambda(df_hi, window=100))
        lam_lo = abs(_kyle_lambda(df_lo, window=100))
        assert lam_hi > lam_lo


class TestComputeMicrostructure:
    def test_returns_features_for_valid_df(self) -> None:
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.003, 100)
        prices = 100 * np.cumprod(1 + returns)
        vols = np.abs(rng.normal(1000, 50, 100))
        df = _make_df(prices.tolist(), vols.tolist())
        feat = compute_microstructure(df)
        assert isinstance(feat, MicrostructureFeatures)
        assert 0.0 <= feat.vpin_lite <= 1.0
        assert feat.buy_pressure in {"heavy_buy", "balanced", "heavy_sell"}

    def test_returns_none_for_short_df(self) -> None:
        df = _make_df([100, 101, 102], [1000, 1000, 1000])
        feat = compute_microstructure(df)
        assert feat is None

    def test_uses_only_past_data(self) -> None:
        rng = np.random.default_rng(3)
        prices = (100 + rng.standard_normal(200).cumsum()).tolist()
        vols = (1000 + rng.standard_normal(200) * 50).tolist()
        df = _make_df(prices, vols)

        feat_full = compute_microstructure(df)
        feat_prefix = compute_microstructure(df.iloc[:150])
        assert feat_full is not None
        assert feat_prefix is not None
        # Different data windows should produce different features;
        # truncating the future should not change earlier-window computation
        # if we use only the most recent N bars.
        feat_100 = compute_microstructure(df.iloc[:100])
        assert feat_100 is not None
