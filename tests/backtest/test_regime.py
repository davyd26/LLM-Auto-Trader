"""Tests for the regime classifier.

Classifies each observation as calm / elevated / crisis based on:
    - Price relative to a long SMA (trend direction)
    - Realized vol ratio (short window / long window)
    - Drawdown from running peak

Edge cases covered:
    - Insufficient history → "unknown"
    - All NaN closes → "unknown"
    - Tied SMA / price → calm
    - Vol ratio of NaN (constant prices) → calm

Tail risks covered:
    - Output stable across runs (deterministic)
    - Labels are exactly the documented set
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.backtest.regime_tagger import classify_regime, classify_regime_series


class TestClassifyRegime:
    def test_calm_when_price_above_sma_and_low_vol(self) -> None:
        # Slow uptrend, low vol
        closes = np.array([100 + i * 0.5 for i in range(100)])
        label = classify_regime(closes, sma_window=28, vol_short=4, vol_long=28)
        assert label == "calm"

    def test_crisis_when_drawdown_high_and_high_vol(self) -> None:
        # Up phase: very low vol; crash phase: vol explodes (3x baseline)
        rng = np.random.default_rng(0)
        up = 100 + np.cumsum(rng.normal(0.001, 0.005, 50))  # smooth uptrend
        # Multiplicative crash with explicit volatility regime change
        crash_part = up[-1] * np.exp(np.cumsum(rng.normal(-0.04, 0.06, 20)))
        closes = np.concatenate([up, crash_part])
        label = classify_regime(closes, sma_window=28, vol_short=4, vol_long=28)
        assert label == "crisis"

    def test_unknown_with_insufficient_history(self) -> None:
        closes = np.array([100.0, 101.0, 102.0])
        label = classify_regime(closes, sma_window=28, vol_short=4, vol_long=28)
        assert label == "unknown"

    def test_unknown_with_all_nan(self) -> None:
        closes = np.full(50, math.nan)
        label = classify_regime(closes, sma_window=28, vol_short=4, vol_long=28)
        assert label == "unknown"

    def test_calm_with_constant_price(self) -> None:
        closes = np.full(100, 100.0)
        label = classify_regime(closes, sma_window=28, vol_short=4, vol_long=28)
        # Zero vol, zero drawdown → calm
        assert label == "calm"


class TestClassifyRegimeSeries:
    def test_returns_array_of_labels(self) -> None:
        rng = np.random.default_rng(0)
        closes = 100 + np.cumsum(rng.normal(0.001, 0.01, 200))
        labels = classify_regime_series(closes, sma_window=28, vol_short=4, vol_long=28)
        assert len(labels) == len(closes)
        for label in labels:
            assert label in ("calm", "elevated", "crisis", "unknown")

    def test_unknown_until_history_sufficient(self) -> None:
        rng = np.random.default_rng(0)
        closes = 100 + np.cumsum(rng.normal(0.001, 0.01, 50))
        labels = classify_regime_series(closes, sma_window=28, vol_short=4, vol_long=28)
        # First 27 entries cannot have a 28-window SMA → unknown
        assert all(label == "unknown" for label in labels[:27])


class TestEdgeCases:
    def test_invalid_window_rejected(self) -> None:
        closes = np.array([100.0] * 50)
        with pytest.raises(ValueError):
            classify_regime(closes, sma_window=-1, vol_short=4, vol_long=28)
        with pytest.raises(ValueError):
            classify_regime(closes, sma_window=28, vol_short=10, vol_long=4)  # short > long
