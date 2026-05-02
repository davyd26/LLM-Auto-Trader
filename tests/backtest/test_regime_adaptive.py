"""Tests for the regime_overrides feature in run_baseline_fast.

When regime_overrides is None: behavior identical to the prior engine
(verified by parity with a fixed config).

When regime_overrides is set: per-bar regime label drives size_pct and
vol_target_pct overrides, leaving everything else identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.baseline_strategy import BaselineConfig
from src.backtest.fast_baseline import run_baseline_fast
from src.backtest.fill_model import FillConfig


def _bars_from_closes(closes: list[float], start_ms: int = 1_700_000_000_000) -> pd.DataFrame:
    """Synthesize OHLCV with open=close[i-1], high=max, low=min."""
    rows = []
    week_ms = 7 * 86_400_000
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i > 0 else c
        bar_high = max(prev, c) * 1.005
        bar_low = min(prev, c) * 0.995
        rows.append({
            "ts_ms": start_ms + i * week_ms,
            "open": prev, "high": bar_high, "low": bar_low, "close": c, "volume": 1_000.0,
        })
    return pd.DataFrame(rows)


def _config(**kwargs) -> BaselineConfig:
    base = dict(
        momentum_lookback_bars=4,
        drawdown_hedge_threshold=0.20,
        long_size_usd_pct=1.0,
        primary_symbol="BTC/USDT:USDT",
        hedge_symbol="PAXG/USDT",
    )
    base.update(kwargs)
    return BaselineConfig(**base)


_FILL = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0, funding_rate_per_day=0.0)


class TestRegimeOverridesParity:
    def test_none_overrides_matches_fixed_config(self) -> None:
        """When regime_overrides is None, output is byte-for-byte identical
        to the engine without it."""
        rng = np.random.default_rng(42)
        closes = (100.0 * np.exp(np.cumsum(rng.normal(0.005, 0.04, 60)))).tolist()
        bars = _bars_from_closes(closes)

        a = run_baseline_fast(
            primary_bars=bars, hedge_bars=None, config=_config(),
            fill_config=_FILL, initial_equity=2000.0,
        )
        b = run_baseline_fast(
            primary_bars=bars, hedge_bars=None, config=_config(),
            fill_config=_FILL, initial_equity=2000.0,
            regime_overrides=None,
        )
        assert a.final_equity == pytest.approx(b.final_equity, rel=1e-12)
        assert len(a.trades) == len(b.trades)
        for t1, t2 in zip(a.trades, b.trades):
            assert t1.pnl == pytest.approx(t2.pnl, rel=1e-12)


class TestRegimeOverridesEffect:
    def test_crisis_zero_size_blocks_new_longs(self) -> None:
        """If the override forces long_size_usd_pct=0 in every regime, no
        positions should ever open."""
        # Steady uptrend → momentum signal would normally trigger longs.
        closes = [100.0 * (1.01 ** i) for i in range(40)]
        bars = _bars_from_closes(closes)
        result = run_baseline_fast(
            primary_bars=bars, hedge_bars=None, config=_config(),
            fill_config=_FILL, initial_equity=2000.0,
            regime_overrides={
                "calm":     {"long_size_usd_pct": 0.0},
                "elevated": {"long_size_usd_pct": 0.0},
                "crisis":   {"long_size_usd_pct": 0.0},
                "unknown":  {"long_size_usd_pct": 0.0},
            },
        )
        assert len(result.trades) == 0
        assert result.final_equity == pytest.approx(2000.0, rel=1e-9)

    def test_calm_full_size_matches_fixed_full_size(self) -> None:
        """When all regimes resolve to the same size as fixed config,
        the regime-aware engine matches the fixed engine."""
        rng = np.random.default_rng(7)
        closes = (100.0 * np.exp(np.cumsum(rng.normal(0.003, 0.02, 80)))).tolist()
        bars = _bars_from_closes(closes)

        a = run_baseline_fast(
            primary_bars=bars, hedge_bars=None,
            config=_config(long_size_usd_pct=2.0),
            fill_config=_FILL, initial_equity=2000.0,
        )
        b = run_baseline_fast(
            primary_bars=bars, hedge_bars=None,
            config=_config(long_size_usd_pct=2.0),
            fill_config=_FILL, initial_equity=2000.0,
            regime_overrides={
                "calm":     {"long_size_usd_pct": 2.0},
                "elevated": {"long_size_usd_pct": 2.0},
                "crisis":   {"long_size_usd_pct": 2.0},
                "unknown":  {"long_size_usd_pct": 2.0},
            },
        )
        # Identical size in every regime → identical equity curve.
        assert a.final_equity == pytest.approx(b.final_equity, rel=1e-12)


class TestRegimeOverridesValidates:
    def test_unknown_regime_falls_back_to_config_defaults(self) -> None:
        """A regime label not in the overrides dict uses the config's
        own size_pct (no error)."""
        rng = np.random.default_rng(0)
        closes = (100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 40)))).tolist()
        bars = _bars_from_closes(closes)
        result = run_baseline_fast(
            primary_bars=bars, hedge_bars=None,
            config=_config(long_size_usd_pct=1.0),
            fill_config=_FILL, initial_equity=2000.0,
            regime_overrides={"crisis": {"long_size_usd_pct": 0.0}},  # no calm/elevated/unknown
        )
        # Should not raise, should produce a valid equity curve
        assert result.final_equity > 0
        assert isinstance(result.equity_curve, list)
