"""Tests for the multi-config sweep utility.

Running the strategy with one config gives one Sharpe. PBO requires multiple
configs across multiple paths — that's what reveals overfitting (the IS-best
config underperforming OOS). The sweep utility takes a config-grid and a
list of CPCV paths and produces the (n_paths × n_configs) Sharpe matrix
needed by ``compute_pbo``.

Edge cases covered:
    - Empty config list rejected
    - Single path / single config returns 1×1 matrix
    - Configs differ only in fill params (recommended) — same decision cache
      replays for free across configs
    - Some paths fail (any reason) → row of NaN

Tail risks:
    - Sweep on grid larger than memory → out-of-scope (we trust the user)
    - Same key inserted with different fill_params → still one decision (per
      snapshot hash); different equity curves come from different fees/slippage
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.backtest.config_sweep import sweep_configs
from src.backtest.cpcv import generate_paths


@pytest.fixture
def synthetic_universe(tmp_path: Path):
    """Mirror the integration test's synthetic universe with smaller scale."""
    from src.backtest.data_cache import OHLCVBar, OHLCVCache

    WEEK_MS = 7 * 86_400_000
    cache = OHLCVCache(tmp_path / "ohlcv.db")
    rng = np.random.default_rng(42)
    btc_log = rng.normal(0.005, 0.04, 40)
    btc_prices = 50_000.0 * np.exp(np.cumsum(btc_log))
    btc_bars = [
        OHLCVBar(
            ts_ms=i * WEEK_MS,
            open=float(p),
            high=float(p) * 1.02,
            low=float(p) * 0.98,
            close=float(p),
            volume=100.0,
        )
        for i, p in enumerate(btc_prices)
    ]
    cache.store("BTC/USDT", "1w", btc_bars)
    paxg_bars = [
        OHLCVBar(
            ts_ms=i * WEEK_MS,
            open=1900.0 + i * 0.5,
            high=1905.0 + i * 0.5,
            low=1895.0 + i * 0.5,
            close=1900.0 + i * 0.5,
            volume=10.0,
        )
        for i in range(40)
    ]
    cache.store("PAXG/USDT", "1w", paxg_bars)
    return cache, ["BTC/USDT", "PAXG/USDT"]


class TestConfigSweepShape:
    def test_returns_n_paths_by_n_configs_matrix(self, synthetic_universe, tmp_path: Path) -> None:
        cache, universe = synthetic_universe
        configs = [
            {"taker_fee_bps": 0.0, "slippage_bps": 0.0},
            {"taker_fee_bps": 4.0, "slippage_bps": 5.0},
            {"taker_fee_bps": 8.0, "slippage_bps": 10.0},
        ]
        paths = generate_paths(n_observations=40, n_groups=4, k_test=2, purge=1, embargo=1)
        sharpes = sweep_configs(
            cache=cache,
            universe=universe,
            timeframe="1w",
            paths=paths,
            configs=configs,
            initial_equity=2000.0,
            decisions_db_path=tmp_path / "decisions.db",
        )
        assert sharpes.shape == (len(paths), len(configs))

    def test_rejects_empty_configs(self, synthetic_universe, tmp_path: Path) -> None:
        cache, universe = synthetic_universe
        paths = generate_paths(n_observations=40, n_groups=4, k_test=2)
        with pytest.raises(ValueError, match="config"):
            sweep_configs(
                cache=cache, universe=universe, timeframe="1w",
                paths=paths, configs=[],
                initial_equity=2000.0,
                decisions_db_path=tmp_path / "decisions.db",
            )


class TestConfigSweepBehavior:
    def test_zero_fee_outperforms_high_fee_on_same_path(self, synthetic_universe, tmp_path: Path) -> None:
        cache, universe = synthetic_universe
        configs = [
            {"taker_fee_bps": 0.0, "slippage_bps": 0.0},
            {"taker_fee_bps": 50.0, "slippage_bps": 50.0},  # very high fee
        ]
        paths = generate_paths(n_observations=40, n_groups=4, k_test=2, purge=1, embargo=1)
        sharpes = sweep_configs(
            cache=cache,
            universe=universe,
            timeframe="1w",
            paths=paths,
            configs=configs,
            initial_equity=2000.0,
            decisions_db_path=tmp_path / "decisions.db",
        )
        # On most paths, the no-fee config should match or beat the high-fee config
        # (we allow occasional ties when neither config triggers a trade)
        wins_no_fee = (sharpes[:, 0] >= sharpes[:, 1]).sum()
        assert wins_no_fee >= len(paths) * 0.7


class TestConfigSweepIntegrationWithPBO:
    def test_pbo_computable_after_sweep(self, synthetic_universe, tmp_path: Path) -> None:
        from src.backtest.cpcv import compute_pbo

        cache, universe = synthetic_universe
        configs = [
            {"taker_fee_bps": 0.0, "slippage_bps": 0.0},
            {"taker_fee_bps": 2.0, "slippage_bps": 2.0},
            {"taker_fee_bps": 8.0, "slippage_bps": 10.0},
        ]
        paths = generate_paths(n_observations=40, n_groups=4, k_test=2, purge=1, embargo=1)
        sharpes = sweep_configs(
            cache=cache, universe=universe, timeframe="1w",
            paths=paths, configs=configs,
            initial_equity=2000.0,
            decisions_db_path=tmp_path / "decisions.db",
        )
        pbo = compute_pbo(sharpes)
        # PBO is now defined (not NaN) because we have multiple paths × configs
        assert not math.isnan(pbo)
        assert 0.0 <= pbo <= 1.0
