"""Bundled tests for Phase 3 small modules: margin, clock, correlation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.risk.margin_monitor import compute_margin_state
from src.utils.clock import check_drift_sync, drift_severity
from src.analysis.correlation import (
    rolling_correlation_matrix,
    effective_concurrent_count,
    build_returns_df,
)


class TestMarginMonitor:
    def test_empty_positions_give_full_buffer(self) -> None:
        state = compute_margin_state(
            positions=[], balance={"total": 1000, "free": 1000, "used": 0},
        )
        assert state.margin_ratio == 0.0
        assert state.min_distance_to_liquidation_pct == 1.0

    def test_open_position_computes_distance(self) -> None:
        state = compute_margin_state(
            positions=[{
                "symbol": "BTC/USDT:USDT", "side": "long", "leverage": 2,
                "entry_price": 40000.0, "mark_price": 40000.0,
                "notional": 10_000.0,
            }],
            balance={"total": 10_000, "free": 5_000, "used": 5_000},
        )
        assert state.min_distance_to_liquidation_pct < 0.51  # ~0.495 at 2x long
        assert state.margin_ratio == pytest.approx(0.5)

    def test_distance_drops_as_price_moves_against_long(self) -> None:
        state_ok = compute_margin_state(
            positions=[{
                "symbol": "BTC/USDT:USDT", "side": "long", "leverage": 3,
                "entry_price": 40000.0, "mark_price": 40000.0,
                "notional": 10_000.0,
            }],
            balance={"total": 10_000, "free": 5_000, "used": 5_000},
        )
        state_bad = compute_margin_state(
            positions=[{
                "symbol": "BTC/USDT:USDT", "side": "long", "leverage": 3,
                "entry_price": 40000.0, "mark_price": 35000.0,
                "notional": 10_000.0,
            }],
            balance={"total": 10_000, "free": 5_000, "used": 5_000},
        )
        assert state_bad.min_distance_to_liquidation_pct < state_ok.min_distance_to_liquidation_pct


class TestClockDrift:
    def test_no_exchange_returns_none(self) -> None:
        assert check_drift_sync(None) is None

    def test_drift_severity_bands(self) -> None:
        assert drift_severity(0.1) == "ok"
        assert drift_severity(3.0) == "warn"
        assert drift_severity(15.0) == "halt"
        assert drift_severity(-12.0) == "halt"

    def test_none_drift_classified_as_halt(self) -> None:
        # Unknown drift = unsafe — should halt, not pass as OK.
        assert drift_severity(None) == "halt"


class TestCorrelation:
    def test_high_correlation_when_identical(self) -> None:
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0, 0.01, 100))
        df = pd.DataFrame({"BTC": r, "ETH": r})
        corr = rolling_correlation_matrix(df, window=50)
        assert corr.loc["BTC", "ETH"] == pytest.approx(1.0, abs=1e-9)

    def test_effective_count_higher_when_correlated(self) -> None:
        # Mock correlation matrix
        corr = pd.DataFrame(
            {"BTC": [1.0, 0.9], "ETH": [0.9, 1.0]},
            index=["BTC", "ETH"],
        )
        low = effective_concurrent_count(
            open_symbols=["BTC"], proposed_symbol="ETH",
            correlation_matrix=pd.DataFrame(
                {"BTC": [1.0, 0.0], "ETH": [0.0, 1.0]},
                index=["BTC", "ETH"],
            ),
        )
        high = effective_concurrent_count(
            open_symbols=["BTC"], proposed_symbol="ETH",
            correlation_matrix=corr,
        )
        assert high > low

    def test_build_returns_df(self) -> None:
        s_btc = pd.Series([100, 101, 102, 103, 104], name="close")
        s_eth = pd.Series([2000, 2010, 2020, 2030, 2040], name="close")
        df = build_returns_df({"BTC": s_btc, "ETH": s_eth})
        assert list(df.columns) == ["BTC", "ETH"]
        assert not df.empty
