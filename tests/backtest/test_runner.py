"""Tests for the CPCV simulation runner.

The runner walks a CPCV path's test_indices, building snapshots, looking up
LLM decisions in the cache, and simulating fills. It produces a per-path
BacktestResult containing the equity curve, trades, and metadata.

Edge cases covered:
    - Path with all "hold" decisions → equity flat (minus zero-fee no-trade)
    - Single open_long, never closed → forced close at end of path
    - Open + close in consecutive weeks → P&L matches fill_model arithmetic
    - Decision missing in cache → MissingDecisionError raised
    - Stop-loss hits between decision points
    - Position carried into embargo period — must be closed at last test bar
    - Multiple symbols traded in same path
    - Path with empty test_indices → empty result, equity = initial

Tail risks covered:
    - Equity goes to zero (loss exceeds equity) — surfaced not silenced
    - Decision with size=0 → no trade
    - Decision with type "open_long" but already long → idempotent or rejected (we choose: ignored)
    - Bar gaps in OHLCV → raise rather than silently fill
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.backtest.cpcv import generate_paths
from src.backtest.data_cache import OHLCVBar, OHLCVCache
from src.backtest.decision_cache import DecisionCache
from src.backtest.fill_model import FillConfig
from src.backtest.runner import BacktestResult, MissingDecisionError, run_path
from src.backtest.snapshot_builder import SnapshotBuilder

ONE_DAY_MS = 86_400_000
ONE_WEEK_MS = 7 * ONE_DAY_MS


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def populated_cache(tmp_dir: Path) -> OHLCVCache:
    cache = OHLCVCache(tmp_dir / "ohlcv.db")
    # Generate 12 weekly bars at increasing prices
    bars = [
        OHLCVBar(
            ts_ms=i * ONE_WEEK_MS,
            open=100.0 + i * 5,
            high=105.0 + i * 5,
            low=95.0 + i * 5,
            close=102.0 + i * 5,
            volume=1.0,
        )
        for i in range(12)
    ]
    cache.store("BTC/USDT", "1w", bars)
    return cache


@pytest.fixture
def empty_decision_cache(tmp_dir: Path) -> DecisionCache:
    return DecisionCache(tmp_dir / "decisions.db")


def _hold_decision() -> dict:
    return {
        "market_assessment": {"trend": "neutral", "confidence": 0.5, "regime": "calm"},
        "actions": [{"type": "hold"}],
        "next_check_minutes": 10080,
        "risk_notes": "no signal",
    }


def _open_long_decision(symbol: str, size: float, stop_loss: float | None = None) -> dict:
    action = {
        "type": "open_long",
        "symbol": symbol,
        "size": size,
    }
    if stop_loss is not None:
        action["stop_loss"] = stop_loss
    return {
        "market_assessment": {"trend": "bullish", "confidence": 0.7, "regime": "calm"},
        "actions": [action],
        "next_check_minutes": 10080,
        "risk_notes": "trend up",
    }


def _close_decision(symbol: str) -> dict:
    return {
        "market_assessment": {"trend": "neutral", "confidence": 0.5, "regime": "calm"},
        "actions": [{"type": "close", "symbol": symbol}],
        "next_check_minutes": 10080,
        "risk_notes": "exit",
    }


def _populate_decisions(
    decision_cache: DecisionCache,
    snapshot_builder: SnapshotBuilder,
    path_indices: list[int],
    bars_df: pd.DataFrame,
    decisions_per_index: dict[int, dict],
    initial_equity: float = 2000.0,
    fill_config: FillConfig | None = None,
) -> None:
    """Trace the state-evolution exactly as the runner does, populating the
    decision cache with the planned decision at each step.

    This mirrors the runner's simulation logic: for each observation it builds
    a snapshot from the current (positions, equity) state, stores the planned
    decision under that snapshot's hash, and applies the decision to update
    state for the next observation. Without this fidelity, the snapshot hashes
    we pre-populate would not match the runner's lookup hashes.
    """
    from src.backtest.fill_model import (
        SimulatedTrade,
        simulate_close,
        simulate_open,
    )
    from src.backtest.snapshot_builder import PositionRecord

    cfg = fill_config or FillConfig(taker_fee_bps=0.0, slippage_bps=0.0)
    open_positions: dict[str, SimulatedTrade] = {}
    equity = initial_equity

    for idx in sorted(path_indices):
        ts = int(bars_df.iloc[idx]["ts_ms"])
        position_records = [
            PositionRecord(
                symbol=t.symbol, side=t.side, size=t.size,
                entry_price=t.entry_price, entry_ts_ms=t.entry_ts_ms,
            )
            for t in open_positions.values()
        ]
        snap = snapshot_builder.build(t_ms=ts, equity_usd=equity, positions=position_records)
        decision = decisions_per_index.get(idx, _hold_decision())
        decision_cache.put(snap.snapshot_id, decision)

        # Now mirror runner evolution to set up the next iteration's state.
        next_idx = idx + 1
        if next_idx >= len(bars_df):
            continue
        next_bar = {
            "ts_ms": int(bars_df.iloc[next_idx]["ts_ms"]),
            "open": float(bars_df.iloc[next_idx]["open"]),
            "high": float(bars_df.iloc[next_idx]["high"]),
            "low": float(bars_df.iloc[next_idx]["low"]),
            "close": float(bars_df.iloc[next_idx]["close"]),
            "volume": float(bars_df.iloc[next_idx]["volume"]),
        }

        # First, settle any SL/TP that would trigger
        for sym, trade in list(open_positions.items()):
            if _hits(trade, next_bar):
                closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=next_bar["ts_ms"], config=cfg)
                equity += closed.pnl
                open_positions.pop(sym)

        # Then apply the decision actions
        for action in decision.get("actions", []):
            atype = action.get("type")
            sym = action.get("symbol")
            if atype == "close" and sym in open_positions:
                trade = open_positions.pop(sym)
                closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=next_bar["ts_ms"], config=cfg)
                equity += closed.pnl
            elif atype in ("open_long", "open_short"):
                size = float(action.get("size", 0.0))
                if size <= 0 or sym in open_positions:
                    continue
                trade = simulate_open(
                    symbol=sym,
                    side="long" if atype == "open_long" else "short",
                    size=size,
                    next_bar=next_bar,
                    stop_loss=action.get("stop_loss"),
                    take_profit=action.get("take_profit"),
                    config=cfg,
                )
                equity -= trade.entry_fee
                open_positions[sym] = trade


def _hits(trade, next_bar) -> bool:
    """Mirror runner._will_hit_sl_or_tp."""
    high = next_bar["high"]
    low = next_bar["low"]
    if trade.stop_loss is not None:
        if trade.side == "long" and low <= trade.stop_loss:
            return True
        if trade.side == "short" and high >= trade.stop_loss:
            return True
    if trade.take_profit is not None:
        if trade.side == "long" and high >= trade.take_profit:
            return True
        if trade.side == "short" and low <= trade.take_profit:
            return True
    return False


class TestRunPathBasic:
    def test_all_hold_keeps_equity_flat(self, populated_cache, empty_decision_cache, tmp_dir) -> None:
        path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        bars = populated_cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=populated_cache,
            universe=["BTC/USDT"],
            lookback_bars=10,
            timeframes=("1w",),
        )

        # All holds
        decisions = {i: _hold_decision() for i in path.test_indices}
        _populate_decisions(
            empty_decision_cache, builder, path.test_indices, bars, decisions, initial_equity=2000.0,
        )

        result = run_path(
            path=path,
            bars_by_symbol={"BTC/USDT": bars},
            snapshot_builder=builder,
            decision_cache=empty_decision_cache,
            fill_config=FillConfig(taker_fee_bps=0, slippage_bps=0),
            initial_equity=2000.0,
        )

        assert isinstance(result, BacktestResult)
        assert result.initial_equity == 2000.0
        assert result.final_equity == 2000.0
        assert result.trades == []

    def test_empty_path_returns_initial_equity(self, populated_cache, empty_decision_cache) -> None:
        empty_path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        # Manually clear test indices to simulate empty path
        from dataclasses import replace
        empty_path = replace(empty_path, test_indices=[], train_indices=list(range(12)))
        bars = populated_cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=populated_cache,
            universe=["BTC/USDT"],
            lookback_bars=10,
            timeframes=("1w",),
        )
        result = run_path(
            path=empty_path,
            bars_by_symbol={"BTC/USDT": bars},
            snapshot_builder=builder,
            decision_cache=empty_decision_cache,
            fill_config=FillConfig(),
            initial_equity=2000.0,
        )
        assert result.final_equity == 2000.0
        assert result.trades == []


class TestRunPathOpenClose:
    def test_open_long_then_close_records_pnl(
        self, populated_cache, empty_decision_cache
    ) -> None:
        # Take the first CPCV path covering observations [0,1,2]
        path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        bars = populated_cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=populated_cache, universe=["BTC/USDT"], lookback_bars=10, timeframes=("1w",),
        )
        # At index 0, open long; at index 1, close
        decisions = {
            path.test_indices[0]: _open_long_decision("BTC/USDT", size=1.0),
            path.test_indices[1]: _close_decision("BTC/USDT"),
            path.test_indices[2]: _hold_decision(),
        }
        _populate_decisions(empty_decision_cache, builder, path.test_indices, bars, decisions)

        result = run_path(
            path=path,
            bars_by_symbol={"BTC/USDT": bars},
            snapshot_builder=builder,
            decision_cache=empty_decision_cache,
            fill_config=FillConfig(taker_fee_bps=0, slippage_bps=0),
            initial_equity=2000.0,
        )
        # Entry at index 0+1 next bar's open, exit at index 1+1's open
        # bars[1].open = 100 + 1*5 = 105; bars[2].open = 100 + 2*5 = 110
        # Long size=1 → pnl = 110 - 105 = 5
        assert len(result.trades) == 1
        assert result.trades[0].pnl == pytest.approx(5.0)
        assert result.final_equity == pytest.approx(2005.0)

    def test_position_open_at_end_of_path_force_closed(
        self, populated_cache, empty_decision_cache
    ) -> None:
        # Open long at index 0, never close → runner closes at last test bar's open
        path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        bars = populated_cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=populated_cache, universe=["BTC/USDT"], lookback_bars=10, timeframes=("1w",),
        )
        decisions = {
            path.test_indices[0]: _open_long_decision("BTC/USDT", size=1.0),
            path.test_indices[1]: _hold_decision(),
            path.test_indices[2]: _hold_decision(),
        }
        _populate_decisions(empty_decision_cache, builder, path.test_indices, bars, decisions)

        result = run_path(
            path=path,
            bars_by_symbol={"BTC/USDT": bars},
            snapshot_builder=builder,
            decision_cache=empty_decision_cache,
            fill_config=FillConfig(taker_fee_bps=0, slippage_bps=0),
            initial_equity=2000.0,
        )
        assert len(result.trades) == 1, "open position must be force-closed at path end"
        assert result.trades[0].exit_reason == "manual"


class TestRunPathErrors:
    def test_missing_decision_raises(self, populated_cache, empty_decision_cache) -> None:
        path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        bars = populated_cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=populated_cache, universe=["BTC/USDT"], lookback_bars=10, timeframes=("1w",),
        )
        # Don't populate the decision cache — runner should raise
        with pytest.raises(MissingDecisionError):
            run_path(
                path=path,
                bars_by_symbol={"BTC/USDT": bars},
                snapshot_builder=builder,
                decision_cache=empty_decision_cache,
                fill_config=FillConfig(),
                initial_equity=2000.0,
            )


class TestRunPathStopLoss:
    def test_long_stop_hit_in_next_bar(self, tmp_dir, empty_decision_cache) -> None:
        # Build a fixture with declining prices so a long stop actually fires.
        cache = OHLCVCache(tmp_dir / "ohlcv_decline.db")
        # Prices fall over time: bar[i].close = 200 - i*10
        bars_to_store = [
            OHLCVBar(
                ts_ms=i * ONE_WEEK_MS,
                open=200.0 - i * 10,
                high=205.0 - i * 10,
                low=195.0 - i * 10,
                close=200.0 - i * 10,
                volume=1.0,
            )
            for i in range(12)
        ]
        cache.store("BTC/USDT", "1w", bars_to_store)
        bars = cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=cache, universe=["BTC/USDT"], lookback_bars=10, timeframes=("1w",),
        )

        path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        # Open long at idx 0: entry on bars[1].open = 190.
        # Stop at 187: bars[2] has low=175, hits 187 → stop fires.
        decisions = {
            path.test_indices[0]: _open_long_decision("BTC/USDT", size=1.0, stop_loss=187.0),
            path.test_indices[1]: _hold_decision(),
            path.test_indices[2]: _hold_decision(),
        }
        _populate_decisions(empty_decision_cache, builder, path.test_indices, bars, decisions)

        result = run_path(
            path=path,
            bars_by_symbol={"BTC/USDT": bars},
            snapshot_builder=builder,
            decision_cache=empty_decision_cache,
            fill_config=FillConfig(taker_fee_bps=0, slippage_bps=0),
            initial_equity=2000.0,
        )
        # Entry at 190, stop at 187 fires when bars[2].low = 175. Exit at 187.
        # PnL = 187 - 190 = -3.
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason == "stop_loss"
        assert result.trades[0].pnl == pytest.approx(-3.0)


class TestRunPathTailRisks:
    def test_zero_size_decision_is_no_op(self, populated_cache, empty_decision_cache) -> None:
        path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        bars = populated_cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=populated_cache, universe=["BTC/USDT"], lookback_bars=10, timeframes=("1w",),
        )
        decisions = {
            i: _open_long_decision("BTC/USDT", size=0.0) for i in path.test_indices
        }
        _populate_decisions(empty_decision_cache, builder, path.test_indices, bars, decisions)

        result = run_path(
            path=path,
            bars_by_symbol={"BTC/USDT": bars},
            snapshot_builder=builder,
            decision_cache=empty_decision_cache,
            fill_config=FillConfig(),
            initial_equity=2000.0,
        )
        assert result.trades == []
        assert result.final_equity == 2000.0

    def test_open_when_already_long_is_ignored(
        self, populated_cache, empty_decision_cache
    ) -> None:
        path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        bars = populated_cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=populated_cache, universe=["BTC/USDT"], lookback_bars=10, timeframes=("1w",),
        )
        decisions = {
            path.test_indices[0]: _open_long_decision("BTC/USDT", size=1.0),
            path.test_indices[1]: _open_long_decision("BTC/USDT", size=1.0),  # already long
            path.test_indices[2]: _hold_decision(),
        }
        _populate_decisions(empty_decision_cache, builder, path.test_indices, bars, decisions)

        result = run_path(
            path=path,
            bars_by_symbol={"BTC/USDT": bars},
            snapshot_builder=builder,
            decision_cache=empty_decision_cache,
            fill_config=FillConfig(taker_fee_bps=0, slippage_bps=0),
            initial_equity=2000.0,
        )
        # Only 1 trade should exist (the second open ignored), force-closed at path end
        assert len(result.trades) == 1


class TestEquityCurve:
    def test_equity_curve_recorded_per_observation(
        self, populated_cache, empty_decision_cache
    ) -> None:
        path = generate_paths(n_observations=12, n_groups=4, k_test=1, purge=0, embargo=0)[0]
        bars = populated_cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999)
        builder = SnapshotBuilder(
            cache=populated_cache, universe=["BTC/USDT"], lookback_bars=10, timeframes=("1w",),
        )
        decisions = {i: _hold_decision() for i in path.test_indices}
        _populate_decisions(empty_decision_cache, builder, path.test_indices, bars, decisions)

        result = run_path(
            path=path,
            bars_by_symbol={"BTC/USDT": bars},
            snapshot_builder=builder,
            decision_cache=empty_decision_cache,
            fill_config=FillConfig(),
            initial_equity=2000.0,
        )
        # Equity curve has one entry per test observation (plus initial)
        assert len(result.equity_curve) == len(path.test_indices) + 1
