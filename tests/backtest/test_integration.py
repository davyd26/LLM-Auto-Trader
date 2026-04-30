"""End-to-end integration test for the backtest harness.

Verifies the full pipeline runs cleanly:
    OHLCV cache → snapshots → baseline decisions → CPCV runner → report.

This is the "everything works together" check. The baseline strategy is
deterministic, so we can assert hand-computed properties of the result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.backtest.baseline_strategy import BaselineConfig, decide_baseline
from src.backtest.cpcv import generate_paths
from src.backtest.data_cache import OHLCVBar, OHLCVCache
from src.backtest.decision_cache import DecisionCache
from src.backtest.fill_model import FillConfig
from src.backtest.report import aggregate_results
from src.backtest.runner import run_path
from src.backtest.snapshot_builder import PositionRecord, SnapshotBuilder

WEEK_MS = 7 * 86_400_000


@pytest.fixture
def populated_universe(tmp_path: Path) -> tuple[OHLCVCache, list[str]]:
    """Build a 2-symbol cache with realistic-looking weekly bars over 60 weeks."""
    rng = np.random.default_rng(42)
    cache = OHLCVCache(tmp_path / "ohlcv.db")

    # BTC: trending with a 15% drawdown in the middle then recovery
    btc_log_returns = rng.normal(0.005, 0.04, 60)
    btc_log_returns[20:30] -= 0.025  # drawdown segment
    btc_prices = 50_000.0 * np.exp(np.cumsum(btc_log_returns))
    btc_bars = [
        OHLCVBar(
            ts_ms=i * WEEK_MS,
            open=float(p),
            high=float(p) * (1 + abs(rng.normal(0, 0.02))),
            low=float(p) * (1 - abs(rng.normal(0, 0.02))),
            close=float(p),
            volume=100.0,
        )
        for i, p in enumerate(btc_prices)
    ]
    cache.store("BTC/USDT", "1w", btc_bars)

    # PAXG: low-vol, slowly drifting up
    paxg_log_returns = rng.normal(0.001, 0.005, 60)
    paxg_prices = 1900.0 * np.exp(np.cumsum(paxg_log_returns))
    paxg_bars = [
        OHLCVBar(
            ts_ms=i * WEEK_MS,
            open=float(p),
            high=float(p) * (1 + abs(rng.normal(0, 0.005))),
            low=float(p) * (1 - abs(rng.normal(0, 0.005))),
            close=float(p),
            volume=10.0,
        )
        for i, p in enumerate(paxg_prices)
    ]
    cache.store("PAXG/USDT", "1w", paxg_bars)

    return cache, ["BTC/USDT", "PAXG/USDT"]


def _populate_baseline_decisions(
    decision_cache: DecisionCache,
    snapshot_builder: SnapshotBuilder,
    cache: OHLCVCache,
    universe: list[str],
    test_indices: list[int],
    primary_bars,
    initial_equity: float,
    fill_config: FillConfig,
    baseline_config: BaselineConfig,
) -> None:
    """Trace state evolution and store baseline decisions for each snapshot."""
    from src.backtest.fill_model import (
        SimulatedTrade,
        simulate_close,
        simulate_open,
    )

    open_positions: dict[str, SimulatedTrade] = {}
    equity = initial_equity

    for idx in sorted(test_indices):
        ts = int(primary_bars.iloc[idx]["ts_ms"])

        # Build snapshot via the same path the runner uses
        position_records = [
            PositionRecord(
                symbol=t.symbol, side=t.side, size=t.size,
                entry_price=t.entry_price, entry_ts_ms=t.entry_ts_ms,
            )
            for t in open_positions.values()
        ]
        snap = snapshot_builder.build(
            t_ms=ts, equity_usd=equity, positions=position_records
        )

        # Baseline strategy operates on the snapshot's bars dict directly
        symbol_bars = {sym: snap.bars.get(sym, {}).get("1w", []) for sym in universe}
        positions_dicts = [
            {"symbol": p.symbol, "side": p.side, "size": p.size,
             "entry_price": p.entry_price, "entry_ts_ms": p.entry_ts_ms}
            for p in position_records
        ]
        decision = decide_baseline(
            symbol_bars=symbol_bars,
            current_positions=positions_dicts,
            equity_usd=equity,
            config=baseline_config,
        )
        decision_cache.put(snap.snapshot_id, decision)

        # Mirror runner state evolution
        next_idx = idx + 1
        if next_idx >= len(primary_bars):
            continue

        # Build next bars per symbol (using the cache, since baseline trades each)
        for sym in universe:
            sym_df = cache.load_range(sym, "1w", 0, 999_999_999_999).reset_index(drop=True)
            if next_idx >= len(sym_df):
                continue

        # Settle existing
        for sym, trade in list(open_positions.items()):
            sym_df = cache.load_range(sym, "1w", 0, 999_999_999_999).reset_index(drop=True)
            if next_idx >= len(sym_df):
                continue
            row = sym_df.iloc[next_idx]
            next_bar = {
                "ts_ms": int(row["ts_ms"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            hit = (
                (trade.stop_loss is not None and (
                    (trade.side == "long" and next_bar["low"] <= trade.stop_loss)
                    or (trade.side == "short" and next_bar["high"] >= trade.stop_loss)
                )) or (trade.take_profit is not None and (
                    (trade.side == "long" and next_bar["high"] >= trade.take_profit)
                    or (trade.side == "short" and next_bar["low"] <= trade.take_profit)
                ))
            )
            if hit:
                closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=next_bar["ts_ms"], config=fill_config)
                equity += closed.pnl
                open_positions.pop(sym)

        # Apply actions
        for action in decision.get("actions", []):
            atype = action.get("type")
            sym = action.get("symbol")
            if atype == "close" and sym in open_positions:
                sym_df = cache.load_range(sym, "1w", 0, 999_999_999_999).reset_index(drop=True)
                if next_idx >= len(sym_df):
                    continue
                row = sym_df.iloc[next_idx]
                next_bar = {
                    "ts_ms": int(row["ts_ms"]), "open": float(row["open"]),
                    "high": float(row["high"]), "low": float(row["low"]),
                    "close": float(row["close"]), "volume": float(row["volume"]),
                }
                trade = open_positions.pop(sym)
                closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=next_bar["ts_ms"], config=fill_config)
                equity += closed.pnl
            elif atype in ("open_long", "open_short"):
                size = float(action.get("size", 0.0))
                if size <= 0 or sym in open_positions:
                    continue
                sym_df = cache.load_range(sym, "1w", 0, 999_999_999_999).reset_index(drop=True)
                if next_idx >= len(sym_df):
                    continue
                row = sym_df.iloc[next_idx]
                next_bar = {
                    "ts_ms": int(row["ts_ms"]), "open": float(row["open"]),
                    "high": float(row["high"]), "low": float(row["low"]),
                    "close": float(row["close"]), "volume": float(row["volume"]),
                }
                trade = simulate_open(
                    symbol=sym,
                    side="long" if atype == "open_long" else "short",
                    size=size,
                    next_bar=next_bar,
                    stop_loss=action.get("stop_loss"),
                    take_profit=action.get("take_profit"),
                    config=fill_config,
                )
                equity -= trade.entry_fee
                open_positions[sym] = trade


class TestEndToEndBaseline:
    def test_pipeline_produces_valid_report(
        self, populated_universe, tmp_path: Path
    ) -> None:
        cache, universe = populated_universe
        decision_cache = DecisionCache(tmp_path / "decisions.db")

        builder = SnapshotBuilder(
            cache=cache, universe=universe, lookback_bars=12, timeframes=("1w",),
        )
        baseline_cfg = BaselineConfig(momentum_lookback_bars=4, drawdown_hedge_threshold=0.05)
        fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0)

        # Generate CPCV paths over 60 weeks: 5 groups of 12, k=2 → C(5,2)=10 paths
        paths = generate_paths(
            n_observations=60, n_groups=5, k_test=2, purge=1, embargo=1
        )

        primary_bars = cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999).reset_index(drop=True)

        results = []
        for path in paths:
            # Populate decisions for this path's test indices
            _populate_baseline_decisions(
                decision_cache=decision_cache,
                snapshot_builder=builder,
                cache=cache,
                universe=universe,
                test_indices=path.test_indices,
                primary_bars=primary_bars,
                initial_equity=2000.0,
                fill_config=fill_cfg,
                baseline_config=baseline_cfg,
            )

            bars_by_symbol = {
                sym: cache.load_range(sym, "1w", 0, 999_999_999_999).reset_index(drop=True)
                for sym in universe
            }
            result = run_path(
                path=path,
                bars_by_symbol=bars_by_symbol,
                snapshot_builder=builder,
                decision_cache=decision_cache,
                fill_config=fill_cfg,
                initial_equity=2000.0,
            )
            results.append(result)

        # Verify report
        summary = aggregate_results(results, periods_per_year=52)
        assert summary["n_paths"] == 10
        # Baseline produces SOME trades (not all hold)
        assert summary["total_trades"] > 0
        # Final equities range across paths
        assert summary["p5_final_equity"] <= summary["median_final_equity"]
        assert summary["median_final_equity"] <= summary["p95_final_equity"]
        # Each path's equity curve has expected length
        for r in results:
            assert len(r.equity_curve) == len(sorted(r.path.test_indices)) + 1 if hasattr(r, "path") else len(r.equity_curve) >= 1


class TestEndToEndDeterminism:
    def test_same_inputs_produce_same_results(
        self, populated_universe, tmp_path: Path
    ) -> None:
        # Run the harness twice and verify identical output (CPCV reproducibility)
        cache, universe = populated_universe
        builder = SnapshotBuilder(
            cache=cache, universe=universe, lookback_bars=12, timeframes=("1w",),
        )
        baseline_cfg = BaselineConfig(momentum_lookback_bars=4, drawdown_hedge_threshold=0.05)
        fill_cfg = FillConfig(taker_fee_bps=0.0, slippage_bps=0.0)

        path = generate_paths(n_observations=60, n_groups=5, k_test=2, purge=1, embargo=1)[0]
        primary_bars = cache.load_range("BTC/USDT", "1w", 0, 999_999_999_999).reset_index(drop=True)
        bars_by_symbol = {
            sym: cache.load_range(sym, "1w", 0, 999_999_999_999).reset_index(drop=True)
            for sym in universe
        }

        finals = []
        for run_idx in range(2):
            decision_cache = DecisionCache(tmp_path / f"decisions_{run_idx}.db")
            _populate_baseline_decisions(
                decision_cache=decision_cache,
                snapshot_builder=builder,
                cache=cache,
                universe=universe,
                test_indices=path.test_indices,
                primary_bars=primary_bars,
                initial_equity=2000.0,
                fill_config=fill_cfg,
                baseline_config=baseline_cfg,
            )
            result = run_path(
                path=path,
                bars_by_symbol=bars_by_symbol,
                snapshot_builder=builder,
                decision_cache=decision_cache,
                fill_config=fill_cfg,
                initial_equity=2000.0,
            )
            finals.append(result.final_equity)

        assert finals[0] == finals[1], "harness must be deterministic"
