"""Multi-config CPCV sweep for PBO computation.

Running the strategy under one config gives one Sharpe per path. To
compute PBO honestly we need multiple configs across the same paths.
This module runs the sweep, sharing the decision cache across configs
(decisions are config-independent because they only depend on the
snapshot — fees and slippage are applied at fill time, not decision
time).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.backtest.baseline_strategy import BaselineConfig, decide_baseline
from src.backtest.cpcv import CPCVPath
from src.backtest.data_cache import OHLCVCache
from src.backtest.decision_cache import DecisionCache
from src.backtest.fill_model import FillConfig, simulate_close, simulate_open
from src.backtest.runner import run_path
from src.backtest.snapshot_builder import PositionRecord, SnapshotBuilder


def sweep_configs(
    cache: OHLCVCache,
    universe: list[str],
    timeframe: str,
    paths: list[CPCVPath],
    configs: list[dict[str, Any]],
    initial_equity: float,
    decisions_db_path: Path,
    lookback_bars: int = 12,
    periods_per_year: int = 52,
) -> np.ndarray:
    """Run the strategy with multiple ``FillConfig`` candidates over the same paths.

    ``configs`` is a list of dicts holding the fields of ``FillConfig``.
    The same decision cache is reused across configs (decisions are
    config-independent). Returns a (n_paths, n_configs) Sharpe array.
    """
    if not configs:
        raise ValueError("configs list must be non-empty")

    builder = SnapshotBuilder(
        cache=cache,
        universe=universe,
        lookback_bars=lookback_bars,
        timeframes=(timeframe,),
    )

    # We compute decisions ONCE under a no-fee config — they don't depend on
    # fees. Then we replay the SAME decisions under different fill configs.
    # However, the runner's snapshot construction uses (positions, equity)
    # state which depends on the fill config (different fees → different
    # equity → different snapshot hashes → cache misses).
    #
    # The pragmatic approach: pre-populate decisions for each config
    # separately. With the baseline strategy this is fast and deterministic.
    decision_cache = DecisionCache(decisions_db_path)

    bars_by_symbol = {
        sym: cache.load_range(sym, timeframe, 0, 999_999_999_999).reset_index(drop=True)
        for sym in universe
    }
    primary_bars = bars_by_symbol[universe[0]]

    n_paths = len(paths)
    n_configs = len(configs)
    sharpes = np.zeros((n_paths, n_configs), dtype=float)

    baseline_cfg = BaselineConfig()

    for c_idx, cfg_dict in enumerate(configs):
        fill_cfg = FillConfig(**cfg_dict)
        for p_idx, path in enumerate(paths):
            _populate_for_path_and_config(
                path=path,
                builder=builder,
                cache=cache,
                decision_cache=decision_cache,
                universe=universe,
                timeframe=timeframe,
                primary_bars=primary_bars,
                initial_equity=initial_equity,
                fill_config=fill_cfg,
                baseline_cfg=baseline_cfg,
            )
            try:
                result = run_path(
                    path=path,
                    bars_by_symbol=bars_by_symbol,
                    snapshot_builder=builder,
                    decision_cache=decision_cache,
                    fill_config=fill_cfg,
                    initial_equity=initial_equity,
                )
            except Exception:
                sharpes[p_idx, c_idx] = np.nan
                continue
            sharpes[p_idx, c_idx] = _path_sharpe(result, periods_per_year)

    return sharpes


def _populate_for_path_and_config(
    path: CPCVPath,
    builder: SnapshotBuilder,
    cache: OHLCVCache,
    decision_cache: DecisionCache,
    universe: list[str],
    timeframe: str,
    primary_bars,
    initial_equity: float,
    fill_config: FillConfig,
    baseline_cfg: BaselineConfig,
) -> None:
    """Populate decisions for a single (path, fill_config) combination using baseline."""
    open_positions: dict = {}
    equity = initial_equity

    for idx in sorted(path.test_indices):
        if idx >= len(primary_bars):
            continue
        ts = int(primary_bars.iloc[idx]["ts_ms"])
        position_records = [
            PositionRecord(
                symbol=t.symbol, side=t.side, size=t.size,
                entry_price=t.entry_price, entry_ts_ms=t.entry_ts_ms,
            )
            for t in open_positions.values()
        ]
        snap = builder.build(t_ms=ts, equity_usd=equity, positions=position_records)

        if not decision_cache.has(snap.snapshot_id):
            symbol_bars = {sym: snap.bars.get(sym, {}).get(timeframe, []) for sym in universe}
            positions_dicts = [
                {"symbol": p.symbol, "side": p.side, "size": p.size,
                 "entry_price": p.entry_price, "entry_ts_ms": p.entry_ts_ms}
                for p in position_records
            ]
            decision = decide_baseline(
                symbol_bars=symbol_bars,
                current_positions=positions_dicts,
                equity_usd=equity,
                config=baseline_cfg,
            )
            decision_cache.put(snap.snapshot_id, decision)
        else:
            decision = decision_cache.get(snap.snapshot_id)

        # Mirror runner state evolution under THIS fill_config
        next_idx = idx + 1
        if next_idx >= len(primary_bars):
            continue

        for sym, trade in list(open_positions.items()):
            sym_df = cache.load_range(sym, timeframe, 0, 999_999_999_999).reset_index(drop=True)
            if next_idx >= len(sym_df):
                continue
            row = sym_df.iloc[next_idx]
            next_bar = _row_to_bar(row)
            if _hits(trade, next_bar):
                closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=next_bar["ts_ms"], config=fill_config)
                equity += closed.pnl
                open_positions.pop(sym)

        for action in decision.get("actions", []):
            atype = action.get("type")
            sym = action.get("symbol")
            if atype == "close" and sym in open_positions:
                sym_df = cache.load_range(sym, timeframe, 0, 999_999_999_999).reset_index(drop=True)
                if next_idx >= len(sym_df):
                    continue
                row = sym_df.iloc[next_idx]
                next_bar = _row_to_bar(row)
                trade = open_positions.pop(sym)
                closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=next_bar["ts_ms"], config=fill_config)
                equity += closed.pnl
            elif atype in ("open_long", "open_short"):
                size = float(action.get("size", 0.0))
                if size <= 0 or sym in open_positions:
                    continue
                sym_df = cache.load_range(sym, timeframe, 0, 999_999_999_999).reset_index(drop=True)
                if next_idx >= len(sym_df):
                    continue
                row = sym_df.iloc[next_idx]
                next_bar = _row_to_bar(row)
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


def _row_to_bar(row) -> dict:
    return {
        "ts_ms": int(row["ts_ms"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }


def _hits(trade, next_bar) -> bool:
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


def _path_sharpe(result, periods_per_year: int) -> float:
    from src.analysis.backtest_stats import sharpe_annualized
    if len(result.equity_curve) < 2:
        return 0.0
    equity = np.array([eq for _, eq in result.equity_curve], dtype=float)
    safe = np.where(equity[:-1] > 0, equity[:-1], 1.0)
    rets = np.diff(equity) / safe
    return float(sharpe_annualized(rets, periods_per_year=periods_per_year))
