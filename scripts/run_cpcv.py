#!/usr/bin/env python3
"""Run CPCV simulation for a prepared backtest run.

Reads the run manifest, loads cached decisions (or generates baseline-stub
decisions on the fly), runs C(N, K) CPCV paths, aggregates results and
writes a summary JSON + per-path equity curves CSV.

Usage:
    python scripts/run_cpcv.py \\
        --run-id low-freq-aggressive-v1 \\
        --runs-dir runs \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db \\
        --decisions-db runs/low-freq-aggressive-v1/decisions.db \\
        --cpcv-n 8 --cpcv-k 2 --purge 2 --embargo 2 \\
        --decisions-source cached
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.backtest.baseline_strategy import BaselineConfig, decide_baseline
from src.backtest.cpcv import generate_paths
from src.backtest.data_cache import OHLCVCache
from src.backtest.decision_cache import DecisionCache
from src.backtest.fill_model import FillConfig, simulate_close, simulate_open
from src.backtest.report import aggregate_results, write_summary_json
from src.backtest.runner import run_path
from src.backtest.snapshot_builder import PositionRecord, SnapshotBuilder


def _ensure_decisions_via_baseline(
    paths,
    builder: SnapshotBuilder,
    cache: OHLCVCache,
    decision_cache: DecisionCache,
    universe: list[str],
    timeframe: str,
    initial_equity: float,
    fill_config: FillConfig,
) -> None:
    """Populate decision_cache with baseline-strategy decisions.

    Walks each path's test_indices the way the runner does, building snapshots
    with the evolving (positions, equity) state and storing the baseline's
    decision under each snapshot's hash. After this call, the runner can run
    the path with no missing decisions.
    """
    baseline_cfg = BaselineConfig()
    primary_bars = cache.load_range(universe[0], timeframe, 0, 999_999_999_999).reset_index(drop=True)

    for path in paths:
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

            # State evolution mirroring the runner
            next_idx = idx + 1
            if next_idx >= len(primary_bars):
                continue

            for sym, trade in list(open_positions.items()):
                sym_df = cache.load_range(sym, timeframe, 0, 999_999_999_999).reset_index(drop=True)
                if next_idx >= len(sym_df):
                    continue
                row = sym_df.iloc[next_idx]
                next_bar = _row_to_bar(row)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--cache-db", required=True)
    parser.add_argument("--decisions-db", required=True)
    parser.add_argument("--cpcv-n", type=int, default=8)
    parser.add_argument("--cpcv-k", type=int, default=2)
    parser.add_argument("--purge", type=int, default=2)
    parser.add_argument("--embargo", type=int, default=2)
    parser.add_argument(
        "--decisions-source",
        choices=("cached", "baseline-stub"),
        default="cached",
        help="cached = require pre-populated decisions; baseline-stub = use deterministic baseline",
    )
    parser.add_argument("--taker-fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())

    universe = manifest["universe"]
    timeframes = manifest["timeframes"]
    primary_tf = timeframes[0]
    n_obs = manifest["n_observations"]
    initial_equity = manifest["initial_equity"]

    cache = OHLCVCache(args.cache_db)
    decision_cache = DecisionCache(args.decisions_db)
    builder = SnapshotBuilder(
        cache=cache,
        universe=universe,
        lookback_bars=manifest["lookback_bars"],
        timeframes=tuple(timeframes),
    )
    fill_cfg = FillConfig(
        taker_fee_bps=args.taker_fee_bps, slippage_bps=args.slippage_bps,
    )

    paths = generate_paths(
        n_observations=n_obs, n_groups=args.cpcv_n, k_test=args.cpcv_k,
        purge=args.purge, embargo=args.embargo,
    )
    print(f"Generated {len(paths)} CPCV paths.")

    if args.decisions_source == "baseline-stub":
        _ensure_decisions_via_baseline(
            paths, builder, cache, decision_cache, universe, primary_tf, initial_equity, fill_cfg,
        )

    # Pre-load bar DataFrames per symbol (avoid re-querying inside the loop)
    bars_by_symbol = {
        sym: cache.load_range(sym, primary_tf, 0, 999_999_999_999).reset_index(drop=True)
        for sym in universe
    }

    results = []
    for path in paths:
        result = run_path(
            path=path,
            bars_by_symbol=bars_by_symbol,
            snapshot_builder=builder,
            decision_cache=decision_cache,
            fill_config=fill_cfg,
            initial_equity=initial_equity,
        )
        results.append(result)
        print(f"Path {result.path_id}: final={result.final_equity:.2f}, trades={len(result.trades)}")

    summary = aggregate_results(results, periods_per_year=52)
    out_summary = run_dir / "summary.json"
    write_summary_json(summary, out_summary)
    print(f"Summary written to {out_summary}")

    # Equity curves CSV
    curves_path = run_dir / "equity_curves.csv"
    with open(curves_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_id", "ts_ms", "equity"])
        for r in results:
            for ts_ms, eq in r.equity_curve:
                w.writerow([r.path_id, ts_ms, eq])
    print(f"Equity curves written to {curves_path}")

    # Trade log CSV
    trades_path = run_dir / "trades.csv"
    with open(trades_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_id", "symbol", "side", "size", "entry_price", "exit_price", "exit_reason", "pnl"])
        for r in results:
            for t in r.trades:
                w.writerow([
                    r.path_id, t.symbol, t.side, t.size,
                    t.entry_price, t.exit_price, t.exit_reason, t.pnl,
                ])
    print(f"Trades written to {trades_path}")

    print()
    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
