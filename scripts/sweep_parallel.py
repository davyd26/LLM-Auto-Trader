#!/usr/bin/env python3
"""Parallel hyperparameter sweep for the baseline strategy via CPCV.

Uses multiprocessing to evaluate independent (lookback, hedge_threshold,
size_pct) configs in parallel. Each worker writes its summary row to a
shared CSV (file locked via O_APPEND). Configs already in the CSV are
skipped, so the sweep is naturally resumable on Ctrl-C / OS interruption.

Usage:
    python scripts/sweep_parallel.py \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db \\
        --symbols BTC/USDT:USDT,ETH/USDT:USDT,PAXG/USDT \\
        --timeframe 1w \\
        --start 2021-01-04 --end 2026-04-27 \\
        --run-id sweep-v2 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.baseline_strategy import BaselineConfig, decide_baseline
from src.backtest.cpcv import CPCVPath, generate_paths
from src.backtest.data_cache import OHLCVCache
from src.backtest.decision_cache import DecisionCache
from src.backtest.fill_model import FillConfig, simulate_close, simulate_open
from src.backtest.report import aggregate_results
from src.backtest.runner import run_path
from src.backtest.snapshot_builder import PositionRecord, SnapshotBuilder

SUMMARY_FIELDS = [
    "config_id", "momentum_lookback_bars", "drawdown_hedge_threshold",
    "long_size_usd_pct", "primary_symbol", "hedge_symbol",
    "n_paths", "median_final_equity", "median_return_pct",
    "median_sharpe", "p5_sharpe", "p95_sharpe",
    "sharpe_ci_lo_95", "sharpe_ci_hi_95",
    "median_max_drawdown", "p95_max_drawdown",
    "psr_on_median_sharpe", "total_trades",
    "aggregate_hit_rate", "aggregate_profit_factor",
    "aggregate_expectancy",
]


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _row_to_bar(row) -> dict:
    return {
        "ts_ms": int(row["ts_ms"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }


def _hits(trade, next_bar: dict) -> bool:
    if trade.stop_loss is not None:
        if trade.side == "long" and next_bar["low"] <= trade.stop_loss:
            return True
        if trade.side == "short" and next_bar["high"] >= trade.stop_loss:
            return True
    if trade.take_profit is not None:
        if trade.side == "long" and next_bar["high"] >= trade.take_profit:
            return True
        if trade.side == "short" and next_bar["low"] <= trade.take_profit:
            return True
    return False


def populate_path(path, builder, decision_cache, universe, timeframe,
                  primary_bars, bars_by_symbol, initial_equity,
                  fill_config, baseline_cfg):
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

        next_idx = idx + 1
        if next_idx >= len(primary_bars):
            continue
        for sym, trade in list(open_positions.items()):
            sym_df = bars_by_symbol.get(sym)
            if sym_df is None or next_idx >= len(sym_df):
                continue
            next_bar = _row_to_bar(sym_df.iloc[next_idx])
            if _hits(trade, next_bar):
                closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=next_bar["ts_ms"], config=fill_config)
                equity += closed.pnl
                open_positions.pop(sym)
        for action in decision.get("actions", []):
            atype = action.get("type")
            sym = action.get("symbol")
            if atype == "close" and sym in open_positions:
                sym_df = bars_by_symbol.get(sym)
                if sym_df is None or next_idx >= len(sym_df):
                    continue
                next_bar = _row_to_bar(sym_df.iloc[next_idx])
                trade = open_positions.pop(sym)
                closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=next_bar["ts_ms"], config=fill_config)
                equity += closed.pnl
            elif atype in ("open_long", "open_short"):
                size = float(action.get("size", 0.0))
                if size <= 0 or sym in open_positions:
                    continue
                sym_df = bars_by_symbol.get(sym)
                if sym_df is None or next_idx >= len(sym_df):
                    continue
                next_bar = _row_to_bar(sym_df.iloc[next_idx])
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


def evaluate_config(args_tuple):
    """Worker function — evaluates one config across all paths."""
    if len(args_tuple) == 14:
        (
            cfg, runs_dir_str, cache_db, universe, timeframe,
            cpcv_n, cpcv_k, purge, embargo, n_obs,
            initial_equity, fill_kwargs, lookback_bars, periods_per_year,
        ) = args_tuple
    else:
        (
            cfg, runs_dir_str, cache_db, universe, timeframe,
            cpcv_n, cpcv_k, purge, embargo, n_obs,
            initial_equity, fill_kwargs, lookback_bars,
        ) = args_tuple
        periods_per_year = 52
    runs_dir = Path(runs_dir_str)
    cfg_id = config_hash(cfg)
    decision_db = runs_dir / f"decisions_{cfg_id}.db"

    # Each worker opens its own SQLite handle (multiprocessing-safe)
    cache = OHLCVCache(cache_db)
    decision_cache = DecisionCache(decision_db)
    fill_config = FillConfig(**fill_kwargs)

    builder = SnapshotBuilder(
        cache=cache, universe=universe,
        lookback_bars=lookback_bars, timeframes=(timeframe,),
    )
    bars_by_symbol = {
        sym: cache.load_range(sym, timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
        for sym in universe
    }
    primary_bars = bars_by_symbol[universe[0]]

    paths = generate_paths(
        n_observations=n_obs, n_groups=cpcv_n, k_test=cpcv_k,
        purge=purge, embargo=embargo,
    )

    baseline_cfg = BaselineConfig(
        momentum_lookback_bars=cfg["momentum_lookback_bars"],
        drawdown_hedge_threshold=cfg["drawdown_hedge_threshold"],
        long_size_usd_pct=cfg["long_size_usd_pct"],
        primary_symbol=cfg["primary_symbol"],
        hedge_symbol=cfg["hedge_symbol"],
    )

    results = []
    for path in paths:
        populate_path(
            path=path, builder=builder, decision_cache=decision_cache,
            universe=universe, timeframe=timeframe,
            primary_bars=primary_bars, bars_by_symbol=bars_by_symbol,
            initial_equity=initial_equity,
            fill_config=fill_config, baseline_cfg=baseline_cfg,
        )
        try:
            res = run_path(
                path=path, bars_by_symbol=bars_by_symbol,
                snapshot_builder=builder, decision_cache=decision_cache,
                fill_config=fill_config, initial_equity=initial_equity,
            )
            results.append(res)
        except Exception:
            continue

    summary = aggregate_results(results, periods_per_year=periods_per_year)
    summary["config_id"] = cfg_id
    summary["config"] = cfg
    return cfg_id, cfg, summary


def append_csv_locked(path: Path, row: list):
    """Atomic append to shared CSV across multiprocess workers."""
    with open(path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            csv.writer(f).writerow(row)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def already_done(path: Path, cfg_id: str) -> bool:
    if not path.exists():
        return False
    with open(path) as f:
        for line in f:
            if line.startswith(cfg_id + ","):
                return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-db", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--timeframe", default="1w")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--cpcv-n", type=int, default=8)
    parser.add_argument("--cpcv-k", type=int, default=2)
    parser.add_argument("--purge", type=int, default=2)
    parser.add_argument("--embargo", type=int, default=2)
    parser.add_argument("--initial-equity", type=float, default=2000.0)
    parser.add_argument("--lookback-bars", type=int, default=12)
    parser.add_argument("--taker-fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--funding-rate-per-day", type=float, default=0.00010)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--periods-per-year", type=int, default=52)
    args = parser.parse_args(argv)

    universe = [s.strip() for s in args.symbols.split(",") if s.strip()]
    runs_dir = Path(args.runs_dir) / args.run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = runs_dir / "sweep_summary.csv"

    cache = OHLCVCache(args.cache_db)
    primary_bars = cache.load_range(universe[0], args.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
    primary_bars = primary_bars[
        (primary_bars["ts_ms"] >= parse_iso(args.start))
        & (primary_bars["ts_ms"] <= parse_iso(args.end))
    ].reset_index(drop=True)
    n_obs = len(primary_bars)
    print(f"Universe: {universe}")
    print(f"Window: {args.start} .. {args.end}  ({n_obs} {args.timeframe} bars)")

    paths = generate_paths(
        n_observations=n_obs, n_groups=args.cpcv_n, k_test=args.cpcv_k,
        purge=args.purge, embargo=args.embargo,
    )
    print(f"CPCV paths: {len(paths)}")

    # Build the parameter grid
    grid = {
        "momentum_lookback_bars": [4, 8, 12, 16, 26],
        "drawdown_hedge_threshold": [0.05, 0.10, 0.15, 0.20],
        "long_size_usd_pct": [0.10, 0.25, 0.50, 0.75, 1.00],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Configs to evaluate: {len(combos)}")

    if not summary_csv.exists():
        with open(summary_csv, "w", newline="") as f:
            csv.writer(f).writerow(SUMMARY_FIELDS)

    fill_kwargs = dict(
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        funding_rate_per_day=args.funding_rate_per_day,
    )

    tasks = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        cfg["primary_symbol"] = universe[0]
        cfg["hedge_symbol"] = universe[2] if len(universe) > 2 else universe[0]
        cfg_id = config_hash(cfg)
        if already_done(summary_csv, cfg_id):
            continue
        tasks.append((
            cfg, str(runs_dir), args.cache_db, universe, args.timeframe,
            args.cpcv_n, args.cpcv_k, args.purge, args.embargo, n_obs,
            args.initial_equity, fill_kwargs, args.lookback_bars,
        ))
    print(f"Pending (after resume check): {len(tasks)}")

    if not tasks:
        print("Nothing to do.")
        return 0

    t_start = time.time()
    completed = 0
    with mp.Pool(args.workers) as pool:
        for cfg_id, cfg, summary in pool.imap_unordered(evaluate_config, tasks):
            row = [
                cfg_id,
                cfg["momentum_lookback_bars"],
                cfg["drawdown_hedge_threshold"],
                cfg["long_size_usd_pct"],
                cfg["primary_symbol"],
                cfg["hedge_symbol"],
                summary["n_paths"],
                summary["median_final_equity"],
                summary["median_return_pct"],
                summary["median_sharpe"],
                summary["p5_sharpe"],
                summary["p95_sharpe"],
                summary.get("sharpe_ci_lo_95"),
                summary.get("sharpe_ci_hi_95"),
                summary["median_max_drawdown"],
                summary["p95_max_drawdown"],
                summary["psr_on_median_sharpe"],
                summary["total_trades"],
                summary["aggregate_hit_rate"],
                summary["aggregate_profit_factor"],
                summary["aggregate_expectancy"],
            ]
            append_csv_locked(summary_csv, row)
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed
            eta = (len(tasks) - completed) / rate if rate > 0 else 0
            print(f"[{completed}/{len(tasks)}] {cfg_id} "
                  f"M={cfg['momentum_lookback_bars']} "
                  f"H={cfg['drawdown_hedge_threshold']} "
                  f"S={cfg['long_size_usd_pct']} "
                  f"-> Sharpe {summary['median_sharpe']:+.2f}  "
                  f"DD {summary['median_max_drawdown']:.1%}  "
                  f"ret {summary['median_return_pct']:+.1%}  "
                  f"trades {summary['total_trades']}  "
                  f"[ETA {eta:.0f}s]")

    print(f"\nDone. Total: {completed} configs in {time.time()-t_start:.0f}s.")
    print(f"Results: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
