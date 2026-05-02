#!/usr/bin/env python3
"""Ultra-fast CPCV sweep using fast_baseline engine.

Skips the snapshot/decision-cache pipeline entirely. For the deterministic
baseline strategy, that pipeline is pure overhead: SQLite, JSON, SHA-256
hashing add 50-100x runtime per config without changing the result.

This script runs `run_baseline_fast` per CPCV path per config in parallel,
producing the same per-config statistics in milliseconds instead of minutes.

For the final reported result, always run the slow pipeline (run_full_period.py)
to verify — fast engine is for sweep ranking, not the canonical headline.

Usage:
    python scripts/sweep_fast.py \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db \\
        --symbols BTC/USDT:USDT,ETH/USDT:USDT,PAXG/USDT \\
        --timeframe 1d --start 2021-01-01 --end 2026-04-30 \\
        --cpcv-n 8 --cpcv-k 2 --purge 14 --embargo 14 \\
        --run-id daily-fast-vt --workers 8
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.backtest_stats import max_drawdown, sharpe_annualized
from src.backtest.baseline_strategy import BaselineConfig
from src.backtest.cpcv import generate_paths
from src.backtest.data_cache import OHLCVCache
from src.backtest.fast_baseline import run_baseline_fast
from src.backtest.fill_model import FillConfig

SUMMARY_FIELDS = [
    "config_id",
    "momentum_lookback_bars", "drawdown_hedge_threshold", "long_size_usd_pct",
    "vol_target_pct", "vol_lookback_bars", "vol_size_max_mult",
    "enable_shorts", "short_stop_loss_pct", "short_min_drawdown",
    "n_paths", "median_final_equity", "median_return_pct",
    "median_sharpe", "p5_sharpe", "p95_sharpe",
    "median_max_drawdown", "p95_max_drawdown",
    "total_trades_mean",
]


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def append_csv_locked(path: Path, row: list) -> None:
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


@dataclass
class WorkerArgs:
    cfg: dict
    cache_db: str
    universe: list[str]
    timeframe: str
    n_obs: int
    cpcv_n: int
    cpcv_k: int
    purge: int
    embargo: int
    initial_equity: float
    fill_kwargs: dict
    snapshot_lookback_bars: int
    periods_per_year: int
    primary_bars_window: tuple[int, int]  # (start_idx, end_idx) in cache


def evaluate_one(w: WorkerArgs):
    cache = OHLCVCache(w.cache_db)
    primary_full = cache.load_range(w.universe[0], w.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
    hedge_sym = w.universe[2] if len(w.universe) > 2 else w.universe[0]
    hedge_full = cache.load_range(hedge_sym, w.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)

    s_idx, e_idx = w.primary_bars_window
    primary_window = primary_full.iloc[s_idx : e_idx + 1].reset_index(drop=True)
    hedge_window = hedge_full.iloc[s_idx : e_idx + 1].reset_index(drop=True) if not hedge_full.empty else pd.DataFrame()

    paths = generate_paths(
        n_observations=w.n_obs, n_groups=w.cpcv_n, k_test=w.cpcv_k,
        purge=w.purge, embargo=w.embargo,
    )

    config = BaselineConfig(
        momentum_lookback_bars=w.cfg["momentum_lookback_bars"],
        drawdown_hedge_threshold=w.cfg["drawdown_hedge_threshold"],
        long_size_usd_pct=w.cfg["long_size_usd_pct"],
        primary_symbol=w.universe[0],
        hedge_symbol=hedge_sym,
        enable_shorts=w.cfg.get("enable_shorts", False),
        short_size_usd_pct=w.cfg.get("short_size_usd_pct"),
        short_stop_loss_pct=w.cfg.get("short_stop_loss_pct", 0.10),
        long_stop_loss_pct=w.cfg.get("long_stop_loss_pct"),
        short_min_drawdown=w.cfg.get("short_min_drawdown", 0.0),
        vol_target_pct=w.cfg.get("vol_target_pct"),
        vol_lookback_bars=w.cfg.get("vol_lookback_bars", 12),
        vol_size_max_mult=w.cfg.get("vol_size_max_mult", 2.0),
    )
    fill_config = FillConfig(**w.fill_kwargs)

    finals = []
    rets = []
    sharpes = []
    drawdowns = []
    trades = []
    for path in paths:
        if not path.test_indices:
            continue
        sub_start = path.test_indices[0]
        sub_end = path.test_indices[-1]
        result = run_baseline_fast(
            primary_bars=primary_window,
            hedge_bars=hedge_window if not hedge_window.empty else None,
            config=config,
            fill_config=fill_config,
            initial_equity=w.initial_equity,
            start_idx=sub_start,
            end_idx=sub_end,
            snapshot_lookback_bars=w.snapshot_lookback_bars,
        )
        # Clamp equity at zero — a blown-up trajectory is just a 100% loss
        final_eq = max(0.0, result.final_equity)
        finals.append(final_eq)
        rets.append((final_eq - result.initial_equity) / result.initial_equity)
        eq = np.array([max(0.0, eq) for _, eq in result.equity_curve], dtype=float)
        if len(eq) >= 2:
            r = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
            sharpes.append(sharpe_annualized(r, periods_per_year=w.periods_per_year))
            try:
                dd, _, _ = max_drawdown(eq)
            except ValueError:
                dd = 1.0  # 100% loss
            drawdowns.append(dd)
        else:
            sharpes.append(0.0)
            drawdowns.append(0.0)
        trades.append(len(result.trades))

    cfg_id = config_hash(w.cfg)
    if not finals:
        return cfg_id, w.cfg, None
    finals_arr = np.array(finals)
    rets_arr = np.array(rets)
    sharpes_arr = np.array(sharpes)
    dd_arr = np.array(drawdowns)
    summary = {
        "config_id": cfg_id,
        "n_paths": len(finals),
        "median_final_equity": float(np.median(finals_arr)),
        "median_return_pct": float(np.median(rets_arr)),
        "median_sharpe": float(np.median(sharpes_arr)),
        "p5_sharpe": float(np.percentile(sharpes_arr, 5)),
        "p95_sharpe": float(np.percentile(sharpes_arr, 95)),
        "median_max_drawdown": float(np.median(dd_arr)),
        "p95_max_drawdown": float(np.percentile(dd_arr, 95)),
        "total_trades_mean": float(np.mean(trades)),
    }
    return cfg_id, w.cfg, summary


def build_grid(timeframe: str) -> list[dict]:
    """Param grid scaled for the timeframe."""
    if timeframe in ("1w", "7d"):
        return list(itertools.product(
            [4, 8, 12, 16],                         # momentum_lookback_bars (weeks)
            [0.10, 0.15, 0.20, 0.99],               # drawdown_hedge_threshold (0.99 disables)
            [0.50, 0.75, 1.00, 1.50, 2.00, 2.50],   # long_size_usd_pct
            [None, 0.04, 0.03, 0.02, 0.015],        # vol_target_pct (None = no vol-targeting)
            [12],                                   # vol_lookback_bars
            [1.5, 1.67, 2.0],                       # vol_size_max_mult
        ))
    elif timeframe == "1d":
        return list(itertools.product(
            [10, 20, 50, 100, 200],                 # momentum_lookback_bars (days)
            [0.10, 0.15, 0.20, 0.99],
            [0.50, 0.75, 1.00, 1.50, 2.00, 2.50],
            [None, 0.04, 0.03, 0.02, 0.015],
            [60],                                    # vol_lookback_bars (days)
            [1.5, 1.67, 2.0],
        ))
    else:
        raise ValueError(f"unknown timeframe {timeframe}")


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
    parser.add_argument("--snapshot-lookback-bars", type=int, default=30)
    parser.add_argument("--taker-fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--funding-rate-per-day", type=float, default=0.00010)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--periods-per-year", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=None,
                        help="cap number of configs (for quick smoke tests)")
    args = parser.parse_args(argv)

    universe = [s.strip() for s in args.symbols.split(",") if s.strip()]
    runs_dir = Path(args.runs_dir) / args.run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = runs_dir / "sweep_summary.csv"

    cache = OHLCVCache(args.cache_db)
    primary_full = cache.load_range(universe[0], args.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
    s_ms = parse_iso(args.start)
    e_ms = parse_iso(args.end)
    in_window = primary_full[(primary_full["ts_ms"] >= s_ms) & (primary_full["ts_ms"] <= e_ms)]
    s_idx = int(in_window.index[0])
    e_idx = int(in_window.index[-1])
    n_obs = e_idx - s_idx + 1
    print(f"Universe: {universe}")
    print(f"Window: {args.start} .. {args.end}  ({n_obs} {args.timeframe} bars)")
    if args.periods_per_year is None:
        periods_per_year = 52 if args.timeframe in ("1w", "7d") else 365
    else:
        periods_per_year = args.periods_per_year

    if not summary_csv.exists():
        with open(summary_csv, "w", newline="") as f:
            csv.writer(f).writerow(SUMMARY_FIELDS)

    fill_kwargs = dict(
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        funding_rate_per_day=args.funding_rate_per_day,
    )

    grid = build_grid(args.timeframe)
    if args.max_configs is not None:
        grid = grid[: args.max_configs]
    print(f"Configs to evaluate: {len(grid)}")

    tasks: list[WorkerArgs] = []
    for combo in grid:
        cfg = {
            "momentum_lookback_bars": combo[0],
            "drawdown_hedge_threshold": combo[1],
            "long_size_usd_pct": combo[2],
            "vol_target_pct": combo[3],
            "vol_lookback_bars": combo[4],
            "vol_size_max_mult": combo[5],
        }
        cfg_id = config_hash(cfg)
        if already_done(summary_csv, cfg_id):
            continue
        tasks.append(WorkerArgs(
            cfg=cfg,
            cache_db=args.cache_db,
            universe=universe,
            timeframe=args.timeframe,
            n_obs=n_obs,
            cpcv_n=args.cpcv_n,
            cpcv_k=args.cpcv_k,
            purge=args.purge,
            embargo=args.embargo,
            initial_equity=args.initial_equity,
            fill_kwargs=fill_kwargs,
            snapshot_lookback_bars=args.snapshot_lookback_bars,
            periods_per_year=periods_per_year,
            primary_bars_window=(s_idx, e_idx),
        ))
    print(f"Pending: {len(tasks)}")
    if not tasks:
        return 0

    t_start = time.time()
    completed = 0
    with mp.Pool(args.workers) as pool:
        for cfg_id, cfg, summary in pool.imap_unordered(evaluate_one, tasks):
            if summary is None:
                continue
            row = [
                cfg_id,
                cfg["momentum_lookback_bars"],
                cfg["drawdown_hedge_threshold"],
                cfg["long_size_usd_pct"],
                cfg["vol_target_pct"],
                cfg["vol_lookback_bars"],
                cfg["vol_size_max_mult"],
                cfg.get("enable_shorts", False),
                cfg.get("short_stop_loss_pct", 0.10),
                cfg.get("short_min_drawdown", 0.0),
                summary["n_paths"],
                summary["median_final_equity"],
                summary["median_return_pct"],
                summary["median_sharpe"],
                summary["p5_sharpe"],
                summary["p95_sharpe"],
                summary["median_max_drawdown"],
                summary["p95_max_drawdown"],
                summary["total_trades_mean"],
            ]
            append_csv_locked(summary_csv, row)
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                print(f"[{completed}/{len(tasks)}] {elapsed:.0f}s elapsed, "
                      f"{rate:.1f} cfg/s, ETA {eta:.0f}s")

    print(f"\nDone in {time.time()-t_start:.0f}s. Results: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
