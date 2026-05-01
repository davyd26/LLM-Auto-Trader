#!/usr/bin/env python3
"""Daily-cadence parallel sweep variant.

Same as sweep_parallel but with a parameter grid scaled for daily bars
(longer momentum_lookback in days, periods_per_year = 365). Annualization
is automatic via periods_per_year passed to aggregate_results.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.sweep_parallel import (
    SUMMARY_FIELDS,
    already_done,
    append_csv_locked,
    config_hash,
    evaluate_config,
)
from src.backtest.cpcv import generate_paths
from src.backtest.data_cache import OHLCVCache


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-db", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--cpcv-n", type=int, default=8)
    parser.add_argument("--cpcv-k", type=int, default=2)
    parser.add_argument("--purge", type=int, default=14)
    parser.add_argument("--embargo", type=int, default=14)
    parser.add_argument("--initial-equity", type=float, default=2000.0)
    parser.add_argument("--lookback-bars", type=int, default=120)
    parser.add_argument("--taker-fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--funding-rate-per-day", type=float, default=0.00010)
    parser.add_argument("--workers", type=int, default=8)
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

    # Daily-scaled grid: momentum lookback in days, hedge threshold reasonable
    grid = {
        "momentum_lookback_bars": [10, 20, 50, 100],
        "drawdown_hedge_threshold": [0.05, 0.10, 0.15, 0.20],
        "long_size_usd_pct": [0.50, 0.75, 1.00],
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
            args.initial_equity, fill_kwargs, args.lookback_bars, 365,
        ))
    print(f"Pending: {len(tasks)}")

    if not tasks:
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

    print(f"\nDone in {time.time()-t_start:.0f}s. Results: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
