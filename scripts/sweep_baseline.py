#!/usr/bin/env python3
"""Hyperparameter sweep for the baseline strategy via CPCV.

For each (momentum_lookback_bars, drawdown_hedge_threshold, long_size_usd_pct)
combination, runs CPCV across the configured number of paths, aggregates the
per-config statistics, and writes a CSV ranking.

Output:
    runs/<run_id>/sweep_results.csv  — one row per config × path
    runs/<run_id>/sweep_summary.csv — one row per config (median + tails)
    runs/<run_id>/best_config.json — the top config by median Sharpe

Usage example:
    python scripts/sweep_baseline.py \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db \\
        --symbols BTC/USDT:USDT,ETH/USDT:USDT,PAXG/USDT \\
        --timeframe 1w \\
        --start 2021-01-04 --end 2026-04-27 \\
        --cpcv-n 8 --cpcv-k 2 --purge 2 --embargo 2 \\
        --run-id sweep-v1 \\
        --runs-dir runs

Designed to be resumable: the per-config decision cache is keyed by config
hash so repeated runs with the same grid replay cached decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.baseline_strategy import BaselineConfig, decide_baseline
from src.backtest.cpcv import CPCVPath, generate_paths
from src.backtest.data_cache import OHLCVCache
from src.backtest.decision_cache import DecisionCache
from src.backtest.fill_model import (
    FillConfig,
    SimulatedTrade,
    simulate_close,
    simulate_open,
)
from src.backtest.report import aggregate_results
from src.backtest.runner import run_path
from src.backtest.snapshot_builder import PositionRecord, SnapshotBuilder


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


def _hits(trade: SimulatedTrade, next_bar: dict) -> bool:
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


def populate_decisions_for_path(
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
    """Walk the path with the baseline strategy, populating the decision cache."""
    open_positions: dict[str, SimulatedTrade] = {}
    equity = initial_equity

    bars_by_symbol = {
        sym: cache.load_range(sym, timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
        for sym in universe
    }

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


def run_one_config(
    cfg: dict,
    paths: list[CPCVPath],
    cache: OHLCVCache,
    universe: list[str],
    timeframe: str,
    runs_dir: Path,
    initial_equity: float,
    fill_config: FillConfig,
    lookback_bars: int,
) -> dict:
    """Run all CPCV paths for one strategy config; return aggregated stats."""
    cfg_id = config_hash(cfg)
    decision_db = runs_dir / f"decisions_{cfg_id}.db"
    decision_cache = DecisionCache(decision_db)

    builder = SnapshotBuilder(
        cache=cache,
        universe=universe,
        lookback_bars=lookback_bars,
        timeframes=(timeframe,),
    )
    primary_bars = cache.load_range(universe[0], timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
    bars_by_symbol = {
        sym: cache.load_range(sym, timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
        for sym in universe
    }

    baseline_cfg = BaselineConfig(
        momentum_lookback_bars=cfg["momentum_lookback_bars"],
        drawdown_hedge_threshold=cfg["drawdown_hedge_threshold"],
        long_size_usd_pct=cfg["long_size_usd_pct"],
        primary_symbol=cfg.get("primary_symbol", "BTC/USDT:USDT"),
        hedge_symbol=cfg.get("hedge_symbol", "PAXG/USDT"),
    )

    results = []
    for path in paths:
        populate_decisions_for_path(
            path=path,
            builder=builder,
            cache=cache,
            decision_cache=decision_cache,
            universe=universe,
            timeframe=timeframe,
            primary_bars=primary_bars,
            initial_equity=initial_equity,
            fill_config=fill_config,
            baseline_cfg=baseline_cfg,
        )
        try:
            res = run_path(
                path=path,
                bars_by_symbol=bars_by_symbol,
                snapshot_builder=builder,
                decision_cache=decision_cache,
                fill_config=fill_config,
                initial_equity=initial_equity,
            )
            results.append(res)
        except Exception as e:
            print(f"    path {path.test_groups} FAILED: {e}", file=sys.stderr)

    summary = aggregate_results(results, periods_per_year=52)
    summary["config_id"] = cfg_id
    summary["config"] = cfg
    return summary, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-db", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--timeframe", default="1w")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
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
    parser.add_argument("--funding-rate-per-day", type=float, default=0.00010,
                        help="0.0001 = 0.01%/day, ~0.0033%/8h Binance baseline")
    args = parser.parse_args(argv)

    universe = [s.strip() for s in args.symbols.split(",") if s.strip()]
    runs_dir = Path(args.runs_dir) / args.run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    cache = OHLCVCache(args.cache_db)
    primary_bars = cache.load_range(universe[0], args.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)

    # Restrict to the configured window
    start_ms = parse_iso(args.start)
    end_ms = parse_iso(args.end)
    primary_bars = primary_bars[(primary_bars["ts_ms"] >= start_ms) & (primary_bars["ts_ms"] <= end_ms)].reset_index(drop=True)
    n_obs = len(primary_bars)
    print(f"Universe: {universe}")
    print(f"Window: {args.start} .. {args.end}  ({n_obs} {args.timeframe} bars)")
    print(f"CPCV: N={args.cpcv_n}, K={args.cpcv_k}, purge={args.purge}, embargo={args.embargo}")

    paths = generate_paths(
        n_observations=n_obs, n_groups=args.cpcv_n, k_test=args.cpcv_k,
        purge=args.purge, embargo=args.embargo,
    )
    print(f"CPCV paths: {len(paths)}")

    fill_config = FillConfig(
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        funding_rate_per_day=args.funding_rate_per_day,
    )

    # Parameter grid
    grid = {
        "momentum_lookback_bars": [4, 8, 12, 16, 26],
        "drawdown_hedge_threshold": [0.05, 0.10, 0.15, 0.20],
        "long_size_usd_pct": [0.10, 0.25, 0.50, 0.75, 1.00],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Configs to evaluate: {len(combos)}")

    # CSV writers
    summary_csv = runs_dir / "sweep_summary.csv"
    summary_fields = [
        "config_id", "momentum_lookback_bars", "drawdown_hedge_threshold",
        "long_size_usd_pct", "n_paths",
        "median_final_equity", "median_return_pct", "median_sharpe",
        "p5_sharpe", "p95_sharpe",
        "sharpe_ci_lo_95", "sharpe_ci_hi_95",
        "median_max_drawdown", "p95_max_drawdown",
        "psr_on_median_sharpe", "total_trades",
        "aggregate_hit_rate", "aggregate_profit_factor",
        "aggregate_expectancy",
    ]
    with open(summary_csv, "w", newline="") as f:
        csv.writer(f).writerow(summary_fields)

    summaries = []
    t_start = time.time()
    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        cfg["primary_symbol"] = universe[0]
        cfg["hedge_symbol"] = universe[2] if len(universe) > 2 else universe[0]
        print(f"\n[{i+1}/{len(combos)}] {cfg}")
        summary, _ = run_one_config(
            cfg=cfg,
            paths=paths,
            cache=cache,
            universe=universe,
            timeframe=args.timeframe,
            runs_dir=runs_dir,
            initial_equity=args.initial_equity,
            fill_config=fill_config,
            lookback_bars=args.lookback_bars,
        )
        summaries.append(summary)
        # Append to CSV immediately (don't lose results on a crash)
        with open(summary_csv, "a", newline="") as f:
            row = [
                summary["config_id"],
                cfg["momentum_lookback_bars"],
                cfg["drawdown_hedge_threshold"],
                cfg["long_size_usd_pct"],
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
            csv.writer(f).writerow(row)
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (len(combos) - i - 1) / rate if rate > 0 else 0
        print(f"   median_Sharpe={summary['median_sharpe']:+.2f}  "
              f"median_DD={summary['median_max_drawdown']:.1%}  "
              f"median_ret={summary['median_return_pct']:+.1%}  "
              f"trades={summary['total_trades']}  "
              f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    # Find best by median Sharpe (with non-trivial trade count)
    candidates = [s for s in summaries if s["total_trades"] >= 5]
    if not candidates:
        candidates = summaries
    best = max(candidates, key=lambda s: s["median_sharpe"])
    print("\n=== BEST CONFIG (by median Sharpe with >=5 trades) ===")
    print(json.dumps(best["config"], indent=2))
    print(f"  median_sharpe: {best['median_sharpe']:.3f}")
    print(f"  p5_sharpe: {best['p5_sharpe']:.3f}")
    print(f"  median_max_drawdown: {best['median_max_drawdown']:.2%}")
    print(f"  median_return_pct: {best['median_return_pct']:.2%}")
    print(f"  total_trades: {best['total_trades']}")

    (runs_dir / "best_config.json").write_text(json.dumps({
        "config": best["config"],
        "summary": {k: v for k, v in best.items() if k != "config"},
    }, indent=2, default=str))
    print(f"\nResults written to {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
