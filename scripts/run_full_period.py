#!/usr/bin/env python3
"""Run a single contiguous full-period backtest for direct B&H comparison.

CPCV produces a distribution of Sharpes over disjoint test slices — useful for
robustness, but hard to compare against buy-and-hold over the same window.
This script runs the baseline strategy in one continuous walk from start to
end, producing one equity curve and one set of stats directly comparable to
buy-and-hold.

Usage:
    python scripts/run_full_period.py \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db \\
        --symbols BTC/USDT:USDT,ETH/USDT:USDT,PAXG/USDT \\
        --timeframe 1w \\
        --start 2021-01-04 --end 2026-04-27 \\
        --momentum-lookback 4 --hedge-threshold 0.20 --size-pct 1.0 \\
        --run-id full-best
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.analysis.backtest_stats import (
    calmar_ratio,
    max_drawdown,
    pnl_moments,
    probabilistic_sharpe_ratio,
    sharpe_annualized,
    sortino_ratio,
)
from src.analysis.bootstrap import block_bootstrap_sharpe_ci
from src.backtest.baseline_strategy import BaselineConfig
from src.backtest.benchmarks import buy_and_hold, equal_weight
from src.backtest.cpcv import CPCVPath
from src.backtest.data_cache import OHLCVCache
from src.backtest.decision_cache import DecisionCache
from src.backtest.fill_model import FillConfig
from src.backtest.runner import run_path
from src.backtest.snapshot_builder import SnapshotBuilder
from scripts.sweep_parallel import populate_path


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-db", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--timeframe", default="1w")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--momentum-lookback", type=int, default=4)
    parser.add_argument("--hedge-threshold", type=float, default=0.15)
    parser.add_argument("--size-pct", type=float, default=1.0)
    parser.add_argument("--initial-equity", type=float, default=2000.0)
    parser.add_argument("--lookback-bars", type=int, default=30)
    parser.add_argument("--taker-fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--funding-rate-per-day", type=float, default=0.00010)
    parser.add_argument("--enable-shorts", action="store_true")
    parser.add_argument("--short-stop-loss-pct", type=float, default=0.10)
    parser.add_argument("--long-stop-loss-pct", type=float, default=None)
    parser.add_argument("--short-min-drawdown", type=float, default=0.0)
    parser.add_argument("--confluence-lookback", type=int, default=None)
    parser.add_argument("--rotation-symbols", default="",
                        help="comma-separated symbols for momentum-rotation; if empty, uses primary only")
    parser.add_argument("--vol-target-pct", type=float, default=None,
                        help="weekly vol target (e.g. 0.04 = 4%/week); enables vol-targeting if set")
    parser.add_argument("--vol-lookback", type=int, default=12)
    parser.add_argument("--vol-size-max-mult", type=float, default=2.0,
                        help="cap on vol-target size multiplier (peak leverage = S * this)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--periods-per-year", type=int, default=52)
    args = parser.parse_args(argv)

    universe = [s.strip() for s in args.symbols.split(",") if s.strip()]
    runs_dir = Path(args.runs_dir) / args.run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    cache = OHLCVCache(args.cache_db)
    primary_bars = cache.load_range(universe[0], args.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
    primary_bars = primary_bars[
        (primary_bars["ts_ms"] >= parse_iso(args.start))
        & (primary_bars["ts_ms"] <= parse_iso(args.end))
    ].reset_index(drop=True)
    n_obs = len(primary_bars)
    print(f"Window: {args.start} .. {args.end}  ({n_obs} {args.timeframe} bars)")

    # Use the same indices the cache holds for these symbols (assumes alignment).
    full_bars = cache.load_range(universe[0], args.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
    start_idx = int(full_bars[full_bars["ts_ms"] == int(primary_bars.iloc[0]["ts_ms"])].index[0])
    end_idx = int(full_bars[full_bars["ts_ms"] == int(primary_bars.iloc[-1]["ts_ms"])].index[0])

    # One "path" containing every bar — the full contiguous walk.
    full_path = CPCVPath(
        test_groups=(0,),
        test_indices=list(range(start_idx, end_idx + 1)),
        train_indices=[],
    )

    builder = SnapshotBuilder(
        cache=cache, universe=universe,
        lookback_bars=args.lookback_bars, timeframes=(args.timeframe,),
    )
    bars_by_symbol = {
        sym: cache.load_range(sym, args.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
        for sym in universe
    }
    decision_cache = DecisionCache(runs_dir / "decisions.db")
    fill_config = FillConfig(
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        funding_rate_per_day=args.funding_rate_per_day,
    )
    baseline_cfg = BaselineConfig(
        momentum_lookback_bars=args.momentum_lookback,
        drawdown_hedge_threshold=args.hedge_threshold,
        long_size_usd_pct=args.size_pct,
        primary_symbol=universe[0],
        hedge_symbol=universe[2] if len(universe) > 2 else universe[0],
        enable_shorts=args.enable_shorts,
        short_stop_loss_pct=args.short_stop_loss_pct,
        long_stop_loss_pct=args.long_stop_loss_pct,
        short_min_drawdown=args.short_min_drawdown,
        confluence_lookback_bars=args.confluence_lookback,
        rotation_symbols=tuple(s.strip() for s in args.rotation_symbols.split(",") if s.strip()),
        vol_target_pct=args.vol_target_pct,
        vol_lookback_bars=args.vol_lookback,
        vol_size_max_mult=args.vol_size_max_mult,
    )

    # Use the actual full bars (bars_by_symbol[universe[0]]) for the walk
    populate_path(
        path=full_path, builder=builder, decision_cache=decision_cache,
        universe=universe, timeframe=args.timeframe,
        primary_bars=bars_by_symbol[universe[0]],
        bars_by_symbol=bars_by_symbol,
        initial_equity=args.initial_equity,
        fill_config=fill_config, baseline_cfg=baseline_cfg,
    )
    result = run_path(
        path=full_path, bars_by_symbol=bars_by_symbol,
        snapshot_builder=builder, decision_cache=decision_cache,
        fill_config=fill_config, initial_equity=args.initial_equity,
    )

    equity = np.array([eq for _, eq in result.equity_curve], dtype=float)
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
    moments = pnl_moments(returns) if returns.size else {"mean": 0, "std": 0, "skew": 0, "kurtosis": 3}
    sharpe = sharpe_annualized(returns, periods_per_year=args.periods_per_year) if returns.size else 0.0
    sortino = sortino_ratio(returns, periods_per_year=args.periods_per_year) if returns.size else 0.0
    calmar = calmar_ratio(equity, periods_per_year=args.periods_per_year) if len(equity) > 1 else 0.0
    dd, peak_idx, trough_idx = max_drawdown(equity) if len(equity) else (0.0, 0, 0)
    psr = probabilistic_sharpe_ratio(
        sr_hat=sharpe, n=len(returns), skew=moments["skew"], kurt=moments["kurtosis"]
    ) if len(returns) > 1 else 0.0
    boot_lo, boot_hi = block_bootstrap_sharpe_ci(
        returns, periods_per_year=args.periods_per_year,
        n_resamples=2000, block_size=4, seed=0,
    ) if returns.size >= 8 else (math.nan, math.nan)

    # Buy-and-hold benchmarks over the same indices
    bh_btc = buy_and_hold(cache, universe[0], args.timeframe, start_idx, end_idx,
                          args.initial_equity, fill_config)
    bh_eth = buy_and_hold(cache, universe[1], args.timeframe, start_idx, end_idx,
                          args.initial_equity, fill_config) if len(universe) > 1 else None
    ew = equal_weight(cache, universe, args.timeframe, start_idx, end_idx,
                      args.initial_equity, fill_config)

    print(f"\n=== Strategy (M={args.momentum_lookback}, H={args.hedge_threshold}, S={args.size_pct}) ===")
    print(f"Final equity:     ${result.final_equity:,.2f}")
    print(f"Total return:     {(result.final_equity - result.initial_equity) / result.initial_equity:+.1%}")
    n_years = (parse_iso(args.end) - parse_iso(args.start)) / (365.25 * 86_400_000)
    cagr = ((result.final_equity / result.initial_equity) ** (1 / n_years) - 1.0) if n_years > 0 else 0.0
    print(f"CAGR:             {cagr:+.1%}")
    print(f"Sharpe (ann.):    {sharpe:+.2f}")
    print(f"Sortino:          {sortino:+.2f}")
    print(f"Calmar:           {calmar:+.2f}")
    print(f"Max drawdown:     {dd:.1%}")
    print(f"PSR (vs 0):       {psr:.3f}")
    print(f"Bootstrap CI95:   [{boot_lo:+.2f}, {boot_hi:+.2f}]" if not math.isnan(boot_lo) else "")
    print(f"Total trades:     {len(result.trades)}")
    print(f"Skewness:         {moments['skew']:+.2f}")
    print(f"Kurtosis:         {moments['kurtosis']:.2f}")

    print(f"\n=== Benchmarks (same window, same fees) ===")
    print(f"Buy-and-hold BTC: ${bh_btc.final_equity:,.2f}  "
          f"({(bh_btc.final_equity - bh_btc.initial_equity) / bh_btc.initial_equity:+.1%})")
    if bh_eth:
        print(f"Buy-and-hold ETH: ${bh_eth.final_equity:,.2f}  "
              f"({(bh_eth.final_equity - bh_eth.initial_equity) / bh_eth.initial_equity:+.1%})")
    print(f"Equal-weight all: ${ew.final_equity:,.2f}  "
          f"({(ew.final_equity - ew.initial_equity) / ew.initial_equity:+.1%})")

    # Save artifacts
    equity_path = runs_dir / "equity_curve.csv"
    with open(equity_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts_ms", "equity"])
        for ts, eq in result.equity_curve:
            w.writerow([ts, eq])

    trades_path = runs_dir / "trades.csv"
    with open(trades_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "side", "size", "entry_price", "exit_price",
                    "exit_reason", "pnl", "entry_ts_ms", "exit_ts_ms"])
        for t in result.trades:
            w.writerow([t.symbol, t.side, t.size, t.entry_price, t.exit_price,
                        t.exit_reason, t.pnl, t.entry_ts_ms, t.exit_ts_ms])

    summary = {
        "config": {
            "momentum_lookback_bars": args.momentum_lookback,
            "drawdown_hedge_threshold": args.hedge_threshold,
            "long_size_usd_pct": args.size_pct,
            "primary_symbol": universe[0],
            "hedge_symbol": universe[2] if len(universe) > 2 else universe[0],
        },
        "strategy": {
            "final_equity": result.final_equity,
            "total_return_pct": (result.final_equity - result.initial_equity) / result.initial_equity,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": dd,
            "psr": psr,
            "n_trades": len(result.trades),
            "skew": moments["skew"],
            "kurtosis": moments["kurtosis"],
            "bootstrap_ci_lo_95": None if math.isnan(boot_lo) else boot_lo,
            "bootstrap_ci_hi_95": None if math.isnan(boot_hi) else boot_hi,
        },
        "benchmarks": {
            "buy_and_hold_btc": {
                "final_equity": bh_btc.final_equity,
                "return_pct": (bh_btc.final_equity - bh_btc.initial_equity) / bh_btc.initial_equity,
            },
            "buy_and_hold_eth": {
                "final_equity": bh_eth.final_equity,
                "return_pct": (bh_eth.final_equity - bh_eth.initial_equity) / bh_eth.initial_equity,
            } if bh_eth else None,
            "equal_weight": {
                "final_equity": ew.final_equity,
                "return_pct": (ew.final_equity - ew.initial_equity) / ew.initial_equity,
            },
        },
    }
    (runs_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nArtifacts: {runs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
