#!/usr/bin/env python3
"""Compare a strategy's CPCV result against passive benchmarks.

Runs buy-and-hold and equal-weight benchmarks over the same window with the
same fee structure, then reports relative outperformance:

    excess_return = strategy_return - benchmark_return
    information_ratio ≈ excess_return / vol(excess)
    win_vs_benchmark = % of CPCV paths that beat the benchmark

Usage:
    python scripts/compare_to_benchmarks.py \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db \\
        --symbols BTC/USDT:USDT,ETH/USDT:USDT,PAXG/USDT \\
        --timeframe 1w \\
        --start 2021-01-04 --end 2026-04-27 \\
        --strategy-summary runs/sweep-v1/best_config_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.benchmarks import buy_and_hold, equal_weight
from src.backtest.data_cache import OHLCVCache
from src.backtest.fill_model import FillConfig


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-db", required=True)
    parser.add_argument("--symbols", required=True, help="comma-separated; first is primary")
    parser.add_argument("--timeframe", default="1w")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--initial-equity", type=float, default=2000.0)
    parser.add_argument("--taker-fee-bps", type=float, default=4.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--strategy-summary", help="optional JSON file with strategy summary to compare")
    args = parser.parse_args(argv)

    cache = OHLCVCache(args.cache_db)
    universe = [s.strip() for s in args.symbols.split(",") if s.strip()]
    primary = universe[0]
    fill_cfg = FillConfig(taker_fee_bps=args.taker_fee_bps, slippage_bps=args.slippage_bps)

    # Find the bar indices spanning the window
    primary_bars = cache.load_range(primary, args.timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
    start_ms = parse_iso(args.start)
    end_ms = parse_iso(args.end)
    in_window = primary_bars[(primary_bars["ts_ms"] >= start_ms) & (primary_bars["ts_ms"] <= end_ms)]
    if in_window.empty:
        print(f"No bars in window for {primary}", file=sys.stderr)
        return 2
    start_idx = int(in_window.index[0])
    end_idx = int(in_window.index[-1])
    print(f"Window indices in cache: [{start_idx}..{end_idx}] ({end_idx - start_idx + 1} bars)")

    # Buy-and-hold each symbol
    print("\n=== Buy-and-Hold benchmarks ===")
    bh_results = {}
    for sym in universe:
        r = buy_and_hold(
            cache=cache, symbol=sym, timeframe=args.timeframe,
            start_idx=start_idx, end_idx=end_idx,
            initial_equity=args.initial_equity, fill_config=fill_cfg,
        )
        ret_pct = (r.final_equity - r.initial_equity) / r.initial_equity
        bh_results[sym] = {"final_equity": r.final_equity, "return_pct": ret_pct}
        print(f"  {sym}: final=${r.final_equity:.2f}, return={ret_pct:+.1%}")

    # Equal-weight portfolios
    print("\n=== Equal-weight benchmarks ===")
    ew_results = {}
    for combo in [universe, [universe[0], universe[-1]] if len(universe) > 1 else universe]:
        r = equal_weight(
            cache=cache, symbols=combo, timeframe=args.timeframe,
            start_idx=start_idx, end_idx=end_idx,
            initial_equity=args.initial_equity, fill_config=fill_cfg,
        )
        ret_pct = (r.final_equity - r.initial_equity) / r.initial_equity
        key = "+".join(combo)
        ew_results[key] = {"final_equity": r.final_equity, "return_pct": ret_pct}
        print(f"  EW({key}): final=${r.final_equity:.2f}, return={ret_pct:+.1%}")

    # Strategy comparison
    if args.strategy_summary:
        s = json.loads(Path(args.strategy_summary).read_text())
        ssum = s.get("summary", s)
        print("\n=== Strategy vs benchmarks ===")
        s_ret = ssum.get("median_return_pct", 0.0)
        primary_ret = bh_results[primary]["return_pct"]
        print(f"Strategy median return:  {s_ret:+.1%}")
        print(f"BH primary return:       {primary_ret:+.1%}")
        excess = s_ret - primary_ret
        print(f"Excess return:           {excess:+.1%}")
        print(f"Strategy median Sharpe:  {ssum.get('median_sharpe', 0):.2f}")
        print(f"Strategy max drawdown:   {ssum.get('median_max_drawdown', 0):.2%}")

    # Persist
    out = {"buy_and_hold": bh_results, "equal_weight": ew_results}
    print("\nJSON output:")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
