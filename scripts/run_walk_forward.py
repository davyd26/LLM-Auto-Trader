#!/usr/bin/env python3
"""Walk-forward portfolio analysis.

For each (universe × rebalancer):
  - Slice the training window into rolling 2-year train, 13-week test segments
  - At each segment, simulate the strategy through the test bars
  - Report Sharpe / CAGR / DD per segment, then median across segments

This is complementary to CPCV: walk-forward is about temporal stability
(does the strategy work in CONSECUTIVE periods?), CPCV is about regime
robustness (does it work in any random subset of bars?). A robust strategy
should pass both.

All CAGR / DD / Sharpe figures USD-denominated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.analysis.backtest_stats import max_drawdown, sharpe_annualized
from src.backtest.data_cache import OHLCVCache
from src.backtest.fill_model import FillConfig
from src.backtest.llm_portfolio_decisions import (
    equal_weight_decision_source,
    inverse_vol_decision_source,
    markowitz_min_var_decision_source,
    strawman_decision_source,
    strawman_v2_decision_source,
)
from src.backtest.llm_portfolio_strategy import USD_CASH, run_llm_portfolio
from src.backtest.quant_overlay import QuantOverlayConfig

TRAIN_START = "2021-01-04"
TRAIN_END = "2025-09-29"
WEEKLY_MS = 7 * 86_400_000

REBALANCERS = {
    "EqualWeight":  equal_weight_decision_source,
    "InverseVol":   inverse_vol_decision_source,
    "MinVariance":  markowitz_min_var_decision_source,
    "Strawman":     strawman_decision_source,
    "Strawman_v2":  strawman_v2_decision_source,
}

UNIVERSES: dict[str, list[str]] = {
    "pair_BTC_GLD":         ["BTC/USDT:USDT", "GLD"],
    "pair_BTC_USD":         ["BTC/USDT:USDT", USD_CASH],
    "pair_GLD_USD":         ["GLD", USD_CASH],
    "anchor_btc_gld_usd":   ["BTC/USDT:USDT", "GLD", USD_CASH],
    "anchor_plus_sol":      ["BTC/USDT:USDT", "GLD", "SOL/USDT:USDT", USD_CASH],
    "anchor_plus_NVDA":     ["BTC/USDT:USDT", "GLD", "NVDA", USD_CASH],
    "anchor_plus_BNO":      ["BTC/USDT:USDT", "GLD", "BNO", USD_CASH],
    "anchor_plus_eth":      ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT", USD_CASH],
}


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _stats(initial: float, eq_curve: list, periods_per_year: int = 52) -> dict:
    eq = np.array([e for _, e in eq_curve], dtype=float)
    eq = np.maximum(eq, 1e-9)
    final = float(eq[-1])
    log_ret = np.diff(np.log(eq))
    years = eq.shape[0] / periods_per_year
    cagr = (final / initial) ** (1.0 / max(years, 1e-9)) - 1.0 if final > 0 else -1.0
    sharpe = sharpe_annualized(log_ret, periods_per_year=periods_per_year)
    dd, _, _ = max_drawdown(eq)
    return {"cagr_usd": cagr, "sharpe_usd": sharpe, "dd_usd": dd, "final": final}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--test-weeks", type=int, default=13)
    p.add_argument("--n-segments", type=int, default=8)
    p.add_argument("--out-dir", default="runs/portfolio-walk-forward")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0,
                          funding_rate_per_day=0.00010)
    overlay = QuantOverlayConfig(use_garch_vol=False)

    all_syms = set()
    for univ in UNIVERSES.values():
        for s in univ:
            if s != USD_CASH:
                all_syms.add(s)

    train_ms = parse_iso(TRAIN_START)
    end_ms = parse_iso(TRAIN_END)
    test_ms = args.test_weeks * WEEKLY_MS

    # Build segment boundaries: spaced evenly through the training window
    total_ms = end_ms - train_ms
    step = (total_ms - test_ms) / max(1, args.n_segments - 1) if args.n_segments > 1 else 0
    segments = []
    for i in range(args.n_segments):
        seg_start = train_ms + int(i * step)
        seg_end = min(end_ms, seg_start + test_ms)
        segments.append((seg_start, seg_end))

    print(f"\n{'═'*100}")
    print(f"WALK-FORWARD — {args.n_segments} test segments × {args.test_weeks} weeks each")
    print("All figures USD-denominated.")
    print(f"{'═'*100}")
    print(f"Segments: {[(datetime.fromtimestamp(s/1000, tz=timezone.utc).date(), datetime.fromtimestamp(e/1000, tz=timezone.utc).date()) for s, e in segments]}\n")

    rows = []
    for univ_name, univ in UNIVERSES.items():
        for reb_name, src in REBALANCERS.items():
            seg_sharpes = []
            seg_cagrs = []
            seg_dds = []
            for seg_start, seg_end in segments:
                bars_by_symbol = {}
                for sym in all_syms:
                    df = cache.load_range(sym, "1w", seg_start, seg_end)
                    if not df.empty:
                        bars_by_symbol[sym] = df
                avail = [s for s in univ if s == USD_CASH or s in bars_by_symbol]
                if len([s for s in avail if s != USD_CASH]) == 0:
                    continue
                sub_bars = {s: bars_by_symbol[s] for s in avail
                            if s != USD_CASH and s in bars_by_symbol}
                if not sub_bars:
                    continue
                try:
                    res = run_llm_portfolio(
                        universe=avail, bars_by_symbol=sub_bars,
                        decision_source=src, fill_config=fill_cfg,
                        initial_equity_usd=args.initial_equity,
                        anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                        quant_overlay=overlay, rebalance_threshold=0.05,
                    )
                except Exception:
                    continue
                stats = _stats(args.initial_equity, res.equity_curve)
                seg_sharpes.append(stats["sharpe_usd"])
                seg_cagrs.append(stats["cagr_usd"])
                seg_dds.append(stats["dd_usd"])

            if not seg_sharpes:
                continue
            sh_arr = np.asarray(seg_sharpes)
            cg_arr = np.asarray(seg_cagrs)
            dd_arr = np.asarray(seg_dds)
            rows.append({
                "universe": univ_name, "rebalancer": reb_name,
                "n_segments": len(seg_sharpes),
                "med_sharpe": float(np.median(sh_arr)),
                "p25_sharpe": float(np.percentile(sh_arr, 25)),
                "p75_sharpe": float(np.percentile(sh_arr, 75)),
                "med_cagr": float(np.median(cg_arr)),
                "med_dd": float(np.median(dd_arr)),
                "max_dd_across_segments": float(np.max(dd_arr)),
                "frac_segments_positive": float(np.mean(sh_arr > 0)),
                "frac_segments_dd_under_15": float(np.mean(dd_arr < 0.15)),
            })
            print(f"  {univ_name:24s} {reb_name:12s} "
                  f"med_Sh={np.median(sh_arr):+5.2f} p25={np.percentile(sh_arr,25):+5.2f} "
                  f"p75={np.percentile(sh_arr,75):+5.2f}  med_CAGR={np.median(cg_arr)*100:+6.1f}% "
                  f"med_DD={np.median(dd_arr)*100:5.1f}% maxDD={np.max(dd_arr)*100:5.1f}%  "
                  f"frac>0={np.mean(sh_arr>0):.2f}")

    cache.close()

    if rows:
        with (out_dir / "walk_forward.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"\n{'═'*100}")
        print("TOP 10 by walk-forward median Sharpe (USD)")
        print(f"{'═'*100}")
        ranked = sorted(rows, key=lambda r: -r["med_sharpe"])[:10]
        print(f"  {'rank':<4s} {'universe':24s} {'rebalancer':12s} "
              f"{'med_Sh':>7s} {'med_CAGR':>9s} {'med_DD':>7s} {'maxDD':>7s} {'fr>0':>5s}")
        for i, r in enumerate(ranked, start=1):
            print(f"  {i:<4d} {r['universe']:24s} {r['rebalancer']:12s} "
                  f"{r['med_sharpe']:+7.2f} {r['med_cagr']*100:+8.1f}% "
                  f"{r['med_dd']*100:6.1f}% {r['max_dd_across_segments']*100:6.1f}%  "
                  f"{r['frac_segments_positive']:5.2f}")
        print(f"\n→ {out_dir}/walk_forward.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
