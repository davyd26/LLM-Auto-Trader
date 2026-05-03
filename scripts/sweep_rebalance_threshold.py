#!/usr/bin/env python3
"""Sweep rebalance threshold (rebalance frequency) on the OOS-passing config.

Lower thresholds rebalance more often (catch every signal change but pay
more fees). Higher thresholds rebalance less (lose timing precision but
save fees). Find the sweet spot for anchor_plus_BNO/MinVariance.

All CAGR / DD figures USD-denominated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.analysis.backtest_stats import (
    calmar_ratio,
    max_drawdown,
    sharpe_annualized,
)
from src.backtest.data_cache import OHLCVCache
from src.backtest.fill_model import FillConfig
from src.backtest.llm_portfolio_decisions import (
    inverse_vol_decision_source,
    markowitz_min_var_decision_source,
    strawman_decision_source,
)
from src.backtest.llm_portfolio_strategy import USD_CASH, run_llm_portfolio
from src.backtest.quant_overlay import QuantOverlayConfig

TRAIN_START = "2021-01-04"
TRAIN_END = "2025-09-29"
OOS_START = "2025-10-06"
OOS_END = "2026-04-27"

# Sweep these thresholds × the top-3 universes × 3 rebalancers
THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

UNIVERSES = {
    "anchor_plus_BNO":     ["BTC/USDT:USDT", "GLD", "BNO", USD_CASH],
    "anchor_btc_gld_usd":  ["BTC/USDT:USDT", "GLD", USD_CASH],
    "anchor_plus_eth":     ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT", USD_CASH],
}

REBALANCERS = {
    "MinVariance":  markowitz_min_var_decision_source,
    "InverseVol":   inverse_vol_decision_source,
    "Strawman":     strawman_decision_source,
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
    cal = calmar_ratio(eq, periods_per_year=periods_per_year)
    return {"cagr_usd": cagr, "sharpe_usd": sharpe, "dd_usd": dd, "calmar_usd": cal,
            "log_returns": log_ret}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--out-dir", default="runs/portfolio-rebalance-sweep")
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

    # Pre-load both training and full (train+OOS) bars
    bars_train = {}
    bars_full = {}
    train_start, train_end = parse_iso(TRAIN_START), parse_iso(TRAIN_END)
    full_end = parse_iso(OOS_END)
    for sym in all_syms:
        df_t = cache.load_range(sym, "1w", train_start, train_end)
        df_f = cache.load_range(sym, "1w", train_start, full_end)
        if not df_t.empty:
            bars_train[sym] = df_t
        if not df_f.empty:
            bars_full[sym] = df_f
    cache.close()

    print("\n" + "═" * 100)
    print("REBALANCE-THRESHOLD SWEEP (USD-denominated)")
    print("Higher threshold = less frequent rebalancing = lower fees but slower")
    print("response to regime changes.")
    print("═" * 100)
    print(f"  {'universe':24s} {'rebalancer':12s} {'thr':>5s} "
          f"{'TRAIN_CAGR':>11s} {'TRAIN_Sh':>9s} {'TRAIN_DD':>9s}  "
          f"{'FULL_CAGR':>10s} {'FULL_Sh':>8s} {'FULL_DD':>8s}  trades")

    rows = []
    for univ_name, univ in UNIVERSES.items():
        for reb_name, src in REBALANCERS.items():
            for thr in THRESHOLDS:
                # Run on training window only
                sub_t = {s: bars_train[s] for s in univ
                         if s != USD_CASH and s in bars_train}
                if not sub_t:
                    continue
                try:
                    res_t = run_llm_portfolio(
                        universe=univ, bars_by_symbol=sub_t,
                        decision_source=src, fill_config=fill_cfg,
                        initial_equity_usd=args.initial_equity,
                        anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                        quant_overlay=overlay, rebalance_threshold=thr,
                    )
                except Exception:
                    continue
                stats_t = _stats(args.initial_equity, res_t.equity_curve)

                # Run on full (train + OOS) window for comparison
                sub_f = {s: bars_full[s] for s in univ
                         if s != USD_CASH and s in bars_full}
                try:
                    res_f = run_llm_portfolio(
                        universe=univ, bars_by_symbol=sub_f,
                        decision_source=src, fill_config=fill_cfg,
                        initial_equity_usd=args.initial_equity,
                        anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                        quant_overlay=overlay, rebalance_threshold=thr,
                    )
                    stats_f = _stats(args.initial_equity, res_f.equity_curve)
                except Exception:
                    stats_f = stats_t

                row = {
                    "universe": univ_name, "rebalancer": reb_name, "threshold": thr,
                    "train_cagr_usd": stats_t["cagr_usd"],
                    "train_sharpe_usd": stats_t["sharpe_usd"],
                    "train_dd_usd": stats_t["dd_usd"],
                    "train_calmar_usd": stats_t["calmar_usd"],
                    "full_cagr_usd": stats_f["cagr_usd"],
                    "full_sharpe_usd": stats_f["sharpe_usd"],
                    "full_dd_usd": stats_f["dd_usd"],
                    "n_trades": len(res_t.trades),
                }
                rows.append(row)
                print(f"  {univ_name:24s} {reb_name:12s} {thr:>5.2f} "
                      f"{stats_t['cagr_usd']*100:+10.1f}% {stats_t['sharpe_usd']:+9.2f} "
                      f"{stats_t['dd_usd']*100:8.1f}%  "
                      f"{stats_f['cagr_usd']*100:+9.1f}% {stats_f['sharpe_usd']:+8.2f} "
                      f"{stats_f['dd_usd']*100:7.1f}%  {len(res_t.trades):>3d}")

    if rows:
        with (out_dir / "rebalance_sweep.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Best threshold per (universe, rebalancer)
        print("\n" + "═" * 100)
        print("BEST THRESHOLD per (universe × rebalancer) by full-window Sharpe")
        print("═" * 100)
        seen = set()
        ranked = sorted(rows, key=lambda r: -r["full_sharpe_usd"])
        for r in ranked:
            key = (r["universe"], r["rebalancer"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  {r['universe']:24s} {r['rebalancer']:12s} "
                  f"thr={r['threshold']:>5.2f}  "
                  f"FULL: CAGR$={r['full_cagr_usd']*100:+6.1f}% "
                  f"Sh$={r['full_sharpe_usd']:+5.2f} DD$={r['full_dd_usd']*100:5.1f}%  "
                  f"trades={r['n_trades']}")

        print("\n" + "═" * 100)
        print("TOP 10 (universe × rebalancer × threshold) by full-window Sharpe")
        print("═" * 100)
        for i, r in enumerate(ranked[:10], start=1):
            print(f"  {i:<3d} {r['universe']:24s} {r['rebalancer']:12s} "
                  f"thr={r['threshold']:>5.2f}  "
                  f"FULL: CAGR$={r['full_cagr_usd']*100:+6.1f}% "
                  f"Sh$={r['full_sharpe_usd']:+5.2f} DD$={r['full_dd_usd']*100:5.1f}%")
        print(f"\n→ {out_dir}/rebalance_sweep.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
