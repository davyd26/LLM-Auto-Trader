#!/usr/bin/env python3
"""Augmented-universe sweep — what if we extend the OOS-passing config?

Starts from anchor_plus_BNO (BTC+GLD+BNO+USD) and tests adding ONE more
asset at a time across all candidate symbols. If the augmented universe
materially improves Sharpe AND survives OOS, it becomes the new pick.

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
from src.backtest.llm_portfolio_decisions import markowitz_min_var_decision_source
from src.backtest.llm_portfolio_strategy import USD_CASH, run_llm_portfolio
from src.backtest.quant_overlay import QuantOverlayConfig

TRAIN_START = "2021-01-04"
TRAIN_END = "2025-09-29"
OOS_START = "2025-10-06"
OOS_END = "2026-04-27"

BASE_UNIVERSE = ["BTC/USDT:USDT", "GLD", "BNO", USD_CASH]
THRESHOLD = 0.20

CANDIDATES_TO_ADD = [
    "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA", "COIN", "MSTR",
    "SPY", "QQQ", "IWM", "DIA",
    "XLE", "XLF", "XLK",
    "SLV", "USO", "UCO",
]


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
    return {"cagr_usd": cagr, "sharpe_usd": sharpe, "dd_usd": dd, "calmar_usd": cal}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--out-dir", default="runs/portfolio-augmented")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0,
                          funding_rate_per_day=0.00010)
    overlay = QuantOverlayConfig(use_garch_vol=False)

    train_ms = parse_iso(TRAIN_START)
    full_end_ms = parse_iso(OOS_END)

    all_syms = set(BASE_UNIVERSE)
    for c in CANDIDATES_TO_ADD:
        all_syms.add(c)
    all_syms.discard(USD_CASH)

    bars = {}
    for sym in all_syms:
        df = cache.load_range(sym, "1w", train_ms, full_end_ms)
        if not df.empty:
            bars[sym] = df
    cache.close()

    print("\n" + "═" * 100)
    print("AUGMENTED-UNIVERSE SWEEP — base = anchor_plus_BNO (BTC+GLD+BNO+USD)")
    print("All figures USD-denominated. Rebalancer = MinVariance, threshold = 0.20")
    print("═" * 100)
    print(f"  {'added asset':12s} {'n':>2s} {'TRAIN_CAGR':>11s} {'TRAIN_Sh':>9s} "
          f"{'TRAIN_DD':>9s}  {'FULL_CAGR':>10s} {'FULL_Sh':>8s} {'FULL_DD':>8s}")

    rows = []
    # Baseline first
    base_universe = list(BASE_UNIVERSE)
    sub_bars = {s: bars[s] for s in base_universe if s != USD_CASH and s in bars}
    res = run_llm_portfolio(
        universe=base_universe, bars_by_symbol=sub_bars,
        decision_source=markowitz_min_var_decision_source, fill_config=fill_cfg,
        initial_equity_usd=args.initial_equity,
        anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
        quant_overlay=overlay, rebalance_threshold=THRESHOLD,
    )
    base_stats_full = _stats(args.initial_equity, res.equity_curve)

    # Train-only run
    bars_train = {s: bars[s].iloc[:np.searchsorted(
        bars[s]["ts_ms"].to_numpy(dtype=np.int64),
        parse_iso(TRAIN_END), side="right")]
        for s in base_universe if s != USD_CASH and s in bars}
    res_t = run_llm_portfolio(
        universe=base_universe, bars_by_symbol=bars_train,
        decision_source=markowitz_min_var_decision_source, fill_config=fill_cfg,
        initial_equity_usd=args.initial_equity,
        anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
        quant_overlay=overlay, rebalance_threshold=THRESHOLD,
    )
    base_stats_train = _stats(args.initial_equity, res_t.equity_curve)
    rows.append({
        "added": "(BASELINE)", "n_symbols": len(base_universe),
        "train_cagr_usd": base_stats_train["cagr_usd"],
        "train_sharpe_usd": base_stats_train["sharpe_usd"],
        "train_dd_usd": base_stats_train["dd_usd"],
        "full_cagr_usd": base_stats_full["cagr_usd"],
        "full_sharpe_usd": base_stats_full["sharpe_usd"],
        "full_dd_usd": base_stats_full["dd_usd"],
    })
    print(f"  {'(BASELINE)':12s} {len(base_universe):>2d} "
          f"{base_stats_train['cagr_usd']*100:+10.1f}% {base_stats_train['sharpe_usd']:+9.2f} "
          f"{base_stats_train['dd_usd']*100:8.1f}%  "
          f"{base_stats_full['cagr_usd']*100:+9.1f}% {base_stats_full['sharpe_usd']:+8.2f} "
          f"{base_stats_full['dd_usd']*100:7.1f}%")

    for cand in CANDIDATES_TO_ADD:
        if cand not in bars:
            continue
        univ = list(BASE_UNIVERSE)
        # Insert before USD_CASH
        univ.insert(-1, cand)
        sub_full = {s: bars[s] for s in univ if s != USD_CASH and s in bars}
        if not sub_full:
            continue
        # Full window
        try:
            res_f = run_llm_portfolio(
                universe=univ, bars_by_symbol=sub_full,
                decision_source=markowitz_min_var_decision_source, fill_config=fill_cfg,
                initial_equity_usd=args.initial_equity,
                anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                quant_overlay=overlay, rebalance_threshold=THRESHOLD,
            )
        except Exception:
            continue
        stats_f = _stats(args.initial_equity, res_f.equity_curve)
        # Train window
        sub_train = {s: bars[s].iloc[:np.searchsorted(
            bars[s]["ts_ms"].to_numpy(dtype=np.int64),
            parse_iso(TRAIN_END), side="right")]
            for s in univ if s != USD_CASH and s in bars}
        try:
            res_t = run_llm_portfolio(
                universe=univ, bars_by_symbol=sub_train,
                decision_source=markowitz_min_var_decision_source, fill_config=fill_cfg,
                initial_equity_usd=args.initial_equity,
                anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                quant_overlay=overlay, rebalance_threshold=THRESHOLD,
            )
            stats_t = _stats(args.initial_equity, res_t.equity_curve)
        except Exception:
            stats_t = stats_f
        rows.append({
            "added": cand, "n_symbols": len(univ),
            "train_cagr_usd": stats_t["cagr_usd"],
            "train_sharpe_usd": stats_t["sharpe_usd"],
            "train_dd_usd": stats_t["dd_usd"],
            "full_cagr_usd": stats_f["cagr_usd"],
            "full_sharpe_usd": stats_f["sharpe_usd"],
            "full_dd_usd": stats_f["dd_usd"],
        })
        print(f"  {cand:12s} {len(univ):>2d} "
              f"{stats_t['cagr_usd']*100:+10.1f}% {stats_t['sharpe_usd']:+9.2f} "
              f"{stats_t['dd_usd']*100:8.1f}%  "
              f"{stats_f['cagr_usd']*100:+9.1f}% {stats_f['sharpe_usd']:+8.2f} "
              f"{stats_f['dd_usd']*100:7.1f}%")

    if rows:
        with (out_dir / "augmented.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Show top by full Sharpe and identify which ADDS improved over baseline
        print("\n" + "═" * 100)
        print("TOP 10 augmented universes by FULL-window Sharpe")
        print("═" * 100)
        ranked = sorted(rows, key=lambda r: -r["full_sharpe_usd"])[:10]
        for i, r in enumerate(ranked, start=1):
            delta_vs_baseline = r["full_sharpe_usd"] - rows[0]["full_sharpe_usd"]
            note = "(BASELINE)" if r["added"] == "(BASELINE)" else \
                f"(Δ vs base: {delta_vs_baseline:+.2f})"
            print(f"  {i:<3d} +{r['added']:12s} (n={r['n_symbols']:>2d}) "
                  f"Sh$={r['full_sharpe_usd']:+5.2f} CAGR$={r['full_cagr_usd']*100:+6.1f}% "
                  f"DD$={r['full_dd_usd']*100:5.1f}% {note}")
        print(f"\n→ {out_dir}/augmented.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
