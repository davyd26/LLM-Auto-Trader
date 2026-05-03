#!/usr/bin/env python3
"""Final LLM-vs-deterministic comparison on the 4 user-specified subsets.

Reads the opus-populated LLMPortfolioCache and runs the LLM column
against the deterministic baselines (EqualWeight, InverseVol,
MinVariance, Strawman, Strawman_v2) — all 6 columns side-by-side.

If a snapshot has no cached LLM decision, falls back to equal-weight
for that one bar (so partial-cache runs still produce comparable
numbers).

All CAGR / DD / Sharpe figures USD-denominated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.analysis.backtest_stats import (
    calmar_ratio,
    max_drawdown,
    sharpe_annualized,
    sortino_ratio,
)
from src.backtest.data_cache import OHLCVCache
from src.backtest.fill_model import FillConfig
from src.backtest.llm_portfolio_decisions import (
    LLMPortfolioCache,
    equal_weight_decision_source,
    inverse_vol_decision_source,
    markowitz_min_var_decision_source,
    strawman_decision_source,
    strawman_v2_decision_source,
)
from src.backtest.llm_portfolio_strategy import (
    USD_CASH,
    run_llm_portfolio,
    snapshot_hash,
)
from src.backtest.quant_overlay import QuantOverlayConfig

TRAIN_START = "2021-01-04"
TRAIN_END = "2025-09-29"

# The 4 user-specified subsets (the opus dispatcher's targets)
USER_SUBSETS = {
    "pair_BTC_GLD":         ["BTC/USDT:USDT", "GLD"],
    "pair_BTC_USD":         ["BTC/USDT:USDT", USD_CASH],
    "pair_GLD_USD":         ["GLD", USD_CASH],
    "anchor_btc_gld_usd":   ["BTC/USDT:USDT", "GLD", USD_CASH],
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
    sortino = sortino_ratio(log_ret, periods_per_year=periods_per_year)
    dd, _, _ = max_drawdown(eq)
    cal = calmar_ratio(eq, periods_per_year=periods_per_year)
    return {"final_usd": final, "cagr_usd": cagr, "sharpe_usd": sharpe,
            "sortino_usd": sortino, "max_drawdown_usd": dd, "calmar_usd": cal}


def make_cache_only_llm_source(cache: LLMPortfolioCache, fallback):
    """Build a decision source that only consults the cache. If the snapshot
    isn't cached, fall back to ``fallback`` (a deterministic source).

    No live ``claude -p`` calls. Used for fast/free replay in comparisons.
    """
    n_hits = [0]
    n_misses = [0]

    def _src(snapshot: dict) -> dict[str, float]:
        h = snapshot_hash(snapshot)
        cached = cache.get(h)
        if cached is not None:
            n_hits[0] += 1
            return cached
        n_misses[0] += 1
        return fallback(snapshot)

    _src.hits = n_hits  # type: ignore
    _src.misses = n_misses  # type: ignore
    return _src


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--llm-cache-db", default="runs/llm-portfolio/llm_decisions_opus.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--rebalance-threshold", type=float, default=0.03,
                   help="match the value used during opus dispatch (default: 0.03)")
    p.add_argument("--out-dir", default="runs/llm-portfolio")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0,
                          funding_rate_per_day=0.00010)
    overlay = QuantOverlayConfig(use_garch_vol=False)
    llm_cache = LLMPortfolioCache(args.llm_cache_db)

    print(f"\nLLM cache: {llm_cache.total()} decisions")
    print(f"Window: {TRAIN_START} → {TRAIN_END}")
    print("All figures USD-denominated. Rebalance threshold:", args.rebalance_threshold)
    print()

    train_ms = (parse_iso(TRAIN_START), parse_iso(TRAIN_END))
    all_syms = set()
    for univ in USER_SUBSETS.values():
        for s in univ:
            if s != USD_CASH:
                all_syms.add(s)
    bars_by_symbol = {}
    for sym in all_syms:
        df = cache.load_range(sym, "1w", *train_ms)
        if not df.empty:
            bars_by_symbol[sym] = df
    cache.close()

    rows = []
    sources = {
        "EqualWeight":  equal_weight_decision_source,
        "InverseVol":   inverse_vol_decision_source,
        "MinVariance":  markowitz_min_var_decision_source,
        "Strawman":     strawman_decision_source,
        "Strawman_v2":  strawman_v2_decision_source,
    }

    for sub_name, univ in USER_SUBSETS.items():
        sub_bars = {s: bars_by_symbol[s] for s in univ
                    if s != USD_CASH and s in bars_by_symbol}
        if not sub_bars:
            continue

        # Run all 5 deterministic sources
        for src_name, src in sources.items():
            try:
                res = run_llm_portfolio(
                    universe=univ, bars_by_symbol=sub_bars,
                    decision_source=src, fill_config=fill_cfg,
                    initial_equity_usd=args.initial_equity,
                    anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                    quant_overlay=overlay,
                    rebalance_threshold=args.rebalance_threshold,
                )
            except Exception as e:
                print(f"  ERR {sub_name}/{src_name}: {e}")
                continue
            stats = _stats(args.initial_equity, res.equity_curve)
            rows.append({
                "subset": sub_name, "decision_source": src_name,
                **stats, "n_trades": len(res.trades),
                "llm_cache_hits": 0, "llm_cache_misses": 0,
            })

        # Run the LLM column with cache-only fallback
        llm_src = make_cache_only_llm_source(llm_cache, equal_weight_decision_source)
        try:
            res = run_llm_portfolio(
                universe=univ, bars_by_symbol=sub_bars,
                decision_source=llm_src, fill_config=fill_cfg,
                initial_equity_usd=args.initial_equity,
                anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                quant_overlay=overlay,
                rebalance_threshold=args.rebalance_threshold,
            )
            stats = _stats(args.initial_equity, res.equity_curve)
            rows.append({
                "subset": sub_name, "decision_source": "LLM(opus)",
                **stats, "n_trades": len(res.trades),
                "llm_cache_hits": llm_src.hits[0],
                "llm_cache_misses": llm_src.misses[0],
            })
        except Exception as e:
            print(f"  ERR {sub_name}/LLM: {e}")

    llm_cache.close()

    # Pretty tables
    SRC_LABEL = {
        "EqualWeight": "EqualWeight", "InverseVol": "InverseVol",
        "MinVariance": "MinVariance", "Strawman": "Strawman",
        "Strawman_v2": "Strawman_v2", "LLM(opus)": "LLM(opus)",
    }
    src_order = ["EqualWeight", "InverseVol", "MinVariance",
                 "Strawman", "Strawman_v2", "LLM(opus)"]

    def _print_table(title: str, value_fn, value_format: str,
                     better: str = "higher"):
        print("═" * 110)
        print(title)
        print("═" * 110)
        header = f"  {'subset':24s}"
        for src in src_order:
            header += f"│ {SRC_LABEL[src]:>13s}"
        print(header)
        print("  " + "─" * 24 + "".join(["│" + "─" * 14] * len(src_order)))
        for sub in USER_SUBSETS:
            line = f"  {sub:24s}"
            vals = {}
            for src in src_order:
                m = next((r for r in rows
                          if r["subset"] == sub and r["decision_source"] == src), None)
                vals[src] = value_fn(m) if m else None
            valid = [v for v in vals.values() if v is not None]
            if valid:
                best = max(valid) if better == "higher" else min(valid)
            else:
                best = None
            for src in src_order:
                v = vals[src]
                if v is None:
                    line += f"│ {' --- ':>13s}"
                else:
                    star = "★" if v == best else " "
                    line += f"│ {star}{value_format.format(v):>12s}"
            print(line)
        print()

    # LLM cache coverage report
    print("LLM CACHE COVERAGE per subset (hits = real LLM decisions used,")
    print("                              misses = fell back to equal-weight):")
    for r in rows:
        if r["decision_source"] == "LLM(opus)":
            total = r["llm_cache_hits"] + r["llm_cache_misses"]
            pct = (r["llm_cache_hits"] / total * 100) if total > 0 else 0.0
            verdict = "FULL LLM" if pct > 95 else (
                "MOSTLY LLM" if pct > 50 else
                "MOSTLY FALLBACK" if pct > 5 else "NO LLM YET"
            )
            print(f"  {r['subset']:24s}  {r['llm_cache_hits']:>3d} hits / "
                  f"{total:>3d} bars = {pct:>5.1f}%  [{verdict}]")
    print()

    _print_table("TABLE 1 — Sharpe (USD-denominated). ★ = best per universe.",
                 lambda r: r["sharpe_usd"], "{:+.2f}")
    _print_table("TABLE 2 — CAGR_USD (annualized USD return).",
                 lambda r: r["cagr_usd"], "{:+.1%}")
    _print_table("TABLE 3 — Max drawdown_USD (LOWER is better, ★ = safest).",
                 lambda r: r["max_drawdown_usd"], "{:.1%}", better="lower")
    _print_table("TABLE 4 — Calmar (CAGR / DD).",
                 lambda r: r["calmar_usd"], "{:+.2f}")

    if rows:
        with (out_dir / "comparison_with_llm.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Top 5 across all
        print("═" * 110)
        print("TOP 5 across all (subset × strategy) by Sharpe")
        print("═" * 110)
        ranked = sorted(rows, key=lambda r: -r["sharpe_usd"])[:5]
        for i, r in enumerate(ranked, start=1):
            llm_note = f" [{r['llm_cache_hits']} hits / {r['llm_cache_misses']} miss]" \
                if r["decision_source"] == "LLM(opus)" else ""
            print(f"  {i}  {r['subset']:24s} {r['decision_source']:14s} "
                  f"Sh$={r['sharpe_usd']:+.2f}  "
                  f"CAGR$={r['cagr_usd']*100:+.1f}%  "
                  f"DD$={r['max_drawdown_usd']*100:.1f}%{llm_note}")

        meta = {
            "training_window": {"start": TRAIN_START, "end": TRAIN_END},
            "subsets": list(USER_SUBSETS.keys()),
            "decision_sources": src_order,
            "convention": "All CAGR / Sharpe / DD figures USD-denominated.",
            "llm_total_cached": llm_cache.total() if False else None,
            "rebalance_threshold": args.rebalance_threshold,
        }
        (out_dir / "comparison_with_llm_meta.json").write_text(
            json.dumps(meta, indent=2, default=float)
        )
        print(f"\n→ {out_dir}/comparison_with_llm.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
