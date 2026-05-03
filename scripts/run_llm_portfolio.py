#!/usr/bin/env python3
"""Multi-symbol LLM-driven portfolio rebalancer with comprehensive
side-by-side comparison across subsets and decision sources.

The strategy: at each weekly bar, the decision source (real LLM via
``claude -p`` or one of three deterministic baselines) sees a rich JSON
context — per-symbol GARCH/MAs/RSI/MACD/volume + portfolio state +
advisory weight schemes (equal-weight, inverse-vol, Markowitz min-var)
— and outputs target weights. The harness rebalances at the next bar's
open with fees + slippage.

ALL CAGR FIGURES ARE USD-DENOMINATED. The portfolio is funded in USD;
every leg's value is converted to USD using its current price; the
USD_CASH leg has zero return; CAGR_USD = (final_USD / initial_USD)^(1/years) − 1.

Decision sources tested per subset:
  - equal_weight       : 1/N rebalanced weekly
  - inverse_vol        : 1/σ risk-parity, weekly
  - markowitz_min_var  : long-only min-var w/ cash cap, weekly
  - strawman           : MA-cross + GARCH-crisis filter, deterministic
                          rule-based "what an LLM might do"
  - llm                : real Claude Code, dispatched via ``claude -p``,
                          cached by snapshot hash

Usage:
    # Deterministic baselines on all subsets (no LLM cost):
    python scripts/run_llm_portfolio.py --decision-sources \\
        equal_weight,inverse_vol,markowitz_min_var,strawman

    # Add LLM smoke test on a specific subset:
    python scripts/run_llm_portfolio.py --decision-sources llm \\
        --subsets anchor_btc_gld_usd --model haiku
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
    make_llm_decision_source,
    markowitz_min_var_decision_source,
    strawman_decision_source,
)
from src.backtest.llm_portfolio_strategy import USD_CASH, run_llm_portfolio
from src.backtest.quant_overlay import QuantOverlayConfig

# Training window = same as sweep_robust.py (the OOS holdout is reserved)
TRAIN_START = "2021-01-04"
TRAIN_END = "2025-09-29"


# Per user requirement: every multi-symbol subset MUST contain
# BTC + GLD + USD_CASH. The pair_* / triple_* subsets explicitly omit
# the anchor for "what does only this combination do" analysis.
def _anchored(symbols: list[str]) -> list[str]:
    """Inject BTC + GLD + USD_CASH if missing (preserve order)."""
    out = list(symbols)
    if "BTC/USDT:USDT" not in out:
        out.insert(0, "BTC/USDT:USDT")
    if "GLD" not in out:
        out.insert(1, "GLD")
    if USD_CASH not in out:
        out.append(USD_CASH)
    return out


SUBSETS: dict[str, list[str]] = {
    # ----- Anchor only -----
    "anchor_btc_gld_usd":  _anchored([]),
    # ----- Anchor + 1 risk asset -----
    "anchor_plus_eth":     _anchored(["ETH/USDT:USDT"]),
    "anchor_plus_sol":     _anchored(["SOL/USDT:USDT"]),
    "anchor_plus_NVDA":    _anchored(["NVDA"]),
    "anchor_plus_BNO":     _anchored(["BNO"]),
    "anchor_plus_SLV":     _anchored(["SLV"]),
    # ----- Anchor + 2-3 risk assets -----
    "anchor_eth_sol":      _anchored(["ETH/USDT:USDT", "SOL/USDT:USDT"]),
    "anchor_eth_nvda":     _anchored(["ETH/USDT:USDT", "NVDA"]),
    "anchor_eth_oil":      _anchored(["ETH/USDT:USDT", "BNO"]),
    # ----- Anchor + larger panels -----
    "crypto_top_5_anchor": _anchored([
        "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT"]),
    "balanced_8":          _anchored([
        "ETH/USDT:USDT", "SPY", "QQQ", "MSFT", "AAPL", "BNO"]),
    "kitchen_sink":        _anchored([
        "ETH/USDT:USDT", "SOL/USDT:USDT", "MSFT", "AAPL", "GOOGL",
        "NVDA", "SPY", "QQQ", "SLV", "USO", "BNO"]),
    # ----- User-requested explicit pairs/triples (NO anchor injection) -----
    "pair_BTC_GLD":             ["BTC/USDT:USDT", "GLD"],
    "pair_BTC_USD":             ["BTC/USDT:USDT", USD_CASH],
    "pair_GLD_USD":             ["GLD", USD_CASH],
    "pair_GLD_OIL":             ["GLD", "BNO"],
    "pair_BTC_OIL":             ["BTC/USDT:USDT", "BNO"],
    "triple_BTC_GLD_OIL":       ["BTC/USDT:USDT", "GLD", "BNO"],
    "triple_BTC_GLD_ETH":       ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT"],
    "triple_BTC_GLD_SLV":       ["BTC/USDT:USDT", "GLD", "SLV"],
}


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _stats(initial_eq: float, equity_curve: list[tuple[int, float]],
           periods_per_year: int = 52,
           benchmarks: dict[str, np.ndarray] | None = None) -> dict:
    """USD-denominated portfolio stats + CAGR in each benchmark numeraire.

    For each benchmark numeraire (BTC, GLD, ETH, SOL, ...), computes:
      * cagr_<num>             — strategy's CAGR EXPRESSED in units of <num>
                                  (i.e., starting USD bought N units of <num>;
                                  ending USD buys M units of <num>; CAGR = M/N)
      * buyhold_<num>_cagr_usd — buy-and-hold <num> return in USD over the
                                  same window, for direct strategy-vs-benchmark
                                  comparison.

    USD_CAGR is the conventional one: strategy USD value over time.
    """
    eq = np.array([e for _, e in equity_curve], dtype=float)
    eq = np.maximum(eq, 1e-9)
    final = float(eq[-1])
    log_ret = np.diff(np.log(eq))
    years = eq.shape[0] / periods_per_year
    cagr_usd = (final / initial_eq) ** (1.0 / max(years, 1e-9)) - 1.0 if final > 0 else -1.0
    sharpe_usd = sharpe_annualized(log_ret, periods_per_year=periods_per_year)
    sortino_usd = sortino_ratio(log_ret, periods_per_year=periods_per_year)
    dd_usd, _, _ = max_drawdown(eq)
    cal_usd = calmar_ratio(eq, periods_per_year=periods_per_year)

    out = {
        "final_equity_usd": final,
        "cagr_usd": cagr_usd, "sharpe_usd": sharpe_usd, "sortino_usd": sortino_usd,
        "max_drawdown_usd": dd_usd, "calmar_usd": cal_usd,
    }

    def _benchmark_cagr(numeraire_curve: np.ndarray) -> tuple[float | None, float | None]:
        """Return (strategy_CAGR_in_numeraire, buy_and_hold_CAGR_USD)."""
        if numeraire_curve is None or len(numeraire_curve) < 2:
            return (None, None)
        m = min(len(numeraire_curve), len(eq))
        if m < 2:
            return (None, None)
        bench = numeraire_curve[-m:]
        eq_match = eq[-m:]
        strat_units = eq_match / np.maximum(bench, 1e-12)
        if strat_units[0] <= 0:
            return (None, None)
        bench_cagr_usd = (bench[-1] / bench[0]) ** (1.0 / max(years, 1e-9)) - 1.0 \
            if bench[0] > 0 else None
        strat_cagr_in_num = (strat_units[-1] / strat_units[0]) ** (1.0 / max(years, 1e-9)) - 1.0
        return (float(strat_cagr_in_num),
                float(bench_cagr_usd) if bench_cagr_usd is not None else None)

    if benchmarks:
        for label, curve in benchmarks.items():
            cagr_in_num, buyhold = _benchmark_cagr(curve)
            out[f"cagr_{label.lower()}"] = cagr_in_num
            out[f"buyhold_{label.lower()}_cagr_usd"] = buyhold
    return out


def _decision_source_factory(name: str, llm_cache: LLMPortfolioCache | None,
                              model: str):
    if name == "equal_weight":
        return equal_weight_decision_source
    if name == "inverse_vol":
        return inverse_vol_decision_source
    if name == "markowitz_min_var":
        return markowitz_min_var_decision_source
    if name == "strawman":
        return strawman_decision_source
    if name == "llm":
        return make_llm_decision_source(cache=llm_cache, model=model)
    raise ValueError(f"unknown decision source: {name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--decision-sources", default="equal_weight,inverse_vol,markowitz_min_var,strawman",
                   help="comma-separated decision sources to evaluate")
    p.add_argument("--subsets", default="",
                   help="comma-separated subset names; default = all defined subsets")
    p.add_argument("--llm-cache-db", default="runs/llm-portfolio/llm_decisions.db")
    p.add_argument("--model", default="haiku")
    p.add_argument("--rebalance-threshold", type=float, default=0.05)
    p.add_argument("--out-dir", default="runs/llm-portfolio")
    p.add_argument("--limit-bars", type=int, default=0,
                   help="if > 0, only run the first N bars (for LLM smoke tests)")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [s.strip() for s in args.decision_sources.split(",") if s.strip()]
    subsets_to_run = [s.strip() for s in args.subsets.split(",") if s.strip()] or list(SUBSETS.keys())

    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0,
                          funding_rate_per_day=0.00010)
    overlay = QuantOverlayConfig(use_garch_vol=False)  # signals only; sizing is LLM-driven

    # LLM cache (only opened when LLM source is selected)
    llm_cache: LLMPortfolioCache | None = None
    if "llm" in sources:
        Path(args.llm_cache_db).parent.mkdir(parents=True, exist_ok=True)
        llm_cache = LLMPortfolioCache(args.llm_cache_db)

    start_ms, end_ms = parse_iso(TRAIN_START), parse_iso(TRAIN_END)

    # Pre-load all symbol bars referenced by chosen subsets.
    needed_syms = set()
    for sub in subsets_to_run:
        if sub not in SUBSETS:
            print(f"WARNING: unknown subset '{sub}' — skipped", file=sys.stderr)
            continue
        for s in SUBSETS[sub]:
            if s != USD_CASH:
                needed_syms.add(s)

    bars_by_symbol = {}
    for sym in sorted(needed_syms):
        df = cache.load_range(sym, "1w", start_ms, end_ms)
        if df.empty:
            print(f"WARNING: no data for {sym}", file=sys.stderr)
            continue
        bars_by_symbol[sym] = df

    # Always load benchmark curves (BTC, GLD, ETH, SOL) for CAGR
    # in different numeraires per user requirement.
    benchmark_symbols = {
        "BTC": "BTC/USDT:USDT",
        "GLD": "GLD",
        "ETH": "ETH/USDT:USDT",
        "SOL": "SOL/USDT:USDT",
    }
    benchmark_curves: dict[str, np.ndarray] = {}
    for label, sym in benchmark_symbols.items():
        df = cache.load_range(sym, "1w", start_ms, end_ms)
        if not df.empty:
            benchmark_curves[label] = df["close"].to_numpy(dtype=float)

    cache.close()
    print(f"Loaded {len(bars_by_symbol)} symbols.\n"
          f"Train window: {TRAIN_START} → {TRAIN_END}  /  "
          f"{len(subsets_to_run)} universes × {len(sources)} strategies\n"
          "All CAGR / DD figures are USD-denominated (USDT for crypto legs).\n")
    print("How to read these tables:")
    print("  • Each row is a 'universe' — the basket of symbols the strategy can hold.")
    print("  • Each column is a 'rebalancing strategy' — WHO decides the weights:")
    print("      EqualWeight   → rebalance to 1/N each week (deterministic benchmark)")
    print("      InverseVol    → weight ∝ 1/σ (risk parity, deterministic benchmark)")
    print("      MinVariance   → analytic Markowitz min-var (deterministic benchmark)")
    print("      Strawman      → rule-based MA-cross + GARCH-crisis filter (deterministic)")
    print("      LLM(Claude)   → Claude Code decides weights weekly (the actual strategy)")
    print("  • The deterministic columns are BENCHMARKS so we can see whether the LLM")
    print("    actually adds value over plain rules. The LLM is the production strategy.")
    print()

    rows = []
    for sub_name in subsets_to_run:
        if sub_name not in SUBSETS:
            continue
        universe = [s for s in SUBSETS[sub_name] if s == USD_CASH or s in bars_by_symbol]
        non_cash = [s for s in universe if s != USD_CASH]
        if not non_cash:
            print(f"  {sub_name}: no available symbols, skipped")
            continue

        for src_name in sources:
            decision_source = _decision_source_factory(src_name, llm_cache, args.model)
            sub_bars = {s: bars_by_symbol[s] for s in non_cash}
            end_idx = (args.limit_bars - 1) if args.limit_bars > 0 else None
            try:
                res = run_llm_portfolio(
                    universe=universe,
                    bars_by_symbol=sub_bars,
                    decision_source=decision_source,
                    fill_config=fill_cfg,
                    initial_equity_usd=args.initial_equity,
                    anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                    quant_overlay=overlay,
                    rebalance_threshold=args.rebalance_threshold,
                    end_idx=end_idx,
                )
            except Exception as e:
                print(f"  ERR {sub_name}/{src_name}: {e}")
                continue
            stats = _stats(args.initial_equity, res.equity_curve,
                           benchmarks=benchmark_curves)
            row = {
                "subset": sub_name,
                "decision_source": src_name,
                "n_symbols": len(universe),
                "symbols": ",".join(universe),
                "n_decisions": res.n_decisions,
                "n_trades": len(res.trades),
                **stats,
            }
            rows.append(row)
            print(f"  {sub_name:24s} {src_name:18s} "
                  f"CAGR_USD={stats['cagr_usd']*100:+6.1f}% "
                  f"Sharpe_USD={stats['sharpe_usd']:+.2f} "
                  f"DD_USD={stats['max_drawdown_usd']*100:5.1f}%  "
                  f"trades={len(res.trades):3d}")

    if llm_cache is not None:
        llm_cache.close()

    # Save CSV + comparison table
    if rows:
        with (out_dir / "comparison.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        subset_names = sorted({r["subset"] for r in rows},
                              key=lambda n: list(SUBSETS).index(n) if n in SUBSETS else 999)
        source_names = ["equal_weight", "inverse_vol", "markowitz_min_var", "strawman", "llm"]
        source_names = [s for s in source_names if any(r["decision_source"] == s for r in rows)]

        # Friendly source labels
        SRC_LABEL = {
            "equal_weight": "EqualWeight",
            "inverse_vol": "InverseVol",
            "markowitz_min_var": "MinVariance",
            "strawman": "Strawman",
            "llm": "LLM(Claude)",
        }

        # ============================================================
        # TABLE 1: Sharpe (USD) — best Sharpe per row highlighted
        # ============================================================
        print("\n" + "═" * 80)
        print("TABLE 1 — Sharpe ratio (USD-denominated, weekly cadence)")
        print("Rows: portfolio universes  |  Columns: rebalancing strategy")
        print("Higher is better. ★ marks the best strategy for that universe.")
        print("═" * 80)
        col_w = 12
        print(f"  {'universe':28s} ", end="")
        for src in source_names:
            print(f"│ {SRC_LABEL[src]:>{col_w}s} ", end="")
            print()  # newline only after last col
            break
        # actually print all columns properly:
        line = f"  {'universe':28s} "
        for src in source_names:
            line += f"│ {SRC_LABEL[src]:>{col_w}s} "
        print(line)
        print("  " + "─" * 28 + " " + ("│" + "─" * (col_w + 2)) * len(source_names))
        for sub in subset_names:
            sharpes = {src: None for src in source_names}
            for src in source_names:
                m = next((r for r in rows
                          if r["subset"] == sub and r["decision_source"] == src), None)
                if m:
                    sharpes[src] = m["sharpe_usd"]
            best = max((s for s in sharpes.values() if s is not None), default=None)
            line = f"  {sub:28s} "
            for src in source_names:
                v = sharpes[src]
                if v is None:
                    line += f"│ {' --- ':>{col_w}s} "
                else:
                    star = "★" if best is not None and v == best else " "
                    line += f"│ {star} {v:+.2f}      "[:col_w + 3]
            print(line)

        # ============================================================
        # TABLE 2: CAGR_USD — best per row highlighted
        # ============================================================
        print("\n" + "═" * 80)
        print("TABLE 2 — CAGR_USD (annualized USD return, train 2021-01 → 2025-09)")
        print("Higher is better. ★ marks best CAGR per universe.")
        print("═" * 80)
        line = f"  {'universe':28s} "
        for src in source_names:
            line += f"│ {SRC_LABEL[src]:>{col_w}s} "
        print(line)
        print("  " + "─" * 28 + " " + ("│" + "─" * (col_w + 2)) * len(source_names))
        for sub in subset_names:
            cagrs = {src: None for src in source_names}
            for src in source_names:
                m = next((r for r in rows
                          if r["subset"] == sub and r["decision_source"] == src), None)
                if m:
                    cagrs[src] = m["cagr_usd"]
            best = max((c for c in cagrs.values() if c is not None), default=None)
            line = f"  {sub:28s} "
            for src in source_names:
                v = cagrs[src]
                if v is None:
                    line += f"│ {' --- ':>{col_w}s} "
                else:
                    star = "★" if best is not None and v == best else " "
                    line += f"│ {star}{v*100:+6.1f}%    "[:col_w + 3]
            print(line)

        # ============================================================
        # TABLE 3: Max drawdown_USD — LOWER is better, ★ marks safest
        # ============================================================
        print("\n" + "═" * 80)
        print("TABLE 3 — Max drawdown_USD (lower is better — capital preservation)")
        print("Live cap: 15%. Configs ≤ 15% safe to deploy under risk policy.")
        print("═" * 80)
        line = f"  {'universe':28s} "
        for src in source_names:
            line += f"│ {SRC_LABEL[src]:>{col_w}s} "
        print(line)
        print("  " + "─" * 28 + " " + ("│" + "─" * (col_w + 2)) * len(source_names))
        for sub in subset_names:
            dds = {src: None for src in source_names}
            for src in source_names:
                m = next((r for r in rows
                          if r["subset"] == sub and r["decision_source"] == src), None)
                if m:
                    dds[src] = m["max_drawdown_usd"]
            best = min((d for d in dds.values() if d is not None), default=None)
            line = f"  {sub:28s} "
            for src in source_names:
                v = dds[src]
                if v is None:
                    line += f"│ {' --- ':>{col_w}s} "
                else:
                    star = "★" if best is not None and v == best else " "
                    line += f"│ {star}{v*100:5.1f}%     "[:col_w + 3]
            print(line)

        # ============================================================
        # TABLE 4: Top-10 by Sharpe across the whole grid (ranked)
        # ============================================================
        print("\n" + "═" * 80)
        print("TABLE 4 — Top 10 (subset × strategy) by Sharpe_USD")
        print("Best risk-adjusted configs across the entire grid.")
        print("═" * 80)
        ranked = sorted(rows, key=lambda r: -r["sharpe_usd"])[:10]
        print(f"  {'rank':<4s} {'universe':28s} {'strategy':14s} "
              f"{'Sharpe':>7s} {'CAGR$':>8s} {'DD$':>8s} {'Calmar':>7s}")
        print("  " + "─" * 80)
        for i, r in enumerate(ranked, start=1):
            print(f"  {i:<4d} {r['subset']:28s} {SRC_LABEL[r['decision_source']]:14s} "
                  f"{r['sharpe_usd']:+7.2f} {r['cagr_usd']*100:+7.1f}% "
                  f"{r['max_drawdown_usd']*100:6.1f}%  {r.get('calmar_usd', 0):+7.2f}")

        # ============================================================
        # TABLE 5: Top-10 SAFE configs (DD ≤ 15%) by Sharpe
        # ============================================================
        safe = [r for r in rows if r["max_drawdown_usd"] <= 0.15]
        ranked_safe = sorted(safe, key=lambda r: -r["sharpe_usd"])[:10]
        print("\n" + "═" * 80)
        print("TABLE 5 — Top 10 SAFE configs (DD_USD ≤ 15% live cap)")
        print("Eligible for live deployment under the 15% drawdown ceiling.")
        print("═" * 80)
        if not ranked_safe:
            print("  (none — no config kept DD under 15% on the training window)")
        else:
            print(f"  {'rank':<4s} {'universe':28s} {'strategy':14s} "
                  f"{'Sharpe':>7s} {'CAGR$':>8s} {'DD$':>8s} {'Calmar':>7s}")
            print("  " + "─" * 80)
            for i, r in enumerate(ranked_safe, start=1):
                print(f"  {i:<4d} {r['subset']:28s} {SRC_LABEL[r['decision_source']]:14s} "
                      f"{r['sharpe_usd']:+7.2f} {r['cagr_usd']*100:+7.1f}% "
                      f"{r['max_drawdown_usd']*100:6.1f}%  {r.get('calmar_usd', 0):+7.2f}")

        # ============================================================
        # TABLE 6: Multi-numeraire CAGR for top-5 by Sharpe
        # ============================================================
        print("\n" + "═" * 80)
        print("TABLE 6 — Top 5 by Sharpe — CAGR in different numeraires")
        print("USD: USD value of portfolio. BTC/ETH/SOL/GLD: portfolio")
        print("expressed in units of that asset. Negative = strategy")
        print("underperformed buy-and-hold of that asset.")
        print("═" * 80)
        top5 = sorted(rows, key=lambda r: -r["sharpe_usd"])[:5]
        print(f"  {'rank':<4s} {'universe':28s} {'strategy':14s} "
              f"{'CAGR$':>7s} {'BTCcg':>7s} {'ETHcg':>7s} {'SOLcg':>7s} {'GLDcg':>7s}")
        print("  " + "─" * 80)
        def _pct(v):
            return f"{v*100:+6.1f}%" if v is not None else "    -- "
        for i, r in enumerate(top5, start=1):
            print(f"  {i:<4d} {r['subset']:28s} {SRC_LABEL[r['decision_source']]:14s} "
                  f"{_pct(r.get('cagr_usd'))} "
                  f"{_pct(r.get('cagr_btc'))} "
                  f"{_pct(r.get('cagr_eth'))} "
                  f"{_pct(r.get('cagr_sol'))} "
                  f"{_pct(r.get('cagr_gld'))}")
        print("\n  Buy-and-hold benchmarks over the same window (USD):")
        if top5:
            r0 = top5[0]
            print(f"    BTC: {_pct(r0.get('buyhold_btc_cagr_usd'))}   "
                  f"ETH: {_pct(r0.get('buyhold_eth_cagr_usd'))}   "
                  f"SOL: {_pct(r0.get('buyhold_sol_cagr_usd'))}   "
                  f"GLD: {_pct(r0.get('buyhold_gld_cagr_usd'))}")

        meta = {
            "training_window": {"start": TRAIN_START, "end": TRAIN_END},
            "decision_sources": sources,
            "subsets": subset_names,
            "n_rows": len(rows),
            "convention_note": (
                "All CAGR / DD / Sharpe figures USD-denominated. "
                "USDT positions valued at USDT==USD parity. "
                "Stocks/ETFs/commodities valued at their USD close. "
                "USD_CASH leg has zero return."
            ),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\nWrote → {out_dir}/comparison.csv  meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
