#!/usr/bin/env python3
"""Out-of-sample validation for the multi-symbol portfolio strategies.

Runs the top configs from CPCV/walk-forward on the held-out OOS window
(2025-10-06 → 2026-04-27, ~30 weekly bars never referenced during
selection or sweep tuning). The OOS pass criterion is:

  - OOS Sharpe ≥ 0.5 × CPCV-median Sharpe
  - OOS Max DD ≤ 1.5 × CPCV-median DD
  - OOS CAGR_USD > 0

Configs that pass are eligible for the layered live deployment design.
All CAGR / DD figures USD-denominated.
"""

from __future__ import annotations

import argparse
import csv
import json
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

OOS_START = "2025-10-06"
OOS_END = "2026-04-27"

REBALANCERS = {
    "EqualWeight":  equal_weight_decision_source,
    "InverseVol":   inverse_vol_decision_source,
    "MinVariance":  markowitz_min_var_decision_source,
    "Strawman":     strawman_decision_source,
    "Strawman_v2":  strawman_v2_decision_source,
}

# All universes — we test every combination on OOS so we know which
# survive the held-out gate.
UNIVERSES: dict[str, list[str]] = {
    "pair_BTC_GLD":         ["BTC/USDT:USDT", "GLD"],
    "pair_BTC_USD":         ["BTC/USDT:USDT", USD_CASH],
    "pair_GLD_USD":         ["GLD", USD_CASH],
    "anchor_btc_gld_usd":   ["BTC/USDT:USDT", "GLD", USD_CASH],
    "anchor_plus_eth":      ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT", USD_CASH],
    "anchor_plus_sol":      ["BTC/USDT:USDT", "GLD", "SOL/USDT:USDT", USD_CASH],
    "anchor_plus_NVDA":     ["BTC/USDT:USDT", "GLD", "NVDA", USD_CASH],
    "anchor_plus_BNO":      ["BTC/USDT:USDT", "GLD", "BNO", USD_CASH],
    "anchor_plus_SPY":      ["BTC/USDT:USDT", "GLD", "SPY", USD_CASH],
    "anchor_plus_QQQ":      ["BTC/USDT:USDT", "GLD", "QQQ", USD_CASH],
    "anchor_eth_sol":       ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT",
                             "SOL/USDT:USDT", USD_CASH],
    "anchor_eth_nvda":      ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT", "NVDA", USD_CASH],
    "anchor_eth_oil":       ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT", "BNO", USD_CASH],
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
    p.add_argument("--cpcv-summary",
                   default="runs/portfolio-cpcv/cpcv.csv",
                   help="path to CPCV summary CSV (used for pass thresholds)")
    p.add_argument("--out-dir", default="runs/portfolio-oos")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0,
                          funding_rate_per_day=0.00010)
    overlay = QuantOverlayConfig(use_garch_vol=False)

    # Load CPCV thresholds for pass criteria
    cpcv_thresholds = {}
    cpcv_path = Path(args.cpcv_summary)
    if cpcv_path.exists():
        for row in csv.DictReader(cpcv_path.open()):
            key = (row["universe"], row["rebalancer"])
            cpcv_thresholds[key] = {
                "med_sharpe": float(row["med_sharpe"]),
                "med_dd": float(row["med_dd"]),
            }

    start_ms, end_ms = parse_iso(OOS_START), parse_iso(OOS_END)
    all_syms = set()
    for univ in UNIVERSES.values():
        for s in univ:
            if s != USD_CASH:
                all_syms.add(s)
    bars_by_symbol = {}
    for sym in all_syms:
        df = cache.load_range(sym, "1w", start_ms, end_ms)
        if not df.empty:
            bars_by_symbol[sym] = df
    cache.close()

    n_oos_bars = max((len(d) for d in bars_by_symbol.values()), default=0)
    print(f"\n{'═'*100}")
    print(f"OOS HOLDOUT — {OOS_START} → {OOS_END}, ~{n_oos_bars} weekly bars")
    print("All figures USD-denominated.")
    print(f"{'═'*100}")
    print(f"  {'universe':24s} {'rebalancer':12s} "
          f"{'OOS_CAGR':>9s} {'OOS_Sh':>8s} {'OOS_DD':>7s}  "
          f"{'CPCV_Sh':>8s} {'pass':>6s}")

    rows = []
    for univ_name, univ in UNIVERSES.items():
        avail = [s for s in univ if s == USD_CASH or s in bars_by_symbol]
        if len([s for s in avail if s != USD_CASH]) == 0:
            continue
        sub_bars = {s: bars_by_symbol[s] for s in avail
                    if s != USD_CASH and s in bars_by_symbol}
        if not sub_bars:
            continue
        for reb_name, src in REBALANCERS.items():
            try:
                res = run_llm_portfolio(
                    universe=avail, bars_by_symbol=sub_bars,
                    decision_source=src, fill_config=fill_cfg,
                    initial_equity_usd=args.initial_equity,
                    anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                    quant_overlay=overlay, rebalance_threshold=0.05,
                )
            except Exception as e:
                print(f"  ERR {univ_name}/{reb_name}: {e}", file=sys.stderr)
                continue
            stats = _stats(args.initial_equity, res.equity_curve)
            cpcv = cpcv_thresholds.get((univ_name, reb_name), {})
            cpcv_sh = cpcv.get("med_sharpe", float("nan"))
            cpcv_dd = cpcv.get("med_dd", float("nan"))

            pass_sh = stats["sharpe_usd"] >= 0.5 * cpcv_sh \
                if not np.isnan(cpcv_sh) else stats["sharpe_usd"] > 0
            pass_dd = stats["dd_usd"] <= 1.5 * cpcv_dd \
                if not np.isnan(cpcv_dd) and cpcv_dd > 0 else stats["dd_usd"] <= 0.20
            pass_cagr = stats["cagr_usd"] > 0
            passed = pass_sh and pass_dd and pass_cagr

            rows.append({
                "universe": univ_name, "rebalancer": reb_name,
                "oos_cagr_usd": stats["cagr_usd"],
                "oos_sharpe_usd": stats["sharpe_usd"],
                "oos_dd_usd": stats["dd_usd"],
                "cpcv_med_sharpe": cpcv_sh,
                "cpcv_med_dd": cpcv_dd,
                "pass_sharpe": pass_sh,
                "pass_dd": pass_dd,
                "pass_cagr": pass_cagr,
                "passed": passed,
                "n_trades": len(res.trades),
            })

            mark = "YES" if passed else "no"
            print(f"  {univ_name:24s} {reb_name:12s} "
                  f"{stats['cagr_usd']*100:+8.1f}% {stats['sharpe_usd']:+8.2f} "
                  f"{stats['dd_usd']*100:6.1f}%  {cpcv_sh:+8.2f}  {mark:>6s}")

    if rows:
        with (out_dir / "oos.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        # Summary
        n_pass = sum(1 for r in rows if r["passed"])
        print(f"\n{'═'*100}")
        print(f"PASS SUMMARY: {n_pass}/{len(rows)} configs passed the OOS gate")
        print(f"{'═'*100}")
        passing = [r for r in rows if r["passed"]]
        passing.sort(key=lambda r: -r["oos_sharpe_usd"])
        print("\nTop passing configs by OOS Sharpe (USD-denominated):")
        print(f"  {'rank':<4s} {'universe':24s} {'rebalancer':12s} "
              f"{'CAGR$':>8s} {'Sharpe':>8s} {'DD$':>7s}")
        for i, r in enumerate(passing[:10], start=1):
            print(f"  {i:<4d} {r['universe']:24s} {r['rebalancer']:12s} "
                  f"{r['oos_cagr_usd']*100:+7.1f}% {r['oos_sharpe_usd']:+8.2f} "
                  f"{r['oos_dd_usd']*100:6.1f}%")
        meta = {
            "oos_window": {"start": OOS_START, "end": OOS_END},
            "n_configs_tested": len(rows),
            "n_passed": n_pass,
            "best_oos_passing": passing[0] if passing else None,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        print(f"\n→ {out_dir}/oos.csv  meta.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
