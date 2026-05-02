#!/usr/bin/env python3
"""Out-of-sample validation gate for the top-K configs from sweep_robust.py.

Reads ``runs/robust-sweep/stats.json`` to get the top-3 configs that won the
training-window composite score, then runs each on the held-out OOS window
(2025-10-06 → 2026-04-27) — data NEVER referenced during selection.

Pass criteria:
  - OOS Sharpe ≥ 0.5 × CPCV-median Sharpe (residual selection-bias deflator)
  - OOS max DD ≤ 1.5 × training max DD
  - OOS CAGR > 0

The chosen-for-deployment config is the OOS-passing config with the best
training composite score. If no config passes: report honestly, do NOT
deploy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.analysis.backtest_stats import (
    calmar_ratio,
    max_drawdown,
    sharpe_annualized,
    sortino_ratio,
)
from src.backtest.data_cache import OHLCVCache
from src.backtest.fast_baseline import run_baseline_fast
from src.backtest.fill_model import FillConfig

# Mirror sweep_robust.py to reconstruct configs by name.
sys.path.insert(0, str(Path(__file__).parent))
from sweep_robust import (
    HEDGE,
    OOS_END,
    OOS_START,
    PRIMARY,
    REGIME_WINDOWS,  # noqa: F401  — referenced in summary
    all_configs,
    parse_iso,
)


def _stats(initial_eq: float, equity_curve, periods_per_year=52) -> dict:
    eq = np.array([e for _, e in equity_curve], dtype=float)
    eq = np.maximum(eq, 1e-9)
    final = float(eq[-1])
    log_ret = np.diff(np.log(eq))
    years = eq.shape[0] / periods_per_year
    cagr = (final / initial_eq) ** (1.0 / max(years, 1e-9)) - 1.0 if final > 0 else -1.0
    sharpe = sharpe_annualized(log_ret, periods_per_year=periods_per_year)
    sortino = sortino_ratio(log_ret, periods_per_year=periods_per_year)
    dd, _, _ = max_drawdown(eq)
    cal = calmar_ratio(eq, periods_per_year=periods_per_year)
    return {"final_equity": final, "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": dd, "calmar": cal, "n_returns": len(log_ret)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--sweep-dir", default="runs/robust-sweep")
    p.add_argument("--out-dir", default="runs/robust-oos")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    stats_path = sweep_dir / "stats.json"
    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found. Run sweep_robust.py first.", file=sys.stderr)
        return 2

    sweep_stats = json.loads(stats_path.read_text())
    top_names = sweep_stats["top_3"][: args.top_k]
    grid_rows = list(csv.DictReader((sweep_dir / "grid.csv").open()))

    # Train rows for these configs (used for DD comparison and CPCV gate)
    cfg_train_stats = {}
    for n in top_names:
        for row in grid_rows:
            if row["config"] == n and row["window"] == "train-full":
                cfg_train_stats[n] = {k: float(row[k]) if row[k] not in {"", "nan"} else float("nan")
                                      for k in ("cagr", "sharpe", "max_drawdown")}
                break
        for row in grid_rows:
            if row["config"] == n and row["window"] == "cpcv":
                # CPCV "median" Sharpe was stored in the sharpe column.
                cfg_train_stats[n]["cpcv_median_sharpe"] = float(row["sharpe"]) \
                    if row["sharpe"] not in {"", "nan"} else float("nan")
                break

    # Reconstruct config objects.
    by_name = {c.name: c for c in all_configs()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0, funding_rate_per_day=0.00010)

    primary = cache.load_range(PRIMARY, "1w", parse_iso(OOS_START), parse_iso(OOS_END))
    hedge   = cache.load_range(HEDGE,   "1w", parse_iso(OOS_START), parse_iso(OOS_END))
    hedge = hedge if not hedge.empty else None

    if primary.empty:
        print(f"ERROR: no OOS primary data {OOS_START}..{OOS_END}", file=sys.stderr)
        return 2
    print(f"OOS holdout: {OOS_START} → {OOS_END}, {len(primary)} weekly bars")

    print(f"\nValidating top-{args.top_k} configs against OOS holdout:\n")
    print(f"{'config':30s} {'OOS_CAGR':>9s} {'OOS_Sh':>8s} {'OOS_DD':>7s} "
          f"{'CPCV_Sh':>8s} {'pass':>6s}")
    print("-" * 80)

    results = {}
    passing = []
    for name in top_names:
        if name not in by_name:
            print(f"  {name}: not in by_name — skipping", file=sys.stderr)
            continue
        cfg = by_name[name]
        result = run_baseline_fast(
            primary_bars=primary, hedge_bars=hedge,
            config=cfg.config, fill_config=fill_cfg,
            initial_equity=args.initial_equity,
            regime_overrides=cfg.regime_overrides,
        )
        oos_stats = _stats(args.initial_equity, result.equity_curve)
        train = cfg_train_stats.get(name, {})

        # Pass criteria
        cpcv_sh = train.get("cpcv_median_sharpe", float("nan"))
        sh_threshold = 0.5 * cpcv_sh if not np.isnan(cpcv_sh) else 0.0
        pass_sharpe = oos_stats["sharpe"] >= sh_threshold
        pass_dd = oos_stats["max_drawdown"] <= 1.5 * train.get("max_drawdown", 0.15)
        pass_cagr = oos_stats["cagr"] > 0
        passed = pass_sharpe and pass_dd and pass_cagr

        results[name] = {
            "oos": oos_stats,
            "train_cagr": train.get("cagr", float("nan")),
            "train_dd":   train.get("max_drawdown", float("nan")),
            "cpcv_median_sharpe": cpcv_sh,
            "sh_threshold": sh_threshold,
            "passed": passed,
            "passed_sharpe": pass_sharpe,
            "passed_dd": pass_dd,
            "passed_cagr": pass_cagr,
            "n_trades": len(result.trades),
        }
        if passed:
            passing.append(name)

        print(f"{name:30s} {oos_stats['cagr']*100:+8.1f}% {oos_stats['sharpe']:+8.2f} "
              f"{oos_stats['max_drawdown']*100:6.1f}%  {cpcv_sh:+8.2f}  "
              f"{'YES' if passed else 'no':>6s}")

    cache.close()

    # Choose the deployment config: passing AND best composite score
    composite = sweep_stats.get("composite_score", {})
    if passing:
        chosen = max(passing, key=lambda n: composite.get(n, 0.0))
    else:
        chosen = None

    print()
    if chosen:
        print(f"DEPLOYMENT-CANDIDATE: {chosen}")
        print(f"  - Train CAGR: {cfg_train_stats[chosen].get('cagr', float('nan'))*100:+.1f}%")
        print(f"  - OOS CAGR:   {results[chosen]['oos']['cagr']*100:+.1f}%")
        print(f"  - Train DD:   {cfg_train_stats[chosen].get('max_drawdown', float('nan'))*100:.1f}%")
        print(f"  - OOS DD:     {results[chosen]['oos']['max_drawdown']*100:.1f}%")
        print(f"  - Composite:  {composite.get(chosen, 0.0):.4f}")
    else:
        print("NO CONFIG PASSED THE OOS GATE.")
        print("Honest interpretation: the strategy did not generalize from the")
        print("training window to the held-out OOS period. Do NOT deploy.")

    # Save full report
    report = {
        "oos_window": {"start": OOS_START, "end": OOS_END},
        "n_oos_bars": len(primary),
        "tested_configs": top_names,
        "results": results,
        "passing_configs": passing,
        "deployment_candidate": chosen,
    }
    (out_dir / "oos_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote → {out_dir}/oos_report.json")
    return 0 if chosen else 1


if __name__ == "__main__":
    sys.exit(main())
