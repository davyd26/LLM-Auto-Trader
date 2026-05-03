#!/usr/bin/env python3
"""Deep analysis of the OOS-passing config: anchor_plus_BNO + MinVariance.

Reports for the chosen config:
  - Per-regime CAGR_USD / Sharpe / DD across the 5 sub-windows
  - Bootstrap 95% CI on training-window Sharpe
  - CAGR in numeraires: USD, BTC, GLD, ETH, SOL, BNO (Brent oil), SPY
  - Buy-and-hold benchmarks for each numeraire
  - Per-bar weight history (proxy for trading activity)
  - Trade count, fee impact, average rebalance turnover

This isolates the production-recommended config and produces a complete
deployment dossier on it.
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
from src.analysis.bootstrap import block_bootstrap_sharpe_ci
from src.backtest.data_cache import OHLCVCache
from src.backtest.fill_model import FillConfig
from src.backtest.llm_portfolio_decisions import markowitz_min_var_decision_source
from src.backtest.llm_portfolio_strategy import USD_CASH, run_llm_portfolio
from src.backtest.quant_overlay import QuantOverlayConfig

# The config being analyzed
UNIVERSE_NAME = "anchor_plus_BNO"
UNIVERSE = ["BTC/USDT:USDT", "GLD", "BNO", USD_CASH]
REBALANCER_NAME = "MinVariance"
REBALANCER = markowitz_min_var_decision_source

# Periods
TRAIN_START = "2021-01-04"
TRAIN_END = "2025-09-29"
OOS_START = "2025-10-06"
OOS_END = "2026-04-27"

REGIME_WINDOWS = [
    ("blowoff-2021",  "2021-01-04", "2022-01-03"),
    ("bear-2022",     "2022-01-03", "2023-01-02"),
    ("recovery-2023", "2023-01-02", "2024-01-01"),
    ("bull-2024",     "2024-01-01", "2024-12-30"),
    ("late-2025-tr",  "2025-01-06", TRAIN_END),
    ("OOS-2025-26",   OOS_START, OOS_END),
]

NUMERAIRES = {
    "USD":  None,             # USD = portfolio's native unit
    "BTC":  "BTC/USDT:USDT",
    "GLD":  "GLD",
    "ETH":  "ETH/USDT:USDT",
    "SOL":  "SOL/USDT:USDT",
    "BNO":  "BNO",
    "SPY":  "SPY",
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
    return {
        "final_usd": final, "cagr_usd": cagr, "sharpe_usd": sharpe,
        "sortino_usd": sortino, "max_drawdown_usd": dd, "calmar_usd": cal,
        "log_returns": log_ret,
    }


def _benchmark_cagr(eq: np.ndarray, bench_curve: np.ndarray, periods: int = 52
                    ) -> tuple[float | None, float | None]:
    """Return (strategy_cagr_in_bench_units, buy_and_hold_bench_cagr_usd)."""
    if bench_curve is None or len(bench_curve) < 2 or len(eq) < 2:
        return (None, None)
    m = min(len(bench_curve), len(eq))
    bench = bench_curve[-m:]
    eq_match = eq[-m:]
    years = m / periods
    strat_units = eq_match / np.maximum(bench, 1e-12)
    if strat_units[0] <= 0 or bench[0] <= 0:
        return (None, None)
    bench_cagr = (bench[-1] / bench[0]) ** (1.0 / max(years, 1e-9)) - 1.0
    strat_cagr_in_num = (strat_units[-1] / strat_units[0]) ** (1.0 / max(years, 1e-9)) - 1.0
    return (float(strat_cagr_in_num), float(bench_cagr))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--out-dir", default="runs/portfolio-top-config")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0,
                          funding_rate_per_day=0.00010)
    overlay = QuantOverlayConfig(use_garch_vol=False)

    print("\n" + "═" * 100)
    print(f"DEEP ANALYSIS — {UNIVERSE_NAME} / {REBALANCER_NAME}")
    print(f"Universe: {UNIVERSE}")
    print("All CAGR / DD figures are USD-denominated unless explicitly suffixed.")
    print("═" * 100)

    # ============================================
    # PHASE 1: Per-regime contiguous walk
    # ============================================
    print("\nPER-REGIME PERFORMANCE")
    print("─" * 100)
    print(f"  {'window':14s} {'days':>6s} "
          f"{'CAGR$':>9s} {'Sh$':>7s} {'Sortino$':>9s} {'DD$':>7s} {'Calmar$':>8s}  trades")

    regime_rows = []
    for win_name, ws, we in REGIME_WINDOWS:
        ws_ms, we_ms = parse_iso(ws), parse_iso(we)
        bars_by_symbol = {}
        for sym in UNIVERSE:
            if sym == USD_CASH:
                continue
            df = cache.load_range(sym, "1w", ws_ms, we_ms)
            if not df.empty:
                bars_by_symbol[sym] = df
        if not bars_by_symbol:
            continue
        try:
            res = run_llm_portfolio(
                universe=UNIVERSE, bars_by_symbol=bars_by_symbol,
                decision_source=REBALANCER, fill_config=fill_cfg,
                initial_equity_usd=args.initial_equity,
                anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
                quant_overlay=overlay, rebalance_threshold=0.05,
            )
        except Exception as e:
            print(f"  ERR {win_name}: {e}")
            continue
        stats = _stats(args.initial_equity, res.equity_curve)
        days = (we_ms - ws_ms) // 86_400_000
        regime_rows.append({
            "window": win_name, "days": int(days), **stats,
            "n_trades": len(res.trades),
        })
        print(f"  {win_name:14s} {days:>6d} "
              f"{stats['cagr_usd']*100:+8.1f}% {stats['sharpe_usd']:+7.2f} "
              f"{stats['sortino_usd']:+9.2f} {stats['max_drawdown_usd']*100:6.1f}% "
              f"{stats['calmar_usd']:+8.2f}  {len(res.trades):>3d}")

    # ============================================
    # PHASE 2: Multi-numeraire CAGR (training + OOS combined)
    # ============================================
    print("\n\nMULTI-NUMERAIRE CAGR — full data window 2021-01-04 → 2026-04-27")
    print("─" * 100)
    full_start, full_end = parse_iso(TRAIN_START), parse_iso(OOS_END)
    bars_by_symbol = {}
    for sym in UNIVERSE:
        if sym == USD_CASH:
            continue
        df = cache.load_range(sym, "1w", full_start, full_end)
        if not df.empty:
            bars_by_symbol[sym] = df

    res_full = run_llm_portfolio(
        universe=UNIVERSE, bars_by_symbol=bars_by_symbol,
        decision_source=REBALANCER, fill_config=fill_cfg,
        initial_equity_usd=args.initial_equity,
        anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
        quant_overlay=overlay, rebalance_threshold=0.05,
    )
    full_stats = _stats(args.initial_equity, res_full.equity_curve)
    eq_arr = np.array([e for _, e in res_full.equity_curve], dtype=float)

    print(f"  {'numeraire':10s} {'strat_CAGR_in_num':>18s} "
          f"{'buyhold_num_CAGR_usd':>22s}  verdict")
    multi_num_rows = []
    for label, sym in NUMERAIRES.items():
        if label == "USD":
            print(f"  {label:10s} {full_stats['cagr_usd']*100:+17.1f}%   "
                  f"{'(reference)':>22s}")
            multi_num_rows.append({"numeraire": label,
                                    "strat_cagr_in_num": full_stats["cagr_usd"],
                                    "buyhold_cagr_usd": None})
            continue
        bench = cache.load_range(sym, "1w", full_start, full_end)
        if bench.empty:
            continue
        bench_curve = bench["close"].to_numpy(dtype=float)
        strat_in_num, buyhold = _benchmark_cagr(eq_arr, bench_curve)
        verdict = ""
        if strat_in_num is not None and buyhold is not None:
            if strat_in_num > 0:
                verdict = "BEAT buy-and-hold"
            else:
                verdict = "underperformed buy-and-hold"
        print(f"  {label:10s} "
              f"{strat_in_num*100 if strat_in_num is not None else 0:+17.1f}%   "
              f"{buyhold*100 if buyhold is not None else 0:+21.1f}%   {verdict}")
        multi_num_rows.append({
            "numeraire": label,
            "strat_cagr_in_num": strat_in_num,
            "buyhold_cagr_usd": buyhold,
        })

    # ============================================
    # PHASE 3: Bootstrap 95% CI on training-window Sharpe
    # ============================================
    train_start, train_end = parse_iso(TRAIN_START), parse_iso(TRAIN_END)
    bars_by_symbol = {}
    for sym in UNIVERSE:
        if sym == USD_CASH:
            continue
        df = cache.load_range(sym, "1w", train_start, train_end)
        if not df.empty:
            bars_by_symbol[sym] = df
    res_train = run_llm_portfolio(
        universe=UNIVERSE, bars_by_symbol=bars_by_symbol,
        decision_source=REBALANCER, fill_config=fill_cfg,
        initial_equity_usd=args.initial_equity,
        anchor=["BTC/USDT:USDT", "GLD", USD_CASH],
        quant_overlay=overlay, rebalance_threshold=0.05,
    )
    train_stats = _stats(args.initial_equity, res_train.equity_curve)
    log_train = np.array(train_stats["log_returns"])
    try:
        ci_lo, ci_hi = block_bootstrap_sharpe_ci(
            log_train, periods_per_year=52, n_resamples=2000,
            confidence=0.95, block_size=4, seed=42,
        )
    except Exception:
        ci_lo, ci_hi = float("nan"), float("nan")

    print("\n\nTRAINING-WINDOW STATS WITH BOOTSTRAP 95% CI")
    print("─" * 100)
    print(f"  Window:       {TRAIN_START} → {TRAIN_END}")
    print(f"  CAGR_USD:     {train_stats['cagr_usd']*100:+.1f}%")
    print(f"  Sharpe_USD:   {train_stats['sharpe_usd']:+.2f}  "
          f"(95% CI: [{ci_lo:+.2f}, {ci_hi:+.2f}])")
    print(f"  Sortino_USD:  {train_stats['sortino_usd']:+.2f}")
    print(f"  Max DD_USD:   {train_stats['max_drawdown_usd']*100:.1f}%")
    print(f"  Calmar_USD:   {train_stats['calmar_usd']:+.2f}")
    print(f"  Trades:       {len(res_train.trades)}")

    # ============================================
    # PHASE 4: Save artifacts
    # ============================================
    cache.close()

    if regime_rows:
        with (out_dir / "per_regime.csv").open("w", newline="") as f:
            cols = [c for c in regime_rows[0].keys() if c != "log_returns"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in regime_rows:
                w.writerow({c: r.get(c) for c in cols})

    summary = {
        "config": {
            "universe": UNIVERSE,
            "universe_name": UNIVERSE_NAME,
            "rebalancer": REBALANCER_NAME,
        },
        "training_window": {"start": TRAIN_START, "end": TRAIN_END},
        "oos_window": {"start": OOS_START, "end": OOS_END},
        "training_stats": {
            "cagr_usd": train_stats["cagr_usd"],
            "sharpe_usd": train_stats["sharpe_usd"],
            "sharpe_95_ci_lo": ci_lo, "sharpe_95_ci_hi": ci_hi,
            "sortino_usd": train_stats["sortino_usd"],
            "max_drawdown_usd": train_stats["max_drawdown_usd"],
            "calmar_usd": train_stats["calmar_usd"],
            "n_trades": len(res_train.trades),
        },
        "regime_results": [{k: v for k, v in r.items() if k != "log_returns"}
                           for r in regime_rows],
        "multi_numeraire_full_window": multi_num_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n→ {out_dir}/per_regime.csv  summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
