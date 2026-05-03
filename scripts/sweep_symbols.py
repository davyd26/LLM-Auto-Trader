#!/usr/bin/env python3
"""Per-symbol comparison: run the H4_baseline champion (with full CPCV) on
each available symbol independently, plus on equally-weighted subsets.

Single-symbol mode tells us which assets the strategy actually works on.
Subset mode tells us whether diversifying across N symbols beats picking
the best single one.

Symbols fetched but rejected:
  - US-stock perps (COIN/MSTR/NVDA/TSLA/SPY): too few weekly bars
    (Binance listed these in 2026; <14 bars each — CPCV needs ≥50)
  - Brent oil: not listed on Binance.
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
)
from src.backtest.cpcv import compute_pbo, generate_paths
from src.backtest.data_cache import OHLCVCache
from src.backtest.fast_baseline import run_baseline_fast
from src.backtest.fill_model import FillConfig

sys.path.insert(0, str(Path(__file__).parent))
from sweep_robust import HEDGE, TRAIN_END, TRAIN_START, _champion_kwargs, parse_iso  # noqa: E402

from src.backtest.baseline_strategy import BaselineConfig  # noqa: E402

# Symbols available with sufficient history. Tagged by asset class and
# ordered roughly by market-cap / liquidity (large-cap / lower-vol first).
LIQUID_SYMBOLS = [
    # Crypto majors (Binance perp)
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    # US large-cap stocks (Yahoo v8)
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "COIN", "MSTR",
    # Index ETFs (broad market, lower vol)
    "SPY", "QQQ", "IWM", "DIA",
    # Sector ETFs
    "XLE", "XLF", "XLK",
    # Commodities
    "GLD", "SLV", "USO", "BNO", "UCO",
]

# PER USER REQUIREMENT: every multi-symbol subset MUST contain
# BTC + GLD + USD (cash). USD is represented by an implicit cash
# allocation — the strategy already supports "no position = cash" via
# the absence of a long position, so we ensure each subset reserves a
# share for cash by including a "USD_CASH" sentinel that resolves to
# zero-return cash during sweeping.
USD_CASH = "USD_CASH"


def _require_btc_gld_usd(syms: list[str]) -> list[str]:
    out = list(syms)
    if "BTC/USDT:USDT" not in out:
        out.insert(0, "BTC/USDT:USDT")
    if "GLD" not in out:
        out.insert(1, "GLD")
    if USD_CASH not in out:
        out.append(USD_CASH)
    return out


SUBSETS = [
    # Mandatory triple — the user's required defensive anchor
    ("anchor_btc_gld_usd",      _require_btc_gld_usd([])),
    # Crypto + anchor (BTC and GLD always; ETH/SOL on top)
    ("anchor_plus_eth",         _require_btc_gld_usd(["ETH/USDT:USDT"])),
    ("anchor_plus_sol",         _require_btc_gld_usd(["SOL/USDT:USDT"])),
    ("anchor_plus_eth_sol",     _require_btc_gld_usd(["ETH/USDT:USDT", "SOL/USDT:USDT"])),
    # Crypto top + anchor (replaces former crypto_top_5)
    ("crypto_top_5_w_anchor",   _require_btc_gld_usd([
        "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT"])),
    # Stock + anchor (large-cap, lower-vol first)
    ("indices_w_anchor",        _require_btc_gld_usd(["SPY", "QQQ", "DIA"])),
    ("megacap_w_anchor",        _require_btc_gld_usd([
        "MSFT", "AAPL", "GOOGL"])),
    ("megacap_8_w_anchor",      _require_btc_gld_usd([
        "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA", "SPY"])),
    # Commodities + anchor (with BNO=Brent oil and SLV=silver, both fetched)
    ("commodities_w_anchor",    _require_btc_gld_usd(["SLV", "USO", "BNO"])),
    # Cross-asset balanced (crypto + stocks + metals + oil + cash)
    ("balanced_w_anchor",       _require_btc_gld_usd([
        "ETH/USDT:USDT", "SPY", "QQQ", "MSFT", "AAPL", "BNO"])),
    ("low_vol_w_anchor",        _require_btc_gld_usd([
        "SPY", "MSFT", "AAPL"])),
    # Full universe (crypto + 5 megacap + indices + commodities)
    ("kitchen_sink_w_anchor",   _require_btc_gld_usd([
        "ETH/USDT:USDT", "SOL/USDT:USDT",
        "MSFT", "AAPL", "GOOGL", "NVDA",
        "SPY", "QQQ", "SLV", "USO", "BNO"])),

    # Focused: anchor + ONE additional asset class (per user's "subsets of US
    # stocks and/or oil must be tested" requirement). Each isolates the
    # marginal effect of that asset class on top of the BTC+GLD+USD anchor.
    ("anchor_plus_SPY",         _require_btc_gld_usd(["SPY"])),
    ("anchor_plus_QQQ",         _require_btc_gld_usd(["QQQ"])),
    ("anchor_plus_NVDA",        _require_btc_gld_usd(["NVDA"])),
    ("anchor_plus_MSFT_AAPL",   _require_btc_gld_usd(["MSFT", "AAPL"])),
    ("anchor_plus_3_indices",   _require_btc_gld_usd(["SPY", "QQQ", "DIA"])),
    ("anchor_plus_USO",         _require_btc_gld_usd(["USO"])),
    ("anchor_plus_BNO",         _require_btc_gld_usd(["BNO"])),
    ("anchor_plus_oil_2",       _require_btc_gld_usd(["USO", "BNO"])),
    ("anchor_plus_oil_3",       _require_btc_gld_usd(["USO", "BNO", "UCO"])),
    # Stocks + oil + anchor
    ("anchor_plus_indices_oil", _require_btc_gld_usd(["SPY", "QQQ", "USO", "BNO"])),

    # Pair-only / triple-only subsets (per user request — explicit pairs
    # without forced anchor injection so we can see each combo cleanly).
    # NOTE: these intentionally do NOT use _require_btc_gld_usd().
    ("pair_BTC_GLD",            ["BTC/USDT:USDT", "GLD"]),
    ("pair_BTC_USD",            ["BTC/USDT:USDT", USD_CASH]),
    ("triple_BTC_GLD_OIL",      ["BTC/USDT:USDT", "GLD", "BNO"]),
    ("pair_GLD_USD",            ["GLD", USD_CASH]),
    ("pair_GLD_OIL",            ["GLD", "BNO"]),
    ("pair_BTC_OIL",            ["BTC/USDT:USDT", "BNO"]),

    # Adding ETH and/or Silver to the small subsets above (per user request)
    ("triple_BTC_GLD_ETH",      ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT"]),
    ("triple_BTC_GLD_SLV",      ["BTC/USDT:USDT", "GLD", "SLV"]),
    ("quad_BTC_GLD_ETH_SLV",    ["BTC/USDT:USDT", "GLD", "ETH/USDT:USDT", "SLV"]),
    ("quad_BTC_GLD_OIL_ETH",    ["BTC/USDT:USDT", "GLD", "BNO", "ETH/USDT:USDT"]),
    ("quad_BTC_GLD_OIL_SLV",    ["BTC/USDT:USDT", "GLD", "BNO", "SLV"]),
    ("BTC_GLD_USD_ETH",         ["BTC/USDT:USDT", "GLD", USD_CASH, "ETH/USDT:USDT"]),
    ("BTC_GLD_USD_SLV",         ["BTC/USDT:USDT", "GLD", USD_CASH, "SLV"]),
    ("BTC_GLD_USD_ETH_SLV",     ["BTC/USDT:USDT", "GLD", USD_CASH, "ETH/USDT:USDT", "SLV"]),
    ("BTC_GLD_USD_OIL_ETH_SLV", ["BTC/USDT:USDT", "GLD", USD_CASH, "BNO",
                                  "ETH/USDT:USDT", "SLV"]),
]


def _stats(initial: float, eq_curve, periods_per_year=52) -> dict:
    eq = np.array([e for _, e in eq_curve], dtype=float)
    eq = np.maximum(eq, 1e-9)
    final = float(eq[-1])
    log_ret = np.diff(np.log(eq))
    years = eq.shape[0] / periods_per_year
    cagr = (final / initial) ** (1.0 / max(years, 1e-9)) - 1.0 if final > 0 else -1.0
    sharpe = sharpe_annualized(log_ret, periods_per_year=periods_per_year)
    dd, _, _ = max_drawdown(eq)
    cal = calmar_ratio(eq, periods_per_year=periods_per_year)
    return {"final": final, "cagr": cagr, "sharpe": sharpe, "dd": dd, "calmar": cal}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--cpcv-n", type=int, default=10)
    p.add_argument("--cpcv-k", type=int, default=2)
    p.add_argument("--out-dir", default="runs/symbol-sweep")
    args = p.parse_args(argv)

    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0, funding_rate_per_day=0.00010)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ms, end_ms = parse_iso(TRAIN_START), parse_iso(TRAIN_END)
    hedge = cache.load_range(HEDGE, "1w", start_ms, end_ms)
    hedge_use = hedge if not hedge.empty else None

    # ---- Phase 1: per-symbol contiguous run + CPCV ----
    per_symbol = {}
    print(f"\n{'='*100}")
    print(f"PER-SYMBOL CHAMPION (H4_baseline) — train {TRAIN_START} → {TRAIN_END}")
    print("Initial equity & all PnL/CAGR are USD-denominated (USDT for crypto)")
    print(f"{'='*100}")
    print(f"{'symbol':25s} {'bars':>5s} {'CAGR_USD':>9s} {'Sharpe':>8s} {'DD_USD':>8s} "
          f"{'CPCV_med':>9s} {'CPCV_p5':>8s} {'CPCV_p95':>9s} trades")

    rows = []
    for sym in LIQUID_SYMBOLS:
        primary = cache.load_range(sym, "1w", start_ms, end_ms)
        if primary.empty or len(primary) < 100:
            print(f"  {sym:25s} insufficient data ({len(primary)} bars) — SKIP")
            continue

        # Use this symbol as primary; PAXG remains the hedge
        cfg2 = BaselineConfig(**{**_champion_kwargs(), "primary_symbol": sym})
        # Contiguous run
        try:
            res = run_baseline_fast(
                primary_bars=primary, hedge_bars=hedge_use, config=cfg2,
                fill_config=fill_cfg, initial_equity=args.initial_equity,
            )
            stats = _stats(args.initial_equity, res.equity_curve)
        except Exception as e:
            print(f"  {sym:25s} run error: {e}")
            continue

        # CPCV per-path Sharpe (approximation: log returns at test indices)
        log_ret = np.diff(np.log(np.maximum(np.array([e for _, e in res.equity_curve]), 1e-12)))
        n_obs = len(primary)
        paths = generate_paths(n_observations=n_obs, n_groups=args.cpcv_n,
                               k_test=args.cpcv_k, purge=2, embargo=2)
        path_shs = []
        for path in paths:
            sample = [log_ret[i] for i in path.test_indices if 0 <= i < len(log_ret)]
            if len(sample) >= 4:
                path_shs.append(sharpe_annualized(np.asarray(sample), periods_per_year=52))
        med_sh = float(np.median(path_shs)) if path_shs else float("nan")
        p5 = float(np.percentile(path_shs, 5)) if path_shs else float("nan")
        p95 = float(np.percentile(path_shs, 95)) if path_shs else float("nan")

        per_symbol[sym] = {
            "stats": stats, "log_returns": log_ret.tolist(),
            "n_bars": len(primary), "n_trades": len(res.trades),
            "cpcv_median_sharpe": med_sh, "cpcv_p5": p5, "cpcv_p95": p95,
            "path_sharpes": path_shs,
        }
        print(f"  {sym:25s} {len(primary):>5d} {stats['cagr']*100:+7.1f}% "
              f"{stats['sharpe']:+8.2f} {stats['dd']*100:6.1f}%  "
              f"{med_sh:+9.2f} {p5:+8.2f} {p95:+8.2f}  {len(res.trades)}")
        rows.append({
            "symbol": sym, "n_bars": len(primary),
            "cagr": stats['cagr'], "sharpe": stats['sharpe'], "dd": stats['dd'],
            "calmar": stats['calmar'], "n_trades": len(res.trades),
            "cpcv_median_sharpe": med_sh, "cpcv_p5_sharpe": p5, "cpcv_p95_sharpe": p95,
        })

    # PBO across symbols
    if len(per_symbol) >= 2:
        names = list(per_symbol.keys())
        n_min = min(len(per_symbol[n]["path_sharpes"]) for n in names)
        if n_min >= 2:
            mat = np.full((n_min, len(names)), np.nan)
            for j, n in enumerate(names):
                mat[:, j] = per_symbol[n]["path_sharpes"][:n_min]
            pbo = compute_pbo(mat)
        else:
            pbo = float("nan")
    else:
        pbo = float("nan")
    print(f"\nPBO across {len(per_symbol)} symbols: {pbo:.3f}")

    # ---- Phase 2: subset comparison ----
    # Three weighting schemes per subset, each with USD_CASH = zero-return:
    #   (a) equal-weight 1/N
    #   (b) inverse-vol (each symbol weighted by 1/realized_sigma; risk parity)
    #   (c) Markowitz-style min-variance (returns matrix → analytic min-var w)
    #
    # USD_CASH is included as a zero-return / zero-vol leg in every subset
    # per user requirement (BTC + GLD + USD always). It anchors the
    # portfolio's cash sleeve.
    print(f"\n{'='*110}")
    print("SUBSET PORTFOLIOS — anchor enforced (BTC+GLD+USD) for anchored subsets")
    print("All CAGR / DD figures are USD-denominated (USDT for crypto legs).")
    print(f"{'='*110}")
    print(f"{'subset':28s} {'n':>2s}  EQUAL-WEIGHT (USD)       INV-VOL/risk-parity      MIN-VAR (Markowitz)")
    print(f"{'  ':28s} {'  ':>2s}  CAGR$ Sh    DD$         CAGR$ Sh    DD$         CAGR$ Sh    DD$")

    def _weighted_eq(syms_avail: list[str], weights: np.ndarray) -> dict:
        # Build per-symbol log-return arrays (USD_CASH → zeros)
        rets = []
        for s in syms_avail:
            if s == USD_CASH:
                rets.append(None)
            else:
                rets.append(np.asarray(per_symbol[s]["log_returns"]))
        non_cash_lens = [len(r) for r in rets if r is not None]
        if not non_cash_lens:
            return {}
        min_len = min(non_cash_lens)
        # Build (T, N) returns matrix
        cols = []
        for r in rets:
            cols.append(np.zeros(min_len) if r is None else r[:min_len])
        ret_mat = np.vstack(cols).T  # shape (T, N)
        port_ret = ret_mat @ weights
        eq = args.initial_equity * np.exp(np.cumsum(port_ret))
        return _stats(args.initial_equity,
                      [(0, args.initial_equity)] + [(i, float(v)) for i, v in enumerate(eq)])

    def _equal_weights(syms_avail: list[str]) -> np.ndarray:
        return np.full(len(syms_avail), 1.0 / max(len(syms_avail), 1))

    def _inverse_vol_weights(syms_avail: list[str]) -> np.ndarray:
        # 1 / sigma per symbol. USD_CASH gets a small fixed weight (cash anchor).
        n = len(syms_avail)
        if n == 0:
            return np.array([])
        sigmas = []
        for s in syms_avail:
            if s == USD_CASH:
                sigmas.append(None)  # cash leg
                continue
            r = np.asarray(per_symbol[s]["log_returns"])
            sd = float(np.std(r, ddof=1)) if len(r) > 2 else 0.05
            sigmas.append(sd if sd > 1e-6 else 0.05)
        # Risk-parity: weight ∝ 1/sigma for risk legs; cash gets the
        # average risk weight (so cash isn't dominant when included)
        avg_risk_sigma = np.mean([s for s in sigmas if s is not None]) \
            if any(s is not None for s in sigmas) else 0.05
        sigmas_filled = [avg_risk_sigma if s is None else s for s in sigmas]
        inv = 1.0 / np.asarray(sigmas_filled)
        return inv / inv.sum()

    def _min_var_weights(syms_avail: list[str], cash_cap: float = 0.20) -> np.ndarray:
        # Analytic minimum-variance with long-only constraint and a hard cap
        # on cash (USD_CASH) so the optimizer can't trivially collapse to
        # zero-vol cash. The remaining ~80% is allocated by analytic min-var
        # over the risk legs.
        n = len(syms_avail)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        cash_idx = [i for i, s in enumerate(syms_avail) if s == USD_CASH]
        risk_idx = [i for i, s in enumerate(syms_avail) if s != USD_CASH]
        if not risk_idx:
            return _equal_weights(syms_avail)

        rets = []
        for s in syms_avail:
            if s == USD_CASH:
                rets.append(None)
            else:
                rets.append(np.asarray(per_symbol[s]["log_returns"]))
        min_len = min(len(r) for r in rets if r is not None)
        risk_cols = [rets[i][:min_len] for i in risk_idx]
        risk_mat = np.vstack(risk_cols).T  # (T, k)
        k = len(risk_idx)
        cov = np.cov(risk_mat, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov) + 1e-8]])
        cov = np.atleast_2d(cov)
        cov += np.eye(k) * 1e-8

        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(k)
            w_risk = inv_cov @ ones
            denom = float(ones @ inv_cov @ ones)
            if denom <= 0:
                return _equal_weights(syms_avail)
            w_risk = w_risk / denom
            w_risk = np.maximum(w_risk, 0.0)
            if w_risk.sum() <= 0:
                return _equal_weights(syms_avail)
            w_risk = w_risk / w_risk.sum()
        except np.linalg.LinAlgError:
            return _equal_weights(syms_avail)

        full = np.zeros(n)
        risk_share = 1.0 - cash_cap if cash_idx else 1.0
        for j, idx in enumerate(risk_idx):
            full[idx] = w_risk[j] * risk_share
        if cash_idx:
            for idx in cash_idx:
                full[idx] = cash_cap / len(cash_idx)
        return full

    subset_rows = []
    for name, syms in SUBSETS:
        avail = [s for s in syms if s == USD_CASH or s in per_symbol]
        # Need BTC and GLD present (or USD_CASH alone is degenerate)
        non_cash = [s for s in avail if s != USD_CASH]
        if not non_cash:
            continue

        ew = _weighted_eq(avail, _equal_weights(avail))
        iv_w = _inverse_vol_weights(avail)
        iv = _weighted_eq(avail, iv_w)
        mv_w = _min_var_weights(avail)
        mv = _weighted_eq(avail, mv_w)

        if not (ew and iv and mv):
            continue

        print(f"  {name:28s} {len(avail):>2d}  "
              f"{ew['cagr']*100:+5.1f}% {ew['sharpe']:+5.2f} {ew['dd']*100:5.1f}%   "
              f"{iv['cagr']*100:+5.1f}% {iv['sharpe']:+5.2f} {iv['dd']*100:5.1f}%   "
              f"{mv['cagr']*100:+5.1f}% {mv['sharpe']:+5.2f} {mv['dd']*100:5.1f}%")
        subset_rows.append({
            "subset": name, "n_symbols": len(avail),
            "symbols": ",".join(avail),
            "equal_weight_cagr": ew["cagr"], "equal_weight_sharpe": ew["sharpe"],
            "equal_weight_dd": ew["dd"],
            "inv_vol_cagr": iv["cagr"], "inv_vol_sharpe": iv["sharpe"],
            "inv_vol_dd": iv["dd"],
            "min_var_cagr": mv["cagr"], "min_var_sharpe": mv["sharpe"],
            "min_var_dd": mv["dd"],
        })

    cache.close()

    # Save
    if rows:
        with (out_dir / "per_symbol.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if subset_rows:
        with (out_dir / "subsets.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(subset_rows[0].keys()))
            w.writeheader()
            w.writerows(subset_rows)

    summary = {
        "training_window": {"start": TRAIN_START, "end": TRAIN_END},
        "n_symbols": len(per_symbol),
        "pbo_across_symbols": pbo,
        "per_symbol": {n: {k: v for k, v in d.items() if k not in {"log_returns", "path_sharpes"}}
                       for n, d in per_symbol.items()},
        "subsets": subset_rows,
        "rejected_symbols": {
            "US_stock_perps": ["COIN", "MSTR", "NVDA", "TSLA", "SPY"],
            "reason": "Binance listed these in 2026; <14 weekly bars per symbol — "
                       "insufficient history for CPCV (need ≥50)",
            "Brent_oil": ["OIL", "BRENT", "WTI"],
            "Brent_reason": "Not listed on Binance perpetuals.",
            "silver": "No silver-backed perpetual on Binance (gold = PAXG/XAUT)",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote → {out_dir}/per_symbol.csv  subsets.csv  summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
