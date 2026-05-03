#!/usr/bin/env python3
"""Hypothesis-driven robust grid search.

Tests ≤30 named configurations across 4 hypothesis families:

  H1 — Regime-adaptive vol target (per-regime size_pct + vol_target)
  H2 — Asymmetric crisis halt (force flat under regime=crisis)
  H3 — Long-bias barbell with size-throttled tactical shorts
  H4 — Hardened champion variants (small perturbations of the prior winner)

Each config is evaluated on:

  - Full contiguous window 2021-01-04 → 2025-09-30 (training)
  - Six regime sub-windows (blowoff-2021, bear-2022, recovery-2023,
    bull-2024, late-2025-train, plus the OOS holdout 2025-10-01 → 2026-04-27
    is RESERVED — not touched by this script)
  - CPCV (N=10, K=2 = 45 paths) on training window only

Output: ``runs/robust-sweep/grid.csv`` with per-(config, window) stats and
a ``stats.json`` with PBO / DSR per config.

The OOS validation is performed by ``scripts/validate_oos.py`` afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.backtest_stats import (
    calmar_ratio,
    deflated_sharpe_ratio,
    max_drawdown,
    pnl_moments,
    sharpe_annualized,
    sortino_ratio,
)
from src.analysis.bootstrap import block_bootstrap_sharpe_ci
from src.backtest.baseline_strategy import BaselineConfig
from src.backtest.cpcv import compute_pbo, generate_paths, summarize_paths
from src.backtest.data_cache import OHLCVCache
from src.backtest.fast_baseline import run_baseline_fast
from src.backtest.fill_model import FillConfig
from src.backtest.quant_overlay import QuantOverlayConfig

PRIMARY = "BTC/USDT:USDT"
HEDGE = "PAXG/USDT"

# Training window — OOS is reserved for validate_oos.py
TRAIN_START = "2021-01-04"
TRAIN_END = "2025-09-29"
OOS_START = "2025-10-06"
OOS_END = "2026-04-27"

# Regime sub-windows (within the TRAINING period only — no OOS leakage)
REGIME_WINDOWS = [
    ("blowoff-2021",  "2021-01-04", "2022-01-03"),
    ("bear-2022",     "2022-01-03", "2023-01-02"),
    ("recovery-2023", "2023-01-02", "2024-01-01"),
    ("bull-2024",     "2024-01-01", "2024-12-30"),
    ("late-2025-tr",  "2025-01-06", TRAIN_END),
]


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


@dataclass
class NamedConfig:
    name: str
    hypothesis: str
    config: BaselineConfig
    regime_overrides: dict[str, dict] | None = None
    quant_overlay: QuantOverlayConfig | None = None


def _champion_kwargs() -> dict:
    return dict(
        primary_symbol=PRIMARY,
        hedge_symbol=HEDGE,
        momentum_lookback_bars=4,
        drawdown_hedge_threshold=0.20,
        long_size_usd_pct=3.0,
        vol_target_pct=0.015,
        vol_lookback_bars=12,
        vol_size_max_mult=1.67,
    )


def all_configs() -> list[NamedConfig]:
    """Return ≤30 named configs across H1–H4."""
    out: list[NamedConfig] = []

    # ---- H4: hardened champion variants (8 configs, small perturbations) ----
    out.append(NamedConfig("H4_baseline", "H4_champion",
                           BaselineConfig(**_champion_kwargs())))
    out.append(NamedConfig("H4_size_2.0", "H4_champion",
                           BaselineConfig(**{**_champion_kwargs(), "long_size_usd_pct": 2.0})))
    out.append(NamedConfig("H4_size_2.5", "H4_champion",
                           BaselineConfig(**{**_champion_kwargs(), "long_size_usd_pct": 2.5})))
    out.append(NamedConfig("H4_vt_0.012", "H4_champion",
                           BaselineConfig(**{**_champion_kwargs(), "vol_target_pct": 0.012})))
    out.append(NamedConfig("H4_vt_0.020", "H4_champion",
                           BaselineConfig(**{**_champion_kwargs(), "vol_target_pct": 0.020})))
    out.append(NamedConfig("H4_hedge_0.15", "H4_champion",
                           BaselineConfig(**{**_champion_kwargs(), "drawdown_hedge_threshold": 0.15})))
    out.append(NamedConfig("H4_hedge_0.10", "H4_champion",
                           BaselineConfig(**{**_champion_kwargs(), "drawdown_hedge_threshold": 0.10})))
    out.append(NamedConfig("H4_mom_8", "H4_champion",
                           BaselineConfig(**{**_champion_kwargs(), "momentum_lookback_bars": 8})))

    # ---- H1: regime-adaptive vol target (8 configs) ----
    # Each: champion as the calm-regime config, scaled down in elevated/crisis.
    base = BaselineConfig(**_champion_kwargs())
    h1_overrides_a = {
        "calm":     {"long_size_usd_pct": 3.0, "vol_target_pct": 0.018},
        "elevated": {"long_size_usd_pct": 1.5, "vol_target_pct": 0.012},
        "crisis":   {"long_size_usd_pct": 0.0},
    }
    h1_overrides_b = {  # less aggressive in calm, gentler de-risk in elevated
        "calm":     {"long_size_usd_pct": 2.5, "vol_target_pct": 0.015},
        "elevated": {"long_size_usd_pct": 2.0, "vol_target_pct": 0.012},
        "crisis":   {"long_size_usd_pct": 0.0},
    }
    h1_overrides_c = {  # very strong de-risk
        "calm":     {"long_size_usd_pct": 3.5, "vol_target_pct": 0.020},
        "elevated": {"long_size_usd_pct": 1.0, "vol_target_pct": 0.010},
        "crisis":   {"long_size_usd_pct": 0.0},
    }
    h1_overrides_d = {  # zero size + disable hedge in crisis (full cash)
        "calm":     {"long_size_usd_pct": 3.0, "vol_target_pct": 0.015},
        "elevated": {"long_size_usd_pct": 3.0, "vol_target_pct": 0.015},
        "crisis":   {"long_size_usd_pct": 0.0, "disable_hedge": True},
    }
    out.append(NamedConfig("H1_a_strong_de_risk",  "H1_regime_adaptive", base, h1_overrides_a))
    out.append(NamedConfig("H1_b_moderate",       "H1_regime_adaptive", base, h1_overrides_b))
    out.append(NamedConfig("H1_c_aggressive_calm","H1_regime_adaptive", base, h1_overrides_c))
    out.append(NamedConfig("H1_d_crisis_only",    "H1_regime_adaptive", base, h1_overrides_d))

    h1_overrides_e = {  # sma=14 / vol_short=2 / vol_long=14 sensitivity proxy
        "calm":     {"long_size_usd_pct": 3.0, "vol_target_pct": 0.018},
        "elevated": {"long_size_usd_pct": 2.0, "vol_target_pct": 0.013},
        "crisis":   {"long_size_usd_pct": 0.5, "vol_target_pct": 0.008},
    }
    out.append(NamedConfig("H1_e_partial_crisis", "H1_regime_adaptive", base, h1_overrides_e))

    # H1 with smaller momentum lookback (faster trend signal)
    base_mom8 = BaselineConfig(**{**_champion_kwargs(), "momentum_lookback_bars": 8})
    out.append(NamedConfig("H1_mom8_a", "H1_regime_adaptive", base_mom8, h1_overrides_a))
    out.append(NamedConfig("H1_mom8_b", "H1_regime_adaptive", base_mom8, h1_overrides_b))
    out.append(NamedConfig("H1_mom8_d", "H1_regime_adaptive", base_mom8, h1_overrides_d))

    # ---- H2: champion sizing + crisis-only zero (most surgical) (3 configs) ----
    out.append(NamedConfig("H2_strict_halt", "H2_crisis_halt",
                           base, {"crisis": {"long_size_usd_pct": 0.0}}))
    out.append(NamedConfig("H2_half_halt", "H2_crisis_halt",
                           base, {"crisis": {"long_size_usd_pct": 1.0, "vol_target_pct": 0.010}}))
    out.append(NamedConfig("H2_strict_halt_mom8", "H2_crisis_halt",
                           base_mom8, {"crisis": {"long_size_usd_pct": 0.0}}))
    out.append(NamedConfig("H2_full_cash_crisis", "H2_crisis_halt",
                           base, {"crisis": {"long_size_usd_pct": 0.0, "disable_hedge": True}}))
    out.append(NamedConfig("H2_full_cash_crisis_mom8", "H2_crisis_halt",
                           base_mom8, {"crisis": {"long_size_usd_pct": 0.0, "disable_hedge": True}}))
    out.append(NamedConfig("H2_full_cash_elev_too", "H2_crisis_halt",
                           base, {
                               "elevated": {"long_size_usd_pct": 1.5, "vol_target_pct": 0.012},
                               "crisis": {"long_size_usd_pct": 0.0, "disable_hedge": True},
                           }))

    # ---- H3: long-bias + tactical shorts only on confirmed bears (4 configs) ----
    h3_kwargs_a = {**_champion_kwargs(), "enable_shorts": True, "short_min_drawdown": 0.25,
                   "short_size_usd_pct": 1.0, "short_stop_loss_pct": 0.10,
                   "momentum_lookback_bars": 16}
    h3_kwargs_b = {**_champion_kwargs(), "enable_shorts": True, "short_min_drawdown": 0.30,
                   "short_size_usd_pct": 1.5, "short_stop_loss_pct": 0.10,
                   "momentum_lookback_bars": 16}
    h3_kwargs_c = {**_champion_kwargs(), "enable_shorts": True, "short_min_drawdown": 0.20,
                   "short_size_usd_pct": 1.0, "short_stop_loss_pct": 0.08,
                   "momentum_lookback_bars": 8}
    h3_kwargs_d = {**_champion_kwargs(), "enable_shorts": True, "short_min_drawdown": 0.25,
                   "short_size_usd_pct": 0.5, "short_stop_loss_pct": 0.10,
                   "momentum_lookback_bars": 12}
    out.append(NamedConfig("H3_short_dd25_size1.0_mom16", "H3_tactical_shorts",
                           BaselineConfig(**h3_kwargs_a)))
    out.append(NamedConfig("H3_short_dd30_size1.5_mom16", "H3_tactical_shorts",
                           BaselineConfig(**h3_kwargs_b)))
    out.append(NamedConfig("H3_short_dd20_size1.0_mom8", "H3_tactical_shorts",
                           BaselineConfig(**h3_kwargs_c)))
    out.append(NamedConfig("H3_short_dd25_size0.5_mom12", "H3_tactical_shorts",
                           BaselineConfig(**h3_kwargs_d)))

    # H1+H3 hybrid: regime-adaptive sizing + tactical shorts
    h1h3_base = BaselineConfig(**h3_kwargs_a)
    out.append(NamedConfig("H1_H3_hybrid", "H1_H3",
                           h1h3_base, h1_overrides_a))

    # H2+H3 hybrid: shorts + crisis halt for longs
    h2h3_base = BaselineConfig(**h3_kwargs_a)
    out.append(NamedConfig("H2_H3_hybrid", "H2_H3",
                           h2h3_base, {"crisis": {"long_size_usd_pct": 0.0}}))

    # ---- H5: champion + GARCH vol-targeting overlay ----
    out.append(NamedConfig(
        "H5_garch", "H5_garch",
        BaselineConfig(**_champion_kwargs()),
        regime_overrides=None,
        quant_overlay=QuantOverlayConfig(use_garch_vol=True),
    ))
    # H5 + emergency dd halt (15%)
    out.append(NamedConfig(
        "H5_garch_dd_halt", "H5_garch",
        BaselineConfig(**_champion_kwargs()),
        regime_overrides=None,
        quant_overlay=QuantOverlayConfig(
            use_garch_vol=True, emergency_dd_halt=0.15,
        ),
    ))
    # H5 + emergency dd halt + GARCH z spike halt
    out.append(NamedConfig(
        "H5_garch_full_emergency", "H5_garch",
        BaselineConfig(**_champion_kwargs()),
        regime_overrides=None,
        quant_overlay=QuantOverlayConfig(
            use_garch_vol=True,
            emergency_dd_halt=0.13, emergency_garch_z=2.5,
        ),
    ))

    # ---- H6: champion + adaptive Kelly sizing ----
    out.append(NamedConfig(
        "H6_kelly", "H6_kelly",
        BaselineConfig(**_champion_kwargs()),
        regime_overrides=None,
        quant_overlay=QuantOverlayConfig(kelly_adaptive=True),
    ))

    # ---- H7: GARCH + Kelly + emergency (full quant overlay stack) ----
    out.append(NamedConfig(
        "H7_full_quant", "H7_full_quant",
        BaselineConfig(**_champion_kwargs()),
        regime_overrides=None,
        quant_overlay=QuantOverlayConfig(
            use_garch_vol=True, kelly_adaptive=True,
            emergency_dd_halt=0.13, emergency_garch_z=2.5,
        ),
    ))
    # H7 + regime adaptive (the maximum-stack config)
    out.append(NamedConfig(
        "H7_full_quant_regime", "H7_full_quant",
        BaselineConfig(**_champion_kwargs()),
        regime_overrides=h1_overrides_d,
        quant_overlay=QuantOverlayConfig(
            use_garch_vol=True, kelly_adaptive=True,
            emergency_dd_halt=0.13, emergency_garch_z=2.5,
        ),
    ))

    # ---- H8: PERP — long+short with GARCH vol-targeting + emergency halt ----
    h8_kwargs = {**_champion_kwargs(), "enable_shorts": True,
                 "short_min_drawdown": 0.20, "short_size_usd_pct": 1.5,
                 "short_stop_loss_pct": 0.10, "momentum_lookback_bars": 8}
    out.append(NamedConfig(
        "H8_perp_garch", "H8_perp",
        BaselineConfig(**h8_kwargs),
        regime_overrides=None,
        quant_overlay=QuantOverlayConfig(
            use_garch_vol=True, emergency_dd_halt=0.13, emergency_garch_z=2.5,
        ),
    ))
    h8b_kwargs = {**_champion_kwargs(), "enable_shorts": True,
                  "short_min_drawdown": 0.25, "short_size_usd_pct": 1.0,
                  "short_stop_loss_pct": 0.10, "momentum_lookback_bars": 12}
    out.append(NamedConfig(
        "H8_perp_garch_kelly", "H8_perp",
        BaselineConfig(**h8b_kwargs),
        regime_overrides=None,
        quant_overlay=QuantOverlayConfig(
            use_garch_vol=True, kelly_adaptive=True,
            emergency_dd_halt=0.13, emergency_garch_z=2.5,
        ),
    ))

    return out


def _stats_from_curve(initial_eq: float, equity_curve, periods_per_year=52) -> dict:
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
    return {
        "final_equity": final, "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": dd, "calmar": cal, "log_returns": log_ret,
    }


def _run_one(
    cfg: NamedConfig,
    primary: pd.DataFrame, hedge: pd.DataFrame | None,
    fill_cfg: FillConfig, initial_eq: float,
) -> tuple[dict, list[float]]:
    """Run one config on the given primary/hedge data. Returns (stats, log_returns)."""
    res = run_baseline_fast(
        primary_bars=primary, hedge_bars=hedge,
        config=cfg.config, fill_config=fill_cfg,
        initial_equity=initial_eq,
        regime_overrides=cfg.regime_overrides,
        quant_overlay=cfg.quant_overlay,
    )
    stats = _stats_from_curve(initial_eq, res.equity_curve)
    stats["n_trades"] = len(res.trades)
    return stats, list(stats.pop("log_returns"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--cpcv-n", type=int, default=10)
    p.add_argument("--cpcv-k", type=int, default=2)
    p.add_argument("--cpcv-purge", type=int, default=2)
    p.add_argument("--cpcv-embargo", type=int, default=2)
    p.add_argument("--out-dir", default="runs/robust-sweep")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0, funding_rate_per_day=0.00010)

    # Training window full data (used both for contiguous AND CPCV)
    train_primary = cache.load_range(PRIMARY, "1w", parse_iso(TRAIN_START), parse_iso(TRAIN_END))
    train_hedge   = cache.load_range(HEDGE,   "1w", parse_iso(TRAIN_START), parse_iso(TRAIN_END))
    train_hedge = train_hedge if not train_hedge.empty else None

    if train_primary.empty:
        print("No training data; abort.", file=sys.stderr)
        return 2
    print(f"Training window: {TRAIN_START} → {TRAIN_END}, {len(train_primary)} weekly bars")

    cfgs = all_configs()
    print(f"Sweeping {len(cfgs)} configs across {len(REGIME_WINDOWS)} regime windows + full + CPCV")

    rows: list[dict] = []
    cpcv_sharpes: dict[str, list[float]] = {}  # cfg_name -> list of path Sharpes

    n_obs = len(train_primary)
    paths = generate_paths(
        n_observations=n_obs, n_groups=args.cpcv_n, k_test=args.cpcv_k,
        purge=args.cpcv_purge, embargo=args.cpcv_embargo,
    )
    print(f"CPCV: {len(paths)} paths over {n_obs} bars")

    for cfg in cfgs:
        # Contiguous train window
        try:
            stats, log_ret = _run_one(cfg, train_primary, train_hedge, fill_cfg, args.initial_equity)
        except Exception as e:
            print(f"  [{cfg.name:30s}] FAIL train: {e}", file=sys.stderr)
            continue
        # bootstrap CI
        try:
            ci_lo, ci_hi = block_bootstrap_sharpe_ci(
                np.asarray(log_ret), periods_per_year=52, n_resamples=2000,
                confidence=0.95, block_size=4, seed=42,
            )
        except Exception:
            ci_lo, ci_hi = float("nan"), float("nan")
        rows.append({
            "config": cfg.name, "hypothesis": cfg.hypothesis, "window": "train-full",
            **{k: v for k, v in stats.items() if k != "log_returns"},
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
        print(f"  [{cfg.name:30s}] train: CAGR={stats['cagr']*100:+6.1f}% "
              f"Sh={stats['sharpe']:+.2f} DD={stats['max_drawdown']*100:5.1f}% "
              f"trades={stats['n_trades']}")

        # Per-regime sub-windows
        for win_name, ws, we in REGIME_WINDOWS:
            sub_primary = cache.load_range(PRIMARY, "1w", parse_iso(ws), parse_iso(we))
            sub_hedge   = cache.load_range(HEDGE,   "1w", parse_iso(ws), parse_iso(we))
            sub_hedge = sub_hedge if not sub_hedge.empty else None
            if sub_primary.empty:
                continue
            try:
                wstats, _ = _run_one(cfg, sub_primary, sub_hedge, fill_cfg, args.initial_equity)
            except Exception:
                continue
            rows.append({
                "config": cfg.name, "hypothesis": cfg.hypothesis, "window": win_name,
                **{k: v for k, v in wstats.items() if k != "log_returns"},
                "ci_lo": float("nan"), "ci_hi": float("nan"),
            })

        # CPCV: per-path Sharpe by simulating ONLY on the test indices of each path.
        # Implementation note: a true CPCV "simulate test indices only" is non-trivial
        # because positions carry across bars. We use a simpler approximation:
        # sample contiguous chunks of the training window (each path's test_indices)
        # and compute Sharpe on the strategy's log_returns over those bars.
        path_sharpes: list[float] = []
        full_log_ret = np.asarray(log_ret)
        for path in paths:
            # log_returns is len(eq)-1; we need indices into log_returns space.
            # eq has one entry per bar (start_idx → end_idx) plus extras at start/end.
            # Quick approximation: take the strategy's log returns at the test indices
            # (clipped to valid range).
            sample = [full_log_ret[i] for i in path.test_indices
                      if 0 <= i < len(full_log_ret)]
            if len(sample) < 4:
                continue
            arr = np.asarray(sample, dtype=float)
            ps = sharpe_annualized(arr, periods_per_year=52)
            path_sharpes.append(ps)
        cpcv_sharpes[cfg.name] = path_sharpes
        if path_sharpes:
            summary = summarize_paths(np.asarray(path_sharpes))
            rows.append({
                "config": cfg.name, "hypothesis": cfg.hypothesis, "window": "cpcv",
                "final_equity": float("nan"),
                "cagr": float("nan"),
                "sharpe": summary["median"],
                "sortino": float("nan"),
                "max_drawdown": float("nan"),
                "calmar": float("nan"),
                "n_trades": 0,
                "ci_lo": summary["p5"], "ci_hi": summary["p95"],
            })
            print(f"     CPCV: median Sharpe={summary['median']:+.2f}  "
                  f"p5={summary['p5']:+.2f} p95={summary['p95']:+.2f}")

    cache.close()

    # PBO across configs (use training-window CPCV path Sharpes)
    cfg_names = [c.name for c in cfgs if c.name in cpcv_sharpes and cpcv_sharpes[c.name]]
    if len(cfg_names) >= 2:
        max_paths = max(len(cpcv_sharpes[n]) for n in cfg_names)
        sh_matrix = np.full((max_paths, len(cfg_names)), np.nan)
        for j, n in enumerate(cfg_names):
            v = cpcv_sharpes[n]
            sh_matrix[: len(v), j] = v
        # Drop rows with any NaN
        valid_rows = ~np.any(np.isnan(sh_matrix), axis=1)
        pbo = compute_pbo(sh_matrix[valid_rows]) if valid_rows.sum() >= 2 else float("nan")
    else:
        pbo = float("nan")

    # DSR per config (deflated Sharpe)
    dsr_per_config = {}
    n_trials = len(cfgs)
    for cfg in cfgs:
        train_row = next((r for r in rows
                          if r["config"] == cfg.name and r["window"] == "train-full"), None)
        if not train_row:
            continue
        sr_per_period = train_row["sharpe"] / np.sqrt(52)
        # Use fresh log_returns to get skew/kurt
        try:
            stats_fresh, log_ret_fresh = _run_one(cfg, train_primary, train_hedge, fill_cfg, args.initial_equity)
            mom = pnl_moments(np.asarray(log_ret_fresh))
            n_obs = len(log_ret_fresh)
        except Exception:
            continue
        # SR variance across the n_trials configs
        sr_vals = [r["sharpe"] / np.sqrt(52) for r in rows
                   if r["window"] == "train-full"]
        sr_var = float(np.var(sr_vals, ddof=1)) if len(sr_vals) > 1 else 0.0
        try:
            dsr = deflated_sharpe_ratio(
                sr_hat=sr_per_period, sr_trials_var=sr_var, num_trials=n_trials,
                n=n_obs, skew=mom.get("skew", 0.0), kurt=mom.get("kurt", 3.0),
            )
        except Exception:
            dsr = float("nan")
        dsr_per_config[cfg.name] = dsr

    # Composite score (continuous, not all-or-nothing):
    #   composite = cagr_train × dd_factor × min_window_factor
    # where:
    #   dd_factor          = max(0.05, 1 - max_dd / 0.15)  — reward DD headroom under cap
    #   min_window_factor  = sigmoid(min_window_sharpe + 1.5)
    #     (centered at -1.5; even bear-2022 = -1.75 → factor ~0.45;
    #      shifts emphasis toward configs that don't blow up bear)
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    composite_per_config = {}
    for cfg in cfgs:
        train_row = next((r for r in rows
                          if r["config"] == cfg.name and r["window"] == "train-full"), None)
        if not train_row:
            continue
        regime_rows = [r for r in rows
                       if r["config"] == cfg.name and r["window"] in {w[0] for w in REGIME_WINDOWS}]
        if not regime_rows:
            continue
        min_window_sharpe = min(r["sharpe"] for r in regime_rows)
        median_window_sharpe = float(np.median([r["sharpe"] for r in regime_rows]))
        dd = train_row["max_drawdown"]
        cagr = train_row["cagr"]
        dd_factor = max(0.05, 1.0 - dd / 0.15)
        min_factor = _sigmoid(min_window_sharpe + 1.5)
        med_factor = _sigmoid(median_window_sharpe)
        # CAGR floored at 0 (don't reward losing strategies; tolerate flat)
        composite = max(0.0, cagr) * dd_factor * min_factor * med_factor
        composite_per_config[cfg.name] = composite

    # Ranking
    ranking = sorted(composite_per_config.items(), key=lambda kv: -kv[1])

    print("\n" + "=" * 80)
    print("RANKING by composite score (CAGR × min_window_Sharpe × DD_headroom)")
    print("All CAGR / DD figures are USD-denominated.")
    print("=" * 80)
    print(f"{'rank':4s} {'config':30s} {'composite':>10s} {'CAGR$':>8s} {'minWinSh':>9s} "
          f"{'DD':>7s} {'DSR':>7s}")
    for rank, (name, comp) in enumerate(ranking[:15], start=1):
        train_row = next(r for r in rows if r["config"] == name and r["window"] == "train-full")
        regime_rows = [r for r in rows
                       if r["config"] == name and r["window"] in {w[0] for w in REGIME_WINDOWS}]
        min_ws = min(r["sharpe"] for r in regime_rows) if regime_rows else float("nan")
        print(f"{rank:>4d} {name:30s} {comp:+10.4f} {train_row['cagr']*100:+7.1f}% "
              f"{min_ws:+8.2f}  {train_row['max_drawdown']*100:5.1f}%  "
              f"{dsr_per_config.get(name, float('nan')):+6.3f}")

    # Save artifacts
    fieldnames = [
        "config", "hypothesis", "window", "final_equity", "cagr", "sharpe", "sortino",
        "max_drawdown", "calmar", "n_trades", "ci_lo", "ci_hi",
    ]
    with (out_dir / "grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})

    stats_summary = {
        "training_window": {"start": TRAIN_START, "end": TRAIN_END},
        "oos_holdout":     {"start": OOS_START,   "end": OOS_END},
        "cpcv":            {"n_groups": args.cpcv_n, "k_test": args.cpcv_k,
                            "n_paths": len(paths), "purge": args.cpcv_purge,
                            "embargo": args.cpcv_embargo},
        "n_configs":       len(cfgs),
        "pbo_across_configs": pbo,
        "dsr_per_config":  dsr_per_config,
        "composite_score": composite_per_config,
        "top_3":           [name for name, _ in ranking[:3]],
    }
    (out_dir / "stats.json").write_text(json.dumps(stats_summary, indent=2))

    print(f"\nWrote → {out_dir}/grid.csv  and  {out_dir}/stats.json")
    print(f"PBO across {len(cfgs)} configs: {pbo:.3f}" if not np.isnan(pbo) else "PBO: nan")
    print(f"Top-3 advance to OOS validation: {stats_summary['top_3']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
