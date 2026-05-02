#!/usr/bin/env python3
"""True CPCV (Combinatorial Purged Cross-Validation) runner for the
hypothesis-driven configs in ``sweep_robust.py``.

Per López de Prado AFML Ch. 12 (adapted for trading strategies):

  - Split the timeline into ``cpcv_n`` contiguous groups.
  - For each combination of ``cpcv_k`` test groups (C(N,K) paths):
      1. Identify the test indices (purge & embargo applied around each group)
      2. Group adjacent test indices into contiguous segments
      3. Reset strategy state at the start of each segment
      4. Run the deterministic engine on each segment's bars
      5. Concatenate the equity-changes from each segment → one path equity
         curve. Strategy state is RESET between segments — this is what
         distinguishes true CPCV from simply "looking at log-returns at
         test indices".
  - Per-path: compute Sharpe, max DD, CAGR, n_trades.
  - Per-config: aggregate path distribution → median, p5, p95, IQR.
  - Across configs: PBO (Bailey & López de Prado 2014) from per-path
    Sharpe matrix.

This is a more rigorous backtest than the contiguous walk: it produces a
distribution of outcomes across many time-window selections, which is the
right way to bound expected variance for a strategy facing an unknown
future regime mix.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.backtest_stats import sharpe_annualized
from src.backtest.cpcv import compute_pbo, generate_paths
from src.backtest.data_cache import OHLCVCache
from src.backtest.fast_baseline import run_baseline_fast
from src.backtest.fill_model import FillConfig

sys.path.insert(0, str(Path(__file__).parent))
from sweep_robust import (
    HEDGE,
    PRIMARY,
    TRAIN_END,
    TRAIN_START,
    NamedConfig,
    all_configs,
    parse_iso,
)


def _contiguous_segments(indices: list[int]) -> list[tuple[int, int]]:
    """Group sorted indices into contiguous (start, end_inclusive) segments."""
    if not indices:
        return []
    sorted_idx = sorted(indices)
    segs = []
    seg_start = sorted_idx[0]
    prev = seg_start
    for i in sorted_idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segs.append((seg_start, prev))
            seg_start = i
            prev = i
    segs.append((seg_start, prev))
    return segs


def _path_equity_log_returns(
    cfg: NamedConfig,
    primary_bars: pd.DataFrame,
    hedge_bars: pd.DataFrame | None,
    fill_cfg: FillConfig,
    initial_equity: float,
    test_indices: list[int],
    snapshot_lookback_bars: int = 30,
) -> tuple[np.ndarray, float, float, int]:
    """Run the strategy on each contiguous test segment with state reset.

    Returns (concatenated_log_returns, final_equity_relative, max_dd, n_trades).

    ``final_equity_relative`` = product of segment-wise equity multipliers,
    not the absolute USD amount, because each segment starts at 1.0.
    """
    segments = _contiguous_segments(test_indices)
    if not segments:
        return np.array([], dtype=float), 1.0, 0.0, 0

    # Need enough lead-in bars before each segment for the strategy's
    # signals (momentum lookback, vol lookback, regime classifier).
    # Lead-in is included in the simulation but its returns are NOT
    # counted toward the path's PnL — only bars in [seg_start, seg_end]
    # contribute returns. Lead-in is read-only for signal computation.
    cfg_obj = cfg.config
    lead_in = max(
        snapshot_lookback_bars,
        cfg_obj.momentum_lookback_bars + 2,
        cfg_obj.vol_lookback_bars + 2 if cfg_obj.vol_target_pct is not None else 0,
        28,  # regime classifier sma_window default
    )

    all_log_rets: list[float] = []
    cumulative_mult = 1.0
    cumulative_max = 1.0
    running_dd = 0.0
    total_trades = 0
    n = len(primary_bars)

    for (seg_start, seg_end) in segments:
        # Effective start including lead-in (clipped to >= 0). The strategy
        # will compute signals from these bars but the equity curve at
        # those bars is discarded.
        sim_start = max(0, seg_start - lead_in)
        sim_end = min(n - 1, seg_end + 1)  # +1 so the last bar can be settled

        seg_primary = primary_bars.iloc[: sim_end + 1].reset_index(drop=True)
        seg_hedge = (
            hedge_bars.iloc[: sim_end + 1].reset_index(drop=True)
            if hedge_bars is not None and not hedge_bars.empty else None
        )
        # Run the strategy starting at sim_start (state resets here)
        try:
            res = run_baseline_fast(
                primary_bars=seg_primary,
                hedge_bars=seg_hedge,
                config=cfg.config,
                fill_config=fill_cfg,
                initial_equity=1.0,  # normalized; we track relative multipliers
                start_idx=sim_start,
                end_idx=sim_end,
                regime_overrides=cfg.regime_overrides,
                snapshot_lookback_bars=snapshot_lookback_bars,
            )
        except Exception:
            continue

        # Equity curve: list[(ts_ms, equity)]. Index aligns with bars in
        # [sim_start..sim_end], plus a trailing final entry. We want only
        # the entries corresponding to bars in [seg_start..seg_end].
        # The equity curve from run_baseline_fast has one entry per
        # iteration plus initial + final (so len = (sim_end - sim_start + 1) + 1
        # in normal cases). We slice by ts_ms to be robust.
        eq_curve = np.array([e for _, e in res.equity_curve], dtype=float)
        if len(eq_curve) < 2:
            continue
        # Take only the bars at and after seg_start. The first lead-in
        # bars contribute zero returns (no trading happened during them
        # if the strategy didn't enter, or initial-state churn that we
        # discard by anchoring at seg_start's equity value).
        # eq_curve[0] = sim_start equity = 1.0; eq_curve[k] = equity after iter k
        eq_curve = np.maximum(eq_curve, 1e-12)
        # Indices into eq_curve corresponding to the segment-start bar:
        # iter k corresponds to bar sim_start + k. seg_start = sim_start + lead_offset
        lead_offset = seg_start - sim_start
        if lead_offset >= len(eq_curve):
            continue
        seg_eq = eq_curve[lead_offset:]
        if len(seg_eq) < 2:
            continue

        # Anchor at 1.0 for this segment so the path is a CONCATENATION of
        # independent test-segment journeys (state reset). Compute log
        # returns relative to the segment's starting equity.
        seg_eq_norm = seg_eq / seg_eq[0]
        seg_log = np.diff(np.log(seg_eq_norm))
        all_log_rets.extend(seg_log.tolist())

        # Track cumulative multiplier (path's hypothetical equity if compounded)
        for r in seg_log:
            cumulative_mult *= float(np.exp(r))
            cumulative_max = max(cumulative_max, cumulative_mult)
            running_dd = max(running_dd, (cumulative_max - cumulative_mult) / cumulative_max)

        total_trades += len(res.trades)

    return np.asarray(all_log_rets), cumulative_mult, running_dd, total_trades


@dataclass
class CPCVResult:
    config_name: str
    hypothesis: str
    path_sharpes: list[float]
    path_drawdowns: list[float]
    path_cagrs: list[float]
    n_paths: int
    median_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    iqr_sharpe: float
    median_dd: float
    p95_dd: float
    median_cagr: float
    p5_cagr: float
    fraction_paths_positive: float
    fraction_paths_dd_under_15: float


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--cpcv-n", type=int, default=10)
    p.add_argument("--cpcv-k", type=int, default=2)
    p.add_argument("--cpcv-purge", type=int, default=2)
    p.add_argument("--cpcv-embargo", type=int, default=2)
    p.add_argument("--out-dir", default="runs/robust-cpcv")
    p.add_argument("--limit-configs", type=int, default=0,
                   help="if > 0, only run the first N configs (for debugging)")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0, funding_rate_per_day=0.00010)

    primary = cache.load_range(PRIMARY, "1w", parse_iso(TRAIN_START), parse_iso(TRAIN_END))
    hedge   = cache.load_range(HEDGE,   "1w", parse_iso(TRAIN_START), parse_iso(TRAIN_END))
    hedge = hedge if not hedge.empty else None
    if primary.empty:
        print("No training data; abort.", file=sys.stderr)
        return 2

    n_obs = len(primary)
    paths = generate_paths(
        n_observations=n_obs, n_groups=args.cpcv_n, k_test=args.cpcv_k,
        purge=args.cpcv_purge, embargo=args.cpcv_embargo,
    )
    print(f"Training window: {TRAIN_START} → {TRAIN_END}, {n_obs} bars")
    print(f"CPCV: N={args.cpcv_n}, K={args.cpcv_k}, "
          f"purge={args.cpcv_purge}, embargo={args.cpcv_embargo} → "
          f"{len(paths)} paths")

    cfgs = all_configs()
    if args.limit_configs > 0:
        cfgs = cfgs[: args.limit_configs]
    print(f"Configs: {len(cfgs)}")

    # path_sharpes[config_name] = list of Sharpe per path
    all_path_sharpes: dict[str, list[float]] = {}
    cpcv_results: list[CPCVResult] = []
    weeks_per_year = 52

    for ci, cfg in enumerate(cfgs):
        path_shs: list[float] = []
        path_dds: list[float] = []
        path_cagrs: list[float] = []
        for path in paths:
            log_rets, final_mult, max_dd_path, n_trades = _path_equity_log_returns(
                cfg, primary, hedge, fill_cfg, args.initial_equity, path.test_indices,
            )
            if len(log_rets) < 4:
                continue
            sh = sharpe_annualized(log_rets, periods_per_year=weeks_per_year)
            # CAGR from segment-concatenated multiplier — annualized by the
            # actual number of test bars (not calendar years)
            n_bars = len(log_rets)
            years = n_bars / weeks_per_year
            cagr = (final_mult ** (1.0 / max(years, 1e-9))) - 1.0 if final_mult > 0 else -1.0
            path_shs.append(sh)
            path_dds.append(max_dd_path)
            path_cagrs.append(cagr)

        if not path_shs:
            print(f"  [{cfg.name:30s}] no valid paths", file=sys.stderr)
            continue

        all_path_sharpes[cfg.name] = path_shs
        sh_arr = np.asarray(path_shs)
        dd_arr = np.asarray(path_dds)
        cg_arr = np.asarray(path_cagrs)

        result = CPCVResult(
            config_name=cfg.name, hypothesis=cfg.hypothesis,
            path_sharpes=path_shs, path_drawdowns=path_dds, path_cagrs=path_cagrs,
            n_paths=len(path_shs),
            median_sharpe=float(np.median(sh_arr)),
            p5_sharpe=float(np.percentile(sh_arr, 5)),
            p95_sharpe=float(np.percentile(sh_arr, 95)),
            iqr_sharpe=float(np.percentile(sh_arr, 75) - np.percentile(sh_arr, 25)),
            median_dd=float(np.median(dd_arr)),
            p95_dd=float(np.percentile(dd_arr, 95)),
            median_cagr=float(np.median(cg_arr)),
            p5_cagr=float(np.percentile(cg_arr, 5)),
            fraction_paths_positive=float(np.mean(sh_arr > 0)),
            fraction_paths_dd_under_15=float(np.mean(dd_arr < 0.15)),
        )
        cpcv_results.append(result)
        print(f"  [{ci+1:2d}/{len(cfgs)}] {cfg.name:30s} "
              f"med_Sh={result.median_sharpe:+.2f} "
              f"p5={result.p5_sharpe:+.2f} p95={result.p95_sharpe:+.2f}  "
              f"med_DD={result.median_dd*100:5.1f}% "
              f"med_CAGR={result.median_cagr*100:+6.1f}%  "
              f"frac_DD<15={result.fraction_paths_dd_under_15:.2f}")

    cache.close()

    # PBO across configs (need same number of paths per config)
    n_paths_min = min(len(v) for v in all_path_sharpes.values()) if all_path_sharpes else 0
    if len(all_path_sharpes) >= 2 and n_paths_min >= 2:
        names = list(all_path_sharpes.keys())
        sh_matrix = np.full((n_paths_min, len(names)), np.nan)
        for j, n in enumerate(names):
            sh_matrix[:, j] = all_path_sharpes[n][:n_paths_min]
        pbo = compute_pbo(sh_matrix)
    else:
        pbo = float("nan")

    # ------ Composite ranking under TRUE CPCV ------
    # Score: median_Sharpe × p5_Sharpe(positive-only) × frac_DD<15 × med_CAGR(positive-only)
    # All multiplicative; honest robustness emphasis.
    print("\n" + "=" * 100)
    print("CPCV RANKING (true segment-reset path simulation)")
    print("=" * 100)
    print(f"{'rank':4s} {'config':30s} {'med_Sh':>7s} {'p5_Sh':>7s} {'IQR':>5s} "
          f"{'med_CAGR':>9s} {'med_DD':>7s} {'p<DD15':>7s} {'p>0':>5s} {'comp':>7s}")

    def score(r: CPCVResult) -> float:
        # Reward: high median Sharpe, controlled p5 Sharpe (downside protection),
        # high fraction of paths with DD < 15%, positive median CAGR.
        # Each factor is a continuous, multiplicative term — no all-or-nothing
        # zeros even when bear paths produce deeply negative p5.
        sh_factor = max(0.0, r.median_sharpe)                 # median Sharpe
        p5_factor = max(0.05, (r.p5_sharpe + 1.5) / 1.5)      # softened p5
        dd_factor = max(0.05, r.fraction_paths_dd_under_15)
        cagr_factor = max(0.0, r.median_cagr)
        return sh_factor * p5_factor * dd_factor * cagr_factor

    cpcv_results.sort(key=score, reverse=True)
    rows = []
    for rank, r in enumerate(cpcv_results, start=1):
        s = score(r)
        rows.append({
            "rank": rank, "config": r.config_name, "hypothesis": r.hypothesis,
            "median_sharpe": r.median_sharpe, "p5_sharpe": r.p5_sharpe,
            "p95_sharpe": r.p95_sharpe, "iqr_sharpe": r.iqr_sharpe,
            "median_dd": r.median_dd, "p95_dd": r.p95_dd,
            "median_cagr": r.median_cagr, "p5_cagr": r.p5_cagr,
            "fraction_paths_positive": r.fraction_paths_positive,
            "fraction_paths_dd_under_15": r.fraction_paths_dd_under_15,
            "n_paths": r.n_paths,
            "composite": s,
        })
        print(f"{rank:>4d} {r.config_name:30s} {r.median_sharpe:+7.2f} {r.p5_sharpe:+7.2f} "
              f"{r.iqr_sharpe:5.2f} {r.median_cagr*100:+8.1f}% {r.median_dd*100:6.1f}% "
              f"{r.fraction_paths_dd_under_15:6.2f}  {r.fraction_paths_positive:5.2f}  "
              f"{s:+.3f}")

    # Save CSV + JSON
    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        with (out_dir / "cpcv_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    summary = {
        "training_window": {"start": TRAIN_START, "end": TRAIN_END},
        "cpcv": {
            "n_groups": args.cpcv_n, "k_test": args.cpcv_k,
            "purge": args.cpcv_purge, "embargo": args.cpcv_embargo,
            "n_paths": len(paths), "n_obs": n_obs,
        },
        "n_configs": len(cpcv_results),
        "pbo_across_configs": pbo,
        "ranking": rows,
        "best_config": rows[0]["config"] if rows else None,
        "all_path_sharpes": all_path_sharpes,
    }
    (out_dir / "cpcv_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote → {out_dir}/cpcv_summary.csv  and  cpcv_summary.json")
    print(f"PBO across {len(rows)} configs: {pbo:.3f}" if not np.isnan(pbo) else "PBO: nan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
