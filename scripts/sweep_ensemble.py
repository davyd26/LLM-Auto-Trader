#!/usr/bin/env python3
"""Sweep ensemble configurations on cached real Binance weekly data.

Tries the cross-product of:
    - voter sets: default | champion-tilted | mixed-aggressive
    - aggregators: majority | weighted | veto
    - min_agreement thresholds: 0, 2, 3, 4
    - crisis_vetos_longs: True | False (only meaningful for veto / when relevant)

Walks each config across the full window AND a list of regime sub-windows
(bear-2022, recovery-2023, bull-2024, full). Reports CAGR, Sharpe, MaxDD,
Calmar per (config, window). Saves the full grid as CSV for later analysis.
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
    sortino_ratio,
)
from src.backtest.baseline_strategy import BaselineConfig
from src.backtest.data_cache import OHLCVCache
from src.backtest.ensemble_decision import (
    AggregateConfig,
    aggregate_majority,
    aggregate_veto,
    aggregate_weighted,
)
from src.backtest.fast_ensemble import run_ensemble_fast
from src.backtest.fill_model import FillConfig
from src.backtest.stub_voters import (
    VoterSpec,
    champion_plus_veto_voter_specs,
    champion_tilted_voter_specs,
    default_weekly_voter_specs,
)


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def mixed_aggressive_voter_specs(
    primary_symbol: str = "BTC/USDT:USDT",
    hedge_symbol: str = "PAXG/USDT",
) -> list[VoterSpec]:
    """Voter set with deliberately heterogeneous SIZING + lookbacks.

    Each voter has a different (size_pct, momentum_lookback, hedge_threshold,
    vol_target) combination so disagreement is structural, not just on lookback.
    Mimics the real-LLM persona panel where sizing reflects each persona's
    risk appetite (the trend follower bets bigger; risk-off bets smaller).
    """
    base = dict(primary_symbol=primary_symbol, hedge_symbol=hedge_symbol)
    return [
        # Aggressive trend
        VoterSpec("aggressive_trend", BaselineConfig(
            momentum_lookback_bars=4, drawdown_hedge_threshold=0.15,
            long_size_usd_pct=2.5, vol_target_pct=0.02, vol_lookback_bars=12,
            vol_size_max_mult=2.0, **base,
        )),
        # Conservative trend with strong vol-target
        VoterSpec("conservative_trend", BaselineConfig(
            momentum_lookback_bars=8, drawdown_hedge_threshold=0.10,
            long_size_usd_pct=1.5, vol_target_pct=0.015, vol_lookback_bars=12,
            vol_size_max_mult=1.5, **base,
        )),
        # Slow trend, no vol-targeting
        VoterSpec("slow_trend", BaselineConfig(
            momentum_lookback_bars=16, drawdown_hedge_threshold=0.20,
            long_size_usd_pct=1.0, **base,
        )),
        # Risk-off heavy hedger
        VoterSpec("risk_off", BaselineConfig(
            momentum_lookback_bars=6, drawdown_hedge_threshold=0.08,
            long_size_usd_pct=0.5, vol_target_pct=0.012, vol_lookback_bars=8,
            vol_size_max_mult=1.0, **base,
        )),
        # Vol-targeted technical
        VoterSpec("vol_targeted", BaselineConfig(
            momentum_lookback_bars=4, drawdown_hedge_threshold=0.20,
            long_size_usd_pct=2.0, vol_target_pct=0.018, vol_lookback_bars=12,
            vol_size_max_mult=1.67, **base,
        )),
    ]


_VOTER_SETS = {
    "default": default_weekly_voter_specs,
    "champion-tilted": champion_tilted_voter_specs,
    "mixed-aggressive": mixed_aggressive_voter_specs,
    "champion-plus-veto": champion_plus_veto_voter_specs,
}

_AGGREGATORS = {
    "majority": aggregate_majority,
    "weighted": aggregate_weighted,
    "veto": aggregate_veto,
}

_WINDOWS = [
    ("full", "2021-01-04", "2026-04-27"),
    ("blowoff-2021", "2021-01-04", "2022-01-03"),
    ("bear-2022", "2022-01-03", "2023-01-02"),
    ("recovery-2023", "2023-01-02", "2024-01-01"),
    ("bull-2024", "2024-01-01", "2024-12-30"),
    ("late-2025", "2025-01-06", "2026-04-27"),
]


def _stats(initial_eq: float, equity_curve: list, periods_per_year: int = 52) -> dict:
    eq = np.array([e for _, e in equity_curve], dtype=float)
    eq = np.maximum(eq, 1e-9)
    final = float(eq[-1])
    log_returns = np.diff(np.log(eq))
    years = eq.shape[0] / periods_per_year
    cagr = (final / initial_eq) ** (1.0 / max(years, 1e-9)) - 1.0 if final > 0 else -1.0
    sharpe = sharpe_annualized(log_returns, periods_per_year=periods_per_year)
    sortino = sortino_ratio(log_returns, periods_per_year=periods_per_year)
    dd, _, _ = max_drawdown(eq)
    calmar = calmar_ratio(eq, periods_per_year=periods_per_year)
    return {
        "final_equity": final,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": dd,
        "calmar": calmar,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--primary", default="BTC/USDT:USDT")
    p.add_argument("--hedge", default="PAXG/USDT")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--out-csv", default="runs/ensemble-sweep/grid.csv")
    args = p.parse_args(argv)

    cache = OHLCVCache(args.cache_db)
    fill_cfg = FillConfig(taker_fee_bps=4.0, slippage_bps=5.0, funding_rate_per_day=0.00010)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "voter_set", "aggregator", "min_agreement", "crisis_vetos_longs",
        "window", "start", "end",
        "final_equity", "cagr", "sharpe", "sortino", "max_drawdown", "calmar",
        "n_trades", "n_consensus", "n_decisions",
    ]

    n_configs = 0
    rows = []
    for vset_name, vset_fn in _VOTER_SETS.items():
        voters = vset_fn(primary_symbol=args.primary, hedge_symbol=args.hedge)
        for agg_name, agg_fn in _AGGREGATORS.items():
            for min_agree in (0, 2, 3, 4):
                for veto_longs in (False, True):
                    # Veto-longs only meaningful with veto aggregator (otherwise duplicates)
                    if veto_longs and agg_name != "veto":
                        continue
                    if not veto_longs and agg_name == "veto":
                        # ensure veto runs at least once with crisis_vetos_longs=True
                        # but also skip the "veto + no veto" combo (it's same as majority)
                        continue
                    agg_cfg = AggregateConfig(
                        crisis_vetos_longs=veto_longs,
                        min_agreement_count=min_agree,
                    )
                    for win_name, win_start, win_end in _WINDOWS:
                        primary = cache.load_range(args.primary, "1w",
                                                   parse_iso(win_start), parse_iso(win_end))
                        hedge = cache.load_range(args.hedge, "1w",
                                                 parse_iso(win_start), parse_iso(win_end))
                        if primary.empty:
                            continue
                        hedge_use = hedge if not hedge.empty else None
                        try:
                            res = run_ensemble_fast(
                                primary_bars=primary,
                                hedge_bars=hedge_use,
                                voter_specs=voters,
                                fill_config=fill_cfg,
                                initial_equity=args.initial_equity,
                                aggregate_fn=agg_fn,
                                aggregate_config=agg_cfg,
                            )
                            stats = _stats(args.initial_equity, res.equity_curve)
                            row = {
                                "voter_set": vset_name,
                                "aggregator": agg_name,
                                "min_agreement": min_agree,
                                "crisis_vetos_longs": veto_longs,
                                "window": win_name, "start": win_start, "end": win_end,
                                **stats,
                                "n_trades": len(res.trades),
                                "n_consensus": res.n_decisions_with_consensus,
                                "n_decisions": res.n_decisions_total,
                            }
                            rows.append(row)
                            n_configs += 1
                            print(f"  {vset_name:20s} {agg_name:9s} ma={min_agree} veto={int(veto_longs)} "
                                  f"{win_name:14s} CAGR={stats['cagr']*100:+6.1f}% "
                                  f"Sharpe={stats['sharpe']:+.2f} DD={stats['max_drawdown']*100:5.1f}%")
                        except (ValueError, IndexError) as e:
                            print(f"  ERR ({vset_name}/{agg_name}/{win_name}): {e}", file=sys.stderr)

    cache.close()

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows ({n_configs} configs × windows) → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
