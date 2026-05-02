#!/usr/bin/env python3
"""Run a weekly ensemble backtest on cached Binance perp data and compare
to the single-config champion (CAGR 44% / DD 14.2% / Sharpe 0.94).

Each voter is a deterministic baseline-strategy variant (see stub_voters.py).
The aggregator collapses N votes into one decision; execution is identical
to fast_baseline (next-bar open + slippage + fees + funding + stops).

Stub voters are the deterministic counterpart of dispatching N LLM personas
in parallel. The ensemble architecture (snapshot → N votes → aggregator)
is identical for both.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.analysis.backtest_stats import (
    calmar_ratio,
    max_drawdown,
    pnl_moments,
    probabilistic_sharpe_ratio,
    sharpe_annualized,
    sortino_ratio,
)
from src.analysis.bootstrap import block_bootstrap_sharpe_ci
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
    champion_tilted_voter_specs,
    default_weekly_voter_specs,
)


def parse_iso(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


_AGGREGATORS = {
    "majority": aggregate_majority,
    "weighted": aggregate_weighted,
    "veto": aggregate_veto,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-db", default="/home/erfan/.cache/llm-auto-trader/ohlcv.db")
    p.add_argument("--primary", default="BTC/USDT:USDT")
    p.add_argument("--hedge", default="PAXG/USDT")
    p.add_argument("--timeframe", default="1w")
    p.add_argument("--start", default="2021-01-04")
    p.add_argument("--end", default="2026-04-27")
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--taker-fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--funding-rate-per-day", type=float, default=0.00010)
    p.add_argument("--aggregator", choices=list(_AGGREGATORS), default="majority")
    p.add_argument("--min-agreement", type=int, default=0,
                   help="if >0, fall back to hold when fewer than N voters agree")
    p.add_argument("--crisis-vetos-longs", action="store_true",
                   help="(veto aggregator only) crisis regime from any voter blocks longs")
    p.add_argument("--voter-set", choices=["default", "champion-tilted"], default="default",
                   help="default: 5 baseline-strategy variants; "
                        "champion-tilted: 5 momentum-lookback variants over champion's vol-targeted sizing")
    p.add_argument("--run-id", default="ensemble-weekly-v1")
    args = p.parse_args(argv)

    start_ms = parse_iso(args.start)
    end_ms = parse_iso(args.end)

    cache = OHLCVCache(args.cache_db)
    primary = cache.load_range(args.primary, args.timeframe, start_ms, end_ms)
    hedge = cache.load_range(args.hedge, args.timeframe, start_ms, end_ms)
    cache.close()

    if primary.empty:
        print(f"No primary data for {args.primary} {args.timeframe} {args.start}..{args.end}", file=sys.stderr)
        return 1
    if hedge.empty:
        print(f"WARNING: no hedge data for {args.hedge}; running primary-only", file=sys.stderr)
        hedge = None

    if args.voter_set == "champion-tilted":
        voter_specs = champion_tilted_voter_specs(
            primary_symbol=args.primary, hedge_symbol=args.hedge,
        )
    else:
        voter_specs = default_weekly_voter_specs(
            primary_symbol=args.primary, hedge_symbol=args.hedge,
        )
    fill_cfg = FillConfig(
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        funding_rate_per_day=args.funding_rate_per_day,
    )
    agg_cfg = AggregateConfig(
        crisis_vetos_longs=args.crisis_vetos_longs,
        min_agreement_count=args.min_agreement,
    )
    aggregate_fn = _AGGREGATORS[args.aggregator]

    print(f"Running ensemble: {len(voter_specs)} voters, aggregator={args.aggregator}, "
          f"min_agreement={args.min_agreement}, "
          f"period={args.start}..{args.end}, primary={args.primary}, hedge={args.hedge}")
    print(f"Voters: {[v.voter_id for v in voter_specs]}")

    res = run_ensemble_fast(
        primary_bars=primary,
        hedge_bars=hedge,
        voter_specs=voter_specs,
        fill_config=fill_cfg,
        initial_equity=args.initial_equity,
        aggregate_fn=aggregate_fn,
        aggregate_config=agg_cfg,
    )

    # Stats
    eq = np.array([e for _, e in res.equity_curve], dtype=float)
    eq = np.maximum(eq, 1e-9)
    log_returns = np.diff(np.log(eq))
    years = eq.shape[0] / 52.0  # weekly
    total_return = res.final_equity / args.initial_equity - 1.0
    cagr = (res.final_equity / args.initial_equity) ** (1.0 / max(years, 1e-9)) - 1.0 if res.final_equity > 0 else -1.0
    sharpe = sharpe_annualized(log_returns, periods_per_year=52)
    sortino = sortino_ratio(log_returns, periods_per_year=52)
    dd, _, _ = max_drawdown(eq)
    calmar = calmar_ratio(eq, periods_per_year=52)
    moments = pnl_moments(log_returns)
    psr = probabilistic_sharpe_ratio(
        sr_hat=sharpe / np.sqrt(52),  # PSR expects per-period Sharpe
        n=len(log_returns),
        skew=moments.get("skew", 0.0),
        kurt=moments.get("kurt", 3.0),
        sr_benchmark=0.0,
    )
    try:
        ci_lo, ci_hi = block_bootstrap_sharpe_ci(
            log_returns, periods_per_year=52, n_resamples=2000,
            confidence=0.95, block_size=4, seed=42,
        )
    except Exception:
        ci_lo, ci_hi = float("nan"), float("nan")

    out_dir = Path("runs") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": args.run_id,
        "voters": [v.voter_id for v in voter_specs],
        "aggregator": args.aggregator,
        "min_agreement": args.min_agreement,
        "crisis_vetos_longs": args.crisis_vetos_longs,
        "period": {"start": args.start, "end": args.end, "timeframe": args.timeframe},
        "fill": {
            "taker_fee_bps": args.taker_fee_bps,
            "slippage_bps": args.slippage_bps,
            "funding_rate_per_day": args.funding_rate_per_day,
        },
        "results": {
            "initial_equity": args.initial_equity,
            "final_equity": res.final_equity,
            "total_return_pct": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": dd,
            "calmar": calmar,
            "psr": psr,
            "skew": moments.get("skew"),
            "kurtosis": moments.get("kurtosis"),
            "bootstrap_ci_lo_95": ci_lo,
            "bootstrap_ci_hi_95": ci_hi,
            "n_trades": len(res.trades),
            "n_decisions_total": res.n_decisions_total,
            "n_decisions_with_consensus": res.n_decisions_with_consensus,
        },
        "champion_reference": {
            "config": "M=4 H=0.20 S=3.0 vt=0.015 max_mult=1.67 (single-config)",
            "cagr": 0.4395,
            "sharpe": 0.9448,
            "max_drawdown": 0.1419,
            "calmar": 3.071,
            "n_trades": 22,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Equity curve CSV
    with (out_dir / "equity_curve.csv").open("w") as f:
        f.write("ts_ms,equity\n")
        for ts, e in res.equity_curve:
            f.write(f"{ts},{e:.6f}\n")

    # Trades CSV
    with (out_dir / "trades.csv").open("w") as f:
        f.write("symbol,side,size,entry_price,entry_ts_ms,exit_price,exit_ts_ms,exit_reason,pnl\n")
        for t in res.trades:
            f.write(f"{t.symbol},{t.side},{t.size:.6f},{t.entry_price:.4f},"
                    f"{t.entry_ts_ms},{t.exit_price:.4f},{t.exit_ts_ms},"
                    f"{t.exit_reason},{t.pnl:.4f}\n")

    print()
    print("=" * 70)
    print(f"ENSEMBLE RESULT  ({args.aggregator}, {len(voter_specs)} voters)")
    print("=" * 70)
    print(f"  Final equity     : ${res.final_equity:,.2f}  (initial ${args.initial_equity:,.0f})")
    print(f"  Total return     : {total_return*100:+.2f}%")
    print(f"  CAGR             : {cagr*100:+.2f}%")
    print(f"  Sharpe (annual)  : {sharpe:.3f}   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Sortino          : {sortino:.3f}")
    print(f"  Max drawdown     : {dd*100:.2f}%")
    print(f"  Calmar           : {calmar:.3f}")
    print(f"  PSR              : {psr:.6f}")
    print(f"  Trades           : {len(res.trades)}")
    print(f"  Consensus / total: {res.n_decisions_with_consensus} / {res.n_decisions_total}")
    print()
    print("Champion (single-config) for comparison:")
    print("  CAGR 43.96%, Sharpe 0.945, MaxDD 14.19%, Calmar 3.07, 22 trades")
    print()
    print(f"Saved → {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
