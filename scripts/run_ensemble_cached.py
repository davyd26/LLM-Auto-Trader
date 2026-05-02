#!/usr/bin/env python3
"""Simulate the ensemble strategy from a populated VoteCache.

Companion to ``scripts/dispatch_ensemble.py``: that script invokes
``claude -p`` once per (snapshot, voter) and stores votes; this script reads
them back, aggregates them for each snapshot, and walks the same fill model
as ``run_ensemble.py``. No LLM calls during simulation.

The aggregator is configurable. Re-running with a different aggregator over
the same cache costs nothing — that's the point of caching raw votes
instead of aggregated decisions.

Usage:
    python scripts/run_ensemble_cached.py \\
        --run-id ensemble-real-llm-v1 --runs-dir runs \\
        --votes-db runs/.../votes.db \\
        --aggregator majority --min-agreement 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from src.backtest.ensemble_decision import (
    AggregateConfig,
    aggregate_majority,
    aggregate_veto,
    aggregate_weighted,
)
from src.backtest.fill_model import FillConfig
from src.backtest.funding import compute_funding_cost
from src.backtest.vote_cache import VoteCache

_AGGREGATORS = {
    "majority": aggregate_majority,
    "weighted": aggregate_weighted,
    "veto": aggregate_veto,
}


def _hash_snapshot_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--votes-db", required=True)
    p.add_argument("--initial-equity", type=float, default=2000.0)
    p.add_argument("--taker-fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--funding-rate-per-day", type=float, default=0.00010)
    p.add_argument("--aggregator", choices=list(_AGGREGATORS), default="majority")
    p.add_argument("--min-agreement", type=int, default=0)
    p.add_argument("--crisis-vetos-longs", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cache = VoteCache(args.votes_db)
    aggregate_fn = _AGGREGATORS[args.aggregator]
    agg_cfg = AggregateConfig(
        crisis_vetos_longs=args.crisis_vetos_longs,
        min_agreement_count=args.min_agreement,
    )
    fill_cfg = FillConfig(
        taker_fee_bps=args.taker_fee_bps,
        slippage_bps=args.slippage_bps,
        funding_rate_per_day=args.funding_rate_per_day,
    )
    fee_factor = fill_cfg.taker_fee_bps / 10_000.0
    slip = fill_cfg.slippage_bps / 10_000.0

    # Walk: for each observation in order, get aggregated decision, simulate fill on next bar's open.
    obs = sorted(manifest["observations"], key=lambda e: e["week_idx"])
    equity = args.initial_equity
    pos = None  # dict or None
    trades = []
    equity_curve = []
    n_consensus = 0
    n_total = 0

    for i, entry in enumerate(obs):
        snap_text = (run_dir / entry["path"]).read_text()
        snap_hash = _hash_snapshot_text(snap_text)
        votes = cache.get_all(snap_hash)
        if not votes:
            print(f"  [{entry['week_idx']:03d}] no votes cached — treating as hold", file=sys.stderr)
            decision = {"action_type": "hold", "symbol": None, "size": 0.0,
                        "stop_loss": None, "take_profit": None}
        else:
            decision = aggregate_fn(votes, agg_cfg)
        n_total += 1
        if decision["action_type"] != "hold":
            n_consensus += 1

        # Need next-bar open to fill. Read primary symbol from first entry's bars file.
        bars_path = run_dir / entry["path"]
        snap = json.loads(bars_path.read_text())
        # Snapshot stores recent bars under symbol_bars[primary][-1] = current; we
        # don't have the future open here, so we rely on the manifest's
        # next_open_price hint. If absent, fall back to current close.
        next_open = entry.get("next_open_price")
        ts_ms = int(entry.get("ts_ms", 0))
        if next_open is None:
            primary_sym = snap.get("primary_symbol") or manifest.get("primary_symbol")
            primary_bars = snap.get("symbol_bars", {}).get(primary_sym, [])
            if primary_bars:
                next_open = primary_bars[-1].get("close")
        if next_open is None:
            print(f"  [{entry['week_idx']:03d}] no next_open_price; skipping fill", file=sys.stderr)
            equity_curve.append((ts_ms, equity))
            continue

        # Apply decision (matches fast_ensemble's fill model)
        atype = decision["action_type"]
        sym = decision.get("symbol")
        if atype == "close" and pos is not None:
            exit_p = next_open * (1 - slip if pos["side"] == "long" else 1 + slip)
            exit_fee = exit_p * pos["size"] * fee_factor
            gross = (exit_p - pos["entry_price"]) * pos["size"] if pos["side"] == "long" \
                else (pos["entry_price"] - exit_p) * pos["size"]
            days_held = max(0.0, (ts_ms - pos["entry_ts_ms"]) / 86_400_000.0)
            funding = compute_funding_cost(
                side=pos["side"], notional=pos["entry_price"] * pos["size"],
                days_held=days_held, funding_rate_per_day=fill_cfg.funding_rate_per_day,
            )
            pnl = gross - pos["entry_fee"] - exit_fee - funding
            equity += pnl
            trades.append({**pos, "exit_price": exit_p, "exit_ts_ms": ts_ms,
                           "exit_reason": "manual", "pnl": pnl})
            pos = None
        elif atype in ("open_long", "open_short"):
            if pos is not None:
                exit_p = next_open * (1 - slip if pos["side"] == "long" else 1 + slip)
                exit_fee = exit_p * pos["size"] * fee_factor
                gross = (exit_p - pos["entry_price"]) * pos["size"] if pos["side"] == "long" \
                    else (pos["entry_price"] - exit_p) * pos["size"]
                days_held = max(0.0, (ts_ms - pos["entry_ts_ms"]) / 86_400_000.0)
                funding = compute_funding_cost(
                    side=pos["side"], notional=pos["entry_price"] * pos["size"],
                    days_held=days_held, funding_rate_per_day=fill_cfg.funding_rate_per_day,
                )
                pnl = gross - pos["entry_fee"] - exit_fee - funding
                equity += pnl
                trades.append({**pos, "exit_price": exit_p, "exit_ts_ms": ts_ms,
                               "exit_reason": "flip", "pnl": pnl})
                pos = None
            side = "long" if atype == "open_long" else "short"
            entry_price = next_open * (1 + slip if side == "long" else 1 - slip)
            size = float(decision.get("size") or 0.0)
            if size > 0:
                fee = entry_price * size * fee_factor
                equity -= fee
                pos = {"symbol": sym, "side": side, "size": size,
                       "entry_price": entry_price, "entry_ts_ms": ts_ms, "entry_fee": fee}

        equity_curve.append((ts_ms, equity))

    cache.close()

    eq = np.array([e for _, e in equity_curve], dtype=float)
    eq = np.maximum(eq, 1e-9)
    log_returns = np.diff(np.log(eq))
    years = eq.shape[0] / 52.0
    total_return = equity / args.initial_equity - 1.0
    cagr = (equity / args.initial_equity) ** (1.0 / max(years, 1e-9)) - 1.0 if equity > 0 else -1.0
    sharpe = sharpe_annualized(log_returns, periods_per_year=52)
    sortino = sortino_ratio(log_returns, periods_per_year=52)
    dd, _, _ = max_drawdown(eq)
    calmar = calmar_ratio(eq, periods_per_year=52)
    moments = pnl_moments(log_returns)
    psr = probabilistic_sharpe_ratio(
        sr_hat=sharpe / np.sqrt(52), n=len(log_returns),
        skew=moments.get("skew", 0.0), kurt=moments.get("kurt", 3.0),
        sr_benchmark=0.0,
    )
    try:
        ci_lo, ci_hi = block_bootstrap_sharpe_ci(
            log_returns, periods_per_year=52, n_resamples=2000,
            confidence=0.95, block_size=4, seed=42,
        )
    except Exception:
        ci_lo, ci_hi = float("nan"), float("nan")

    print(f"\nEnsemble (cached votes, {args.aggregator}):")
    print(f"  Final equity     : ${equity:,.2f}")
    print(f"  Total return     : {total_return*100:+.2f}%")
    print(f"  CAGR             : {cagr*100:+.2f}%")
    print(f"  Sharpe (annual)  : {sharpe:.3f}   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Sortino          : {sortino:.3f}")
    print(f"  Max drawdown     : {dd*100:.2f}%")
    print(f"  Calmar           : {calmar:.3f}")
    print(f"  PSR              : {psr:.6f}")
    print(f"  Trades           : {len(trades)}")
    print(f"  Consensus / total: {n_consensus} / {n_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
