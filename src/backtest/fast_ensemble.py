"""Fast ensemble backtest engine.

Walks the timeline, at each decision point:
    1. Builds a per-bar bars dict for each voter's input
    2. Collects votes from all configured voters (deterministic stubs OR
       cached real-LLM decisions from VoteCache)
    3. Aggregates via the configured aggregator (majority / weighted / veto)
    4. Executes the aggregated decision with the same fill model as
       fast_baseline (next-bar open, slippage, fees, funding, stops)

For deterministic-stub voters, the whole simulation is a few thousand Python
ops per voter — fast enough to sweep ensembles in seconds.

For real-LLM voters, VoteCache supplies cached votes; if a vote is missing
the engine raises so the dispatcher can pre-populate. Same caching
discipline as the existing decision_cache: snapshot_hash keys, immutable
votes, lossless persistence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.ensemble_decision import (
    AggregateConfig,
    Vote,
    aggregate_majority,
)
from src.backtest.fill_model import FillConfig
from src.backtest.funding import compute_funding_cost
from src.backtest.stub_voters import VoterSpec, collect_stub_votes


@dataclass
class EnsembleTrade:
    symbol: str
    side: str
    size: float
    entry_price: float
    entry_idx: int
    entry_ts_ms: int
    entry_fee: float
    exit_price: float
    exit_idx: int
    exit_ts_ms: int
    exit_fee: float
    exit_reason: str
    pnl: float


@dataclass
class EnsembleResult:
    initial_equity: float
    final_equity: float
    equity_curve: list[tuple[int, float]]
    trades: list[EnsembleTrade]
    n_decisions_with_consensus: int
    n_decisions_total: int


def _snapshot_hash_from_bars(
    primary_symbol: str,
    bars_window: list[dict],
    positions_state: tuple,
    equity_usd: float,
) -> str:
    """Deterministic hash for a (bars, positions, equity) state.

    The real snapshot_builder uses richer content; this shorter version is
    enough for cache key purposes when voters are stub functions over
    closes only.
    """
    payload = {
        "primary": primary_symbol,
        "bars": bars_window,
        "positions": list(positions_state),
        "equity": round(float(equity_usd), 6),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_ensemble_fast(
    primary_bars: pd.DataFrame,
    hedge_bars: pd.DataFrame | None,
    voter_specs: list[VoterSpec],
    fill_config: FillConfig,
    initial_equity: float,
    aggregate_fn: Callable[[list[Vote], AggregateConfig | None], dict] = aggregate_majority,
    aggregate_config: AggregateConfig | None = None,
    snapshot_lookback_bars: int = 30,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> EnsembleResult:
    """Walk the timeline and simulate the ensemble strategy."""
    if end_idx is None:
        end_idx = len(primary_bars) - 1
    end_idx = min(end_idx, len(primary_bars) - 1)
    if aggregate_config is None:
        aggregate_config = AggregateConfig()

    primary_sym = voter_specs[0].config.primary_symbol if voter_specs else "BTC/USDT:USDT"
    hedge_sym = voter_specs[0].config.hedge_symbol if voter_specs else "PAXG/USDT"

    closes = primary_bars["close"].to_numpy(dtype=float)
    opens = primary_bars["open"].to_numpy(dtype=float)
    highs = primary_bars["high"].to_numpy(dtype=float)
    lows = primary_bars["low"].to_numpy(dtype=float)
    ts_ms = primary_bars["ts_ms"].to_numpy(dtype=np.int64)

    have_hedge = hedge_bars is not None and not hedge_bars.empty
    if have_hedge:
        hedge_closes = hedge_bars["close"].to_numpy(dtype=float)
        hedge_opens = hedge_bars["open"].to_numpy(dtype=float)
        hedge_ts = hedge_bars["ts_ms"].to_numpy(dtype=np.int64)

    fee_factor = fill_config.taker_fee_bps / 10_000.0
    slip = fill_config.slippage_bps / 10_000.0

    # Position state (single-position)
    pos_symbol: str | None = None
    pos_side: str | None = None
    pos_size: float = 0.0
    pos_entry_price: float = 0.0
    pos_entry_idx: int = -1
    pos_entry_ts: int = 0
    pos_entry_fee: float = 0.0
    pos_stop: float | None = None

    equity = initial_equity
    equity_curve: list[tuple[int, float]] = [(int(ts_ms[start_idx]), equity)]
    trades: list[EnsembleTrade] = []
    n_consensus = 0
    n_decisions = 0

    def _close(exit_price: float, exit_idx: int, exit_reason: str) -> None:
        nonlocal equity, pos_symbol, pos_side, pos_size, pos_entry_price
        nonlocal pos_entry_idx, pos_entry_ts, pos_entry_fee, pos_stop
        if pos_symbol is None:
            return
        exit_fee = exit_price * pos_size * fee_factor
        if pos_side == "long":
            gross = (exit_price - pos_entry_price) * pos_size
        else:
            gross = (pos_entry_price - exit_price) * pos_size
        if pos_symbol == primary_sym or not have_hedge:
            exit_ts = int(ts_ms[exit_idx])
        else:
            exit_ts = int(hedge_ts[exit_idx])
        days_held = max(0.0, (exit_ts - pos_entry_ts) / 86_400_000.0)
        notional = pos_entry_price * pos_size
        funding = compute_funding_cost(
            side=pos_side, notional=notional,
            days_held=days_held, funding_rate_per_day=fill_config.funding_rate_per_day,
        )
        pnl = gross - pos_entry_fee - exit_fee - funding
        equity += pnl
        trades.append(EnsembleTrade(
            symbol=pos_symbol, side=pos_side, size=pos_size,
            entry_price=pos_entry_price, entry_idx=pos_entry_idx,
            entry_ts_ms=pos_entry_ts, entry_fee=pos_entry_fee,
            exit_price=exit_price, exit_idx=exit_idx, exit_ts_ms=exit_ts,
            exit_fee=exit_fee, exit_reason=exit_reason, pnl=pnl,
        ))
        pos_symbol = pos_side = None
        pos_size = pos_entry_price = pos_entry_fee = 0.0
        pos_stop = None

    def _open(symbol: str, side: str, idx: int, raw_price: float, size: float, stop: float | None) -> None:
        nonlocal equity, pos_symbol, pos_side, pos_size, pos_entry_price
        nonlocal pos_entry_idx, pos_entry_ts, pos_entry_fee, pos_stop
        if size <= 0 or not math.isfinite(size):
            return
        entry = raw_price * (1 + slip if side == "long" else 1 - slip)
        if stop is not None and not math.isnan(stop):
            if side == "long" and stop >= entry:
                return
            if side == "short" and stop <= entry:
                return
        fee = entry * size * fee_factor
        equity -= fee
        ts_at = int(ts_ms[idx]) if symbol == primary_sym or not have_hedge else int(hedge_ts[idx])
        pos_symbol = symbol
        pos_side = side
        pos_size = size
        pos_entry_price = entry
        pos_entry_idx = idx
        pos_entry_ts = ts_at
        pos_entry_fee = fee
        pos_stop = stop

    # Walk
    for i in range(start_idx, end_idx):
        # Settle stops on i+1
        if pos_symbol is not None and pos_stop is not None and i + 1 <= end_idx:
            if pos_symbol == primary_sym:
                bar_h = highs[i + 1]
                bar_l = lows[i + 1]
            elif have_hedge and i + 1 < len(hedge_closes):
                bar_h = hedge_closes[i + 1]  # use close as proxy if no high/low
                bar_l = bar_h
            else:
                bar_h = bar_l = pos_entry_price
            stop_hit = False
            if pos_side == "long" and bar_l <= pos_stop:
                stop_hit = True
            elif pos_side == "short" and bar_h >= pos_stop:
                stop_hit = True
            if stop_hit:
                _close(float(pos_stop), i + 1, "stop_loss")

        # Build symbol_bars for voters: closes for primary up to i (last lookback)
        voter_lookback = max(snapshot_lookback_bars, 30)
        win_start = max(0, i - voter_lookback + 1)
        prim_bars_dict = [
            {"ts_ms": int(ts_ms[k]), "open": float(opens[k]),
             "high": float(highs[k]), "low": float(lows[k]),
             "close": float(closes[k]), "volume": 1.0}
            for k in range(win_start, i + 1)
        ]
        symbol_bars = {primary_sym: prim_bars_dict}
        if have_hedge:
            hwin_start = max(0, i - voter_lookback + 1)
            hedge_dict = [
                {"ts_ms": int(hedge_ts[k]), "open": float(hedge_opens[k]),
                 "high": float(hedge_opens[k]), "low": float(hedge_opens[k]),
                 "close": float(hedge_closes[k]), "volume": 1.0}
                for k in range(hwin_start, min(i + 1, len(hedge_closes)))
            ]
            symbol_bars[hedge_sym] = hedge_dict

        # Build positions list
        positions_dicts = []
        if pos_symbol is not None:
            positions_dicts.append({
                "symbol": pos_symbol, "side": pos_side, "size": pos_size,
                "entry_price": pos_entry_price, "entry_ts_ms": pos_entry_ts,
            })

        # Collect votes from all stub voters
        votes = collect_stub_votes(
            voter_specs, symbol_bars, positions_dicts, equity,
        )
        decision = aggregate_fn(votes, aggregate_config)
        n_decisions += 1
        if decision["action_type"] != "hold":
            n_consensus += 1

        next_idx = i + 1
        if next_idx > end_idx:
            equity_curve.append((int(ts_ms[i]), equity))
            continue

        atype = decision["action_type"]
        sym = decision.get("symbol")

        # Apply aggregated action
        if atype == "close" and pos_symbol is not None:
            if pos_symbol == primary_sym:
                exit_p = opens[next_idx] * (1 - slip if pos_side == "long" else 1 + slip)
            elif have_hedge:
                exit_p = hedge_opens[next_idx] * (1 - slip if pos_side == "long" else 1 + slip)
            else:
                exit_p = closes[next_idx]
            _close(exit_p, next_idx, "manual")
        elif atype == "open_long":
            if pos_symbol is not None:
                # Close any existing position before flipping
                if pos_symbol == primary_sym:
                    exit_p = opens[next_idx] * (1 - slip if pos_side == "long" else 1 + slip)
                elif have_hedge:
                    exit_p = hedge_opens[next_idx] * (1 - slip if pos_side == "long" else 1 + slip)
                else:
                    exit_p = closes[next_idx]
                _close(exit_p, next_idx, "manual")
            target_sym = sym or primary_sym
            if target_sym == primary_sym:
                _open(primary_sym, "long", next_idx, opens[next_idx],
                      decision["size"], decision.get("stop_loss"))
            elif have_hedge and target_sym == hedge_sym:
                _open(hedge_sym, "long", next_idx, hedge_opens[next_idx],
                      decision["size"], decision.get("stop_loss"))
        elif atype == "open_short":
            if pos_symbol is not None:
                if pos_symbol == primary_sym:
                    exit_p = opens[next_idx] * (1 - slip if pos_side == "long" else 1 + slip)
                elif have_hedge:
                    exit_p = hedge_opens[next_idx] * (1 - slip if pos_side == "long" else 1 + slip)
                else:
                    exit_p = closes[next_idx]
                _close(exit_p, next_idx, "manual")
            _open(primary_sym, "short", next_idx, opens[next_idx],
                  decision["size"], decision.get("stop_loss"))
        # else: hold — no action

        equity_curve.append((int(ts_ms[i]), equity))

    # Force-close at end
    if pos_symbol is not None:
        if pos_symbol == primary_sym:
            exit_p = opens[end_idx] * (1 - slip if pos_side == "long" else 1 + slip)
        elif have_hedge:
            exit_p = hedge_opens[end_idx] * (1 - slip if pos_side == "long" else 1 + slip)
        else:
            exit_p = closes[end_idx]
        _close(exit_p, end_idx, "manual")

    equity_curve.append((int(ts_ms[end_idx]), equity))

    return EnsembleResult(
        initial_equity=initial_equity,
        final_equity=equity,
        equity_curve=equity_curve,
        trades=trades,
        n_decisions_with_consensus=n_consensus,
        n_decisions_total=n_decisions,
    )
