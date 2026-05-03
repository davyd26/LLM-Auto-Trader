"""LLM-driven multi-symbol portfolio rebalancer.

The strategy = the LLM. At each weekly snapshot we:
    1. Build a rich JSON context with every quantitative signal we have:
         - per-symbol: GARCH(1,1), MA panel, RSI, MACD, volume profile,
           recent N-bar OHLCV, return z-scores
         - portfolio-level: equity, cash, current weights, MTM, recent
           realized return, win rate, profit factor, running max-DD
         - reference weight schemes (the LLM may follow, ignore, or blend):
             * equal-weight
             * inverse-vol risk-parity
             * Markowitz min-variance long-only
    2. Dispatch ``claude -p`` (or another decision source) with that context.
       The LLM responds with a target-weight vector for the universe.
    3. Validate weights (sum ≈ 1, long-only, BTC+GLD+USD anchor preserved).
    4. Rebalance to those weights at the next bar's open with fees + slippage.

CAGR figures are USD-denominated throughout. Every weight applies to USD
notional; cash leg has zero return; equity is total USD value of all legs.

Decision sources are pluggable. A deterministic "strawman" source is
provided for fast dev / harness validation; the real LLM source dispatches
``claude -p`` and caches results by snapshot hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.fill_model import FillConfig
from src.backtest.quant_overlay import (
    QuantOverlayConfig,
    compute_quant_signals,
    portfolio_optimization_context,
    portfolio_state_to_llm_context,
    signals_to_llm_context,
)

USD_CASH = "USD_CASH"


@dataclass
class PortfolioTrade:
    """A single rebalance leg from one symbol-weight delta."""

    bar_idx: int
    ts_ms: int
    symbol: str
    side: str  # "buy" | "sell"
    delta_units: float
    fill_price: float
    fee_paid: float


@dataclass
class PortfolioResult:
    """Final result of an LLM-driven backtest run."""

    initial_equity_usd: float
    final_equity_usd: float
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    weights_history: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    trades: list[PortfolioTrade] = field(default_factory=list)
    n_decisions: int = 0
    n_cache_hits: int = 0


def snapshot_hash(payload: dict) -> str:
    """Deterministic SHA-256 of a snapshot dict for caching."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_rich_snapshot(
    *,
    ts_ms: int,
    universe: list[str],
    bars_by_symbol: dict[str, pd.DataFrame],
    current_weights: dict[str, float],
    current_holdings_units: dict[str, float],
    current_prices: dict[str, float],
    equity_usd: float,
    initial_equity_usd: float,
    quant_signals_by_symbol: dict[str, object],
    recent_pnls: list[float],
    equity_curve_recent: list[float],
    bar_idx_by_symbol: dict[str, int],
    rebalance_lookback_for_optimization: int = 26,
    bars_to_include_per_symbol: int = 12,
) -> dict:
    """Build the JSON payload sent to the LLM at one snapshot.

    Includes:
      • Universe tickers and asset-class tags
      • Recent N-bar OHLCV per symbol (12 bars by default)
      • Quant-signal panel per symbol (GARCH, MAs, RSI, MACD, z-scores, volume)
      • Current portfolio state (cash, weights, MTM, recent realized return,
        win rate, profit factor, running max-DD)
      • Reference weight schemes (equal-weight, inv-vol, Markowitz min-var)
        — explicitly described as ADVISORY, the LLM is free to deviate.
    """
    universe_signals = {}
    for sym in universe:
        if sym == USD_CASH:
            continue
        idx = bar_idx_by_symbol.get(sym)
        sig = quant_signals_by_symbol.get(sym)
        if idx is None or sig is None:
            continue
        universe_signals[sym] = signals_to_llm_context(sig, idx)

    # Recent N bars per symbol for raw price view
    recent_bars = {}
    for sym in universe:
        if sym == USD_CASH:
            continue
        bars = bars_by_symbol.get(sym)
        idx = bar_idx_by_symbol.get(sym)
        if bars is None or idx is None:
            continue
        start = max(0, idx - bars_to_include_per_symbol + 1)
        slice_df = bars.iloc[start:idx + 1]
        recent_bars[sym] = [
            {
                "ts_ms": int(row["ts_ms"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0) or 0.0),
            }
            for _, row in slice_df.iterrows()
        ]

    # Reference weight schemes (advisory inputs for the LLM)
    ret_inputs: dict[str, np.ndarray] = {}
    lb = rebalance_lookback_for_optimization
    for sym in universe:
        if sym == USD_CASH:
            continue
        bars = bars_by_symbol.get(sym)
        idx = bar_idx_by_symbol.get(sym)
        if bars is None or idx is None or idx < 4:
            continue
        closes = bars["close"].to_numpy(dtype=float)[max(0, idx - lb):idx + 1]
        closes = np.maximum(closes, 1e-12)
        if len(closes) < 4:
            continue
        ret_inputs[sym] = np.diff(np.log(closes))
    if USD_CASH in universe:
        ret_inputs[USD_CASH] = np.zeros(max(4, lb))
    optim = portfolio_optimization_context(ret_inputs, cash_cap=0.20) if ret_inputs else {}

    # Portfolio state
    positions = []
    for sym, units in current_holdings_units.items():
        if sym == USD_CASH or units <= 0:
            continue
        entry_price = current_prices.get(sym, 0.0)
        positions.append({
            "symbol": sym,
            "side": "long",
            "size": units,
            "entry_price": entry_price,  # not exact entry — it's MTM proxy
            "entry_ts_ms": 0,
        })
    portfolio_state = portfolio_state_to_llm_context(
        equity_usd=equity_usd,
        positions=positions,
        current_prices=current_prices,
        recent_pnls=recent_pnls,
        equity_curve_recent=equity_curve_recent,
    )

    return {
        "schema_version": 1,
        "decision_cadence": "weekly",
        "ts_ms": ts_ms,
        "universe": universe,
        "current_weights": current_weights,
        "initial_equity_usd": initial_equity_usd,
        "portfolio_state": portfolio_state,
        "recent_bars": recent_bars,
        "quant_signals": universe_signals,
        "reference_weight_schemes_advisory": optim,
        "instruction_for_llm": (
            "You are a disciplined multi-asset portfolio manager. Output target "
            "weights for the next week. Constraints: long-only; sum to 1.0; "
            "BTC + GLD + USD anchor must be respected (each > 0). Use the "
            "advisory reference weights as one input among many; they are not "
            "binding."
        ),
    }


def _normalize_weights(
    weights: dict[str, float], universe: list[str], anchor: list[str] | None = None,
) -> dict[str, float]:
    """Validate and normalize a weight dict.

    - Drop unknown symbols
    - Clamp negatives to 0 (long-only)
    - If anchor symbols are missing, inject them at small positive weight
    - Renormalize to sum=1
    """
    out = {s: 0.0 for s in universe}
    for s, w in weights.items():
        if s in out and w is not None and math.isfinite(float(w)):
            out[s] = max(0.0, float(w))
    if anchor:
        for a in anchor:
            if a in out and out[a] <= 0:
                out[a] = 0.05  # tiny default to satisfy the constraint
    total = sum(out.values())
    if total <= 0:
        # Degenerate input — fall back to equal weight
        n = max(1, len(out))
        return {s: 1.0 / n for s in out}
    return {s: v / total for s, v in out.items()}


# Decision source: callable that takes a snapshot dict, returns weights dict.
DecisionSource = Callable[[dict], dict[str, float]]


def run_llm_portfolio(
    universe: list[str],
    bars_by_symbol: dict[str, pd.DataFrame],
    decision_source: DecisionSource,
    fill_config: FillConfig,
    initial_equity_usd: float,
    anchor: list[str] | None = None,
    quant_overlay: QuantOverlayConfig | None = None,
    rebalance_threshold: float = 0.02,
    start_idx: int = 0,
    end_idx: int | None = None,
    snapshot_lookback_bars: int = 30,
) -> PortfolioResult:
    """Walk the timeline, calling ``decision_source`` weekly to get target
    weights, then rebalancing the portfolio with fees + slippage.

    The harness assumes all symbols' bars are time-aligned (same ts_ms grid).
    The first symbol's index drives the loop. USD_CASH is treated as a
    zero-return cash leg (bars not required).

    Args:
        universe: list of symbol tickers (may include USD_CASH).
        bars_by_symbol: dict symbol → DataFrame with ts_ms/open/high/low/close.
        decision_source: callable that turns a snapshot into weights.
        fill_config: fees + slippage.
        initial_equity_usd: starting USD equity.
        anchor: list of symbols that must always have positive weight
            (e.g., ["BTC/USDT:USDT", "GLD", USD_CASH]).
        quant_overlay: signal config; if None, uses defaults.
        rebalance_threshold: rebalance only if any weight differs by ≥ this.
        snapshot_lookback_bars: how many recent bars to compute signals over
            (signals use the full history but the LLM-visible window is this).

    Returns:
        PortfolioResult with full equity curve, weight history, trades.
    """
    if quant_overlay is None:
        quant_overlay = QuantOverlayConfig(use_garch_vol=False)

    # Find the longest aligned timeline
    risk_syms = [s for s in universe if s != USD_CASH]
    if not risk_syms:
        # Pure cash — no work
        return PortfolioResult(
            initial_equity_usd=initial_equity_usd,
            final_equity_usd=initial_equity_usd,
        )

    # Use the FIRST risk symbol as the timing anchor
    anchor_sym = risk_syms[0]
    anchor_df = bars_by_symbol[anchor_sym]
    n = len(anchor_df)
    if n < 8:
        return PortfolioResult(
            initial_equity_usd=initial_equity_usd,
            final_equity_usd=initial_equity_usd,
        )
    if end_idx is None:
        end_idx = n - 1
    end_idx = min(end_idx, n - 1)

    ts_ms_arr = anchor_df["ts_ms"].to_numpy(dtype=np.int64)

    # Pre-compute quant signals once per symbol (frozen GARCH, etc.)
    quant_signals: dict[str, object] = {}
    for sym in risk_syms:
        df = bars_by_symbol.get(sym)
        if df is None or df.empty:
            continue
        closes = df["close"].to_numpy(dtype=float)
        volumes = df.get("volume").to_numpy(dtype=float) if "volume" in df.columns else None
        try:
            quant_signals[sym] = compute_quant_signals(closes, quant_overlay, volumes=volumes)
        except Exception:
            quant_signals[sym] = None

    # Per-symbol close arrays (for MTM)
    close_by_symbol = {
        s: bars_by_symbol[s]["close"].to_numpy(dtype=float)
        for s in risk_syms
    }
    open_by_symbol = {
        s: bars_by_symbol[s]["open"].to_numpy(dtype=float)
        for s in risk_syms
    }
    # Each symbol's bar grid may differ slightly — map anchor index → per-symbol
    # index by ts_ms search. For weekly aligned crypto+stock data this is
    # nearly always identity, but we do the lookup defensively.
    sym_ts = {
        s: bars_by_symbol[s]["ts_ms"].to_numpy(dtype=np.int64)
        for s in risk_syms
    }

    def _idx_for(sym: str, anchor_idx: int) -> int | None:
        ts = int(ts_ms_arr[anchor_idx])
        sym_arr = sym_ts.get(sym)
        if sym_arr is None or len(sym_arr) == 0:
            return None
        # Largest sym index with ts <= ts
        pos = int(np.searchsorted(sym_arr, ts, side="right")) - 1
        if pos < 0 or pos >= len(sym_arr):
            return None
        return pos

    # Portfolio state
    holdings_units: dict[str, float] = {s: 0.0 for s in universe}  # USD_CASH tracked as USD
    holdings_units[USD_CASH] = initial_equity_usd if USD_CASH in universe else 0.0
    cash_usd = initial_equity_usd if USD_CASH not in universe else 0.0

    fee_factor = fill_config.taker_fee_bps / 10_000.0
    slip = fill_config.slippage_bps / 10_000.0

    equity_curve: list[tuple[int, float]] = []
    weights_history: list[tuple[int, dict[str, float]]] = []
    trades: list[PortfolioTrade] = []
    recent_pnls: list[float] = []

    def _equity_now(anchor_idx: int) -> float:
        e = cash_usd + holdings_units.get(USD_CASH, 0.0)
        for sym in risk_syms:
            units = holdings_units.get(sym, 0.0)
            if units == 0:
                continue
            pos_idx = _idx_for(sym, anchor_idx)
            if pos_idx is None:
                continue
            e += units * float(close_by_symbol[sym][pos_idx])
        return e

    def _current_weights(anchor_idx: int, equity: float) -> dict[str, float]:
        if equity <= 0:
            return {s: 0.0 for s in universe}
        weights = {}
        for sym in universe:
            if sym == USD_CASH:
                weights[sym] = (cash_usd + holdings_units.get(USD_CASH, 0.0)) / equity
                continue
            units = holdings_units.get(sym, 0.0)
            pos_idx = _idx_for(sym, anchor_idx)
            if pos_idx is None or units == 0:
                weights[sym] = 0.0
                continue
            weights[sym] = units * float(close_by_symbol[sym][pos_idx]) / equity
        return weights

    n_decisions = 0
    n_cache_hits = 0
    last_target_weights: dict[str, float] | None = None
    initial_ts = int(ts_ms_arr[start_idx])
    equity_curve.append((initial_ts, initial_equity_usd))

    for i in range(start_idx, end_idx):
        eq_now = _equity_now(i)
        cur_w = _current_weights(i, eq_now)

        # Build current prices for portfolio-state context
        cur_prices: dict[str, float] = {}
        for sym in risk_syms:
            pos_idx = _idx_for(sym, i)
            if pos_idx is not None:
                cur_prices[sym] = float(close_by_symbol[sym][pos_idx])

        # Build snapshot
        bar_idx_by_symbol = {s: _idx_for(s, i) for s in risk_syms if _idx_for(s, i) is not None}
        eq_recent = [e for _, e in equity_curve[-26:]]
        snap = build_rich_snapshot(
            ts_ms=int(ts_ms_arr[i]),
            universe=universe,
            bars_by_symbol=bars_by_symbol,
            current_weights=cur_w,
            current_holdings_units=holdings_units,
            current_prices=cur_prices,
            equity_usd=eq_now,
            initial_equity_usd=initial_equity_usd,
            quant_signals_by_symbol=quant_signals,
            recent_pnls=recent_pnls,
            equity_curve_recent=eq_recent,
            bar_idx_by_symbol=bar_idx_by_symbol,
        )

        # Decision: get target weights from the source
        try:
            target_raw = decision_source(snap)
        except Exception:
            target_raw = cur_w  # fail safe: don't rebalance
        target = _normalize_weights(target_raw, universe, anchor)

        # If first decision OR weights differ enough, rebalance at next-bar open
        next_idx = i + 1
        if next_idx > end_idx:
            equity_curve.append((int(ts_ms_arr[i]), eq_now))
            continue

        diff_norm = sum(abs(target.get(s, 0.0) - cur_w.get(s, 0.0)) for s in universe) / 2.0
        do_rebalance = (last_target_weights is None) or (diff_norm >= rebalance_threshold)

        if do_rebalance:
            # Compute current value per symbol at NEXT-bar open prices
            next_prices: dict[str, float] = {}
            for sym in risk_syms:
                pos_idx = _idx_for(sym, next_idx)
                if pos_idx is None:
                    continue
                next_prices[sym] = float(open_by_symbol[sym][pos_idx])
            # Re-mark current portfolio at next-bar open (cash unchanged)
            eq_open = cash_usd + holdings_units.get(USD_CASH, 0.0)
            for sym in risk_syms:
                u = holdings_units.get(sym, 0.0)
                px = next_prices.get(sym, 0.0)
                eq_open += u * px

            # Target USD per symbol
            target_usd: dict[str, float] = {s: target[s] * eq_open for s in universe}

            # Apply trades. Track total available USD (cash_usd + USD_CASH leg)
            # in a SINGLE variable so we don't double-count or lose cash that
            # lives in the variable not present in the universe.
            new_holdings = dict(holdings_units)
            available_cash = cash_usd + holdings_units.get(USD_CASH, 0.0)
            for sym in risk_syms:
                px = next_prices.get(sym, 0.0)
                if px <= 0:
                    continue
                cur_units = new_holdings.get(sym, 0.0)
                cur_value = cur_units * px
                tgt_value = target_usd.get(sym, 0.0)
                diff_value = tgt_value - cur_value
                if abs(diff_value) < 1e-6:
                    continue
                if diff_value > 0:
                    fill_px = px * (1 + slip)
                    units_to_buy = diff_value / fill_px
                    fee = diff_value * fee_factor
                    new_holdings[sym] = cur_units + units_to_buy
                    available_cash -= (units_to_buy * fill_px + fee)
                    trades.append(PortfolioTrade(
                        bar_idx=next_idx, ts_ms=int(ts_ms_arr[next_idx]),
                        symbol=sym, side="buy",
                        delta_units=units_to_buy, fill_price=fill_px, fee_paid=fee,
                    ))
                else:
                    fill_px = px * (1 - slip)
                    units_to_sell = -diff_value / fill_px
                    units_to_sell = min(units_to_sell, cur_units)
                    fee = (units_to_sell * fill_px) * fee_factor
                    new_holdings[sym] = cur_units - units_to_sell
                    available_cash += (units_to_sell * fill_px - fee)
                    trades.append(PortfolioTrade(
                        bar_idx=next_idx, ts_ms=int(ts_ms_arr[next_idx]),
                        symbol=sym, side="sell",
                        delta_units=units_to_sell, fill_price=fill_px, fee_paid=fee,
                    ))
            # Reconcile cash. The single `available_cash` already includes
            # both the USD_CASH leg AND the USD-not-in-universe `cash_usd`,
            # so split them back: if USD_CASH is in the universe, ALL cash
            # is held there; otherwise it lives in `cash_usd`.
            available_cash = max(0.0, available_cash)
            if USD_CASH in universe:
                new_holdings[USD_CASH] = available_cash
                cash_usd = 0.0
            else:
                new_holdings[USD_CASH] = 0.0
                cash_usd = available_cash
            holdings_units = new_holdings
            last_target_weights = dict(target)
            n_decisions += 1
            weights_history.append((int(ts_ms_arr[next_idx]), dict(target)))

        # Mark equity at end of bar i (using bar i close for risk legs)
        eq_after = _equity_now(i)
        equity_curve.append((int(ts_ms_arr[i]), eq_after))

    # Final close at end_idx
    final_eq = _equity_now(end_idx)
    equity_curve.append((int(ts_ms_arr[end_idx]), final_eq))

    return PortfolioResult(
        initial_equity_usd=initial_equity_usd,
        final_equity_usd=final_eq,
        equity_curve=equity_curve,
        weights_history=weights_history,
        trades=trades,
        n_decisions=n_decisions,
        n_cache_hits=n_cache_hits,
    )
