"""Backtest fill model.

Converts a high-level decision (open long size 1 BTC with stop at 95) into a
simulated trade with realistic execution: entry on the *next* bar's open
(no within-bar lookahead), taker fees, configurable bps slippage, and SL/TP
trigger logic.

When both SL and TP are touchable within the same bar, the conservative
convention is assumed: SL fires first. This is the standard backtest
assumption — without tick data we cannot prove which came first, and
assuming the worst prevents over-optimistic results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FillConfig:
    taker_fee_bps: float = 4.0      # 0.04% Binance perp taker default
    slippage_bps: float = 5.0        # extra bps of adverse price movement on fills
    funding_rate_per_day: float = 0.0  # daily funding rate (positive favors shorts)


@dataclass
class SimulatedTrade:
    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    entry_ts_ms: int
    entry_fee: float
    stop_loss: float | None
    take_profit: float | None
    # Filled at close
    exit_price: float | None = None
    exit_ts_ms: int | None = None
    exit_fee: float = 0.0
    exit_reason: str | None = None  # "manual" | "stop_loss" | "take_profit"
    pnl: float = 0.0


def simulate_open(
    symbol: str,
    side: str,
    size: float,
    next_bar: dict,
    stop_loss: float | None,
    take_profit: float | None,
    config: FillConfig,
) -> SimulatedTrade:
    """Open a position at next_bar's open with slippage and fees applied."""
    if size <= 0:
        raise ValueError("size must be strictly positive")
    if side not in ("long", "short"):
        raise ValueError(f"unknown side {side!r}")

    open_price = float(next_bar["open"])
    slip_factor = config.slippage_bps / 10_000.0
    if side == "long":
        entry_price = open_price * (1 + slip_factor)
    else:
        entry_price = open_price * (1 - slip_factor)

    # Validate stop placement against the *entry* price, not the bar open.
    if stop_loss is not None and not math.isnan(stop_loss):
        if side == "long" and stop_loss >= entry_price:
            raise ValueError("long stop_loss must be below entry_price")
        if side == "short" and stop_loss <= entry_price:
            raise ValueError("short stop_loss must be above entry_price")

    fee_factor = config.taker_fee_bps / 10_000.0
    entry_fee = entry_price * size * fee_factor

    return SimulatedTrade(
        symbol=symbol,
        side=side,
        size=size,
        entry_price=entry_price,
        entry_ts_ms=int(next_bar.get("ts_ms", 0)),
        entry_fee=entry_fee,
        stop_loss=stop_loss if (stop_loss is not None and not math.isnan(stop_loss)) else None,
        take_profit=take_profit if (take_profit is not None and not math.isnan(take_profit)) else None,
    )


def simulate_close(
    trade: SimulatedTrade,
    next_bar: dict,
    exit_ts_ms: int,
    config: FillConfig,
) -> SimulatedTrade:
    """Close ``trade`` at the next_bar, respecting SL/TP if triggered.

    Returns the same trade object with exit fields populated.
    """
    bar_open = float(next_bar["open"])
    bar_high = float(next_bar["high"])
    bar_low = float(next_bar["low"])
    slip_factor = config.slippage_bps / 10_000.0
    fee_factor = config.taker_fee_bps / 10_000.0

    exit_price: float
    exit_reason: str

    sl_hit = _stop_hit(trade.side, trade.stop_loss, bar_high, bar_low)
    tp_hit = _take_hit(trade.side, trade.take_profit, bar_high, bar_low)

    if sl_hit:
        # Conservative: SL fires first when both touchable in same bar
        exit_price = float(trade.stop_loss)  # type: ignore[arg-type]
        exit_reason = "stop_loss"
    elif tp_hit:
        exit_price = float(trade.take_profit)  # type: ignore[arg-type]
        exit_reason = "take_profit"
    else:
        # Market close at next bar's open with slippage on adverse side
        if trade.side == "long":
            exit_price = bar_open * (1 - slip_factor)
        else:
            exit_price = bar_open * (1 + slip_factor)
        exit_reason = "manual"

    exit_fee = exit_price * trade.size * fee_factor

    if trade.side == "long":
        gross = (exit_price - trade.entry_price) * trade.size
    else:
        gross = (trade.entry_price - exit_price) * trade.size

    # Funding cost: notional × rate × days_held; longs pay positive funding,
    # shorts receive it (sign handled inside compute_funding_cost).
    from src.backtest.funding import compute_funding_cost
    days_held = max(0.0, (exit_ts_ms - trade.entry_ts_ms) / 86_400_000.0)
    notional = trade.entry_price * trade.size
    funding_cost = compute_funding_cost(
        side=trade.side,
        notional=notional,
        days_held=days_held,
        funding_rate_per_day=config.funding_rate_per_day,
    )

    trade.exit_price = exit_price
    trade.exit_ts_ms = exit_ts_ms
    trade.exit_fee = exit_fee
    trade.exit_reason = exit_reason
    trade.pnl = gross - trade.entry_fee - exit_fee - funding_cost
    return trade


def _stop_hit(side: str, stop: float | None, high: float, low: float) -> bool:
    if stop is None or math.isnan(stop):
        return False
    if side == "long":
        return low <= stop
    return high >= stop  # short


def _take_hit(side: str, tp: float | None, high: float, low: float) -> bool:
    if tp is None or math.isnan(tp):
        return False
    if side == "long":
        return high >= tp
    return low <= tp  # short
