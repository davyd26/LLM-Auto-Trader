"""Unified cost + funding + slippage model (Phase 2 §3A.3).

Every trade PnL is reported gross AND net. The LLM sees the NET numbers so it
can't be tricked by a gross-profitable trade that paid away the entire edge in
funding drag. Slippage is estimated from the order-book depth available at
trade time; funding is accrued from the realized 8h funding rate.

No look-ahead: all inputs are quantities knowable at or before trade time
(fees from the configured ExchangeProfile — constants; funding rate from the
exchange's current schedule; depth from the live order book).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeCosts:
    fee_entry: float
    fee_exit: float
    funding_paid: float           # cumulative USDT paid in funding over holding window
    slippage_entry_est: float     # estimated USDT lost to slippage on entry
    slippage_exit_est: float      # estimated USDT lost to slippage on exit
    total_cost_pct: float         # all-in cost as fraction of notional

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _slippage_usdt(notional: float, spread_pct: float, depth_usdt: float) -> float:
    """Bar-level slippage estimator.

    - Half the spread is always paid (midpoint → one side of the book).
    - Beyond the available depth (here assumed at best quote), slippage grows
      as sqrt(notional / depth) — a common square-root price-impact model.

    Inputs are positive and finite; callers should guard upstream.
    """
    if notional <= 0:
        return 0.0
    base = 0.5 * spread_pct * notional
    if depth_usdt <= 0:
        return base
    excess = max(0.0, notional - depth_usdt)
    if excess <= 0:
        return base
    # sqrt-law impact on the excess portion, scaled by spread as a unit.
    impact = spread_pct * notional * (excess / depth_usdt) ** 0.5
    return base + impact


def estimate_trade_cost(
    notional: float,
    leverage: int,
    holding_seconds: float,
    funding_rate_per_8h: float,
    spread_pct: float,
    depth: float,
    taker_fee: float = 0.0001,
) -> TradeCosts:
    """Project the all-in round-trip cost for a trade of given `notional`.

    `funding_rate_per_8h` is signed (positive means longs pay shorts); we take
    abs value because the LLM pays it either way when it's on the paying side
    and receives when on the other — for a conservative forward estimate we
    bill the absolute drag against both directions (conservative).
    """
    fee_entry = max(0.0, notional) * max(0.0, taker_fee)
    fee_exit = fee_entry
    # Funding: pro-rate the 8h rate across the actual holding period.
    funding_paid = (
        abs(funding_rate_per_8h) * max(0.0, notional)
        * (max(0.0, holding_seconds) / (8.0 * 3600.0))
    )
    slippage_entry = _slippage_usdt(notional, spread_pct, depth)
    slippage_exit = _slippage_usdt(notional, spread_pct, depth)

    total = fee_entry + fee_exit + funding_paid + slippage_entry + slippage_exit
    total_pct = total / notional if notional > 0 else 0.0

    return TradeCosts(
        fee_entry=fee_entry,
        fee_exit=fee_exit,
        funding_paid=funding_paid,
        slippage_entry_est=slippage_entry,
        slippage_exit_est=slippage_exit,
        total_cost_pct=total_pct,
    )


def net_pnl(gross_pnl: float, costs: TradeCosts) -> float:
    """Subtract all-in trade costs from gross PnL (USDT-denominated)."""
    return gross_pnl - (
        costs.fee_entry + costs.fee_exit + costs.funding_paid
        + costs.slippage_entry_est + costs.slippage_exit_est
    )


def required_edge_pct(
    leverage: int,
    holding_seconds: float,
    funding_rate_per_8h: float,
    spread_pct: float,
    depth: float,
    taker_fee: float = 0.0001,
    notional: float = 10_000.0,
) -> float:
    """Minimum gross-return % of MARGIN that breaks even on round-trip cost.

    `notional` is a reference (costs scale with notional; return is relative to
    margin = notional / leverage, so the answer is effectively leverage-sensitive).
    """
    c = estimate_trade_cost(
        notional=notional, leverage=leverage,
        holding_seconds=holding_seconds,
        funding_rate_per_8h=funding_rate_per_8h,
        spread_pct=spread_pct, depth=depth, taker_fee=taker_fee,
    )
    if notional <= 0 or leverage <= 0:
        return 0.0
    margin = notional / leverage
    total_cost = (c.fee_entry + c.fee_exit + c.funding_paid
                  + c.slippage_entry_est + c.slippage_exit_est)
    return total_cost / margin
