"""Funding rate accounting for perpetual futures backtests.

On Binance USDT-perp the funding rate is paid every 8 hours. Long positions
pay positive funding (the longs subsidize the shorts when sentiment is
bullish); shorts receive it. Negative funding flips the direction.

Unmodeled, a backtest overstates returns by 1-3% per month at typical
funding levels — material at $2K AUM. This module charges/credits funding
to the position notional based on time held.

The flat per-day rate is a coarse approximation. A future extension can
plug in historical funding-rate data per (symbol, timestamp).
"""

from __future__ import annotations


def compute_funding_cost(
    side: str,
    notional: float,
    days_held: float,
    funding_rate_per_day: float,
) -> float:
    """Return the funding cost as a positive number for longs (a debit) and
    negative for shorts (a credit), assuming positive funding_rate_per_day.

    Sign flips when funding_rate_per_day is negative (e.g., heavy short bias).
    """
    if days_held < 0:
        raise ValueError("days_held must be non-negative")
    if side not in ("long", "short"):
        raise ValueError(f"unknown side {side!r}")

    raw = notional * funding_rate_per_day * days_held
    if side == "long":
        return raw
    return -raw
