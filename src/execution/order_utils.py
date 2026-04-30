"""Order utilities: size rounding, price rounding, slippage estimation."""

from __future__ import annotations

import math


def round_size(size: float, lot_size: float) -> float:
    """Round order size down to exchange lot size."""
    if lot_size <= 0:
        return size
    decimals = max(0, -int(math.floor(math.log10(lot_size))))
    return math.floor(size / lot_size) * lot_size
    # Round to avoid floating point issues
    return round(math.floor(size / lot_size) * lot_size, decimals)


def round_price(price: float, tick_size: float) -> float:
    """Round price to exchange tick size."""
    if tick_size <= 0:
        return price
    decimals = max(0, -int(math.floor(math.log10(tick_size))))
    return round(round(price / tick_size) * tick_size, decimals)


def estimate_slippage(
    side: str,
    size_usd: float,
    bid_depth: float,
    ask_depth: float,
    spread_pct: float,
) -> float:
    """Estimate slippage as a percentage based on order book.

    Args:
        side: "buy" or "sell"
        size_usd: Order size in USDT
        bid_depth: Total bid depth in top N levels (USDT)
        ask_depth: Total ask depth in top N levels (USDT)
        spread_pct: Current spread as percentage

    Returns:
        Estimated slippage as percentage (e.g., 0.01 = 1%)
    """
    # Base slippage = half the spread
    base = spread_pct / 2

    # Impact: proportion of order vs available depth
    depth = ask_depth if side == "buy" else bid_depth
    if depth > 0:
        impact = (size_usd / depth) * 0.1  # 10% of depth ratio
    else:
        impact = 0.001  # conservative default

    return base + impact


def notional_to_contracts(
    notional_usd: float,
    current_price: float,
    contract_size: float = 1.0,
) -> float:
    """Convert USDT notional to contract count."""
    if current_price <= 0:
        return 0
    return notional_usd / (current_price * contract_size)


def calculate_liquidation_price(
    entry_price: float,
    leverage: int,
    side: str,
    maintenance_margin_rate: float = 0.004,
) -> float:
    """Estimate liquidation price for a position.

    This is approximate — actual liquidation depends on exchange-specific rules.
    """
    if side == "long":
        return entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
    else:
        return entry_price * (1 + (1 / leverage) - maintenance_margin_rate)
