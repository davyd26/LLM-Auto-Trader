"""Margin-ratio monitor.

Computes aggregate margin state from open positions + account balance. Surfaces
a ``distance_to_liquidation_pct`` for every position so risk gates can
force-close when the buffer is too thin. The default maintenance-margin
fraction (0.5%) is a conservative overestimate; configure ``EXCHANGE_MAINT_MARGIN``
or override the ``ExchangeProfile`` to match your venue/market tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PositionMargin:
    symbol: str
    side: str
    mark_price: float
    entry_price: float
    notional: float
    leverage: int
    liquidation_price: float
    distance_to_liquidation_pct: float  # positive number; smaller = more dangerous


@dataclass
class MarginState:
    used_margin: float
    free_margin: float
    equity: float
    margin_ratio: float                                  # used / equity
    min_distance_to_liquidation_pct: float               # min across positions
    positions: list[PositionMargin] = field(default_factory=list)


def _liquidation_price(
    entry_price: float, side: str, leverage: int,
    maintenance_margin_fraction: float = 0.005,
) -> float:
    """Approximate liquidation price (isolated margin).

    For long: liq = entry * (1 - 1/L + maintenance)
    For short: liq = entry * (1 + 1/L - maintenance)
    """
    if entry_price <= 0 or leverage <= 0:
        return 0.0
    adj = (1.0 / leverage) - maintenance_margin_fraction
    if side == "long":
        return float(entry_price * (1.0 - adj))
    return float(entry_price * (1.0 + adj))


def compute_margin_state(
    positions: list[dict],
    balance: dict,
    maintenance_margin_fraction: float = 0.005,
) -> MarginState:
    """Given live positions and a balance dict, produce a MarginState."""
    equity = float(balance.get("total", 0.0))
    free = float(balance.get("free", equity))
    used = float(balance.get("used", max(0.0, equity - free)))

    per_position: list[PositionMargin] = []
    min_dist = 1.0  # 100% means "infinite buffer"; will be shrunk by real positions

    for p in positions:
        symbol = str(p.get("symbol", ""))
        side = str(p.get("side", "long"))
        mark = float(p.get("mark_price") or 0.0)
        entry = float(p.get("entry_price") or 0.0)
        notional = float(p.get("notional") or 0.0)
        leverage = int(p.get("leverage") or 1)
        liq = _liquidation_price(entry, side, leverage, maintenance_margin_fraction)

        if mark > 0 and liq > 0:
            if side == "long":
                dist = max(0.0, (mark - liq) / mark)
            else:
                dist = max(0.0, (liq - mark) / mark)
        else:
            dist = 1.0

        min_dist = min(min_dist, dist)
        per_position.append(PositionMargin(
            symbol=symbol, side=side, mark_price=mark, entry_price=entry,
            notional=notional, leverage=leverage, liquidation_price=liq,
            distance_to_liquidation_pct=dist,
        ))

    margin_ratio = used / equity if equity > 0 else 0.0

    return MarginState(
        used_margin=used,
        free_margin=free,
        equity=equity,
        margin_ratio=margin_ratio,
        min_distance_to_liquidation_pct=min_dist if per_position else 1.0,
        positions=per_position,
    )
