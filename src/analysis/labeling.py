"""Triple-barrier labeling (AFML Ch. 3.3-3.4).

Replays the price path of a closed trade against three barriers and assigns a
canonical label:
- `+1` (profit-take) — price reached the profit-take barrier first
- `-1` (stop-loss)  — price reached the stop-loss barrier first
- `0`  (time-out)   — neither barrier hit before max holding period

Computed ONLY at trade close. Labels feed evaluation / future meta-labeling;
never fed back into live LLM context for an open position.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TripleBarrierLabel:
    label: int                    # +1 / -1 / 0
    barrier_touched: str          # "pt" / "sl" / "time"
    touch_time: float             # unix seconds
    return_at_touch: float        # signed return at the touched barrier (sign flipped for shorts)


def compute_triple_barrier(
    entry_price: float,
    side: str,
    pt_pct: float,
    sl_pct: float,
    max_holding_s: float,
    path: list[tuple[float, float]],
) -> TripleBarrierLabel:
    """Compute the triple-barrier label for a closed trade.

    Args:
        entry_price: Fill price at trade open.
        side: "long" or "short".
        pt_pct: Profit-take distance as positive fraction (e.g., 0.02 = +2%).
        sl_pct: Stop-loss distance as positive fraction (e.g., 0.01 = -1%).
        max_holding_s: Max holding duration in seconds (time barrier).
        path: Ordered list of (timestamp, price) tuples for the trade window.
              Should include the entry price as the first entry or start just
              after. Timestamps must be monotonically non-decreasing.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side}")
    if pt_pct < 0 or sl_pct < 0:
        raise ValueError("pt_pct and sl_pct must be non-negative")

    if not path:
        return TripleBarrierLabel(label=0, barrier_touched="time", touch_time=0.0, return_at_touch=0.0)

    start_ts = path[0][0]
    deadline = start_ts + max_holding_s

    sign = 1.0 if side == "long" else -1.0

    for ts, price in path:
        # Signed return from entry, with short convention flipped so "profitable" is always positive.
        if entry_price <= 0:
            break
        signed_ret = sign * (price - entry_price) / entry_price
        if signed_ret >= pt_pct:
            return TripleBarrierLabel(
                label=1, barrier_touched="pt", touch_time=ts, return_at_touch=signed_ret
            )
        if signed_ret <= -sl_pct:
            return TripleBarrierLabel(
                label=-1, barrier_touched="sl", touch_time=ts, return_at_touch=signed_ret
            )
        if ts >= deadline:
            return TripleBarrierLabel(
                label=0, barrier_touched="time", touch_time=ts, return_at_touch=signed_ret
            )

    # Path ended without hitting any barrier — treat as time-out at last ts.
    last_ts, last_price = path[-1]
    last_ret = sign * (last_price - entry_price) / entry_price if entry_price > 0 else 0.0
    return TripleBarrierLabel(
        label=0, barrier_touched="time", touch_time=last_ts, return_at_touch=last_ret
    )
