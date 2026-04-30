"""Dynamic stop-loss and take-profit calculation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StopLevels:
    """Calculated stop-loss and take-profit levels."""

    stop_loss: float
    take_profit_1: float        # first target (partial exit)
    take_profit_2: float | None  # second target (remainder)
    trailing_activation: float   # price at which trailing SL activates
    risk_reward_ratio: float


def calculate_stop_levels(
    entry_price: float,
    side: str,  # "long" or "short"
    atr: float,
    regime: str = "trending_up",  # trending_up, trending_down, ranging, volatile
    sl_atr_multiplier: float = 1.5,
    partial_tp_rr: float = 2.0,
    trailing_activation_atr: float = 1.0,
) -> StopLevels:
    """Calculate SL/TP levels based on ATR and market regime.

    Args:
        entry_price: Expected entry price
        side: "long" or "short"
        atr: Current ATR value
        regime: Current market regime
        sl_atr_multiplier: SL distance in ATR multiples
        partial_tp_rr: Reward:risk ratio for first TP target
        trailing_activation_atr: ATR distance for trailing SL activation
    """
    # Adjust multiplier based on regime
    regime_adjustments = {
        "trending_up": 1.0 if side == "long" else 1.2,    # tighter SL with trend, wider against
        "trending_down": 1.2 if side == "long" else 1.0,
        "ranging": 0.8,          # tighter in range
        "volatile": 1.5,        # wider in volatile
    }
    adjusted_mult = sl_atr_multiplier * regime_adjustments.get(regime, 1.0)

    sl_distance = atr * adjusted_mult

    if side == "long":
        stop_loss = entry_price - sl_distance
        tp1 = entry_price + sl_distance * partial_tp_rr
        tp2 = entry_price + sl_distance * partial_tp_rr * 1.5
        trailing_act = entry_price + atr * trailing_activation_atr
    else:
        stop_loss = entry_price + sl_distance
        tp1 = entry_price - sl_distance * partial_tp_rr
        tp2 = entry_price - sl_distance * partial_tp_rr * 1.5
        trailing_act = entry_price - atr * trailing_activation_atr

    rr = partial_tp_rr  # by construction

    return StopLevels(
        stop_loss=round(stop_loss, 2),
        take_profit_1=round(tp1, 2),
        take_profit_2=round(tp2, 2),
        trailing_activation=round(trailing_act, 2),
        risk_reward_ratio=round(rr, 2),
    )


def validate_sl_tp(
    entry_price: float,
    side: str,
    stop_loss: float,
    take_profit: float | None,
    max_risk_pct: float = 0.03,
    min_rr_ratio: float = 1.5,
) -> tuple[bool, str]:
    """Validate that SL/TP levels are sane.

    Returns (is_valid, reason).
    """
    if side == "long":
        if stop_loss >= entry_price:
            return False, "SL must be below entry for long"
        if take_profit is not None and take_profit <= entry_price:
            return False, "TP must be above entry for long"

        risk_pct = (entry_price - stop_loss) / entry_price
        if take_profit is not None:
            reward_pct = (take_profit - entry_price) / entry_price
            rr = reward_pct / risk_pct if risk_pct > 0 else 0
            if rr < min_rr_ratio:
                return False, f"R:R ratio {rr:.1f} below minimum {min_rr_ratio}"

    elif side == "short":
        if stop_loss <= entry_price:
            return False, "SL must be above entry for short"
        if take_profit is not None and take_profit >= entry_price:
            return False, "TP must be below entry for short"

        risk_pct = (stop_loss - entry_price) / entry_price
        if take_profit is not None:
            reward_pct = (entry_price - take_profit) / entry_price
            rr = reward_pct / risk_pct if risk_pct > 0 else 0
            if rr < min_rr_ratio:
                return False, f"R:R ratio {rr:.1f} below minimum {min_rr_ratio}"
    else:
        return False, f"Invalid side: {side}"

    if risk_pct > max_risk_pct:
        return False, f"Risk {risk_pct*100:.1f}% exceeds max {max_risk_pct*100:.1f}%"

    return True, "OK"


def calculate_trailing_stop(
    side: str,
    current_price: float,
    highest_since_entry: float,
    lowest_since_entry: float,
    trailing_distance: float,
) -> float:
    """Calculate current trailing stop-loss level.

    Args:
        side: "long" or "short"
        current_price: Current market price
        highest_since_entry: Highest price since position opened
        lowest_since_entry: Lowest price since position opened
        trailing_distance: Distance to trail in price units

    Returns:
        Trailing stop-loss price
    """
    if side == "long":
        return highest_since_entry - trailing_distance
    else:
        return lowest_since_entry + trailing_distance


def should_time_stop(
    entry_time: float,
    current_time: float,
    entry_price: float,
    current_price: float,
    side: str,
    atr: float,
    time_limit_hours: float = 6.0,
    min_progress_atr: float = 0.5,
) -> bool:
    """Check if a time-based stop should tighten the SL.

    If position hasn't made minimum progress (0.5 ATR in favorable direction)
    within time_limit_hours, signal to tighten the stop.
    """
    hours_held = (current_time - entry_time) / 3600
    if hours_held < time_limit_hours:
        return False

    if side == "long":
        progress = current_price - entry_price
    else:
        progress = entry_price - current_price

    return progress < atr * min_progress_atr
