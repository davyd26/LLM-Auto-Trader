"""Validate an LLM decision against live-trading risk caps.

Every decision produced by Claude Code subscription (live or backtest)
passes through this validator before any side effect. The validator
rejects malformed, impossible, or out-of-policy decisions and clamps
edge-case numeric values into safe ranges.

This is the live-side complement to the backtest fill model's preflight:
the backtest can be more lenient (test fixtures legitimately probe edges),
but live trading must reject anything questionable rather than silently
substitute a default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class DecisionValidationError(ValueError):
    """Raised when a decision violates live-trading constraints."""


@dataclass(frozen=True)
class ValidatedAction:
    type: str  # hold | open_long | open_short | close
    symbol: str | None
    size: float
    stop_loss: float | None
    take_profit: float | None
    leverage: float


@dataclass(frozen=True)
class ValidatedDecision:
    trend: str
    confidence: float
    regime: str
    actions: list[ValidatedAction]
    next_check_minutes: int
    risk_notes: str


_VALID_ACTIONS = {"hold", "open_long", "open_short", "close"}
_VALID_TRENDS = {"bullish", "bearish", "neutral"}
_VALID_REGIMES = {"calm", "elevated", "crisis"}


def validate_decision(
    raw: dict,
    current_price: float,
    equity_usd: float,
    max_position_pct: float = 0.25,
    max_leverage: float = 5.0,
    max_size_usd: float = 1_000_000.0,
) -> ValidatedDecision:
    """Validate a raw decision dict and return a typed `ValidatedDecision`."""
    if not isinstance(raw, dict):
        raise DecisionValidationError("decision must be a dict")

    if "market_assessment" not in raw:
        raise DecisionValidationError("missing market_assessment")
    if "actions" not in raw:
        raise DecisionValidationError("missing actions")

    ma = raw["market_assessment"]
    if not isinstance(ma, dict):
        raise DecisionValidationError("market_assessment must be a dict")

    trend = str(ma.get("trend", "neutral")).lower()
    if trend not in _VALID_TRENDS:
        raise DecisionValidationError(f"unknown trend {trend!r}")

    regime = str(ma.get("regime", "calm")).lower()
    if regime not in _VALID_REGIMES:
        raise DecisionValidationError(f"unknown regime {regime!r}")

    confidence_raw = ma.get("confidence", 0.5)
    confidence = _clamp_confidence(confidence_raw)

    actions_raw = raw["actions"]
    if not isinstance(actions_raw, list) or not actions_raw:
        raise DecisionValidationError("actions must be a non-empty list")

    actions: list[ValidatedAction] = []
    for action_raw in actions_raw:
        actions.append(_validate_action(
            action_raw,
            current_price=current_price,
            equity_usd=equity_usd,
            max_position_pct=max_position_pct,
            max_leverage=max_leverage,
            max_size_usd=max_size_usd,
        ))

    return ValidatedDecision(
        trend=trend,
        confidence=confidence,
        regime=regime,
        actions=actions,
        next_check_minutes=int(raw.get("next_check_minutes", 10080)),
        risk_notes=str(raw.get("risk_notes", "")),
    )


def _validate_action(
    action: dict,
    current_price: float,
    equity_usd: float,
    max_position_pct: float,
    max_leverage: float,
    max_size_usd: float,
) -> ValidatedAction:
    if not isinstance(action, dict):
        raise DecisionValidationError("action must be a dict")

    atype = str(action.get("type", "")).lower()
    if atype not in _VALID_ACTIONS:
        raise DecisionValidationError(f"unknown action type {atype!r}")

    if atype == "hold":
        return ValidatedAction(
            type="hold", symbol=None, size=0.0,
            stop_loss=None, take_profit=None, leverage=1.0,
        )

    symbol = action.get("symbol")
    if not symbol:
        raise DecisionValidationError(f"action {atype!r} requires a symbol")

    if atype == "close":
        return ValidatedAction(
            type="close", symbol=str(symbol), size=0.0,
            stop_loss=None, take_profit=None, leverage=1.0,
        )

    # open_long / open_short
    size_raw = action.get("size", 0.0)
    try:
        size = float(size_raw)
    except (TypeError, ValueError):
        raise DecisionValidationError(f"invalid size {size_raw!r}") from None
    if math.isnan(size) or math.isinf(size):
        raise DecisionValidationError("size is NaN/inf")
    if size < 0:
        raise DecisionValidationError("size must be non-negative")

    # A zero-size open is effectively a hold; normalize rather than execute.
    if size == 0:
        return ValidatedAction(
            type="hold", symbol=None, size=0.0,
            stop_loss=None, take_profit=None, leverage=1.0,
        )

    notional = size * current_price
    if notional > max_size_usd:
        raise DecisionValidationError(f"notional ${notional:.0f} exceeds max ${max_size_usd:.0f}")

    if size > 0 and notional > equity_usd * max_position_pct:
        raise DecisionValidationError(
            f"position notional {notional:.2f} exceeds {max_position_pct:.0%} of equity {equity_usd:.2f}"
        )

    leverage_raw = action.get("leverage", 1.0)
    leverage = max(1.0, min(max_leverage, float(leverage_raw)))

    stop_loss = action.get("stop_loss")
    if size > 0 and stop_loss is None:
        raise DecisionValidationError(f"{atype} requires a stop_loss")
    if stop_loss is not None:
        sl = float(stop_loss)
        if math.isnan(sl) or sl <= 0:
            raise DecisionValidationError(f"invalid stop_loss {stop_loss!r}")
        if atype == "open_long" and sl >= current_price:
            raise DecisionValidationError("long stop_loss must be below current price")
        if atype == "open_short" and sl <= current_price:
            raise DecisionValidationError("short stop_loss must be above current price")

    take_profit = action.get("take_profit")
    if take_profit is not None:
        tp = float(take_profit)
        if math.isnan(tp) or tp <= 0:
            raise DecisionValidationError(f"invalid take_profit {take_profit!r}")
        if atype == "open_long" and tp <= current_price:
            raise DecisionValidationError("long take_profit must be above current price")
        if atype == "open_short" and tp >= current_price:
            raise DecisionValidationError("short take_profit must be below current price")
        take_profit = tp

    return ValidatedAction(
        type=atype, symbol=str(symbol), size=size,
        stop_loss=float(stop_loss) if stop_loss is not None else None,
        take_profit=float(take_profit) if take_profit is not None else None,
        leverage=leverage,
    )


def _clamp_confidence(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return max(0.0, min(1.0, f))
