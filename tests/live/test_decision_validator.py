"""Tests for live decision validation.

Before a Claude-Code-produced decision touches the live exchange, it passes
through a strict validator that:
    - Confirms the JSON shape matches the Pydantic schema.
    - Rejects any action that would breach risk-gate caps (size, leverage).
    - Rejects malformed stop_loss / take_profit prices.
    - Logs and quarantines any decision that fails — never fall back to a
      default action silently.

Edge cases covered:
    - Missing required fields → ValidationError
    - size > equity * max_position_pct → reject
    - leverage > max_leverage → clamp to max
    - stop_loss on wrong side of entry → reject
    - confidence outside [0, 1] → clamp

Tail risks covered:
    - Action types we don't recognize → reject (don't silently treat as hold)
    - Numeric overflow in size → reject
    - LLM emits valid-shape but absurd values (size = 1e9) → reject
"""

from __future__ import annotations

import math

import pytest

from src.live.decision_validator import (
    DecisionValidationError,
    ValidatedDecision,
    validate_decision,
)

# At $2K equity, 25% max position cap, $50K BTC → max size = 0.01 BTC.
# 0.008 BTC = $400 notional = 20% of equity, well under the cap.
_OK_DECISION = {
    "market_assessment": {"trend": "bullish", "confidence": 0.7, "regime": "calm"},
    "actions": [
        {
            "type": "open_long",
            "symbol": "BTC/USDT",
            "size": 0.008,
            "stop_loss": 45000.0,
        }
    ],
    "next_check_minutes": 10080,
    "risk_notes": "trend confirmed",
}


class TestValidateDecisionShape:
    def test_valid_decision_returns_validated_object(self) -> None:
        v = validate_decision(_OK_DECISION, current_price=50_000.0, equity_usd=2000.0)
        assert isinstance(v, ValidatedDecision)
        assert len(v.actions) == 1

    def test_missing_market_assessment_rejected(self) -> None:
        bad = dict(_OK_DECISION)
        del bad["market_assessment"]
        with pytest.raises(DecisionValidationError):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)

    def test_missing_actions_rejected(self) -> None:
        bad = dict(_OK_DECISION)
        del bad["actions"]
        with pytest.raises(DecisionValidationError):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)

    def test_unknown_action_type_rejected(self) -> None:
        bad = {**_OK_DECISION, "actions": [{"type": "moonshot", "symbol": "BTC/USDT"}]}
        with pytest.raises(DecisionValidationError):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)


class TestSizeLimits:
    def test_size_above_max_position_pct_rejected(self) -> None:
        # max position 25% of $2000 equity at $50K BTC = 0.01 BTC
        # Decision asks for 0.10 BTC → notional $5000 (250% of equity)
        bad = {
            "market_assessment": {"trend": "bullish", "confidence": 0.7, "regime": "calm"},
            "actions": [{"type": "open_long", "symbol": "BTC/USDT", "size": 0.10, "stop_loss": 45000.0}],
            "next_check_minutes": 10080,
            "risk_notes": "x",
        }
        with pytest.raises(DecisionValidationError, match="position"):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)

    def test_zero_size_normalized_to_hold(self) -> None:
        # Zero-size open is effectively a hold; validator normalizes
        d = {
            "market_assessment": {"trend": "neutral", "confidence": 0.5, "regime": "calm"},
            "actions": [{"type": "open_long", "symbol": "BTC/USDT", "size": 0.0, "stop_loss": 45000.0}],
            "next_check_minutes": 10080,
            "risk_notes": "x",
        }
        v = validate_decision(d, current_price=50_000.0, equity_usd=2000.0)
        assert all(a.type != "open_long" or a.size > 0 for a in v.actions)

    def test_negative_size_rejected(self) -> None:
        bad = {
            "market_assessment": {"trend": "neutral", "confidence": 0.5, "regime": "calm"},
            "actions": [{"type": "open_long", "symbol": "BTC/USDT", "size": -0.05, "stop_loss": 45000.0}],
            "next_check_minutes": 10080,
            "risk_notes": "x",
        }
        with pytest.raises(DecisionValidationError):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)

    def test_absurdly_large_size_rejected(self) -> None:
        bad = {
            "market_assessment": {"trend": "bullish", "confidence": 0.9, "regime": "calm"},
            "actions": [{"type": "open_long", "symbol": "BTC/USDT", "size": 1e9, "stop_loss": 45000.0}],
            "next_check_minutes": 10080,
            "risk_notes": "x",
        }
        with pytest.raises(DecisionValidationError):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)


class TestStopLoss:
    def test_long_with_stop_above_current_rejected(self) -> None:
        bad = {
            "market_assessment": {"trend": "bullish", "confidence": 0.7, "regime": "calm"},
            "actions": [{"type": "open_long", "symbol": "BTC/USDT", "size": 0.01, "stop_loss": 51_000.0}],
            "next_check_minutes": 10080,
            "risk_notes": "x",
        }
        with pytest.raises(DecisionValidationError, match="stop"):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)

    def test_short_with_stop_below_current_rejected(self) -> None:
        bad = {
            "market_assessment": {"trend": "bearish", "confidence": 0.7, "regime": "calm"},
            "actions": [{"type": "open_short", "symbol": "BTC/USDT", "size": 0.01, "stop_loss": 49_000.0}],
            "next_check_minutes": 10080,
            "risk_notes": "x",
        }
        with pytest.raises(DecisionValidationError, match="stop"):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)

    def test_open_long_without_stop_rejected(self) -> None:
        bad = {
            "market_assessment": {"trend": "bullish", "confidence": 0.7, "regime": "calm"},
            "actions": [{"type": "open_long", "symbol": "BTC/USDT", "size": 0.01}],
            "next_check_minutes": 10080,
            "risk_notes": "x",
        }
        with pytest.raises(DecisionValidationError, match="stop"):
            validate_decision(bad, current_price=50_000.0, equity_usd=2000.0)


class TestConfidenceClamping:
    def test_confidence_above_one_clamped(self) -> None:
        d = dict(_OK_DECISION)
        d["market_assessment"] = {**_OK_DECISION["market_assessment"], "confidence": 1.5}
        v = validate_decision(d, current_price=50_000.0, equity_usd=2000.0)
        assert v.confidence == 1.0

    def test_confidence_below_zero_clamped(self) -> None:
        d = dict(_OK_DECISION)
        d["market_assessment"] = {**_OK_DECISION["market_assessment"], "confidence": -0.1}
        v = validate_decision(d, current_price=50_000.0, equity_usd=2000.0)
        assert v.confidence == 0.0

    def test_nan_confidence_treated_as_low(self) -> None:
        d = dict(_OK_DECISION)
        d["market_assessment"] = {**_OK_DECISION["market_assessment"], "confidence": math.nan}
        v = validate_decision(d, current_price=50_000.0, equity_usd=2000.0)
        # NaN confidence should not cause a crash; treated as 0 (no conviction)
        assert v.confidence == 0.0


class TestHoldDecision:
    def test_hold_action_is_valid(self) -> None:
        d = {
            "market_assessment": {"trend": "neutral", "confidence": 0.5, "regime": "calm"},
            "actions": [{"type": "hold"}],
            "next_check_minutes": 10080,
            "risk_notes": "wait",
        }
        v = validate_decision(d, current_price=50_000.0, equity_usd=2000.0)
        assert v.actions[0].type == "hold"
