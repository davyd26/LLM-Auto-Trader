"""Tests for risk manager validation."""

import pytest
from src.risk.risk_manager import RiskManager, ValidationResult
from src.risk.circuit_breakers import CircuitBreakers
from src.risk.position_sizer import PositionSizer
from src.llm.response_parser import (
    TradeAction, ActionType, LLMResponse, MarketAssessment,
)


def make_action(
    action: str = "open_long",
    symbol: str = "BTC/USDT:USDT",
    size_pct: float = 10,
    leverage: int = 2,
    stop_loss: float = 50000,
    take_profit: float = 55000,
) -> TradeAction:
    # Use model_construct to bypass Pydantic field constraints,
    # so we can test risk manager validation independently.
    return TradeAction.model_construct(
        action=ActionType(action),
        symbol=symbol,
        size_pct=size_pct,
        leverage=leverage,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning="Test reasoning for this trade action",
        order_type="market",
        entry_price=None,
    )


@pytest.fixture
def risk_manager():
    return RiskManager(
        max_leverage=5,
        max_position_pct=0.25,
        max_total_exposure_pct=0.60,
        max_risk_per_trade_pct=0.02,
        min_reward_risk_ratio=1.5,
        max_concurrent_positions=3,
        allowed_symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
    )


def test_approve_valid_long(risk_manager):
    action = make_action()
    result = risk_manager.validate_action(
        action=action,
        equity=1000,
        open_positions=[],
        current_prices={"BTC/USDT:USDT": 52000},
    )
    assert result.approved


def test_reject_invalid_symbol(risk_manager):
    action = make_action(symbol="DOGE/USDT:USDT")
    result = risk_manager.validate_action(
        action=action,
        equity=1000,
        open_positions=[],
        current_prices={"DOGE/USDT:USDT": 0.1},
    )
    assert not result.approved
    assert "not in allowed list" in result.rejection_reason


def test_reject_excessive_leverage(risk_manager):
    action = make_action(leverage=10)
    result = risk_manager.validate_action(
        action=action,
        equity=1000,
        open_positions=[],
        current_prices={"BTC/USDT:USDT": 52000},
    )
    assert not result.approved
    assert "Leverage" in result.rejection_reason


def test_reject_too_many_positions(risk_manager):
    positions = [
        {"symbol": f"SYM{i}/USDT:USDT", "notional": 100}
        for i in range(3)
    ]
    action = make_action()
    result = risk_manager.validate_action(
        action=action,
        equity=1000,
        open_positions=positions,
        current_prices={"BTC/USDT:USDT": 52000},
    )
    assert not result.approved
    assert "positions" in result.rejection_reason.lower()


def test_reduce_oversized_position(risk_manager):
    action = make_action(size_pct=50)  # 50% is over 25% limit
    result = risk_manager.validate_action(
        action=action,
        equity=1000,
        open_positions=[],
        current_prices={"BTC/USDT:USDT": 52000},
    )
    assert result.approved
    assert result.adjusted_size_pct <= 25  # capped at max


def test_hold_always_approved(risk_manager):
    action = TradeAction(
        action=ActionType.HOLD,
        symbol="BTC/USDT:USDT",
        reasoning="Waiting for setup",
    )
    result = risk_manager.validate_action(
        action=action,
        equity=1000,
        open_positions=[],
        current_prices={},
    )
    assert result.approved


def test_close_always_approved(risk_manager):
    action = TradeAction(
        action=ActionType.CLOSE,
        symbol="BTC/USDT:USDT",
        reasoning="Taking profits",
    )
    result = risk_manager.validate_action(
        action=action,
        equity=1000,
        open_positions=[],
        current_prices={},
    )
    assert result.approved
