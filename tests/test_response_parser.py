"""Tests for LLM response parsing and validation."""

import pytest

from src.llm.response_parser import ActionType, LLMResponse, parse_llm_response
from src.utils.errors import LLMParseError

VALID_RESPONSE = """{
    "market_assessment": {
        "trend": "bullish",
        "confidence": 0.7,
        "regime": "trending_up",
        "key_factors": ["RSI oversold bounce", "EMA9 cross above EMA21"]
    },
    "actions": [
        {
            "action": "open_long",
            "symbol": "BTC/USDT:USDT",
            "size_pct": 15,
            "leverage": 3,
            "order_type": "market",
            "entry_price": null,
            "stop_loss": 50000.0,
            "take_profit": 55000.0,
            "reasoning": "RSI bouncing from oversold with bullish EMA cross, strong volume confirmation"
        }
    ],
    "next_check_minutes": 15,
    "risk_notes": "Watch for resistance at 54k level"
}"""


HOLD_RESPONSE = """{
    "market_assessment": {
        "trend": "neutral",
        "confidence": 0.4,
        "regime": "ranging",
        "key_factors": ["Low ADX", "No clear direction"]
    },
    "actions": [
        {
            "action": "hold",
            "symbol": "BTC/USDT:USDT",
            "reasoning": "Market is ranging with no clear setup. Waiting for breakout."
        }
    ],
    "next_check_minutes": 60,
    "risk_notes": "Low volatility, no urgency"
}"""


def test_parse_valid_response():
    result = parse_llm_response(VALID_RESPONSE)
    assert isinstance(result, LLMResponse)
    assert result.market_assessment.trend == "bullish"
    assert result.market_assessment.confidence == 0.7
    assert len(result.actions) == 1
    assert result.actions[0].action == ActionType.OPEN_LONG
    assert result.actions[0].stop_loss == 50000.0
    assert result.next_check_minutes == 15


def test_parse_hold_response():
    result = parse_llm_response(HOLD_RESPONSE)
    assert result.actions[0].action == ActionType.HOLD
    assert result.next_check_minutes == 60


def test_parse_with_markdown_fences():
    wrapped = f"```json\n{VALID_RESPONSE}\n```"
    result = parse_llm_response(wrapped)
    assert result.market_assessment.trend == "bullish"


def test_reject_missing_stop_loss():
    bad = """{
        "market_assessment": {
            "trend": "bullish", "confidence": 0.7,
            "regime": "trending_up", "key_factors": ["test"]
        },
        "actions": [{
            "action": "open_long",
            "symbol": "BTC/USDT:USDT",
            "size_pct": 10, "leverage": 2,
            "reasoning": "test reasoning"
        }],
        "next_check_minutes": 15,
        "risk_notes": ""
    }"""
    with pytest.raises(LLMParseError):
        parse_llm_response(bad)


def test_reject_invalid_leverage():
    bad = """{
        "market_assessment": {
            "trend": "bullish", "confidence": 0.7,
            "regime": "trending_up", "key_factors": ["test"]
        },
        "actions": [{
            "action": "open_long",
            "symbol": "BTC/USDT:USDT",
            "size_pct": 10, "leverage": 100,
            "stop_loss": 50000, "reasoning": "test"
        }],
        "next_check_minutes": 15,
        "risk_notes": ""
    }"""
    with pytest.raises(LLMParseError):
        parse_llm_response(bad)


def test_reject_invalid_json():
    with pytest.raises(LLMParseError):
        parse_llm_response("not json at all")


def test_reject_empty_key_factors():
    bad = """{
        "market_assessment": {
            "trend": "bullish", "confidence": 0.7,
            "regime": "trending_up", "key_factors": []
        },
        "actions": [],
        "next_check_minutes": 15,
        "risk_notes": ""
    }"""
    with pytest.raises(LLMParseError):
        parse_llm_response(bad)
