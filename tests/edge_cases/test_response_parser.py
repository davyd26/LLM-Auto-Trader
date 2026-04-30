"""Response parser edge cases — malformed JSON, prose-prefixed JSON, code fences."""

from __future__ import annotations

import pytest

from src.llm.response_parser import LLMResponse, parse_llm_response
from src.utils.errors import LLMParseError

_VALID = """\
{
  "market_assessment": {
    "trend": "neutral",
    "confidence": 0.6,
    "regime": "ranging",
    "key_factors": ["a", "b"]
  },
  "actions": [
    {
      "action": "hold",
      "symbol": "BTC/USDT:USDT",
      "reasoning": "watching"
    }
  ],
  "next_check_minutes": 30,
  "risk_notes": ""
}
"""


def test_valid_json_round_trips():
    r = parse_llm_response(_VALID)
    assert isinstance(r, LLMResponse)
    assert r.actions[0].symbol == "BTC/USDT:USDT"


def test_markdown_code_fences_stripped():
    raw = f"```json\n{_VALID}\n```"
    r = parse_llm_response(raw)
    assert r.next_check_minutes == 30


def test_prose_prefixed_json_extracted():
    """Some models prepend 'Here is my analysis:' despite instructions."""
    raw = "Here is my analysis:\n" + _VALID
    r = parse_llm_response(raw)
    assert r.market_assessment.trend == "neutral"


def test_completely_malformed_raises():
    with pytest.raises(LLMParseError):
        parse_llm_response("I am not JSON")


def test_open_long_without_stop_loss_raises():
    bad = """
    {
      "market_assessment": {"trend":"bullish","confidence":0.7,"regime":"trending_up","key_factors":["a"]},
      "actions": [{"action":"open_long","symbol":"BTC/USDT:USDT","size_pct":10,"leverage":2,"reasoning":"going long because trend"}],
      "next_check_minutes": 30,
      "risk_notes": ""
    }
    """
    with pytest.raises(LLMParseError, match="stop_loss"):
        parse_llm_response(bad)


def test_unrelated_quote_currency_rejected():
    """Symbol regex enforces the configured quote — non-USDT pairs must fail."""
    bad = """
    {
      "market_assessment": {"trend":"bullish","confidence":0.7,"regime":"trending_up","key_factors":["a"]},
      "actions": [{"action":"hold","symbol":"BTC/USD","reasoning":"holding for now"}],
      "next_check_minutes": 30,
      "risk_notes": ""
    }
    """
    with pytest.raises(LLMParseError):
        parse_llm_response(bad)


def test_leverage_out_of_range_rejected():
    bad = """
    {
      "market_assessment": {"trend":"bullish","confidence":0.7,"regime":"trending_up","key_factors":["a"]},
      "actions": [{"action":"open_long","symbol":"BTC/USDT:USDT","size_pct":10,"leverage":50,"stop_loss":1,"reasoning":"way too leveraged"}],
      "next_check_minutes": 30,
      "risk_notes": ""
    }
    """
    with pytest.raises(LLMParseError):
        parse_llm_response(bad)


def test_too_many_actions_rejected():
    """The schema caps action arrays at 4."""
    actions = ", ".join(
        '{"action":"hold","symbol":"BTC/USDT:USDT","reasoning":"holding here"}'
        for _ in range(5)
    )
    bad = f"""
    {{
      "market_assessment": {{"trend":"neutral","confidence":0.5,"regime":"ranging","key_factors":["a"]}},
      "actions": [{actions}],
      "next_check_minutes": 30,
      "risk_notes": ""
    }}
    """
    with pytest.raises(LLMParseError):
        parse_llm_response(bad)


def test_short_long_inverted_stop_loss_rejected():
    """An open_short with stop_loss BELOW entry (and TP also wrong) should fail.

    Note: the existing validator only enforces SL/TP direction when *both*
    are present, so we include both here.
    """
    bad = """
    {
      "market_assessment": {"trend":"bearish","confidence":0.7,"regime":"trending_down","key_factors":["a"]},
      "actions": [{"action":"open_short","symbol":"BTC/USDT:USDT","size_pct":10,"leverage":2,"entry_price":50000,"stop_loss":40000,"take_profit":45000,"reasoning":"shorting the rip"}],
      "next_check_minutes": 30,
      "risk_notes": ""
    }
    """
    with pytest.raises(LLMParseError):
        parse_llm_response(bad)
