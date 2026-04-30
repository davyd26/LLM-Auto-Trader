"""LLM provider abstraction tests.

We don't want CI to hit real LLM endpoints, so providers are tested with
mocked clients. Anthropic / OpenAI / Google SDKs are imported lazily, so
their absence is not a test failure — we skip those provider tests when the
SDK isn't installed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.llm.providers.base import (
    BaseLLMProvider,
    LLMCallResult,
    LLMUsage,
    Pricing,
)
from src.llm.providers.factory import build_provider


class _MockProvider(BaseLLMProvider):
    name = "mock"

    def __init__(self, response_text: str = '{"ok":true}'):
        self._text = response_text

    @property
    def model_id(self) -> str:
        return "mock-model"

    @property
    def pricing(self) -> Pricing:
        return Pricing(input_per_m=1.0, cached_input_per_m=0.1, output_per_m=2.0)

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def complete(self, *, system, user, max_output_tokens, temperature=0.0):
        return LLMCallResult(
            text=self._text,
            usage=LLMUsage(
                input_tokens=100, output_tokens=50, cached_input_tokens=20
            ),
            model="mock-model",
        )


def test_factory_unknown_provider_raises():
    settings = SimpleNamespace(provider="quantum-llm-9000", api_key="x", model="")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        build_provider(settings)


def test_factory_routes_anthropic():
    """Default provider routes through Anthropic. Skip if anthropic not installed."""
    pytest.importorskip("anthropic")
    settings = SimpleNamespace(
        provider="anthropic",
        api_key="sk-test-not-real",
        model="claude-opus-4-7",
        timeout_seconds=30,
        extended_thinking=False,
        thinking_budget_tokens=4000,
        request_long_context=False,
        input_price_per_m=0,
        cached_input_price_per_m=0,
        output_price_per_m=0,
    )
    p = build_provider(settings)
    assert p.name == "anthropic"
    assert p.model_id == "claude-opus-4-7"


def test_factory_routes_openai_compat_with_local_url():
    settings = SimpleNamespace(
        provider="openai-compat",
        api_key="not-needed",
        model="qwen2.5:7b",
        base_url="http://localhost:11434/v1",
        timeout_seconds=30,
        input_price_per_m=0,
        cached_input_price_per_m=0,
        output_price_per_m=0,
    )
    p = build_provider(settings)
    assert p.name == "openai-compat"
    assert p.model_id == "qwen2.5:7b"
    # Free local model — pricing zeroed.
    assert p.pricing.input_per_m == 0


def test_pricing_overrides_take_precedence():
    settings = SimpleNamespace(
        provider="openai-compat",
        api_key="x",
        model="m",
        base_url="http://x/v1",
        timeout_seconds=30,
        input_price_per_m=5.0,
        cached_input_price_per_m=0.5,
        output_price_per_m=15.0,
    )
    p = build_provider(settings)
    assert p.pricing.input_per_m == 5.0
    assert p.pricing.output_per_m == 15.0


def test_mock_provider_complete_round_trip():
    """End-to-end: mock provider returns a usage dict the cost tracker can consume."""
    p = _MockProvider()
    res = asyncio.run(
        p.complete(system="sys", user="u", max_output_tokens=100, temperature=0.0)
    )
    assert res.text == '{"ok":true}'
    assert res.usage.input_tokens == 100
    assert res.usage.output_tokens == 50
    assert res.usage.cached_input_tokens == 20


def test_anthropic_provider_uses_extended_thinking_for_opus_47(monkeypatch):
    """Opus 4.7 + extended_thinking should set the thinking kwarg; older models must not."""
    pytest.importorskip("anthropic")
    from src.llm.providers.anthropic_provider import AnthropicProvider

    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"ok":1}')],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_read_input_tokens=0
            ),
        )

    p = AnthropicProvider(
        api_key="sk-not-real",
        model="claude-opus-4-7",
        extended_thinking=True,
        thinking_budget_tokens=4000,
    )
    p._client.messages.create = fake_create  # type: ignore[assignment]

    asyncio.run(
        p.complete(system="sys", user="u", max_output_tokens=200, temperature=0.0)
    )
    assert "thinking" in captured
    assert captured["thinking"]["budget_tokens"] == 4000

    # Older model: thinking must NOT be passed.
    captured.clear()
    p2 = AnthropicProvider(
        api_key="sk-not-real",
        model="claude-sonnet-4-6",
        extended_thinking=True,
    )
    p2._client.messages.create = fake_create  # type: ignore[assignment]
    asyncio.run(
        p2.complete(system="sys", user="u", max_output_tokens=200, temperature=0.0)
    )
    assert "thinking" not in captured


def test_anthropic_provider_long_context_header(monkeypatch):
    """Opus models with request_long_context should send the 1M-context beta."""
    pytest.importorskip("anthropic")
    from src.llm.providers.anthropic_provider import AnthropicProvider

    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(
                input_tokens=1, output_tokens=1, cache_read_input_tokens=0
            ),
        )

    p = AnthropicProvider(
        api_key="sk-not-real",
        model="claude-opus-4-7",
        extended_thinking=False,
        request_long_context=True,
    )
    p._client.messages.create = fake_create  # type: ignore[assignment]
    asyncio.run(p.complete(system="s", user="u", max_output_tokens=10))
    assert captured.get("extra_headers", {}).get("anthropic-beta", "").startswith(
        "context-1m"
    )
