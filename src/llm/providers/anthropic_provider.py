"""Anthropic provider — default. Supports Claude Opus 4.7 with extended
thinking and the 1M-context-window beta.

Prompt caching: a single ephemeral breakpoint on the system prompt. The user
context is re-sent each call (it changes every call), so caching just the
system prompt is the only safe optimization.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.llm.providers.base import (
    BaseLLMProvider,
    LLMCallResult,
    LLMProviderError,
    LLMTimeoutError,
    LLMUsage,
    Pricing,
)

# Pricing in USD per million tokens. Override via env if your tier differs.
# Sources: anthropic.com/pricing as of model release. Cached-input is the
# read-side cache rate (10x discount on Anthropic).
_PRICING: dict[str, Pricing] = {
    # Opus
    "claude-opus-4-7": Pricing(15.0, 1.50, 75.0),
    "claude-opus-4-6-20250219": Pricing(15.0, 1.50, 75.0),
    "claude-opus-4-6": Pricing(15.0, 1.50, 75.0),
    # Sonnet
    "claude-sonnet-4-6": Pricing(3.0, 0.30, 15.0),
    "claude-sonnet-4-5": Pricing(3.0, 0.30, 15.0),
    # Haiku
    "claude-haiku-4-5": Pricing(1.0, 0.10, 5.0),
}

_DEFAULT_PRICING = Pricing(15.0, 1.50, 75.0)


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: int = 60,
        extended_thinking: bool = True,
        thinking_budget_tokens: int = 8000,
        request_long_context: bool = True,
        input_price_override: float | None = None,
        cached_price_override: float | None = None,
        output_price_override: float | None = None,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise LLMProviderError(
                "Anthropic provider requires `pip install llm-auto-trader[anthropic]`"
            ) from e

        self._anthropic_mod = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._extended_thinking = extended_thinking
        self._thinking_budget = thinking_budget_tokens
        self._request_long_context = request_long_context
        self._price_override = (
            input_price_override,
            cached_price_override,
            output_price_override,
        )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def pricing(self) -> Pricing:
        base = _PRICING.get(self._model, _DEFAULT_PRICING)
        ip, cp, op = self._price_override
        return Pricing(
            input_per_m=ip if ip is not None else base.input_per_m,
            cached_input_per_m=cp if cp is not None else base.cached_input_per_m,
            output_per_m=op if op is not None else base.output_per_m,
        )

    def estimate_tokens(self, text: str) -> int:
        # Conservative ~4 chars per token. Used for budgeting only.
        return max(1, len(text) // 4)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user}],
        }

        # Extended thinking — Opus 4.7+ only. Ignored by older models.
        if self._extended_thinking and "opus-4-7" in self._model:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }
            # Anthropic requires temperature=1 with extended thinking.
            kwargs["temperature"] = 1.0

        # 1M context-window beta header.
        extra_headers: dict[str, str] = {}
        if self._request_long_context and "opus-4" in self._model:
            extra_headers["anthropic-beta"] = "context-1m-2025-08-07"
        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        try:
            response = await asyncio.to_thread(
                self._client.messages.create, **kwargs
            )
        except self._anthropic_mod.APITimeoutError as e:
            raise LLMTimeoutError(f"Anthropic timeout: {e}") from e
        except self._anthropic_mod.APIError as e:
            raise LLMProviderError(f"Anthropic error: {e}") from e

        text = self._extract_text(response)
        usage = self._extract_usage(response)
        return LLMCallResult(text=text, usage=usage, model=self._model)

    def _extract_text(self, response: Any) -> str:
        # Extended thinking returns a list of blocks; the user-visible answer
        # is the last text block. Plain calls just have one.
        for block in reversed(getattr(response, "content", []) or []):
            btype = getattr(block, "type", None)
            if btype == "text":
                return getattr(block, "text", "") or ""
        # Fallback for older SDK shapes.
        if response.content and hasattr(response.content[0], "text"):
            return response.content[0].text
        return ""

    def _extract_usage(self, response: Any) -> LLMUsage:
        u = getattr(response, "usage", None)
        if u is None:
            return LLMUsage(0, 0)
        return LLMUsage(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cached_input_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
            reasoning_tokens=0,
        )
