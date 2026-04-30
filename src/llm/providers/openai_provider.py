"""OpenAI provider — GPT-4.1 / o-series.

Uses the official OpenAI Python SDK. Prompt caching on OpenAI is automatic
when the same prefix is sent within ~10 minutes; ``cached_tokens`` is
returned in ``usage.prompt_tokens_details``.
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

_PRICING: dict[str, Pricing] = {
    "gpt-4.1": Pricing(2.0, 0.50, 8.0),
    "gpt-4.1-mini": Pricing(0.4, 0.10, 1.6),
    "gpt-4o": Pricing(2.5, 1.25, 10.0),
    "o4-mini": Pricing(1.10, 0.275, 4.40, reasoning_per_m=4.40),
    "o3": Pricing(2.0, 0.50, 8.0, reasoning_per_m=8.0),
}

_DEFAULT_PRICING = Pricing(2.0, 0.50, 8.0)


class OpenAIProvider(BaseLLMProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: int = 60,
        base_url: str | None = None,
        input_price_override: float | None = None,
        cached_price_override: float | None = None,
        output_price_override: float | None = None,
    ):
        try:
            import openai
        except ImportError as e:
            raise LLMProviderError(
                "OpenAI provider requires `pip install llm-auto-trader[openai]`"
            ) from e

        self._openai_mod = openai
        client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)
        self._model = model
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
            reasoning_per_m=base.reasoning_per_m,
        )

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        # o-series models (reasoning) reject `temperature` and use
        # `max_completion_tokens` instead of `max_tokens`.
        is_reasoning = self._model.startswith("o") and "-" in self._model

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_output_tokens
        else:
            kwargs["max_tokens"] = max_output_tokens
            kwargs["temperature"] = temperature

        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create, **kwargs
            )
        except self._openai_mod.APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI timeout: {e}") from e
        except self._openai_mod.APIError as e:
            raise LLMProviderError(f"OpenAI error: {e}") from e

        text = response.choices[0].message.content or ""
        usage = self._extract_usage(response)
        return LLMCallResult(text=text, usage=usage, model=self._model)

    def _extract_usage(self, response: Any) -> LLMUsage:
        u = getattr(response, "usage", None)
        if u is None:
            return LLMUsage(0, 0)
        prompt_tokens = int(getattr(u, "prompt_tokens", 0) or 0)
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        completion_details = getattr(u, "completion_tokens_details", None)
        reasoning = 0
        if completion_details is not None:
            reasoning = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
        return LLMUsage(
            input_tokens=prompt_tokens,
            output_tokens=int(getattr(u, "completion_tokens", 0) or 0),
            cached_input_tokens=cached,
            reasoning_tokens=reasoning,
        )
