"""Google Gemini provider.

Uses the ``google-genai`` SDK. Gemini does not natively expose a "cached
input" billing tier in the same way Anthropic / OpenAI do, so
``cached_input_tokens`` is always 0; pricing falls through to the input rate.
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
    "gemini-2.5-pro": Pricing(1.25, 1.25, 10.0),
    "gemini-2.5-flash": Pricing(0.30, 0.30, 2.50),
    "gemini-2.0-flash": Pricing(0.10, 0.10, 0.40),
}

_DEFAULT_PRICING = Pricing(1.25, 1.25, 10.0)


class GoogleProvider(BaseLLMProvider):
    name = "google"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: int = 60,
        input_price_override: float | None = None,
        cached_price_override: float | None = None,
        output_price_override: float | None = None,
    ):
        try:
            from google import genai
        except ImportError as e:
            raise LLMProviderError(
                "Google provider requires `pip install llm-auto-trader[google]`"
            ) from e

        self._genai_mod = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout = timeout
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
        return max(1, len(text) // 4)

    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=user,
                    config={
                        "system_instruction": system,
                        "max_output_tokens": max_output_tokens,
                        "temperature": temperature,
                    },
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as e:
            raise LLMTimeoutError(f"Google timeout after {self._timeout}s") from e
        except Exception as e:
            raise LLMProviderError(f"Google error: {e}") from e

        text = getattr(response, "text", "") or ""
        usage = self._extract_usage(response)
        return LLMCallResult(text=text, usage=usage, model=self._model)

    def _extract_usage(self, response: Any) -> LLMUsage:
        u = getattr(response, "usage_metadata", None)
        if u is None:
            return LLMUsage(0, 0)
        return LLMUsage(
            input_tokens=int(getattr(u, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(u, "candidates_token_count", 0) or 0),
            cached_input_tokens=int(getattr(u, "cached_content_token_count", 0) or 0),
        )
