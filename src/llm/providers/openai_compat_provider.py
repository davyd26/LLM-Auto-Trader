"""Generic OpenAI-compatible provider.

Covers Ollama, vLLM, OpenRouter, Together, Groq, LM Studio, and anything
else that speaks the ``/v1/chat/completions`` shape. Uses ``httpx`` directly
to keep the dependency footprint small and avoid the OpenAI SDK's stricter
schema checks.

Pricing defaults to zero (you're presumably running locally); set
``LLM_INPUT_PRICE_PER_M`` / ``LLM_OUTPUT_PRICE_PER_M`` for hosted services.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from src.llm.providers.base import (
    BaseLLMProvider,
    LLMCallResult,
    LLMProviderError,
    LLMTimeoutError,
    LLMUsage,
    Pricing,
)


class OpenAICompatProvider(BaseLLMProvider):
    name = "openai-compat"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: int = 60,
        input_price_override: float | None = None,
        cached_price_override: float | None = None,
        output_price_override: float | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
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
        ip, cp, op = self._price_override
        return Pricing(
            input_per_m=ip if ip is not None else 0.0,
            cached_input_per_m=cp if cp is not None else 0.0,
            output_per_m=op if op is not None else 0.0,
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
        url = f"{self._base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
            if response.status_code >= 500:
                raise LLMProviderError(
                    f"OpenAI-compat 5xx: {response.status_code} {response.text[:300]}"
                )
            if response.status_code >= 400:
                raise LLMProviderError(
                    f"OpenAI-compat 4xx: {response.status_code} {response.text[:300]}"
                )
            data = response.json()
        except (httpx.TimeoutException, asyncio.TimeoutError) as e:
            raise LLMTimeoutError(f"OpenAI-compat timeout: {e}") from e
        except (httpx.HTTPError, httpx.RequestError) as e:
            raise LLMProviderError(f"OpenAI-compat HTTP error: {e}") from e

        choices = data.get("choices") or []
        if not choices:
            raise LLMProviderError(
                f"OpenAI-compat returned no choices: {str(data)[:300]}"
            )
        text = choices[0].get("message", {}).get("content") or ""

        usage_dict = data.get("usage") or {}
        return LLMCallResult(
            text=text,
            usage=LLMUsage(
                input_tokens=int(usage_dict.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage_dict.get("completion_tokens", 0) or 0),
                cached_input_tokens=int(usage_dict.get("cached_tokens", 0) or 0),
            ),
            model=self._model,
        )
