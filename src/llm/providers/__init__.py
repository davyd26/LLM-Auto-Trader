"""LLM provider abstraction layer.

The bot's decision loop talks to ``BaseLLMProvider``; concrete providers
adapt this interface to Anthropic / OpenAI / Google / any OpenAI-compatible
HTTP API (Ollama, vLLM, OpenRouter, Together, ...).

Providers are imported lazily so that installing only the ``[anthropic]``
extra doesn't pull in OpenAI / Google SDKs.
"""

from src.llm.providers.base import (
    BaseLLMProvider,
    LLMCallResult,
    LLMUsage,
    Pricing,
    LLMProviderError,
)
from src.llm.providers.factory import build_provider

__all__ = [
    "BaseLLMProvider",
    "LLMCallResult",
    "LLMUsage",
    "Pricing",
    "LLMProviderError",
    "build_provider",
]
