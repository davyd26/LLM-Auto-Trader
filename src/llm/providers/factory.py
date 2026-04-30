"""Provider factory: maps ``LLMSettings`` to a concrete provider instance.

Imports are lazy so the extras are only required if you actually use them
(``pip install llm-auto-trader[anthropic]`` vs ``[openai]`` vs ``[google]``).
"""

from __future__ import annotations

from src.llm.providers.base import BaseLLMProvider


def build_provider(settings: object) -> BaseLLMProvider:
    """Construct a provider from ``LLMSettings``.

    The ``settings`` argument is duck-typed (we only access a handful of
    attributes) so we don't import ``LLMSettings`` here — that would create
    a circular dependency with ``config.settings``.
    """
    name = (getattr(settings, "provider", "anthropic") or "anthropic").lower()
    api_key = getattr(settings, "api_key", "") or ""
    model = getattr(settings, "model", "") or ""
    base_url = getattr(settings, "base_url", "") or ""
    timeout = int(getattr(settings, "timeout_seconds", 60))

    if name in ("anthropic", "claude"):
        from src.llm.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=api_key,
            model=model or "claude-opus-4-7",
            timeout=timeout,
            extended_thinking=bool(getattr(settings, "extended_thinking", True)),
            thinking_budget_tokens=int(
                getattr(settings, "thinking_budget_tokens", 8000)
            ),
            request_long_context=bool(
                getattr(settings, "request_long_context", True)
            ),
            input_price_override=float(getattr(settings, "input_price_per_m", 0)) or None,
            cached_price_override=float(getattr(settings, "cached_input_price_per_m", 0)) or None,
            output_price_override=float(getattr(settings, "output_price_per_m", 0)) or None,
        )

    if name in ("openai", "gpt"):
        from src.llm.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4.1",
            timeout=timeout,
            base_url=base_url or None,
            input_price_override=float(getattr(settings, "input_price_per_m", 0)) or None,
            cached_price_override=float(getattr(settings, "cached_input_price_per_m", 0)) or None,
            output_price_override=float(getattr(settings, "output_price_per_m", 0)) or None,
        )

    if name in ("google", "gemini"):
        from src.llm.providers.google_provider import GoogleProvider
        return GoogleProvider(
            api_key=api_key,
            model=model or "gemini-2.5-pro",
            timeout=timeout,
            input_price_override=float(getattr(settings, "input_price_per_m", 0)) or None,
            cached_price_override=float(getattr(settings, "cached_input_price_per_m", 0)) or None,
            output_price_override=float(getattr(settings, "output_price_per_m", 0)) or None,
        )

    if name in ("openai-compat", "openai_compat", "ollama", "vllm", "openrouter", "together", "groq"):
        from src.llm.providers.openai_compat_provider import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=api_key or "not-needed",
            model=model or "gpt-4o-mini",
            base_url=base_url or "http://localhost:11434/v1",
            timeout=timeout,
            input_price_override=float(getattr(settings, "input_price_per_m", 0)) or None,
            cached_price_override=float(getattr(settings, "cached_input_price_per_m", 0)) or None,
            output_price_override=float(getattr(settings, "output_price_per_m", 0)) or None,
        )

    raise ValueError(
        f"Unknown LLM provider '{name}'. Set LLM_PROVIDER to one of: "
        f"anthropic, openai, google, openai-compat."
    )
