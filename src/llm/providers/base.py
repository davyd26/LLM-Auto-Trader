"""LLM provider interface.

Every concrete provider implements ``complete()`` and exposes a ``Pricing``
table. The bot calls ``complete()`` from one place — ``TradingLLMClient`` —
which means swapping providers is a config change, not a code change.

We deliberately do NOT abstract away streaming, tool use, batch APIs, or
anything else the trading loop doesn't need: the provider returns one text
blob plus token usage, and the response parser handles the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMProviderError(Exception):
    """Wraps any provider-side failure with a uniform type."""


class LLMTimeoutError(LLMProviderError):
    """Provider request timed out."""


@dataclass(frozen=True)
class LLMUsage:
    """Token counts for a single call.

    ``cached_input_tokens`` is provider-dependent: Anthropic surfaces it
    when a cache breakpoint hits, OpenAI surfaces it via ``cached_tokens``,
    and most other providers return 0.
    """

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class LLMCallResult:
    """One completion call's return value."""

    text: str
    usage: LLMUsage
    model: str


@dataclass(frozen=True)
class Pricing:
    """Per-million-token pricing in USD.

    ``cached_input_per_m`` is typically ~10× cheaper than ``input_per_m``
    for providers that support prompt caching (Anthropic, OpenAI). Set it
    equal to ``input_per_m`` if your provider doesn't.

    ``reasoning_per_m`` covers the hidden-thinking surcharge that some
    extended-reasoning models bill (e.g. OpenAI o-series). Most providers
    bundle reasoning into ``output_per_m`` and leave this at 0.
    """

    input_per_m: float
    cached_input_per_m: float
    output_per_m: float
    reasoning_per_m: float = 0.0


class BaseLLMProvider(ABC):
    """Common surface for any LLM the trading bot can drive."""

    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int,
        temperature: float = 0.0,
    ) -> LLMCallResult:
        """Send a single prompt; return the text response and token usage."""

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Cheap, offline estimate used for budgeting only."""

    @property
    @abstractmethod
    def pricing(self) -> Pricing:
        """USD per million tokens. Override via env if your tier differs."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The model string the provider should report back."""
