"""LLM client wrapper for trading decisions.

Provider-agnostic: swaps Anthropic / OpenAI / Google / any OpenAI-compat
endpoint via ``LLMSettings.provider``. Keeps the prompt-caching, retry,
budget-gate, and memory hooks the rest of the bot depends on.
"""

from __future__ import annotations

from src.llm.context_builder import ContextBuilder
from src.llm.cost_tracker import CostTracker
from src.llm.memory import TradingMemory
from src.llm.prompts import build_user_prompt, render_system_prompt, SYSTEM_PROMPT
from src.llm.providers.base import (
    BaseLLMProvider,
    LLMProviderError,
    LLMTimeoutError as ProviderTimeoutError,
)
from src.llm.providers.factory import build_provider
from src.llm.response_parser import (
    LLMResponse,
    format_retry_prompt,
    parse_llm_response,
)
from src.llm.scheduler import LLMScheduler
from src.data.aggregator import MarketSnapshot
from src.utils.errors import (
    LLMBudgetExceededError,
    LLMParseError,
    LLMTimeoutError,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class TradingLLMClient:
    """Drives one provider through a single ``analyze()`` per cadence tick."""

    def __init__(
        self,
        *,
        provider: BaseLLMProvider | None = None,
        llm_settings: object | None = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        max_retries: int = 2,
        cost_tracker: CostTracker | None = None,
        memory: TradingMemory | None = None,
        scheduler: LLMScheduler | None = None,
        afml_settings: object | None = None,
        market_state: object | None = None,
        symbol_cooldown: object | None = None,
        event_calendar: object | None = None,
        usdt_monitor: object | None = None,
        daily_peak: object | None = None,
        margin_monitor: object | None = None,
        system_prompt: str | None = None,
    ):
        if provider is None:
            if llm_settings is None:
                raise ValueError(
                    "TradingLLMClient needs either a `provider` or `llm_settings`."
                )
            provider = build_provider(llm_settings)

        self._provider = provider
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._max_retries = max_retries
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        self._context_builder = ContextBuilder(
            afml_settings=afml_settings,
            market_state=market_state,
            symbol_cooldown=symbol_cooldown,
            event_calendar=event_calendar,
            usdt_monitor=usdt_monitor,
            daily_peak=daily_peak,
            margin_monitor=margin_monitor,
        )

        self.cost_tracker = cost_tracker or CostTracker.from_provider(provider)
        self.memory = memory or TradingMemory()
        self.scheduler = scheduler or LLMScheduler()
        self._afml = afml_settings
        self._market_state = market_state

        self._consecutive_failures = 0

    @classmethod
    def for_settings(cls, settings: object, **kwargs) -> "TradingLLMClient":
        """Build a client straight from a ``Settings`` object."""
        llm = getattr(settings, "llm", settings)
        return cls(llm_settings=llm, **kwargs)

    @property
    def provider(self) -> BaseLLMProvider:
        return self._provider

    @property
    def model(self) -> str:
        return self._provider.model_id

    async def analyze(self, snapshot: MarketSnapshot) -> LLMResponse:
        """Send the snapshot, get a parsed decision back."""
        if not self.cost_tracker.can_afford_call():
            raise LLMBudgetExceededError(
                f"Budget exhausted. Today: ${self.cost_tracker.today_spend:.2f}/"
                f"${self.cost_tracker.daily_budget:.2f}"
            )

        sections = self._context_builder.build(
            snapshot=snapshot,
            trade_history=self.memory.get_recent_trades(),
            performance=self.memory.get_performance_summary(
                snapshot.balance.get("total", 0),
                afml_settings=self._afml,
            ),
            previous_decision=self.memory.get_previous_decision(),
        )
        user_prompt = build_user_prompt(sections)

        text = await self._call_api(user_prompt)

        try:
            result = parse_llm_response(text)
            self._consecutive_failures = 0
        except LLMParseError as e:
            result = await self._retry_parse(user_prompt, e)

        self.scheduler.record_call()
        self.scheduler.set_llm_interval(result.next_check_minutes)

        if result.actions:
            primary = result.actions[0]
            self.memory.record_decision(
                action=primary.action.value,
                symbol=primary.symbol,
                reasoning=primary.reasoning,
            )
        else:
            self.memory.record_decision(
                action="hold",
                symbol="N/A",
                reasoning=result.risk_notes or "No action taken",
            )

        log.info(
            "llm_analysis_complete",
            actions=[a.action.value for a in result.actions],
            trend=result.market_assessment.trend,
            confidence=result.market_assessment.confidence,
            next_check=result.next_check_minutes,
        )
        return result

    async def _call_api(self, user_prompt: str) -> str:
        try:
            res = await self._provider.complete(
                system=self._system_prompt,
                user=user_prompt,
                max_output_tokens=self._max_output_tokens,
                temperature=self._temperature,
            )
        except ProviderTimeoutError as e:
            self._consecutive_failures += 1
            raise LLMTimeoutError(str(e)) from e
        except LLMProviderError as e:
            self._consecutive_failures += 1
            raise LLMParseError(str(e)) from e

        self.cost_tracker.record_call(
            input_tokens=res.usage.input_tokens,
            cached_tokens=res.usage.cached_input_tokens,
            output_tokens=res.usage.output_tokens,
            reasoning_tokens=res.usage.reasoning_tokens,
        )
        return res.text

    async def _retry_parse(
        self, original_prompt: str, error: LLMParseError
    ) -> LLMResponse:
        for attempt in range(self._max_retries):
            retry_prompt = (
                f"{original_prompt}\n\n"
                f"---\n{format_retry_prompt(error)}"
            )
            try:
                raw = await self._call_api(retry_prompt)
                result = parse_llm_response(raw)
                self._consecutive_failures = 0
                return result
            except LLMParseError as e:
                error = e
                log.warning(
                    "llm_retry_failed", attempt=attempt + 1, error=str(e)
                )

        self._consecutive_failures += 1
        raise error

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures


def render_system_prompt_for(settings: object) -> str:
    """Helper for ``TradingBot``: build a system prompt from Settings."""
    profile = settings.exchange.profile
    return render_system_prompt(
        exchange_name=profile.name,
        quote_currency=profile.quote_currency,
        maker_fee=profile.maker_fee,
        taker_fee=profile.taker_fee,
        max_leverage=settings.risk.max_leverage,
    )
