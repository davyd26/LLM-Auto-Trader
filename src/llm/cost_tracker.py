"""API cost tracking and budget enforcement.

Pricing is provider-driven: ``CostTracker.from_provider()`` pulls the rate
table off whatever ``BaseLLMProvider`` is in use. Manual rates still work
for tests and unusual setups.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CallRecord:
    timestamp: float
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float


class CostTracker:
    """Tracks LLM API costs and enforces daily/monthly budgets."""

    def __init__(
        self,
        daily_budget: float = 5.00,
        monthly_budget: float = 100.0,
        input_price_per_m: float = 15.0,
        cached_price_per_m: float = 1.50,
        output_price_per_m: float = 75.0,
        reasoning_price_per_m: float = 0.0,
    ):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.input_price_per_m = input_price_per_m
        self.cached_price_per_m = cached_price_per_m
        self.output_price_per_m = output_price_per_m
        self.reasoning_price_per_m = reasoning_price_per_m

        self._records: list[CallRecord] = []

    @classmethod
    def from_provider(
        cls,
        provider,
        *,
        daily_budget: float = 5.00,
        monthly_budget: float = 100.0,
    ) -> CostTracker:
        """Pull the pricing table from a provider's ``pricing`` property."""
        p = provider.pricing
        return cls(
            daily_budget=daily_budget,
            monthly_budget=monthly_budget,
            input_price_per_m=p.input_per_m,
            cached_price_per_m=p.cached_input_per_m,
            output_price_per_m=p.output_per_m,
            reasoning_price_per_m=p.reasoning_per_m,
        )

    def calculate_cost(
        self,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> float:
        uncached_input = max(0, input_tokens - cached_tokens)
        cost = (
            (uncached_input / 1_000_000) * self.input_price_per_m
            + (cached_tokens / 1_000_000) * self.cached_price_per_m
            + (output_tokens / 1_000_000) * self.output_price_per_m
            + (reasoning_tokens / 1_000_000) * self.reasoning_price_per_m
        )
        return cost

    def record_call(
        self,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> float:
        cost = self.calculate_cost(
            input_tokens, cached_tokens, output_tokens, reasoning_tokens
        )
        record = CallRecord(
            timestamp=time.time(),
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
        )
        self._records.append(record)

        log.info(
            "llm_call_cost",
            cost_usd=round(cost, 4),
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            daily_total=round(self.today_spend, 4),
        )
        return cost

    @property
    def today_spend(self) -> float:
        today_start = _day_start_ts()
        return sum(r.cost_usd for r in self._records if r.timestamp >= today_start)

    @property
    def month_spend(self) -> float:
        month_start = _month_start_ts()
        return sum(r.cost_usd for r in self._records if r.timestamp >= month_start)

    @property
    def total_spend(self) -> float:
        return sum(r.cost_usd for r in self._records)

    @property
    def calls_today(self) -> int:
        today_start = _day_start_ts()
        return sum(1 for r in self._records if r.timestamp >= today_start)

    def can_afford_call(
        self,
        estimated_input: int = 5000,
        estimated_output: int = 400,
    ) -> bool:
        """Forecast whether the next call fits inside both budgets."""
        estimated_cost = self.calculate_cost(estimated_input, 1500, estimated_output)
        return (
            self.today_spend + estimated_cost <= self.daily_budget
            and self.month_spend + estimated_cost <= self.monthly_budget
        )

    @property
    def daily_budget_remaining(self) -> float:
        return max(0.0, self.daily_budget - self.today_spend)

    @property
    def monthly_budget_remaining(self) -> float:
        return max(0.0, self.monthly_budget - self.month_spend)

    def get_summary(self) -> dict:
        return {
            "today_spend": round(self.today_spend, 4),
            "month_spend": round(self.month_spend, 4),
            "total_spend": round(self.total_spend, 4),
            "calls_today": self.calls_today,
            "daily_remaining": round(self.daily_budget_remaining, 4),
            "monthly_remaining": round(self.monthly_budget_remaining, 4),
        }

    def load_history(self, records: list[dict]) -> None:
        for r in records:
            self._records.append(CallRecord(
                timestamp=r["timestamp"],
                input_tokens=r.get("input_tokens", 0),
                cached_tokens=r.get("cached_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
                reasoning_tokens=r.get("reasoning_tokens", 0),
                cost_usd=r.get("cost_usd", 0),
            ))

    def cleanup_old_records(self, days: int = 90) -> None:
        cutoff = time.time() - (days * 86400)
        self._records = [r for r in self._records if r.timestamp >= cutoff]


def _day_start_ts() -> float:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp()


def _month_start_ts() -> float:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp()
