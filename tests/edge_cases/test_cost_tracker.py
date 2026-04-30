"""CostTracker edge cases — budget gates, overflow handling, reasoning surcharge."""

from __future__ import annotations

from src.llm.cost_tracker import CostTracker
from src.llm.providers.base import BaseLLMProvider, Pricing


class _StubProvider(BaseLLMProvider):
    name = "stub"

    def __init__(self, pricing: Pricing):
        self._pricing = pricing

    @property
    def model_id(self) -> str:
        return "stub"

    @property
    def pricing(self) -> Pricing:
        return self._pricing

    def estimate_tokens(self, text: str) -> int:
        return 1

    async def complete(self, **kw):  # pragma: no cover
        raise NotImplementedError


def test_from_provider_pulls_pricing():
    p = _StubProvider(Pricing(10.0, 1.0, 50.0))
    ct = CostTracker.from_provider(p, daily_budget=2.0, monthly_budget=20.0)
    assert ct.input_price_per_m == 10.0
    assert ct.cached_price_per_m == 1.0
    assert ct.output_price_per_m == 50.0


def test_reasoning_tokens_are_billed_separately():
    """For OpenAI o-series, reasoning is a separate billed bucket."""
    ct = CostTracker(
        daily_budget=10,
        monthly_budget=100,
        input_price_per_m=2.0,
        cached_price_per_m=0.5,
        output_price_per_m=8.0,
        reasoning_price_per_m=8.0,
    )
    cost = ct.calculate_cost(
        input_tokens=1_000_000,
        cached_tokens=0,
        output_tokens=1_000_000,
        reasoning_tokens=1_000_000,
    )
    # 2 + 8 + 8 = 18.
    assert abs(cost - 18.0) < 1e-9


def test_negative_uncached_input_clamps_to_zero():
    """If `cached_tokens > input_tokens` (provider quirk), don't underflow."""
    ct = CostTracker(input_price_per_m=10, cached_price_per_m=1, output_price_per_m=5)
    cost = ct.calculate_cost(
        input_tokens=100, cached_tokens=200, output_tokens=0
    )
    # Uncached input clamped to 0; cost is purely cached: 200/1M * 1 = 0.0002.
    assert abs(cost - 0.0002) < 1e-9


def test_can_afford_call_blocks_when_daily_exceeded():
    ct = CostTracker(
        daily_budget=0.0001,  # tiny — any call exceeds this
        monthly_budget=100,
        input_price_per_m=15,
        cached_price_per_m=1.5,
        output_price_per_m=75,
    )
    assert not ct.can_afford_call()


def test_summary_includes_all_keys():
    ct = CostTracker()
    s = ct.get_summary()
    for k in ("today_spend", "month_spend", "total_spend", "calls_today",
              "daily_remaining", "monthly_remaining"):
        assert k in s


def test_load_history_with_missing_reasoning_field_defaults_to_zero():
    """Old records on disk won't have reasoning_tokens; load gracefully."""
    ct = CostTracker()
    ct.load_history([
        {"timestamp": 1.0, "input_tokens": 100, "cached_tokens": 0,
         "output_tokens": 50, "cost_usd": 0.001},
    ])
    assert len(ct._records) == 1
    assert ct._records[0].reasoning_tokens == 0
