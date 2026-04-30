"""Adaptive LLM call frequency controller."""

from __future__ import annotations

import time

from src.utils.logger import get_logger

log = get_logger(__name__)


class LLMScheduler:
    """Controls when to make LLM API calls using adaptive frequency.

    The scheduler balances responsiveness against API cost:
    - More frequent when positions are open or volatility is high
    - Less frequent in quiet markets or when budget is tight
    - Respects the LLM's own requested interval (next_check_minutes)
    """

    def __init__(
        self,
        base_interval: int = 900,       # 15 min
        min_interval: int = 300,         # 5 min
        max_interval: int = 3600,        # 60 min
        default_no_position: int = 1800,  # 30 min
    ):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.default_no_position = default_no_position

        self._last_call_time: float = 0
        self._llm_requested_interval: int | None = None
        self._force_next: bool = False

    def set_llm_interval(self, minutes: int) -> None:
        """Set the interval requested by the LLM in its last response."""
        self._llm_requested_interval = max(
            self.min_interval,
            min(minutes * 60, self.max_interval)
        )

    def force_next_call(self) -> None:
        """Force the next call to happen immediately (event-driven)."""
        self._force_next = True

    def record_call(self) -> None:
        """Record that an LLM call was just made."""
        self._last_call_time = time.time()
        self._force_next = False

    @property
    def seconds_since_last_call(self) -> float:
        """Seconds since last LLM call."""
        if self._last_call_time == 0:
            return float("inf")
        return time.time() - self._last_call_time

    def should_call(
        self,
        has_open_position: bool = False,
        price_change_15m_pct: float = 0,
        adx: float = 25,
        budget_ok: bool = True,
    ) -> bool:
        """Determine if it's time for another LLM call.

        Args:
            has_open_position: Whether any positions are currently open
            price_change_15m_pct: Absolute price change in last 15 min
            adx: Current ADX value (trend strength)
            budget_ok: Whether we're within API budget
        """
        elapsed = self.seconds_since_last_call

        # Force call (event-driven trigger)
        if self._force_next and elapsed >= self.min_interval:
            return True

        # Budget exceeded: only call at max interval
        if not budget_ok:
            return elapsed >= self.max_interval

        # Determine target interval
        target = self._compute_interval(
            has_open_position=has_open_position,
            price_change_15m_pct=price_change_15m_pct,
            adx=adx,
        )

        return elapsed >= target

    def _compute_interval(
        self,
        has_open_position: bool,
        price_change_15m_pct: float,
        adx: float,
    ) -> int:
        """Compute the target interval based on market conditions."""
        # LLM-requested interval takes priority (if set)
        if self._llm_requested_interval is not None:
            return self._llm_requested_interval

        # Emergency: large price move
        if abs(price_change_15m_pct) > 2.0:
            return self.min_interval

        # Position open: more frequent monitoring
        if has_open_position:
            return self.base_interval

        # Dead market: check less often
        if adx < 15:
            return self.max_interval

        # Default: no position
        return self.default_no_position

    def get_next_call_in(
        self,
        has_open_position: bool = False,
        price_change_15m_pct: float = 0,
        adx: float = 25,
        budget_ok: bool = True,
    ) -> float:
        """Get seconds until next call should happen."""
        if self._force_next:
            return max(0, self.min_interval - self.seconds_since_last_call)

        if not budget_ok:
            target = self.max_interval
        else:
            target = self._compute_interval(has_open_position, price_change_15m_pct, adx)

        remaining = target - self.seconds_since_last_call
        return max(0, remaining)
