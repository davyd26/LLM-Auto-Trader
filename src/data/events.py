"""Event calendar (Phase 3 §3B.14).

Tracks upcoming macro events (FOMC, CPI, PPI, NFP) + token-specific events
(unlocks, hard forks, exchange listings). Real data sources are plug-in — this
module stores the event list + impact classification + pre-event gating logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Event:
    name: str
    ts: float                    # unix seconds
    impact: str                  # "HIGH" | "MEDIUM" | "LOW"
    category: str                # "macro" | "token_unlock" | "cex" | "other"
    symbols: list[str] = field(default_factory=list)  # empty = affects all


@dataclass
class EventState:
    next_event: Event | None = None
    minutes_until: float = float("inf")
    pre_halt_active: bool = False
    warning_active: bool = False


class EventCalendar:
    """In-memory event calendar with pre-event gating."""

    def __init__(
        self,
        pre_halt_minutes: int = 30,
        warning_hours: int = 4,
    ) -> None:
        self.pre_halt_minutes = int(pre_halt_minutes)
        self.warning_hours = int(warning_hours)
        self._events: list[Event] = []

    def set_events(self, events: list[Event]) -> None:
        """Replace the internal list (sorted ascending)."""
        self._events = sorted(events, key=lambda e: e.ts)

    def upcoming(self, now: float | None = None) -> list[Event]:
        t = now if now is not None else time.time()
        return [e for e in self._events if e.ts >= t]

    def state(self, now: float | None = None, symbol: str | None = None) -> EventState:
        t = now if now is not None else time.time()
        relevant = [
            e for e in self._events
            if e.ts >= t and (not e.symbols or symbol is None or symbol in e.symbols)
        ]
        if not relevant:
            return EventState()

        next_event = relevant[0]
        minutes_until = (next_event.ts - t) / 60.0
        pre_halt = (
            next_event.impact == "HIGH"
            and minutes_until <= self.pre_halt_minutes
        )
        warning = (
            next_event.impact in ("HIGH", "MEDIUM")
            and minutes_until <= self.warning_hours * 60
        )
        return EventState(
            next_event=next_event,
            minutes_until=minutes_until,
            pre_halt_active=pre_halt,
            warning_active=warning,
        )

    def purge_expired(self, now: float | None = None) -> int:
        """Remove events older than now; return how many were removed."""
        t = now if now is not None else time.time()
        before = len(self._events)
        self._events = [e for e in self._events if e.ts >= t]
        return before - len(self._events)
