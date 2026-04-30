"""Daily peak equity anchoring (Phase 3 §3B.12).

Tracks equity peak at two time scales:
- **session_peak**: all-time-high since bot started (drives hard CLOSE_ALL
  circuit breaker — "we're in a real drawdown")
- **daily_peak**: high-water mark since 00:00 UTC today (drives HALT_NEW
  circuit breaker — "bad day so far, don't dig deeper")

Separating the two fixes the false-trigger mode where a profitable previous
day's peak makes today's modest loss look like a deep drawdown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class DailyPeakTracker:
    session_peak: float = 0.0
    daily_peak: float = 0.0
    daily_anchor_day: str = ""
    starting_daily_equity: float = 0.0

    def update(self, equity: float, now: float | None = None) -> None:
        t = now if now is not None else time.time()
        today = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")

        # Daily roll-over
        if today != self.daily_anchor_day:
            self.daily_anchor_day = today
            self.daily_peak = equity
            self.starting_daily_equity = equity

        # Update both peaks
        if equity > self.session_peak:
            self.session_peak = equity
        if equity > self.daily_peak:
            self.daily_peak = equity

    def session_drawdown(self, equity: float) -> float:
        if self.session_peak <= 0:
            return 0.0
        return max(0.0, (self.session_peak - equity) / self.session_peak)

    def daily_drawdown(self, equity: float) -> float:
        if self.daily_peak <= 0:
            return 0.0
        return max(0.0, (self.daily_peak - equity) / self.daily_peak)
