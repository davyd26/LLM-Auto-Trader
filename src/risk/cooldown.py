"""Per-symbol SL cooldown + chop detector (Phase 3 §3B.9 + §3B.15).

After a stop-loss hit, a symbol is locked for `base_minutes * (1 + consecutive_SL_count)`.
Consecutive SLs within the cooldown window escalate. After N SLs in a rolling
window (default 3 in 4h), the symbol is classified as "choppy" and requires
higher LLM confidence + ADX to re-enter.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _SymbolRecord:
    sl_history: deque = field(default_factory=lambda: deque(maxlen=20))
    current_cooldown_end: float = 0.0

    def consecutive_sls(self, now: float, window_s: float) -> int:
        """Count SLs within `window_s` seconds looking backward from `now`."""
        return sum(1 for ts in self.sl_history if ts >= now - window_s)


class SymbolCooldown:
    """Manages post-SL cooldowns and chop detection per symbol."""

    def __init__(
        self,
        base_minutes: float = 30.0,
        chop_count: int = 3,
        chop_window_hours: float = 4.0,
        chop_lockout_hours: float = 4.0,
    ) -> None:
        self.base_minutes = float(base_minutes)
        self.chop_count = int(chop_count)
        self.chop_window_s = float(chop_window_hours * 3600)
        self.chop_lockout_s = float(chop_lockout_hours * 3600)
        self._records: dict[str, _SymbolRecord] = {}

    def on_stop_out(self, symbol: str, now: float | None = None) -> None:
        """Record an SL hit on this symbol."""
        t = now if now is not None else time.time()
        rec = self._records.setdefault(symbol, _SymbolRecord())
        rec.sl_history.append(t)

        # Compute cooldown duration (escalating with consecutive SLs within window).
        consecutive = rec.consecutive_sls(t, self.chop_window_s)
        cooldown_s = self.base_minutes * 60 * (1 + max(0, consecutive - 1))

        # If choppy, extend to full chop_lockout.
        if consecutive >= self.chop_count:
            cooldown_s = max(cooldown_s, self.chop_lockout_s)

        rec.current_cooldown_end = t + cooldown_s

    def is_locked(self, symbol: str, now: float | None = None) -> tuple[bool, float]:
        """Return `(locked, unlock_timestamp)`."""
        t = now if now is not None else time.time()
        rec = self._records.get(symbol)
        if rec is None:
            return False, 0.0
        if t < rec.current_cooldown_end:
            return True, rec.current_cooldown_end
        return False, rec.current_cooldown_end

    def is_choppy(self, symbol: str, now: float | None = None) -> bool:
        """True when recent SL count in the rolling window meets chop threshold."""
        t = now if now is not None else time.time()
        rec = self._records.get(symbol)
        if rec is None:
            return False
        return rec.consecutive_sls(t, self.chop_window_s) >= self.chop_count

    def seconds_until_unlock(self, symbol: str, now: float | None = None) -> float:
        t = now if now is not None else time.time()
        rec = self._records.get(symbol)
        if rec is None:
            return 0.0
        return max(0.0, rec.current_cooldown_end - t)

    def lockouts_summary(self, now: float | None = None) -> list[dict]:
        """Return per-symbol cooldown status for LLM context."""
        t = now if now is not None else time.time()
        out = []
        for sym, rec in self._records.items():
            if t < rec.current_cooldown_end or self._recent_sl_count(rec, t) > 0:
                out.append({
                    "symbol": sym,
                    "locked": t < rec.current_cooldown_end,
                    "seconds_until_unlock": max(0.0, rec.current_cooldown_end - t),
                    "sl_count_in_window": rec.consecutive_sls(t, self.chop_window_s),
                    "choppy": rec.consecutive_sls(t, self.chop_window_s) >= self.chop_count,
                })
        return out

    @staticmethod
    def _recent_sl_count(rec: _SymbolRecord, now: float) -> int:
        return sum(1 for ts in rec.sl_history if ts >= now - 86400)
