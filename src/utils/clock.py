"""Exchange-vs-local clock drift check (Phase 3 §3B.5).

Important: failures return `None` (not 0.0) so callers can distinguish
"clock is in sync" from "I could not measure clock drift". A bot should
treat the None case as halt-worthy, not ignore it.
"""

from __future__ import annotations

import time
from typing import Any


async def check_drift_async(exchange: Any) -> float | None:
    """Return drift in seconds (positive = local clock ahead of exchange).

    Returns `None` when drift cannot be determined (no exchange, no
    `fetchTime`, API error). Callers MUST distinguish None from 0.0.
    """
    if exchange is None:
        return None
    fn = getattr(exchange, "fetch_time", None) or getattr(exchange, "fetchTime", None)
    if fn is None:
        return None
    local_ms = time.time() * 1000.0
    try:
        result = fn()
        if hasattr(result, "__await__"):
            exchange_ms = await result
        else:
            exchange_ms = result
        if exchange_ms is None:
            return None
        return (local_ms - float(exchange_ms)) / 1000.0
    except Exception:
        return None


def check_drift_sync(exchange: Any) -> float | None:
    """Synchronous version. Returns `None` on any error."""
    if exchange is None:
        return None
    fn = getattr(exchange, "fetch_time", None) or getattr(exchange, "fetchTime", None)
    if fn is None:
        return None
    local_ms = time.time() * 1000.0
    try:
        exchange_ms = fn()
        if exchange_ms is None:
            return None
        return (local_ms - float(exchange_ms)) / 1000.0
    except Exception:
        return None


def drift_severity(
    drift_seconds: float | None,
    warn_s: float = 2.0,
    halt_s: float = 10.0,
) -> str:
    """Classify drift: 'ok' | 'warn' | 'halt'. `None` → 'halt' (unknown state)."""
    if drift_seconds is None:
        return "halt"
    abs_drift = abs(drift_seconds)
    if abs_drift >= halt_s:
        return "halt"
    if abs_drift >= warn_s:
        return "warn"
    return "ok"
