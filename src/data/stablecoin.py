"""USDT de-peg monitor (Phase 3 §3B.13).

Watches USDT price vs $1 across multiple sources. When deviation exceeds
threshold for a sustained period, flags 'stressed' state so the risk layer
blocks new entries and urges closure.

Sources in preferred order: Chainlink-style on-chain oracle (if available),
Coinbase USDT-USD spot, Kraken USDT-USD. A real integration uses their public
APIs; this module provides the aggregation logic with a pluggable fetcher.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class USDTStatus:
    price: float              # median across sources
    sources: dict[str, float] # per-source prices
    off_peg: bool
    sustained: bool           # off-peg for N consecutive readings
    staleness_s: float


@dataclass
class USDTMonitor:
    """Stateful monitor; call `record()` with multi-source prices each tick."""

    threshold: float = 0.005           # 0.5% off $1
    sustain_count: int = 3
    history: deque = field(default_factory=lambda: deque(maxlen=32))
    last_status: USDTStatus | None = None
    last_update: float = 0.0

    def record(
        self, sources: dict[str, float], now: float | None = None,
        max_staleness_s: float = 300.0,
    ) -> USDTStatus:
        """Update the monitor with fresh multi-source prices.

        If sources are empty and the last successful reading is older than
        `max_staleness_s`, flip to off_peg (silent-failure-hunter #8: we must
        NOT silently preserve stale "on peg" status during a data outage).
        """
        t = now if now is not None else time.time()
        valid_prices = [float(p) for p in sources.values() if p and p > 0]
        if not valid_prices:
            if self.last_status is not None:
                staleness = t - self.last_update
                # After staleness threshold, force off-peg as a halt signal.
                force_off_peg = staleness > max_staleness_s
                self.last_status = USDTStatus(
                    price=self.last_status.price,
                    sources=self.last_status.sources,
                    off_peg=self.last_status.off_peg or force_off_peg,
                    sustained=self.last_status.sustained or force_off_peg,
                    staleness_s=staleness,
                )
                return self.last_status
            return USDTStatus(price=1.0, sources={}, off_peg=False, sustained=False, staleness_s=0.0)

        valid_prices.sort()
        median = valid_prices[len(valid_prices) // 2]
        off_peg = abs(median - 1.0) >= self.threshold
        self.history.append(off_peg)
        sustained = (
            len(self.history) >= self.sustain_count
            and all(list(self.history)[-self.sustain_count:])
        )
        self.last_update = t
        status = USDTStatus(
            price=round(median, 5), sources=dict(sources),
            off_peg=off_peg, sustained=sustained, staleness_s=0.0,
        )
        self.last_status = status
        return status


def default_fetcher_stub() -> dict[str, float]:
    """Placeholder fetcher — real integration wires Coinbase/Kraken APIs."""
    return {}
