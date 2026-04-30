"""CUSUM event filter for event-based LLM triggering (AFML Ch. 2.5).

Fires +1 / -1 when cumulative signed log-returns cross a threshold `h`. `h` can
be fixed (classic CUSUM) or adaptive — `h = k * ewma_std(returns)`. Adaptive
mode keeps sensitivity constant across volatility regimes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _EWMAStd:
    halflife: float
    mean: float = 0.0
    var: float = 0.0
    count: int = 0

    @property
    def alpha(self) -> float:
        return 1.0 - math.exp(-math.log(2.0) / self.halflife)

    def update(self, x: float) -> None:
        if self.count == 0:
            self.mean = x
            self.var = 0.0
        else:
            a = self.alpha
            delta = x - self.mean
            self.mean = self.mean + a * delta
            self.var = (1.0 - a) * (self.var + a * delta * delta)
        self.count += 1

    @property
    def std(self) -> float:
        return math.sqrt(self.var) if self.var > 0 else 0.0


class CUSUMFilter:
    """Symmetric CUSUM filter on log-returns.

    Args:
        h: Fixed absolute threshold. When None, adaptive: `h = k * ewma_std`.
        k: Multiplier for adaptive threshold. Ignored when h is fixed.
        ewma_halflife: Halflife (in updates) for the adaptive std estimator.
        min_periods: Minimum updates before adaptive mode will fire.
    """

    def __init__(
        self,
        h: float | None = 0.02,
        k: float = 2.5,
        ewma_halflife: float = 20.0,
        min_periods: int = 20,
    ) -> None:
        if h is not None and h < 0:
            raise ValueError("h must be non-negative")
        if k < 0:
            raise ValueError("k must be non-negative")
        if ewma_halflife <= 0:
            raise ValueError("ewma_halflife must be positive")

        self._h_fixed = h
        self._k = k
        self._min_periods = min_periods
        self._ewma = _EWMAStd(halflife=ewma_halflife) if h is None else None

        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self._last_price: float | None = None

    @property
    def current_h(self) -> float | None:
        """Current effective threshold (fixed or adaptive)."""
        if self._h_fixed is not None:
            return self._h_fixed
        if self._ewma is None or self._ewma.count < self._min_periods:
            return None
        return self._k * self._ewma.std

    def update(self, price: float) -> int:
        """Feed a new price; return +1/-1 on event, else 0."""
        if self._last_price is None or self._last_price <= 0 or price <= 0:
            self._last_price = price
            return 0

        log_ret = math.log(price / self._last_price)
        self._last_price = price

        if self._ewma is not None:
            self._ewma.update(log_ret)

        h = self.current_h
        if h is None:
            return 0

        self.s_pos = max(0.0, self.s_pos + log_ret)
        self.s_neg = min(0.0, self.s_neg + log_ret)

        if self.s_pos >= h:
            self.s_pos = 0.0
            self.s_neg = 0.0
            return 1
        if self.s_neg <= -h:
            self.s_pos = 0.0
            self.s_neg = 0.0
            return -1
        return 0

    def reset(self) -> None:
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._last_price = None
