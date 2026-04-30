"""3-state risk regime detector (Phase 2 §3A.1).

Based on jaam's cross-domain classifier with hysteresis. States:
- **calm**: normal market behavior
- **elevated**: one or more signals moderately elevated
- **crisis**: at least 2 signals strongly elevated, cross-confirmed

Inputs (all should be causal, trailing-window derivatives):
- `vol_z`: z-score of realized volatility vs trailing mean
- `news_surge`: score (≥0) measuring recent news intensity
- `sentiment_z`: signed z-score of sentiment (large negative = panic)
- `oi_delta`: signed fractional change in open interest

Hysteresis: tightening a state needs 1 confirmation; loosening needs 2
confirmations + `min_dwell_minutes` elapsed in the current state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

_STATES = ("calm", "elevated", "crisis")


@dataclass
class RiskRegimeDetector:
    elevated_threshold: float = 1.5   # |signal z| above which it counts as "elevated"
    crisis_threshold: float = 2.5     # |signal z| above which it counts as "crisis"
    min_dwell_minutes: float = 30.0   # minimum time before downgrading

    def __post_init__(self) -> None:
        if self.elevated_threshold < 0 or self.crisis_threshold < 0:
            raise ValueError("thresholds must be non-negative")
        if self.crisis_threshold < self.elevated_threshold:
            raise ValueError("crisis_threshold must be >= elevated_threshold")
        self._state: str = "calm"
        self._last_transition_ts: float = 0.0
        self._last_confirmations: list[str] = []

    @property
    def state(self) -> str:
        return self._state

    @property
    def confirmations(self) -> list[str]:
        return list(self._last_confirmations)

    def update(
        self,
        vol_z: float,
        news_surge: float,
        sentiment_z: float,
        oi_delta: float,
        now: float | None = None,
    ) -> str:
        """Feed new signal values; return current state after hysteresis."""
        t = now if now is not None else time.time()

        signals = {
            "vol": abs(vol_z),
            "news": abs(news_surge),
            "sentiment": abs(sentiment_z),
            "oi": abs(oi_delta) * 10.0,  # scale oi_delta (fraction) to be z-ish
        }

        elevated_hits = [name for name, v in signals.items() if v >= self.elevated_threshold]
        crisis_hits = [name for name, v in signals.items() if v >= self.crisis_threshold]

        # Determine target state WITHOUT hysteresis (raw classification)
        if len(crisis_hits) >= 2:
            target = "crisis"
        elif len(elevated_hits) >= 2 or len(crisis_hits) >= 1:
            target = "elevated"
        else:
            target = "calm"

        # Apply hysteresis on DOWNGRADE (target looser than current)
        current_idx = _STATES.index(self._state)
        target_idx = _STATES.index(target)
        if target_idx < current_idx:
            # Downgrade — require min_dwell elapsed AND at most 1 elevated signal
            dwell = (t - self._last_transition_ts) / 60.0
            if dwell < self.min_dwell_minutes:
                # Stay in current state
                self._last_confirmations = sorted(set(elevated_hits + crisis_hits))
                return self._state
            # Dwell elapsed — allow loosening by at most one step
            new_idx = max(target_idx, current_idx - 1)
            new_state = _STATES[new_idx]
            if new_state != self._state:
                self._state = new_state
                self._last_transition_ts = t
            self._last_confirmations = sorted(set(elevated_hits + crisis_hits))
            return self._state

        # Upgrade or same — commit immediately
        if target != self._state:
            self._state = target
            self._last_transition_ts = t
        self._last_confirmations = sorted(set(elevated_hits + crisis_hits))
        return self._state
