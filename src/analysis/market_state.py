"""Per-symbol aggregate state for Phase 2 jaam primitives.

One MarketState per bot instance. Holds cached baselines (refreshed daily),
a GARCH state (fit once, updated incrementally), and a risk regime detector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.baselines import Baselines, compute_baselines
from src.analysis.garch import GARCHState, fit_garch_11, garch_state_from_returns, update_garch
from src.analysis.risk_regime import RiskRegimeDetector
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SymbolState:
    baselines: Baselines | None = None
    baselines_refreshed_at: float = 0.0
    garch_state: GARCHState | None = None
    garch_params: tuple[float, float, float] | None = None  # omega, alpha, beta
    last_return: float | None = None


class MarketState:
    """Owns Phase 2 market-regime + baselines + GARCH per symbol."""

    def __init__(self, afml_settings: Any = None) -> None:
        self._afml = afml_settings
        self._symbols: dict[str, SymbolState] = {}
        self.regime = RiskRegimeDetector(
            min_dwell_minutes=float(getattr(afml_settings, "regime_min_dwell_minutes", 30.0))
            if afml_settings is not None else 30.0,
        )
        self._baseline_refresh_s = float(
            getattr(afml_settings, "baseline_refresh_hours", 24) * 3600
            if afml_settings is not None else 86400
        )

    # ---- per-symbol refresh ----

    def refresh_symbol(
        self,
        symbol: str,
        daily_close: pd.Series,
        current_price: float,
        returns_15m: np.ndarray | None = None,
    ) -> SymbolState:
        s = self._symbols.setdefault(symbol, SymbolState())
        now = time.time()

        # Baselines: refresh daily (or when absent)
        if self._baselines_enabled() and (
            s.baselines is None or (now - s.baselines_refreshed_at) > self._baseline_refresh_s
        ):
            try:
                s.baselines = compute_baselines(daily_close, current_price)
                s.baselines_refreshed_at = now
            except Exception as e:
                # Silent-failure-hunter #3 — log so operators see the degradation.
                log.exception("baseline_compute_failed", symbol=symbol, error=str(e))
                s.baselines = None  # make the failure visible via missing section

        # GARCH: fit once on the first sufficiently-long returns window
        if self._garch_enabled() and returns_15m is not None and len(returns_15m) >= 100:
            if s.garch_params is None:
                try:
                    s.garch_params = fit_garch_11(returns_15m)
                    s.garch_state = garch_state_from_returns(returns_15m)
                except Exception as e:
                    log.exception("garch_fit_failed", symbol=symbol, error=str(e))
                    s.garch_params = None
                    s.garch_state = None
            elif s.garch_state is not None and s.last_return is not None:
                try:
                    omega, alpha, beta = s.garch_params
                    new_state = update_garch(s.garch_state, s.last_return, omega, alpha, beta)
                    s.garch_state = new_state
                except Exception as e:
                    log.exception("garch_update_failed", symbol=symbol, error=str(e))
                    # Preserve last-known state so one bad update doesn't wipe it
            # Track last return for next incremental step
            s.last_return = float(returns_15m[-1])

        return s

    # ---- aggregate signals ----

    def regime_update(
        self,
        vol_z: float = 0.0,
        news_surge: float = 0.0,
        sentiment_z: float = 0.0,
        oi_delta: float = 0.0,
    ) -> str:
        """Feed cross-asset signals into the regime detector and return state."""
        return self.regime.update(
            vol_z=vol_z, news_surge=news_surge,
            sentiment_z=sentiment_z, oi_delta=oi_delta,
        )

    @property
    def state(self) -> str:
        return self.regime.state

    def get_symbol(self, symbol: str) -> SymbolState | None:
        return self._symbols.get(symbol)

    # ---- flag helpers ----

    def _baselines_enabled(self) -> bool:
        return self._flag("enable_baselines")

    def _garch_enabled(self) -> bool:
        return self._flag("enable_garch")

    def _flag(self, name: str) -> bool:
        if self._afml is None:
            return False
        return bool(getattr(self._afml, "enable_all", False)
                    and getattr(self._afml, name, False))

    # ---- snapshot for LLM / logging ----

    def snapshot(self, symbol: str) -> dict:
        s = self._symbols.get(symbol)
        out: dict = {"regime": self.regime.state, "regime_confirmations": self.regime.confirmations}
        if s is None:
            return out
        if s.baselines is not None:
            out["baselines"] = s.baselines.to_dict()
        if s.garch_state is not None:
            out["garch"] = s.garch_state.to_dict()
        return out
