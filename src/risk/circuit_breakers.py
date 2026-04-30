"""Circuit breakers: emergency stops independent of LLM."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


class BreakerType(str, Enum):
    DRAWDOWN = "drawdown"
    DAILY_LOSS = "daily_loss"
    RAPID_LOSS = "rapid_loss"
    VOLATILITY_SPIKE = "volatility_spike"
    CONNECTIVITY = "connectivity"
    LLM_FAILURE = "llm_failure"
    SPREAD_ANOMALY = "spread_anomaly"
    FUNDING_RATE = "funding_rate"
    MARGIN_WARNING = "margin_warning"


class BreakerAction(str, Enum):
    CLOSE_ALL = "close_all"         # Close all positions immediately
    HALT_NEW = "halt_new"           # Stop opening new positions
    REDUCE_SIZE = "reduce_size"     # Reduce position sizes by 50%
    TIGHTEN_STOPS = "tighten_stops"  # Move stops to breakeven
    ALERT_ONLY = "alert_only"       # Just send alert


@dataclass
class BreakerEvent:
    """Record of a circuit breaker trigger."""

    breaker_type: BreakerType
    action: BreakerAction
    reason: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)


class CircuitBreakers:
    """Independent circuit breaker system that operates without LLM."""

    def __init__(
        self,
        max_drawdown_pct: float = 0.05,
        drawdown_warning_pct: float = 0.045,
        daily_loss_limit_pct: float = 0.03,
        rapid_loss_count: int = 3,
        rapid_loss_pct: float = 0.02,
        rapid_loss_window_seconds: int = 3600,
        volatility_spike_multiplier: float = 2.0,
        max_spread_pct: float = 0.001,
        max_funding_rate_8h: float = 0.001,
        ws_disconnect_seconds: int = 60,
        llm_failure_threshold: int = 3,
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.drawdown_warning_pct = drawdown_warning_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.rapid_loss_count = rapid_loss_count
        self.rapid_loss_pct = rapid_loss_pct
        self.rapid_loss_window_seconds = rapid_loss_window_seconds
        self.volatility_spike_multiplier = volatility_spike_multiplier
        self.max_spread_pct = max_spread_pct
        self.max_funding_rate_8h = max_funding_rate_8h
        self.ws_disconnect_seconds = ws_disconnect_seconds
        self.llm_failure_threshold = llm_failure_threshold

        self._events: list[BreakerEvent] = []
        self._recent_losses: list[tuple[float, float]] = []  # (timestamp, loss_pct)
        self._trading_halted = False
        self._halt_until: float = 0
        self._size_reduction_factor: float = 1.0

    @property
    def is_trading_halted(self) -> bool:
        """Check if trading is currently halted."""
        if self._halt_until > 0 and time.time() > self._halt_until:
            self._trading_halted = False
            self._halt_until = 0
        return self._trading_halted

    @property
    def size_reduction(self) -> float:
        """Current position size reduction factor (1.0 = normal, 0.5 = halved)."""
        return self._size_reduction_factor

    def check_all(
        self,
        current_equity: float,
        peak_equity: float,
        daily_pnl_pct: float,
        current_atr_pct: float,
        avg_atr_pct: float,
        spread_pct: float,
        funding_rate: float,
        ws_seconds_since_update: float,
        llm_consecutive_failures: int,
        margin_ratio: float | None = None,
    ) -> list[BreakerEvent]:
        """Run all circuit breaker checks.

        Returns list of triggered breaker events.
        """
        triggered: list[BreakerEvent] = []

        # --- 1. Portfolio Drawdown ---
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown >= self.drawdown_warning_pct:
                action = BreakerAction.CLOSE_ALL
                event = BreakerEvent(
                    breaker_type=BreakerType.DRAWDOWN,
                    action=action,
                    reason=f"Drawdown {drawdown*100:.1f}% exceeds warning level {self.drawdown_warning_pct*100:.1f}%",
                    details={"drawdown_pct": drawdown, "equity": current_equity, "peak": peak_equity},
                )
                triggered.append(event)
                self._halt_trading(3600)  # halt for 1 hour
                log.critical("circuit_breaker_drawdown", drawdown=drawdown, equity=current_equity)

        # --- 2. Daily Loss ---
        if daily_pnl_pct < -self.daily_loss_limit_pct:
            event = BreakerEvent(
                breaker_type=BreakerType.DAILY_LOSS,
                action=BreakerAction.HALT_NEW,
                reason=f"Daily loss {daily_pnl_pct*100:.1f}% exceeds limit {self.daily_loss_limit_pct*100:.1f}%",
                details={"daily_pnl_pct": daily_pnl_pct},
            )
            triggered.append(event)
            self._halt_trading(3600)
            log.warning("circuit_breaker_daily_loss", daily_pnl=daily_pnl_pct)

        # --- 3. Rapid Loss ---
        rapid_event = self._check_rapid_loss()
        if rapid_event:
            triggered.append(rapid_event)

        # --- 4. Volatility Spike ---
        if avg_atr_pct > 0 and current_atr_pct > avg_atr_pct * self.volatility_spike_multiplier:
            event = BreakerEvent(
                breaker_type=BreakerType.VOLATILITY_SPIKE,
                action=BreakerAction.REDUCE_SIZE,
                reason=f"ATR {current_atr_pct:.2f}% is {current_atr_pct/avg_atr_pct:.1f}x average",
                details={"current_atr": current_atr_pct, "avg_atr": avg_atr_pct},
            )
            triggered.append(event)
            self._size_reduction_factor = 0.5
            log.warning("circuit_breaker_volatility", atr=current_atr_pct, avg=avg_atr_pct)

        # --- 5. Connectivity ---
        if ws_seconds_since_update > self.ws_disconnect_seconds:
            event = BreakerEvent(
                breaker_type=BreakerType.CONNECTIVITY,
                action=BreakerAction.CLOSE_ALL,
                reason=f"WebSocket disconnected for {ws_seconds_since_update:.0f}s",
                details={"seconds": ws_seconds_since_update},
            )
            triggered.append(event)
            log.critical("circuit_breaker_connectivity", seconds=ws_seconds_since_update)

        # --- 6. LLM Failure ---
        if llm_consecutive_failures >= self.llm_failure_threshold:
            event = BreakerEvent(
                breaker_type=BreakerType.LLM_FAILURE,
                action=BreakerAction.TIGHTEN_STOPS,
                reason=f"LLM failed {llm_consecutive_failures} consecutive calls",
                details={"failures": llm_consecutive_failures},
            )
            triggered.append(event)
            self._halt_trading(900)  # halt new trades for 15 min
            log.warning("circuit_breaker_llm_failure", failures=llm_consecutive_failures)

        # --- 7. Spread Anomaly ---
        if spread_pct > self.max_spread_pct:
            event = BreakerEvent(
                breaker_type=BreakerType.SPREAD_ANOMALY,
                action=BreakerAction.HALT_NEW,
                reason=f"Spread {spread_pct*100:.3f}% exceeds max {self.max_spread_pct*100:.3f}%",
                details={"spread_pct": spread_pct},
            )
            triggered.append(event)

        # --- 8. Funding Rate ---
        if abs(funding_rate) > self.max_funding_rate_8h:
            event = BreakerEvent(
                breaker_type=BreakerType.FUNDING_RATE,
                action=BreakerAction.ALERT_ONLY,
                reason=f"Funding rate {funding_rate*100:.4f}% exceeds threshold",
                details={"funding_rate": funding_rate},
            )
            triggered.append(event)

        # --- 9. Margin Warning ---
        if margin_ratio is not None and margin_ratio > 0.8:
            event = BreakerEvent(
                breaker_type=BreakerType.MARGIN_WARNING,
                action=BreakerAction.CLOSE_ALL,
                reason=f"Margin ratio {margin_ratio*100:.0f}% approaching liquidation",
                details={"margin_ratio": margin_ratio},
            )
            triggered.append(event)
            log.critical("circuit_breaker_margin", ratio=margin_ratio)

        self._events.extend(triggered)
        return triggered

    def record_loss(self, loss_pct: float) -> None:
        """Record a trade loss for rapid loss detection."""
        self._recent_losses.append((time.time(), abs(loss_pct)))
        # Cleanup old entries
        cutoff = time.time() - self.rapid_loss_window_seconds
        self._recent_losses = [(t, l) for t, l in self._recent_losses if t > cutoff]

    def reset_volatility_reduction(self) -> None:
        """Reset size reduction factor after volatility normalizes."""
        self._size_reduction_factor = 1.0

    def check_trade_allowed(self, spread_pct: float = 0) -> tuple[bool, str]:
        """Quick check if a new trade is currently allowed."""
        if self.is_trading_halted:
            return False, "Trading halted by circuit breaker"
        if spread_pct > self.max_spread_pct:
            return False, f"Spread too wide: {spread_pct*100:.3f}%"
        return True, "OK"

    def get_recent_events(self, limit: int = 10) -> list[BreakerEvent]:
        """Get recent circuit breaker events."""
        return self._events[-limit:]

    def _halt_trading(self, duration_seconds: int) -> None:
        self._trading_halted = True
        self._halt_until = time.time() + duration_seconds

    def _check_rapid_loss(self) -> BreakerEvent | None:
        cutoff = time.time() - self.rapid_loss_window_seconds
        recent = [(t, l) for t, l in self._recent_losses if t > cutoff]

        if len(recent) >= self.rapid_loss_count:
            total_loss = sum(l for _, l in recent)
            if total_loss >= self.rapid_loss_pct:
                self._halt_trading(1800)  # halt 30 min
                return BreakerEvent(
                    breaker_type=BreakerType.RAPID_LOSS,
                    action=BreakerAction.HALT_NEW,
                    reason=f"{len(recent)} losses totaling {total_loss*100:.1f}% in {self.rapid_loss_window_seconds//60}min",
                    details={"count": len(recent), "total_pct": total_loss},
                )
        return None
