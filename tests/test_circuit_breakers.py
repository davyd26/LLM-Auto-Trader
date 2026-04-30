"""Tests for circuit breaker system."""

import pytest

from src.risk.circuit_breakers import BreakerAction, BreakerType, CircuitBreakers


@pytest.fixture
def breakers():
    return CircuitBreakers(
        max_drawdown_pct=0.05,
        drawdown_warning_pct=0.045,
        daily_loss_limit_pct=0.03,
        rapid_loss_count=3,
        rapid_loss_pct=0.02,
        rapid_loss_window_seconds=3600,
        volatility_spike_multiplier=2.0,
        max_spread_pct=0.001,
        max_funding_rate_8h=0.001,
        ws_disconnect_seconds=60,
        llm_failure_threshold=3,
    )


def _default_kwargs(**overrides):
    defaults = dict(
        current_equity=1000,
        peak_equity=1000,
        daily_pnl_pct=0,
        current_atr_pct=1.0,
        avg_atr_pct=1.0,
        spread_pct=0.0001,
        funding_rate=0.0001,
        ws_seconds_since_update=1,
        llm_consecutive_failures=0,
    )
    defaults.update(overrides)
    return defaults


def test_no_triggers_normal(breakers):
    events = breakers.check_all(**_default_kwargs())
    assert len(events) == 0


def test_drawdown_trigger(breakers):
    events = breakers.check_all(**_default_kwargs(
        current_equity=950, peak_equity=1000  # 5% drawdown
    ))
    assert any(e.breaker_type == BreakerType.DRAWDOWN for e in events)
    assert breakers.is_trading_halted


def test_daily_loss_trigger(breakers):
    events = breakers.check_all(**_default_kwargs(daily_pnl_pct=-0.04))
    assert any(e.breaker_type == BreakerType.DAILY_LOSS for e in events)


def test_volatility_spike_trigger(breakers):
    events = breakers.check_all(**_default_kwargs(
        current_atr_pct=3.0, avg_atr_pct=1.0  # 3x normal
    ))
    assert any(e.breaker_type == BreakerType.VOLATILITY_SPIKE for e in events)
    assert breakers.size_reduction == 0.5


def test_connectivity_trigger(breakers):
    events = breakers.check_all(**_default_kwargs(ws_seconds_since_update=120))
    assert any(e.breaker_type == BreakerType.CONNECTIVITY for e in events)


def test_llm_failure_trigger(breakers):
    events = breakers.check_all(**_default_kwargs(llm_consecutive_failures=3))
    assert any(e.breaker_type == BreakerType.LLM_FAILURE for e in events)


def test_spread_anomaly(breakers):
    events = breakers.check_all(**_default_kwargs(spread_pct=0.005))
    assert any(e.breaker_type == BreakerType.SPREAD_ANOMALY for e in events)


def test_funding_rate_alert(breakers):
    events = breakers.check_all(**_default_kwargs(funding_rate=0.005))
    assert any(e.breaker_type == BreakerType.FUNDING_RATE for e in events)
    # Funding is alert only, not halt
    fr_event = next(e for e in events if e.breaker_type == BreakerType.FUNDING_RATE)
    assert fr_event.action == BreakerAction.ALERT_ONLY


def test_rapid_loss_detection(breakers):
    breakers.record_loss(0.008)
    breakers.record_loss(0.007)
    breakers.record_loss(0.006)  # 3 losses totaling 2.1% > 2%
    events = breakers.check_all(**_default_kwargs())
    assert any(e.breaker_type == BreakerType.RAPID_LOSS for e in events)


def test_margin_warning(breakers):
    events = breakers.check_all(**_default_kwargs(margin_ratio=0.85))
    assert any(e.breaker_type == BreakerType.MARGIN_WARNING for e in events)


def test_trade_allowed_when_normal(breakers):
    allowed, reason = breakers.check_trade_allowed(spread_pct=0.0001)
    assert allowed


def test_trade_blocked_wide_spread(breakers):
    allowed, reason = breakers.check_trade_allowed(spread_pct=0.005)
    assert not allowed
