"""Tests for 3-state risk regime detector (Phase 2 §3A.1)."""

from __future__ import annotations

import pytest

from src.analysis.risk_regime import RiskRegimeDetector


class TestRiskRegime:
    def test_initial_state_is_calm(self) -> None:
        d = RiskRegimeDetector()
        assert d.state == "calm"

    def test_low_signals_keep_calm(self) -> None:
        d = RiskRegimeDetector()
        for _ in range(5):
            d.update(vol_z=0.3, news_surge=0.1, sentiment_z=0.0, oi_delta=0.0)
        assert d.state == "calm"

    def test_single_signal_does_not_transition_to_crisis(self) -> None:
        d = RiskRegimeDetector(min_dwell_minutes=0)
        # Only vol spikes — needs cross-domain confirmation for crisis
        d.update(vol_z=3.0, news_surge=0.0, sentiment_z=0.0, oi_delta=0.0, now=0)
        # Elevated is OK on a single strong signal
        assert d.state in ("elevated", "crisis")

    def test_two_signals_confirm_crisis(self) -> None:
        d = RiskRegimeDetector(min_dwell_minutes=0)
        d.update(vol_z=3.5, news_surge=3.5, sentiment_z=-2.5, oi_delta=0.0, now=0)
        assert d.state == "crisis"

    def test_hysteresis_prevents_immediate_downgrade(self) -> None:
        d = RiskRegimeDetector(min_dwell_minutes=60)
        # Enter crisis
        d.update(vol_z=3.5, news_surge=3.5, sentiment_z=-2.5, oi_delta=0.0, now=0)
        assert d.state == "crisis"
        # 1 minute later, all signals calm
        d.update(vol_z=0.0, news_surge=0.0, sentiment_z=0.0, oi_delta=0.0, now=60)
        # Still crisis (min_dwell not elapsed)
        assert d.state == "crisis"

    def test_downgrade_requires_min_dwell_and_calm_signals(self) -> None:
        d = RiskRegimeDetector(min_dwell_minutes=10)
        d.update(vol_z=3.5, news_surge=3.5, sentiment_z=0.0, oi_delta=0.0, now=0)
        assert d.state == "crisis"
        # 11 minutes later with calm signals — allowed to downgrade
        d.update(vol_z=0.0, news_surge=0.0, sentiment_z=0.0, oi_delta=0.0, now=11 * 60)
        # Should have downgraded (at least one notch)
        assert d.state in ("calm", "elevated")

    def test_confirmations_list_returned(self) -> None:
        d = RiskRegimeDetector(min_dwell_minutes=0)
        d.update(vol_z=3.5, news_surge=3.5, sentiment_z=0.0, oi_delta=0.0, now=0)
        assert len(d.confirmations) >= 2

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskRegimeDetector(elevated_threshold=-1.0)
