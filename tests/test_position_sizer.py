"""Tests for position sizing logic."""

import pytest
from src.risk.position_sizer import PositionSizer, minimum_trade_size


@pytest.fixture
def sizer():
    return PositionSizer(taker_fee=0.0001)


def test_basic_sizing(sizer):
    size = sizer.calculate_size(
        equity=1000,
        win_rate=0.55,
        avg_win=15,
        avg_loss=10,
        atr_pct=2.0,
        leverage=2,
        current_price=50000,
    )
    assert size > 0
    assert size <= 250  # max 25% of equity


def test_zero_equity_returns_zero(sizer):
    size = sizer.calculate_size(
        equity=0, win_rate=0.55, avg_win=15, avg_loss=10,
        atr_pct=2.0, leverage=2, current_price=50000,
    )
    assert size == 0


def test_negative_edge_returns_zero(sizer):
    """If win rate is too low, Kelly says don't trade."""
    size = sizer.calculate_size(
        equity=1000,
        win_rate=0.30,  # 30% win rate
        avg_win=10,
        avg_loss=15,    # lose more than win
        atr_pct=2.0,
        leverage=2,
        current_price=50000,
    )
    assert size == 0


def test_exposure_limit(sizer):
    """Should respect remaining exposure room."""
    size = sizer.calculate_size(
        equity=1000,
        win_rate=0.55, avg_win=15, avg_loss=10,
        atr_pct=2.0, leverage=2, current_price=50000,
        existing_exposure_pct=0.55,  # only 5% room left
        max_total_exposure_pct=0.60,
    )
    assert size <= 50  # max 5% of equity


def test_correlation_discount(sizer):
    size_no_corr = sizer.calculate_size(
        equity=5000, win_rate=0.55, avg_win=15, avg_loss=10,
        atr_pct=2.0, leverage=2, current_price=50000,
        correlated_positions=0,
    )
    size_with_corr = sizer.calculate_size(
        equity=5000, win_rate=0.55, avg_win=15, avg_loss=10,
        atr_pct=2.0, leverage=2, current_price=50000,
        correlated_positions=2,
    )
    assert size_with_corr < size_no_corr


def test_kelly_fraction_tiers(sizer):
    """Smaller accounts should use more conservative Kelly fraction."""
    size_1k = sizer.calculate_size(
        equity=1000, win_rate=0.60, avg_win=20, avg_loss=10,
        atr_pct=2.0, leverage=2, current_price=50000,
    )
    size_10k = sizer.calculate_size(
        equity=10000, win_rate=0.60, avg_win=20, avg_loss=10,
        atr_pct=2.0, leverage=2, current_price=50000,
    )
    # 10k account should have proportionally larger position
    assert (size_10k / 10000) > (size_1k / 1000)


def test_minimum_trade_size():
    assert minimum_trade_size(1000) >= 10
    assert minimum_trade_size(500) >= 10
