"""Tests for gap risk and volatility-aware slippage in the fill model.

The default fill model assumes stops fill at exactly the stop price — but
real-world fills can be much worse:
    - Gap risk: when a bar's open prints beyond the stop, the actual fill
      happens at the open, not at the stop level. This is common over
      weekends or after major news.
    - Vol-aware slippage: in high-volatility bars (large range/open), the
      effective slippage is worse than baseline.

These two effects matter most precisely when the strategy is hurting (stops
being hit during stress) — exactly when "very very very realistic" reporting
matters most.

Edge cases covered:
    - Long stop with gap-down open below stop → fill at open
    - Long stop without gap → fill at stop (current behavior)
    - Short stop with gap-up open above stop → fill at open
    - Slippage scales with bar volatility (range/open)

Tail risks:
    - Open below all reasonable bounds → still produces a finite fill price
"""

from __future__ import annotations

import pytest

from src.backtest.fill_model import FillConfig, SimulatedTrade, simulate_close


def _bar(open_p: float, high: float, low: float, close: float, ts: int = 86_400_000) -> dict:
    return {"ts_ms": ts, "open": open_p, "high": high, "low": low, "close": close, "volume": 1.0}


class TestGapRisk:
    def test_long_gap_down_through_stop_fills_at_open(self) -> None:
        """Long stop at 95, next bar opens at 90 (gap through). Fill at 90, not 95."""
        cfg = FillConfig(taker_fee_bps=0.0, slippage_bps=0.0, gap_risk_enabled=True)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=95.0, take_profit=None,
        )
        next_bar = _bar(open_p=90.0, high=92.0, low=88.0, close=89.0)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        # With gap-risk enabled, exit at the open (90), worse than the stop (95)
        assert closed.exit_price == pytest.approx(90.0)
        assert closed.exit_reason == "stop_loss_gap"
        assert closed.pnl == pytest.approx(-10.0)  # 90 - 100 = -10

    def test_long_no_gap_fills_at_stop(self) -> None:
        """Long stop at 95, next bar opens at 99 (no gap). Fill at stop."""
        cfg = FillConfig(taker_fee_bps=0.0, slippage_bps=0.0, gap_risk_enabled=True)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=95.0, take_profit=None,
        )
        next_bar = _bar(open_p=99.0, high=100.0, low=94.0, close=98.0)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        # No gap: stop fills at stop level
        assert closed.exit_price == pytest.approx(95.0)
        assert closed.exit_reason == "stop_loss"

    def test_short_gap_up_through_stop_fills_at_open(self) -> None:
        cfg = FillConfig(taker_fee_bps=0.0, slippage_bps=0.0, gap_risk_enabled=True)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="short", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=105.0, take_profit=None,
        )
        next_bar = _bar(open_p=110.0, high=112.0, low=109.0, close=111.0)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.exit_price == pytest.approx(110.0)
        assert closed.exit_reason == "stop_loss_gap"
        assert closed.pnl == pytest.approx(-10.0)  # 100 - 110 = -10

    def test_gap_risk_disabled_falls_back_to_stop_fill(self) -> None:
        # Without gap_risk_enabled, the legacy behavior persists (fill at stop)
        cfg = FillConfig(taker_fee_bps=0.0, slippage_bps=0.0, gap_risk_enabled=False)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=95.0, take_profit=None,
        )
        next_bar = _bar(open_p=90.0, high=92.0, low=88.0, close=89.0)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.exit_price == pytest.approx(95.0)
        assert closed.exit_reason == "stop_loss"


class TestVolatilityAwareSlippage:
    def test_high_vol_bar_increases_market_close_slippage(self) -> None:
        # Same trade, two next_bars: low vol vs high vol
        cfg = FillConfig(
            taker_fee_bps=0.0, slippage_bps=5.0, vol_aware_slippage_enabled=True,
        )
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=None, take_profit=None,
        )
        # Low-vol next bar
        low_vol_bar = _bar(open_p=110.0, high=110.5, low=109.5, close=110.0)
        closed_low = simulate_close(trade, next_bar=low_vol_bar, exit_ts_ms=86_400_000, config=cfg)

        # High-vol next bar (range = 5% of open)
        trade2 = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=None, take_profit=None,
        )
        high_vol_bar = _bar(open_p=110.0, high=115.5, low=104.5, close=111.0)
        closed_high = simulate_close(trade2, next_bar=high_vol_bar, exit_ts_ms=86_400_000, config=cfg)

        # In a long market exit, slippage subtracts from exit price; with vol-aware
        # slippage, the high-vol bar should produce a worse (lower) exit price.
        assert closed_high.exit_price < closed_low.exit_price

    def test_vol_aware_slippage_disabled_unchanged_behavior(self) -> None:
        cfg_no = FillConfig(
            taker_fee_bps=0.0, slippage_bps=5.0, vol_aware_slippage_enabled=False,
        )
        trade_low = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=None, take_profit=None,
        )
        low_vol_bar = _bar(open_p=110.0, high=110.5, low=109.5, close=110.0)
        closed_low = simulate_close(trade_low, next_bar=low_vol_bar, exit_ts_ms=86_400_000, config=cfg_no)

        trade_high = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=None, take_profit=None,
        )
        high_vol_bar = _bar(open_p=110.0, high=115.5, low=104.5, close=111.0)
        closed_high = simulate_close(trade_high, next_bar=high_vol_bar, exit_ts_ms=86_400_000, config=cfg_no)
        # With vol-aware disabled, slippage is constant → both fills at the same price
        assert closed_high.exit_price == pytest.approx(closed_low.exit_price)
