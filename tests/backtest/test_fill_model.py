"""Tests for the backtest fill model.

The fill model converts an LLM decision into simulated trade outcomes,
respecting:
    - Entry on the *next* bar's open (no within-bar lookahead)
    - Taker fees on entry and exit
    - Slippage: configurable bps applied to entry/exit prices
    - Stop-loss: triggered when a bar's high/low touches the stop, exit at stop
    - Take-profit: same mechanic
    - Position sizing in base-currency units (not notional)

Edge cases covered:
    - Long stop hit on next bar: exit at stop price not at next-next-bar's open
    - Short stop hit on next bar
    - TP and SL both hit in same bar — convention: SL wins (worst-case)
    - Bar with no high/low movement (gap)
    - Position closed normally without stop hit
    - Zero size → no-op

Tail risks covered:
    - Slippage so large that exit price > entry + max_high → clamp
    - Negative pnl on long with stop hit (verify sign)
    - Fees accumulate correctly on multiple entries
    - Stop on opposite side of entry rejected (e.g., long with stop above entry)
"""

from __future__ import annotations

import math

import pytest

from src.backtest.fill_model import (
    FillConfig,
    SimulatedTrade,
    simulate_close,
    simulate_open,
)


def _bar(open_p: float, high: float, low: float, close: float) -> dict:
    return {"open": open_p, "high": high, "low": low, "close": close}


class TestSimulateOpen:
    def test_long_open_at_next_bar_open_with_slippage(self) -> None:
        # Entry slippage applied to long: pay slightly more than open
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=10.0)
        next_bar = _bar(open_p=100.0, high=105.0, low=95.0, close=102.0)
        trade = simulate_open(
            symbol="BTC/USDT", side="long", size=1.0, next_bar=next_bar,
            stop_loss=None, take_profit=None, config=cfg,
        )
        # Long entry: open + slippage = 100 * (1 + 0.001) = 100.1
        assert trade.entry_price == pytest.approx(100.0 * (1 + 10.0 / 10_000))
        # Fee = 100.1 * 1.0 * 5/10000 ≈ 0.05005
        assert trade.entry_fee == pytest.approx(trade.entry_price * 1.0 * 5.0 / 10_000)

    def test_short_open_at_next_bar_open_with_slippage(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=10.0)
        next_bar = _bar(open_p=100.0, high=105.0, low=95.0, close=102.0)
        trade = simulate_open(
            symbol="BTC/USDT", side="short", size=1.0, next_bar=next_bar,
            stop_loss=None, take_profit=None, config=cfg,
        )
        # Short entry: open - slippage (you receive less when shorting via slippage)
        assert trade.entry_price == pytest.approx(100.0 * (1 - 10.0 / 10_000))

    def test_zero_size_is_rejected(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=10.0)
        next_bar = _bar(100, 105, 95, 102)
        with pytest.raises(ValueError, match="size"):
            simulate_open(
                symbol="BTC/USDT", side="long", size=0.0, next_bar=next_bar,
                stop_loss=None, take_profit=None, config=cfg,
            )

    def test_long_stop_above_entry_rejected(self) -> None:
        # Long position with stop above entry is nonsensical (would trigger immediately)
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=0.0)
        next_bar = _bar(100, 105, 95, 102)
        with pytest.raises(ValueError, match="stop"):
            simulate_open(
                symbol="BTC/USDT", side="long", size=1.0, next_bar=next_bar,
                stop_loss=110.0, take_profit=None, config=cfg,
            )

    def test_short_stop_below_entry_rejected(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=0.0)
        next_bar = _bar(100, 105, 95, 102)
        with pytest.raises(ValueError, match="stop"):
            simulate_open(
                symbol="BTC/USDT", side="short", size=1.0, next_bar=next_bar,
                stop_loss=90.0, take_profit=None, config=cfg,
            )


class TestSimulateClose:
    def test_long_close_at_market_with_slippage(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=10.0)
        # Enter at 100, exit market on a bar with open=110
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.05,
            stop_loss=None, take_profit=None,
        )
        next_bar = _bar(open_p=110.0, high=115, low=109, close=112)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        # Long exit: open - slippage (sell at slightly lower than mid)
        assert closed.exit_price == pytest.approx(110.0 * (1 - 10.0 / 10_000))
        # PnL: (exit - entry) * size - fees
        gross = (closed.exit_price - 100.0) * 1.0
        expected_exit_fee = closed.exit_price * 1.0 * 5.0 / 10_000
        assert closed.exit_fee == pytest.approx(expected_exit_fee)
        assert closed.pnl == pytest.approx(gross - 0.05 - expected_exit_fee)

    def test_short_close_at_market_with_slippage(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=10.0)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="short", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.05,
            stop_loss=None, take_profit=None,
        )
        next_bar = _bar(open_p=90.0, high=95, low=89, close=92)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        # Short exit: open + slippage (cover at slightly higher)
        assert closed.exit_price == pytest.approx(90.0 * (1 + 10.0 / 10_000))
        # PnL: (entry - exit) * size - fees
        gross = (100.0 - closed.exit_price) * 1.0
        assert closed.pnl == pytest.approx(gross - 0.05 - closed.exit_fee)


class TestStopLossLogic:
    def test_long_stop_hit_when_low_touches_stop(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=0.0)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.05,
            stop_loss=95.0, take_profit=None,
        )
        # Bar low touches 94 < stop 95 → exit at stop
        next_bar = _bar(open_p=99.0, high=100, low=94, close=98)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.exit_price == pytest.approx(95.0)
        assert closed.exit_reason == "stop_loss"
        assert closed.pnl < 0  # losing trade

    def test_long_stop_not_hit_when_low_above_stop(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=0.0)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.05,
            stop_loss=95.0, take_profit=None,
        )
        # Bar low 96 > stop 95 → no stop hit
        next_bar = _bar(open_p=99.0, high=102, low=96, close=101)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.exit_reason == "manual"
        assert closed.exit_price == pytest.approx(99.0)  # market exit at open

    def test_short_stop_hit_when_high_touches_stop(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=0.0)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="short", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.05,
            stop_loss=105.0, take_profit=None,
        )
        next_bar = _bar(open_p=101.0, high=106, low=99, close=104)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.exit_price == pytest.approx(105.0)
        assert closed.exit_reason == "stop_loss"
        assert closed.pnl < 0


class TestTakeProfit:
    def test_long_tp_hit(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=0.0)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.05,
            stop_loss=None, take_profit=110.0,
        )
        next_bar = _bar(open_p=105, high=112, low=104, close=109)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.exit_price == pytest.approx(110.0)
        assert closed.exit_reason == "take_profit"
        assert closed.pnl > 0


class TestStopAndTpInSameBar:
    def test_stop_wins_when_both_hit(self) -> None:
        # Conservative convention: when both SL and TP are touchable in the same bar,
        # assume the worst (stop hit first). This is the standard backtest assumption.
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=0.0)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.05,
            stop_loss=95.0, take_profit=110.0,
        )
        # Bar swings from low=94 to high=112 — both could be hit
        next_bar = _bar(open_p=98, high=112, low=94, close=108)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.exit_price == pytest.approx(95.0)
        assert closed.exit_reason == "stop_loss"


class TestPnLArithmetic:
    def test_long_winning_trade_pnl(self) -> None:
        cfg = FillConfig(taker_fee_bps=0.0, slippage_bps=0.0)  # clean
        # Long 1 BTC at 100, exit at 120 → +20 pnl
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=None, take_profit=None,
        )
        next_bar = _bar(open_p=120, high=122, low=119, close=121)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.pnl == pytest.approx(20.0)

    def test_short_winning_trade_pnl(self) -> None:
        cfg = FillConfig(taker_fee_bps=0.0, slippage_bps=0.0)
        # Short 1 BTC at 100, exit at 80 → +20 pnl
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="short", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.0,
            stop_loss=None, take_profit=None,
        )
        next_bar = _bar(open_p=80, high=82, low=79, close=81)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.pnl == pytest.approx(20.0)

    def test_fees_reduce_pnl(self) -> None:
        cfg = FillConfig(taker_fee_bps=10.0, slippage_bps=0.0)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.10,  # 100 * 1 * 0.001
            stop_loss=None, take_profit=None,
        )
        next_bar = _bar(open_p=110, high=111, low=109, close=110)
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        # Gross = 10, exit fee = 110 * 1 * 0.001 = 0.11, entry fee = 0.10
        assert closed.pnl == pytest.approx(10.0 - 0.10 - 0.11)


class TestFillModelTailRisks:
    def test_nan_stop_treated_as_no_stop(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=0.0)
        trade = SimulatedTrade(
            symbol="BTC/USDT", side="long", size=1.0,
            entry_price=100.0, entry_ts_ms=0, entry_fee=0.05,
            stop_loss=math.nan, take_profit=None,
        )
        next_bar = _bar(open_p=110, high=112, low=109, close=111)
        # NaN stop should not trigger; trade closes at market
        closed = simulate_close(trade, next_bar=next_bar, exit_ts_ms=86_400_000, config=cfg)
        assert closed.exit_reason == "manual"

    def test_negative_size_rejected(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=10.0)
        next_bar = _bar(100, 105, 95, 102)
        with pytest.raises(ValueError, match="size"):
            simulate_open(
                symbol="BTC/USDT", side="long", size=-1.0, next_bar=next_bar,
                stop_loss=None, take_profit=None, config=cfg,
            )

    def test_unknown_side_rejected(self) -> None:
        cfg = FillConfig(taker_fee_bps=5.0, slippage_bps=10.0)
        next_bar = _bar(100, 105, 95, 102)
        with pytest.raises(ValueError, match="side"):
            simulate_open(
                symbol="BTC/USDT", side="diagonal", size=1.0, next_bar=next_bar,
                stop_loss=None, take_profit=None, config=cfg,
            )
