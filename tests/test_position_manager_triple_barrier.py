"""End-to-end triple-barrier label inference at close_position."""

from __future__ import annotations

import pytest

from src.execution.position_manager import PositionManager


def _open(pm: PositionManager, **kwargs) -> None:
    defaults = dict(
        symbol="BTC/USDT:USDT",
        side="long",
        size=0.01,
        notional=1000.0,
        entry_price=100.0,
        leverage=2,
        stop_loss=99.0,
        take_profit=102.0,
        fees=0.1,
        reasoning="test",
        llm_confidence=0.7,
        max_holding_s=3600.0,
    )
    defaults.update(kwargs)
    pm.open_position(**defaults)


class TestTripleBarrierAtClose:
    def test_long_exit_above_tp_labels_pt(self) -> None:
        pm = PositionManager()
        _open(pm)
        pm.update_prices({"BTC/USDT:USDT": 102.5})  # touch above TP
        trade = pm.close_position("BTC/USDT:USDT", exit_price=102.5, fees=0.1)
        assert trade["tb_label"] == 1
        assert trade["tb_barrier_touched"] == "pt"

    def test_long_exit_below_sl_labels_sl(self) -> None:
        pm = PositionManager()
        _open(pm)
        pm.update_prices({"BTC/USDT:USDT": 98.5})
        trade = pm.close_position("BTC/USDT:USDT", exit_price=98.5, fees=0.1)
        assert trade["tb_label"] == -1
        assert trade["tb_barrier_touched"] == "sl"

    def test_long_exit_between_barriers_labels_time(self) -> None:
        pm = PositionManager()
        _open(pm)
        pm.update_prices({"BTC/USDT:USDT": 101.0})
        trade = pm.close_position("BTC/USDT:USDT", exit_price=101.0, fees=0.1)
        assert trade["tb_label"] == 0
        assert trade["tb_barrier_touched"] == "time"

    def test_short_exit_below_tp_labels_pt(self) -> None:
        pm = PositionManager()
        _open(pm, side="short", stop_loss=101.0, take_profit=98.0)
        pm.update_prices({"BTC/USDT:USDT": 97.5})
        trade = pm.close_position("BTC/USDT:USDT", exit_price=97.5, fees=0.1)
        assert trade["tb_label"] == 1
        assert trade["tb_barrier_touched"] == "pt"

    def test_records_confidence_and_barriers(self) -> None:
        pm = PositionManager()
        _open(pm, llm_confidence=0.82)
        pm.update_prices({"BTC/USDT:USDT": 101.0})
        trade = pm.close_position("BTC/USDT:USDT", exit_price=101.0, fees=0.1)
        assert trade["llm_confidence"] == 0.82
        assert trade["pt_pct"] == pytest.approx(0.02)
        assert trade["sl_pct"] == pytest.approx(0.01)

    def test_no_labels_when_sl_tp_missing(self) -> None:
        pm = PositionManager()
        _open(pm, stop_loss=None, take_profit=None)
        pm.update_prices({"BTC/USDT:USDT": 102.0})
        trade = pm.close_position("BTC/USDT:USDT", exit_price=102.0, fees=0.1)
        assert "tb_label" not in trade
