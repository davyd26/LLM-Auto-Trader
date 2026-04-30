"""Tests for triple-barrier labeling (AFML Ch. 3.3-3.4)."""

from __future__ import annotations

import pytest

from src.analysis.labeling import compute_triple_barrier


def _path(start_ts: float, prices: list[float], step_s: float = 60.0) -> list[tuple[float, float]]:
    return [(start_ts + i * step_s, p) for i, p in enumerate(prices)]


class TestComputeTripleBarrier:
    def test_long_hits_profit_take(self) -> None:
        # Long at 100, PT +2%, SL -1%; price goes 100 -> 101 -> 102.5
        p = _path(0, [100.0, 101.0, 102.5])
        label = compute_triple_barrier(
            entry_price=100.0, side="long",
            pt_pct=0.02, sl_pct=0.01, max_holding_s=3600,
            path=p,
        )
        assert label.label == 1
        assert label.barrier_touched == "pt"
        assert label.return_at_touch > 0.02

    def test_long_hits_stop_loss(self) -> None:
        p = _path(0, [100.0, 99.5, 98.9])
        label = compute_triple_barrier(
            entry_price=100.0, side="long",
            pt_pct=0.02, sl_pct=0.01, max_holding_s=3600,
            path=p,
        )
        assert label.label == -1
        assert label.barrier_touched == "sl"

    def test_long_times_out(self) -> None:
        # Stays inside barriers for entire path, hits time limit
        p = _path(0, [100.0, 100.5, 100.3, 100.4], step_s=600.0)
        label = compute_triple_barrier(
            entry_price=100.0, side="long",
            pt_pct=0.02, sl_pct=0.02, max_holding_s=1800,  # 30min
            path=p,
        )
        assert label.label == 0
        assert label.barrier_touched == "time"

    def test_short_hits_profit_take_on_down_move(self) -> None:
        p = _path(0, [100.0, 99.0, 97.5])
        label = compute_triple_barrier(
            entry_price=100.0, side="short",
            pt_pct=0.02, sl_pct=0.01, max_holding_s=3600,
            path=p,
        )
        assert label.label == 1
        assert label.barrier_touched == "pt"
        # Short return is positive when price drops
        assert label.return_at_touch > 0.02

    def test_short_hits_stop_on_up_move(self) -> None:
        p = _path(0, [100.0, 100.5, 101.2])
        label = compute_triple_barrier(
            entry_price=100.0, side="short",
            pt_pct=0.02, sl_pct=0.01, max_holding_s=3600,
            path=p,
        )
        assert label.label == -1
        assert label.barrier_touched == "sl"

    def test_first_touch_wins_when_both_hit_in_same_bar(self) -> None:
        # PT and SL both crossable — PT touched at earlier timestamp
        p = _path(0, [100.0, 102.5, 98.5])
        label = compute_triple_barrier(
            entry_price=100.0, side="long",
            pt_pct=0.02, sl_pct=0.01, max_holding_s=3600,
            path=p,
        )
        assert label.label == 1

    def test_entry_bar_not_counted_as_touch(self) -> None:
        # If entry price already outside PT (impossible in practice but boundary)
        p = _path(0, [100.0, 100.0])  # no movement
        label = compute_triple_barrier(
            entry_price=100.0, side="long",
            pt_pct=0.02, sl_pct=0.01, max_holding_s=60,
            path=p,
        )
        assert label.label == 0  # time-out

    def test_empty_path_returns_time_label(self) -> None:
        label = compute_triple_barrier(
            entry_price=100.0, side="long",
            pt_pct=0.02, sl_pct=0.01, max_holding_s=0,
            path=[],
        )
        assert label.label == 0

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_triple_barrier(
                entry_price=100.0, side="diagonal",
                pt_pct=0.02, sl_pct=0.01, max_holding_s=60,
                path=_path(0, [100, 101]),
            )

    def test_time_out_touch_time_equals_last_ts(self) -> None:
        p = _path(0, [100.0, 100.5, 100.3])
        label = compute_triple_barrier(
            entry_price=100.0, side="long",
            pt_pct=0.05, sl_pct=0.05, max_holding_s=120,
            path=p,
        )
        assert label.label == 0
        assert label.touch_time == p[-1][0]
