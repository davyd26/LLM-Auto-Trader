"""Tests for report aggregation across CPCV paths.

The report combines per-path BacktestResults into a single summary dict
suitable for JSON export. The summary is the user's primary readout; it
must be honest, distribution-aware (median/p5/p95 across paths), and
include the metrics needed to evaluate Phase-2 pass criteria.

Edge cases covered:
    - Empty results list → safe defaults, no crash
    - Single path → degenerate distribution (median == p5 == p95)
    - Path with zero trades → counted but contributes 0 to trade-level stats
    - Inconsistent initial_equity across paths → flagged
    - All-losing paths → metrics still computed, no division by zero

Tail risks covered:
    - JSON serializability (no NaN / Inf in raw output without flag)
    - Trade list aggregation does not lose trades
    - Equity-curve return computation handles single-point curves
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backtest.fill_model import SimulatedTrade
from src.backtest.report import (
    aggregate_results,
    write_summary_json,
)
from src.backtest.runner import BacktestResult


def _result(
    path_id: str,
    initial: float,
    final: float,
    curve_pts: list[tuple[int, float]] | None = None,
    trades: list[SimulatedTrade] | None = None,
) -> BacktestResult:
    return BacktestResult(
        path_id=path_id,
        initial_equity=initial,
        final_equity=final,
        equity_curve=curve_pts or [(0, initial), (1, final)],
        trades=trades or [],
    )


def _trade(pnl: float, entry: float = 100.0, exit: float = 110.0, side: str = "long") -> SimulatedTrade:
    return SimulatedTrade(
        symbol="BTC/USDT",
        side=side,
        size=1.0,
        entry_price=entry,
        entry_ts_ms=0,
        entry_fee=0.0,
        stop_loss=None,
        take_profit=None,
        exit_price=exit,
        exit_ts_ms=86_400_000,
        exit_fee=0.0,
        exit_reason="manual",
        pnl=pnl,
    )


class TestAggregateBasic:
    def test_empty_results_returns_safe_defaults(self) -> None:
        summary = aggregate_results([], periods_per_year=52)
        assert summary["n_paths"] == 0
        assert summary["median_final_equity"] == 0
        assert summary["total_trades"] == 0

    def test_single_path_degenerate_distribution(self) -> None:
        r = _result("0", 2000.0, 2200.0)
        summary = aggregate_results([r], periods_per_year=52)
        assert summary["n_paths"] == 1
        assert summary["median_final_equity"] == 2200.0
        assert summary["p5_final_equity"] == 2200.0
        assert summary["p95_final_equity"] == 2200.0

    def test_three_paths_median(self) -> None:
        rs = [
            _result("0", 2000.0, 2100.0),
            _result("1", 2000.0, 2300.0),
            _result("2", 2000.0, 1800.0),
        ]
        summary = aggregate_results(rs, periods_per_year=52)
        assert summary["n_paths"] == 3
        assert summary["median_final_equity"] == 2100.0


class TestAggregateReturns:
    def test_total_return_per_path(self) -> None:
        rs = [_result("0", 2000.0, 2200.0), _result("1", 2000.0, 1800.0)]
        summary = aggregate_results(rs, periods_per_year=52)
        # +10% and -10% returns
        assert summary["median_return_pct"] == pytest.approx(0.0)

    def test_inconsistent_initial_equity_flagged(self) -> None:
        rs = [_result("0", 2000.0, 2200.0), _result("1", 1000.0, 1100.0)]
        summary = aggregate_results(rs, periods_per_year=52)
        # The summary should warn — use a flag field
        assert summary.get("warnings", [])


class TestAggregateTrades:
    def test_total_trades_summed_across_paths(self) -> None:
        rs = [
            _result("0", 2000.0, 2010.0, trades=[_trade(5), _trade(5)]),
            _result("1", 2000.0, 1995.0, trades=[_trade(-5)]),
        ]
        summary = aggregate_results(rs, periods_per_year=52)
        assert summary["total_trades"] == 3

    def test_aggregate_hit_rate(self) -> None:
        rs = [
            _result("0", 2000.0, 2010.0, trades=[_trade(10), _trade(-5)]),  # 1/2
            _result("1", 2000.0, 2000.0, trades=[_trade(0)]),                # 0/1
        ]
        summary = aggregate_results(rs, periods_per_year=52)
        # 1 win out of 3 total → 0.333...
        assert summary["aggregate_hit_rate"] == pytest.approx(1 / 3, abs=1e-6)

    def test_aggregate_profit_factor(self) -> None:
        rs = [
            _result("0", 2000.0, 2010.0, trades=[_trade(10), _trade(-2), _trade(5)]),  # gross win 15, gross loss 2
        ]
        summary = aggregate_results(rs, periods_per_year=52)
        assert summary["aggregate_profit_factor"] == pytest.approx(15 / 2)


class TestAggregateDrawdown:
    def test_drawdown_per_path(self) -> None:
        # Path 1 has a 10% drawdown, path 2 has 20%
        path1_curve = [(0, 2000.0), (1, 2200.0), (2, 1980.0), (3, 2100.0)]  # peak 2200, trough 1980 → 10%
        path2_curve = [(0, 2000.0), (1, 2500.0), (2, 2000.0)]  # peak 2500, trough 2000 → 20%
        rs = [
            BacktestResult(path_id="0", initial_equity=2000, final_equity=2100,
                           equity_curve=path1_curve, trades=[]),
            BacktestResult(path_id="1", initial_equity=2000, final_equity=2000,
                           equity_curve=path2_curve, trades=[]),
        ]
        summary = aggregate_results(rs, periods_per_year=52)
        assert summary["median_max_drawdown"] == pytest.approx(0.15, abs=0.01)  # median of 0.10 and 0.20


class TestAggregateSharpe:
    def test_sharpe_recorded_for_each_path(self) -> None:
        # Build a curve of returns from equity points
        path1_curve = [(0, 2000.0), (1, 2050.0), (2, 2100.0), (3, 2080.0), (4, 2150.0)]
        rs = [BacktestResult(path_id="0", initial_equity=2000, final_equity=2150,
                             equity_curve=path1_curve, trades=[])]
        summary = aggregate_results(rs, periods_per_year=52)
        assert "median_sharpe" in summary
        assert isinstance(summary["median_sharpe"], float)


class TestJSONExport:
    def test_write_summary_json_round_trip(self, tmp_path: Path) -> None:
        rs = [_result("0", 2000.0, 2200.0)]
        summary = aggregate_results(rs, periods_per_year=52)
        out = tmp_path / "summary.json"
        write_summary_json(summary, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["median_final_equity"] == 2200.0

    def test_summary_has_no_nan_or_inf_in_json(self, tmp_path: Path) -> None:
        # Empty results force NaN paths through computations — verify safe replacement
        summary = aggregate_results([], periods_per_year=52)
        out = tmp_path / "summary.json"
        write_summary_json(summary, out)
        text = out.read_text()
        assert "NaN" not in text
        assert "Infinity" not in text
        assert "-Infinity" not in text


class TestAggregateTailRisks:
    def test_all_losing_paths_does_not_crash(self) -> None:
        rs = [_result(str(i), 2000.0, 1900.0 - i * 50, trades=[_trade(-50)]) for i in range(5)]
        summary = aggregate_results(rs, periods_per_year=52)
        assert summary["n_paths"] == 5
        assert summary["median_final_equity"] < 2000

    def test_path_with_single_equity_point(self) -> None:
        # Pathological curve with one point — runner shouldn't normally produce this
        # but the aggregator must not crash
        rs = [BacktestResult(path_id="0", initial_equity=2000, final_equity=2000,
                             equity_curve=[(0, 2000.0)], trades=[])]
        summary = aggregate_results(rs, periods_per_year=52)
        assert summary["n_paths"] == 1
