"""Aggregate per-path BacktestResults into a single distribution-aware summary.

The summary is what the user reads to evaluate Phase-2 pass criteria. It must
be honest: median + tails + per-trade aggregate stats, not just the best path.
JSON output replaces NaN/Inf with safe sentinels so files round-trip cleanly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.backtest_stats import (
    expectancy,
    hit_rate,
    max_drawdown,
    probabilistic_sharpe_ratio,
    profit_factor,
    sharpe_annualized,
)
from src.backtest.runner import BacktestResult


def aggregate_results(
    results: list[BacktestResult], periods_per_year: int = 52
) -> dict[str, Any]:
    """Combine per-path results into a single summary dict."""
    if not results:
        return _empty_summary()

    initial_equities = [r.initial_equity for r in results]
    final_equities = np.array([r.final_equity for r in results], dtype=float)
    returns_pct = (final_equities - np.array(initial_equities)) / np.array(initial_equities)

    warnings: list[str] = []
    if len(set(initial_equities)) > 1:
        warnings.append("inconsistent initial_equity across paths")

    sharpes = np.array([_path_sharpe(r, periods_per_year) for r in results])
    drawdowns = np.array([_path_max_drawdown(r) for r in results])

    all_trades = [t.__dict__ if hasattr(t, "__dict__") else t for r in results for t in r.trades]
    total_trades = len(all_trades)
    aggregate_hit = hit_rate(all_trades) if all_trades else 0.0
    aggregate_pf = profit_factor(all_trades) if all_trades else math.nan
    aggregate_exp = expectancy(all_trades) if all_trades else 0.0

    median_sharpe = float(np.median(sharpes))
    moments = _flatten_returns_for_psr(results)
    psr_on_median = (
        probabilistic_sharpe_ratio(
            sr_hat=median_sharpe,
            n=len(moments),
            skew=0.0,
            kurt=3.0,
        ) if len(moments) > 1 else 0.0
    )

    summary: dict[str, Any] = {
        "n_paths": len(results),
        "median_final_equity": float(np.median(final_equities)),
        "p5_final_equity": float(np.percentile(final_equities, 5)),
        "p95_final_equity": float(np.percentile(final_equities, 95)),
        "median_return_pct": float(np.median(returns_pct)),
        "p5_return_pct": float(np.percentile(returns_pct, 5)),
        "p95_return_pct": float(np.percentile(returns_pct, 95)),
        "median_sharpe": median_sharpe,
        "p5_sharpe": float(np.percentile(sharpes, 5)),
        "p95_sharpe": float(np.percentile(sharpes, 95)),
        "median_max_drawdown": float(np.median(drawdowns)),
        "p95_max_drawdown": float(np.percentile(drawdowns, 95)),
        "psr_on_median_sharpe": float(psr_on_median),
        "total_trades": total_trades,
        "aggregate_hit_rate": float(aggregate_hit),
        "aggregate_profit_factor": float(aggregate_pf) if math.isfinite(aggregate_pf) else None,
        "aggregate_expectancy": float(aggregate_exp),
        "warnings": warnings,
    }
    return summary


def _empty_summary() -> dict[str, Any]:
    return {
        "n_paths": 0,
        "median_final_equity": 0,
        "p5_final_equity": 0,
        "p95_final_equity": 0,
        "median_return_pct": 0.0,
        "p5_return_pct": 0.0,
        "p95_return_pct": 0.0,
        "median_sharpe": 0.0,
        "p5_sharpe": 0.0,
        "p95_sharpe": 0.0,
        "median_max_drawdown": 0.0,
        "p95_max_drawdown": 0.0,
        "psr_on_median_sharpe": 0.0,
        "total_trades": 0,
        "aggregate_hit_rate": 0.0,
        "aggregate_profit_factor": None,
        "aggregate_expectancy": 0.0,
        "warnings": ["no paths"],
    }


def _path_returns(result: BacktestResult) -> np.ndarray:
    """Period returns derived from a path's equity curve."""
    if len(result.equity_curve) < 2:
        return np.array([], dtype=float)
    equity = np.array([eq for _, eq in result.equity_curve], dtype=float)
    # Guard against zero-equity divisions
    safe = np.where(equity[:-1] > 0, equity[:-1], 1.0)
    rets = np.diff(equity) / safe
    return rets


def _path_sharpe(result: BacktestResult, periods_per_year: int) -> float:
    rets = _path_returns(result)
    if rets.size == 0:
        return 0.0
    return float(sharpe_annualized(rets, periods_per_year=periods_per_year))


def _path_max_drawdown(result: BacktestResult) -> float:
    if len(result.equity_curve) < 2:
        return 0.0
    equity = np.array([eq for _, eq in result.equity_curve], dtype=float)
    dd, _, _ = max_drawdown(equity)
    return float(dd)


def _flatten_returns_for_psr(results: list[BacktestResult]) -> np.ndarray:
    rets = []
    for r in results:
        rets.extend(_path_returns(r).tolist())
    return np.array(rets, dtype=float)


def write_summary_json(summary: dict[str, Any], path: str | Path) -> None:
    """Write summary to JSON with NaN/Inf replaced for safe parsing."""
    cleaned = _sanitize(summary)
    Path(path).write_text(json.dumps(cleaned, indent=2, sort_keys=True))


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
        return obj
    return obj
