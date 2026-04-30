"""Rolling decision history and performance feedback for LLM context."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DecisionRecord:
    """Record of a single LLM decision and its outcome."""

    timestamp: float
    action: str
    symbol: str
    reasoning: str
    outcome: str | None = None      # filled retrospectively
    pnl: float | None = None        # filled when trade closes
    time_ago_str: str = ""           # computed on access


class TradingMemory:
    """Manages the LLM's recurrent context: past decisions and performance."""

    def __init__(self, max_decisions: int = 5, max_trades: int = 10):
        self._max_decisions = max_decisions
        self._max_trades = max_trades
        self._decisions: deque[DecisionRecord] = deque(maxlen=max_decisions)
        self._trade_history: list[dict] = []
        self._peak_equity: float = 0
        self._starting_capital: float = 0

    def set_starting_capital(self, capital: float) -> None:
        self._starting_capital = capital
        self._peak_equity = max(self._peak_equity, capital)

    def record_decision(
        self, action: str, symbol: str, reasoning: str
    ) -> None:
        """Record a new LLM decision."""
        self._decisions.append(DecisionRecord(
            timestamp=time.time(),
            action=action,
            symbol=symbol,
            reasoning=reasoning,
        ))

    def update_last_decision_outcome(self, outcome: str, pnl: float | None = None) -> None:
        """Update the most recent decision with its outcome."""
        if self._decisions:
            self._decisions[-1].outcome = outcome
            self._decisions[-1].pnl = pnl

    def record_trade(self, trade: dict) -> None:
        """Record a completed trade. Memory retains the full session-level
        trade list so that AFML metrics (DSR, CSCV/PBO) can operate over the
        full sample. `get_recent_trades` still returns only the last N for
        LLM context. Older history lives in the DB."""
        self._trade_history.append(trade)
        # Hard cap to avoid unbounded growth in a very long session — keep the
        # most recent 2000 trades (more than enough for robust AFML metrics
        # at any realistic trading cadence).
        if len(self._trade_history) > 2000:
            self._trade_history = self._trade_history[-2000:]

    def update_equity(self, equity: float) -> None:
        """Update peak equity for drawdown tracking."""
        self._peak_equity = max(self._peak_equity, equity)

    def get_previous_decision(self) -> dict | None:
        """Get the most recent decision for LLM context."""
        if not self._decisions:
            return None

        d = self._decisions[-1]
        elapsed = time.time() - d.timestamp
        if elapsed < 60:
            time_ago = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            time_ago = f"{elapsed/60:.0f}m"
        else:
            time_ago = f"{elapsed/3600:.1f}h"

        return {
            "action": d.action,
            "symbol": d.symbol,
            "reasoning": d.reasoning,
            "outcome": d.outcome or "pending",
            "time_ago": time_ago,
        }

    def get_recent_trades(self) -> list[dict]:
        """Get recent trades for LLM context."""
        trades = self._trade_history[-self._max_trades:]
        for t in trades:
            # Compute human-readable duration
            if "timestamp_open" in t and "timestamp_close" in t:
                duration_s = t["timestamp_close"] - t["timestamp_open"]
                if duration_s < 3600:
                    t["duration"] = f"{duration_s/60:.0f}m"
                else:
                    t["duration"] = f"{duration_s/3600:.1f}h"
        return trades

    def get_performance_summary(
        self,
        current_equity: float,
        afml_settings: object | None = None,
    ) -> dict:
        """Compute performance summary for LLM context.

        Adds AFML-aware metrics (DSR, PSR, PBO, skew/kurtosis) when enabled.
        """
        trades = self._trade_history
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "max_drawdown_pct": 0,
                "current_drawdown_pct": 0,
                "sharpe": 0,
                "pnl_today": 0,
                "pnl_total": 0,
                "streak": "N/A",
            }

        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        pnls = [t.get("pnl", 0) for t in trades]

        # Win rate
        win_rate = len(wins) / len(trades) if trades else 0

        # Average PnL
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0

        # Current drawdown
        current_dd = 0
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - current_equity) / self._peak_equity

        # Max drawdown (from trade history)
        max_dd = current_dd
        running_equity = self._starting_capital
        running_peak = self._starting_capital
        for t in trades:
            running_equity += t.get("pnl", 0)
            running_peak = max(running_peak, running_equity)
            dd = (running_peak - running_equity) / running_peak if running_peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified, using trade PnLs)
        import numpy as np
        if len(pnls) > 2:
            pnl_arr = np.array(pnls)
            sharpe = float(pnl_arr.mean() / pnl_arr.std()) if pnl_arr.std() > 0 else 0
        else:
            sharpe = 0

        # Today's PnL
        today_start = time.time() - (time.time() % 86400)
        pnl_today = sum(
            t.get("pnl", 0) for t in trades
            if t.get("timestamp_close", 0) >= today_start
        )

        # Current streak
        streak = _compute_streak(trades)

        summary = {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "avg_pnl": round(avg_pnl, 2),
            "max_drawdown_pct": round(max_dd, 4),
            "current_drawdown_pct": round(current_dd, 4),
            "sharpe": round(sharpe, 2),
            "pnl_today": round(pnl_today, 2),
            "pnl_total": round(sum(pnls), 2),
            "streak": streak,
        }

        # Phase 1 §3.6 + §3.7 — AFML-aware performance metrics
        afml_on = (
            afml_settings is not None
            and getattr(afml_settings, "enable_all", False)
            and getattr(afml_settings, "enable_dsr", False)
        )
        if afml_on and len(pnls) >= 3:
            try:
                from src.analysis.backtest_stats import (
                    pnl_moments,
                    probabilistic_sharpe_ratio,
                    deflated_sharpe_ratio,
                    sharpe_annualized,
                )

                arr = np.array([p / 100 for p in pnls], dtype=float)  # pnl_pct in [%] → fraction
                moments = pnl_moments(arr)
                sharpe_ann = sharpe_annualized(arr, periods_per_year=365)

                # Determine num_trials: CSCV-driven if enabled + enough trades.
                num_trials = int(getattr(afml_settings, "fallback_trials", 50))
                sr_trials_var = 0.25
                cscv_pbo_val: float | None = None
                cscv_on = getattr(afml_settings, "enable_cscv", False)
                min_trades = int(getattr(afml_settings, "cscv_min_trades", 20))
                if cscv_on and len(trades) >= min_trades:
                    try:
                        from src.analysis.cscv import (
                            cscv_pbo,
                            variants_from_realized_trades,
                        )
                        grid = list(getattr(afml_settings, "cscv_min_p_grid",
                                            (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)))
                        matrix = variants_from_realized_trades(trades, min_p_grid=grid)
                        if matrix.shape[0] >= 10 and matrix.shape[1] >= 2:
                            result = cscv_pbo(matrix)
                            cscv_pbo_val = float(result["pbo"])
                            num_trials = max(1, int(result.get("num_trials", num_trials)))
                            # sr_trials_var from the IS distribution if enough pairs.
                            pairs = result.get("sr_is_oos", [])
                            if pairs:
                                sr_is_series = np.array([p[0] for p in pairs], dtype=float)
                                if sr_is_series.std(ddof=1) > 0:
                                    sr_trials_var = float(sr_is_series.var(ddof=1))
                    except Exception:
                        pass

                psr = probabilistic_sharpe_ratio(
                    sr_hat=sharpe_ann / max(1e-9, (365 ** 0.5)),  # back to per-period
                    n=len(arr),
                    skew=moments["skew"],
                    kurt=moments["kurtosis"],
                )
                dsr = deflated_sharpe_ratio(
                    sr_hat=sharpe_ann / max(1e-9, (365 ** 0.5)),
                    n=len(arr),
                    skew=moments["skew"],
                    kurt=moments["kurtosis"],
                    num_trials=num_trials,
                    sr_trials_var=sr_trials_var,
                )

                summary["sharpe_annualized"] = round(sharpe_ann, 2)
                summary["psr_vs_zero"] = round(psr, 3)
                summary["dsr"] = round(dsr, 3)
                summary["pnl_skew"] = round(moments["skew"], 3)
                summary["pnl_kurtosis"] = round(moments["kurtosis"], 3)
                summary["num_trials_for_dsr"] = num_trials
                if cscv_pbo_val is not None:
                    summary["pbo"] = round(cscv_pbo_val, 3)
            except Exception as e:
                # Never let metric computation break the trading loop.
                import logging
                logging.getLogger(__name__).exception("dsr_computation_failed: %s", e)

        return summary


def _compute_streak(trades: list[dict]) -> str:
    """Compute current win/loss streak."""
    if not trades:
        return "N/A"

    streak = 0
    streak_type = ""
    for t in reversed(trades):
        pnl = t.get("pnl", 0)
        if pnl > 0:
            if streak_type == "" or streak_type == "W":
                streak_type = "W"
                streak += 1
            else:
                break
        elif pnl < 0:
            if streak_type == "" or streak_type == "L":
                streak_type = "L"
                streak += 1
            else:
                break
        else:
            break

    return f"{streak_type}{streak}" if streak_type else "N/A"
