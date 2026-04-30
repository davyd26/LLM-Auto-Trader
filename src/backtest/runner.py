"""CPCV simulation runner — walks a path's test_indices, executes decisions.

The runner is the keystone of the backtest harness. It connects:
    snapshot_builder → decision_cache → fill_model → equity tracking

For each weekly observation in a CPCV path, the runner:
    1. Builds a snapshot of the world at simulated time t.
    2. Looks up the LLM's decision for that snapshot in the decision cache.
       If no decision is cached, raises ``MissingDecisionError`` — silent
       skipping would silently distort the backtest.
    3. Translates the decision's actions into trade open/close calls via the
       fill model. Slippage, fees, and SL/TP are applied at fill time.
    4. Updates equity from realized P&L on closed trades.

Positions still open at the end of the path are force-closed at the next bar
to prevent leakage into the embargo period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.backtest.cpcv import CPCVPath
from src.backtest.decision_cache import DecisionCache
from src.backtest.fill_model import (
    FillConfig,
    SimulatedTrade,
    simulate_close,
    simulate_open,
)
from src.backtest.snapshot_builder import PositionRecord, SnapshotBuilder


class MissingDecisionError(KeyError):
    """Raised when a snapshot has no cached decision — would corrupt the backtest."""


@dataclass
class BacktestResult:
    path_id: str
    initial_equity: float
    final_equity: float
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    trades: list[SimulatedTrade] = field(default_factory=list)


def run_path(
    path: CPCVPath,
    bars_by_symbol: dict[str, pd.DataFrame],
    snapshot_builder: SnapshotBuilder,
    decision_cache: DecisionCache,
    fill_config: FillConfig,
    initial_equity: float,
) -> BacktestResult:
    """Simulate the strategy along ``path`` using cached decisions."""
    path_id = "+".join(str(g) for g in path.test_groups)
    result = BacktestResult(
        path_id=path_id,
        initial_equity=initial_equity,
        final_equity=initial_equity,
    )

    # Use the first symbol's bars as the timing index. All symbols are assumed
    # bar-aligned at the weekly cadence.
    if not path.test_indices or not bars_by_symbol:
        result.equity_curve.append((0, initial_equity))
        return result

    primary_symbol = next(iter(bars_by_symbol))
    primary_bars = bars_by_symbol[primary_symbol].reset_index(drop=True)

    open_positions: dict[str, SimulatedTrade] = {}
    equity = initial_equity

    # Initial equity-curve point uses the first test bar's timestamp.
    first_idx = path.test_indices[0]
    if first_idx < len(primary_bars):
        result.equity_curve.append((int(primary_bars.iloc[first_idx]["ts_ms"]), equity))

    sorted_test = sorted(path.test_indices)
    for k, idx in enumerate(sorted_test):
        if idx >= len(primary_bars):
            continue
        ts_ms = int(primary_bars.iloc[idx]["ts_ms"])

        # Build snapshot using current positions
        position_records = [
            PositionRecord(
                symbol=t.symbol, side=t.side, size=t.size,
                entry_price=t.entry_price, entry_ts_ms=t.entry_ts_ms,
            )
            for t in open_positions.values()
        ]
        snap = snapshot_builder.build(t_ms=ts_ms, equity_usd=equity, positions=position_records)
        decision = decision_cache.get(snap.snapshot_id)
        if decision is None:
            raise MissingDecisionError(
                f"no cached decision for snapshot {snap.snapshot_id[:12]} (path {path_id}, idx {idx})"
            )

        # Determine the next bar to use for fills (entry/exit on next bar's open).
        # We use the primary symbol's next bar; per-symbol bars must be aligned.
        next_idx = idx + 1
        if next_idx >= len(primary_bars):
            # No next bar — cannot fill. Skip actions but still record equity.
            result.equity_curve.append((ts_ms, equity))
            continue

        # Apply stops/TPs against the next bar BEFORE evaluating new decisions.
        equity = _settle_open_positions(
            open_positions, bars_by_symbol, next_idx, fill_config, equity, result.trades
        )

        # Process the decision's actions (mutates equity)
        equity = _apply_decision_actions(
            decision, open_positions, bars_by_symbol, next_idx, fill_config, ts_ms,
            result.trades, equity,
        )

        # Mark equity curve at this observation
        result.equity_curve.append((ts_ms, equity))

    # Force-close any remaining positions at the bar after the last test index
    last_idx = sorted_test[-1] if sorted_test else 0
    final_close_idx = min(last_idx + 1, len(primary_bars) - 1)
    if final_close_idx > last_idx and open_positions:
        equity = _force_close_all(
            open_positions, bars_by_symbol, final_close_idx, fill_config, equity, result.trades
        )

    result.final_equity = equity
    if result.equity_curve:
        result.equity_curve[-1] = (result.equity_curve[-1][0], equity)
    return result


def _settle_open_positions(
    open_positions: dict[str, SimulatedTrade],
    bars_by_symbol: dict[str, pd.DataFrame],
    next_idx: int,
    fill_config: FillConfig,
    equity: float,
    trades_log: list[SimulatedTrade],
) -> float:
    """If any open position's stop/TP triggers in the next bar, exit there."""
    closed_now: list[str] = []
    for symbol, trade in list(open_positions.items()):
        bars = bars_by_symbol.get(symbol)
        if bars is None or next_idx >= len(bars):
            continue
        next_bar = _bar_dict(bars, next_idx)
        sl_or_tp_hit = _will_hit_sl_or_tp(trade, next_bar)
        if not sl_or_tp_hit:
            continue
        closed = simulate_close(
            trade, next_bar=next_bar, exit_ts_ms=int(next_bar["ts_ms"]), config=fill_config
        )
        equity += closed.pnl
        trades_log.append(closed)
        closed_now.append(symbol)
    for s in closed_now:
        open_positions.pop(s, None)
    return equity


def _will_hit_sl_or_tp(trade: SimulatedTrade, next_bar: dict) -> bool:
    """Check if the next bar's high/low would trigger SL or TP."""
    bar_high = float(next_bar["high"])
    bar_low = float(next_bar["low"])
    if trade.stop_loss is not None:
        if trade.side == "long" and bar_low <= trade.stop_loss:
            return True
        if trade.side == "short" and bar_high >= trade.stop_loss:
            return True
    if trade.take_profit is not None:
        if trade.side == "long" and bar_high >= trade.take_profit:
            return True
        if trade.side == "short" and bar_low <= trade.take_profit:
            return True
    return False


def _apply_decision_actions(
    decision: dict,
    open_positions: dict[str, SimulatedTrade],
    bars_by_symbol: dict[str, pd.DataFrame],
    next_idx: int,
    fill_config: FillConfig,
    snapshot_ts_ms: int,
    trades_log: list[SimulatedTrade],
    equity: float,
) -> float:
    for action in decision.get("actions", []):
        action_type = action.get("type")
        if action_type == "hold":
            continue
        symbol = action.get("symbol")
        if action_type == "close" and symbol in open_positions:
            bars = bars_by_symbol.get(symbol)
            if bars is None or next_idx >= len(bars):
                continue
            trade = open_positions.pop(symbol)
            next_bar = _bar_dict(bars, next_idx)
            closed = simulate_close(
                trade, next_bar=next_bar, exit_ts_ms=int(next_bar["ts_ms"]), config=fill_config
            )
            equity += closed.pnl
            trades_log.append(closed)
        elif action_type in ("open_long", "open_short"):
            size = float(action.get("size", 0.0))
            if size <= 0:
                continue
            if symbol in open_positions:
                continue  # ignore: already in a position for this symbol
            bars = bars_by_symbol.get(symbol)
            if bars is None or next_idx >= len(bars):
                continue
            next_bar = _bar_dict(bars, next_idx)
            side = "long" if action_type == "open_long" else "short"
            trade = simulate_open(
                symbol=symbol,
                side=side,
                size=size,
                next_bar=next_bar,
                stop_loss=action.get("stop_loss"),
                take_profit=action.get("take_profit"),
                config=fill_config,
            )
            equity -= trade.entry_fee
            open_positions[symbol] = trade
    return equity


def _force_close_all(
    open_positions: dict[str, SimulatedTrade],
    bars_by_symbol: dict[str, pd.DataFrame],
    next_idx: int,
    fill_config: FillConfig,
    equity: float,
    trades_log: list[SimulatedTrade],
) -> float:
    for symbol, trade in list(open_positions.items()):
        bars = bars_by_symbol.get(symbol)
        if bars is None or next_idx >= len(bars):
            continue
        next_bar = _bar_dict(bars, next_idx)
        closed = simulate_close(
            trade, next_bar=next_bar, exit_ts_ms=int(next_bar["ts_ms"]), config=fill_config
        )
        equity += closed.pnl
        trades_log.append(closed)
    open_positions.clear()
    return equity


def _bar_dict(bars: pd.DataFrame, idx: int) -> dict:
    row = bars.iloc[idx]
    return {
        "ts_ms": int(row["ts_ms"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }
