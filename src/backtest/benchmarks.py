"""Passive benchmark strategies the LLM strategy must beat.

If the LLM can't beat buy-and-hold BTC and equal-weight portfolio after
fees, the LLM doesn't have edge — the user should use the benchmark.

Benchmarks share the fill model with the runner so fees and slippage are
applied identically. They produce ``BacktestResult`` so they slot into the
report layer without special-casing.
"""

from __future__ import annotations

from src.backtest.data_cache import OHLCVCache
from src.backtest.fill_model import (
    FillConfig,
    SimulatedTrade,
    simulate_close,
    simulate_open,
)
from src.backtest.runner import BacktestResult


def buy_and_hold(
    cache: OHLCVCache,
    symbol: str,
    timeframe: str,
    start_idx: int,
    end_idx: int,
    initial_equity: float,
    fill_config: FillConfig,
) -> BacktestResult:
    """Allocate the full equity to ``symbol`` at ``start_idx``, exit at ``end_idx+1``.

    Equity curve is the mark-to-market value at each bar in [start_idx, end_idx].
    """
    bars = cache.load_range(symbol, timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
    result = BacktestResult(
        path_id=f"buy_hold:{symbol}",
        initial_equity=initial_equity,
        final_equity=initial_equity,
    )

    if bars.empty or start_idx >= len(bars):
        result.equity_curve.append((0, initial_equity))
        return result

    end_idx = min(end_idx, len(bars) - 1)
    if end_idx <= start_idx:
        result.equity_curve.append((int(bars.iloc[start_idx]["ts_ms"]), initial_equity))
        return result

    # Buy-and-hold entry: at the start of the window, at start_idx's open.
    # No "next bar" lookahead concern here because no decision is being made —
    # this is the passive benchmark, allocating at t0.
    entry_idx = start_idx
    entry_bar = _row_to_bar(bars.iloc[entry_idx])
    entry_price = entry_bar["open"] * (1 + fill_config.slippage_bps / 10_000.0)
    size = initial_equity / entry_price  # full-equity, no leverage
    trade = simulate_open(
        symbol=symbol,
        side="long",
        size=size,
        next_bar=entry_bar,
        stop_loss=None,
        take_profit=None,
        config=fill_config,
    )
    equity = initial_equity - trade.entry_fee
    result.equity_curve.append((int(bars.iloc[start_idx]["ts_ms"]), equity))

    # Mark-to-market at each bar
    for i in range(entry_idx, end_idx + 1):
        bar = _row_to_bar(bars.iloc[i])
        mtm_equity = equity + (bar["close"] - trade.entry_price) * trade.size
        result.equity_curve.append((bar["ts_ms"], mtm_equity))

    # Exit at end_idx (close at next bar's open if available; else use end_idx open)
    exit_idx = min(end_idx + 1, len(bars) - 1)
    exit_bar = _row_to_bar(bars.iloc[exit_idx])
    closed = simulate_close(
        trade, next_bar=exit_bar, exit_ts_ms=exit_bar["ts_ms"], config=fill_config
    )
    equity += closed.pnl
    result.trades.append(closed)
    result.equity_curve[-1] = (result.equity_curve[-1][0], equity)
    result.final_equity = equity
    return result


def equal_weight(
    cache: OHLCVCache,
    symbols: list[str],
    timeframe: str,
    start_idx: int,
    end_idx: int,
    initial_equity: float,
    fill_config: FillConfig,
) -> BacktestResult:
    """Allocate ``initial_equity`` equally across ``symbols`` and hold to ``end_idx``."""
    result = BacktestResult(
        path_id=f"equal_weight:{','.join(symbols)}",
        initial_equity=initial_equity,
        final_equity=initial_equity,
    )
    if not symbols:
        result.equity_curve.append((0, initial_equity))
        return result

    per_symbol_equity = initial_equity / len(symbols)
    trades: list[SimulatedTrade] = []
    total_equity = 0.0
    final_ts = 0

    # Open positions at start_idx (no lookahead concern — this is passive entry)
    for sym in symbols:
        bars = cache.load_range(sym, timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
        if bars.empty or start_idx >= len(bars):
            total_equity += per_symbol_equity
            continue
        entry_bar = _row_to_bar(bars.iloc[start_idx])
        entry_price = entry_bar["open"] * (1 + fill_config.slippage_bps / 10_000.0)
        size = per_symbol_equity / entry_price
        trade = simulate_open(
            symbol=sym, side="long", size=size, next_bar=entry_bar,
            stop_loss=None, take_profit=None, config=fill_config,
        )
        trades.append(trade)
        total_equity += per_symbol_equity - trade.entry_fee
        final_ts = max(final_ts, entry_bar["ts_ms"])

    result.equity_curve.append((int(final_ts), total_equity))

    # Close at end_idx + 1
    closed_pnl = 0.0
    for trade in trades:
        bars = cache.load_range(trade.symbol, timeframe, 0, 9_999_999_999_999).reset_index(drop=True)
        exit_idx = min(end_idx + 1, len(bars) - 1)
        if exit_idx >= len(bars):
            continue
        exit_bar = _row_to_bar(bars.iloc[exit_idx])
        closed = simulate_close(
            trade, next_bar=exit_bar, exit_ts_ms=exit_bar["ts_ms"], config=fill_config
        )
        closed_pnl += closed.pnl
        result.trades.append(closed)
        final_ts = max(final_ts, exit_bar["ts_ms"])

    final_equity = total_equity + closed_pnl
    result.final_equity = final_equity
    result.equity_curve.append((int(final_ts), final_equity))
    return result


def _row_to_bar(row) -> dict:
    return {
        "ts_ms": int(row["ts_ms"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    }
