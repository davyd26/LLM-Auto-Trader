"""Deterministic baseline strategy: 4-week momentum + PAXG drawdown hedge.

The baseline serves two purposes:
    1. End-to-end harness validation. Running the harness with this strategy
       in stub mode produces a sanity-check equity curve we can verify by
       hand. If the harness has bugs, this is where they surface.
    2. The benchmark the LLM strategy must beat. If the LLM doesn't beat a
       4-week momentum + drawdown hedge, it doesn't have edge — kill the
       strategy and use buy-and-hold instead.

Rules (intentionally simple):
    - Long BTC if BTC.close[-1] > BTC.close[-N] (momentum); else flat.
    - When the recent peak-to-current drawdown exceeds the threshold, rotate
       to PAXG (gold proxy) instead of BTC.
    - Single-position cadence: at most one open position at a time.
    - Defensive on missing or NaN data: return hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    momentum_lookback_bars: int = 5
    drawdown_hedge_threshold: float = 0.05
    hedge_symbol: str = "PAXG/USDT"
    primary_symbol: str = "BTC/USDT"
    long_size_usd_pct: float = 0.10  # 10% of equity per position


def _hold() -> dict:
    return {
        "market_assessment": {"trend": "neutral", "confidence": 0.5, "regime": "calm"},
        "actions": [{"type": "hold"}],
        "next_check_minutes": 10080,
        "risk_notes": "baseline: hold",
    }


def _has_nan(closes: list[float]) -> bool:
    return any(math.isnan(c) or math.isinf(c) for c in closes)


def _drawdown_pct(closes: list[float]) -> float:
    if len(closes) < 2:
        return 0.0
    peak = max(closes)
    if peak <= 0:
        return 0.0
    return (peak - closes[-1]) / peak


def _momentum_positive(closes: list[float], lookback: int) -> bool:
    if len(closes) <= lookback:
        return False
    return closes[-1] > closes[-1 - lookback]


def decide_baseline(
    symbol_bars: dict[str, list[dict]],
    current_positions: list[dict],
    equity_usd: float,
    config: BaselineConfig,
) -> dict:
    """Return a decision in the same schema the LLM uses, deterministically."""
    primary_bars = symbol_bars.get(config.primary_symbol, [])
    if not primary_bars or len(primary_bars) <= config.momentum_lookback_bars:
        return _hold()

    closes = [b["close"] for b in primary_bars]
    if _has_nan(closes):
        return _hold()

    momentum_up = _momentum_positive(closes, config.momentum_lookback_bars)
    dd = _drawdown_pct(closes)
    needs_hedge = dd >= config.drawdown_hedge_threshold

    has_btc_long = any(
        p.get("symbol") == config.primary_symbol and p.get("side") == "long"
        for p in current_positions
    )
    has_paxg_long = any(
        p.get("symbol") == config.hedge_symbol and p.get("side") == "long"
        for p in current_positions
    )

    actions: list[dict] = []
    target_size_usd = equity_usd * config.long_size_usd_pct

    if needs_hedge:
        if config.hedge_symbol in symbol_bars and not has_paxg_long:
            paxg_bars = symbol_bars.get(config.hedge_symbol, [])
            if paxg_bars:
                paxg_price = paxg_bars[-1]["close"]
                if paxg_price > 0 and not math.isnan(paxg_price):
                    actions.append({
                        "type": "open_long",
                        "symbol": config.hedge_symbol,
                        "size": target_size_usd / paxg_price,
                    })
        if has_btc_long:
            actions.append({"type": "close", "symbol": config.primary_symbol})
    elif momentum_up:
        if not has_btc_long:
            btc_price = closes[-1]
            actions.append({
                "type": "open_long",
                "symbol": config.primary_symbol,
                "size": target_size_usd / btc_price,
            })
    else:
        # Momentum flat/down, no hedge needed → close any open BTC long
        if has_btc_long:
            actions.append({"type": "close", "symbol": config.primary_symbol})

    if not actions:
        actions = [{"type": "hold"}]

    regime = "elevated" if needs_hedge else ("calm" if momentum_up else "calm")
    return {
        "market_assessment": {
            "trend": "bullish" if momentum_up else "neutral",
            "confidence": 0.6 if momentum_up else 0.5,
            "regime": regime,
        },
        "actions": actions,
        "next_check_minutes": 10080,
        "risk_notes": (
            f"momentum {'up' if momentum_up else 'flat/down'}; "
            f"drawdown {dd:.1%}; "
            f"hedge={'on' if needs_hedge else 'off'}"
        ),
    }
