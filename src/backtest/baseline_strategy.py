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
    enable_shorts: bool = False  # when True, go short on negative momentum
    short_size_usd_pct: float | None = None  # override for short size; defaults to long_size_usd_pct
    short_stop_loss_pct: float = 0.10  # stop above entry for shorts (e.g. 10% above)
    long_stop_loss_pct: float | None = None  # if set, place stop below entry for longs
    short_min_drawdown: float = 0.0  # only short when running drawdown >= this (regime gate)
    confluence_lookback_bars: int | None = None  # if set, also requires close > close[-1-confluence] for longs
    rotation_symbols: tuple[str, ...] = ()  # if set, picks strongest-momentum from these, replaces primary
    vol_target_pct: float | None = None  # if set, scale size by target_vol / realized_vol
    vol_lookback_bars: int = 12  # window for realized vol calc (when vol_target_pct is set)
    vol_size_min_mult: float = 0.25  # min multiplier on size_pct from vol-targeting
    vol_size_max_mult: float = 2.0  # max multiplier on size_pct from vol-targeting


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


def _momentum_negative(closes: list[float], lookback: int) -> bool:
    if len(closes) <= lookback:
        return False
    return closes[-1] < closes[-1 - lookback]


def _strongest_momentum_symbol(
    symbol_bars: dict[str, list[dict]],
    candidates: tuple[str, ...],
    lookback: int,
) -> str | None:
    """Return the symbol with strongest positive momentum, else None."""
    best_sym = None
    best_ret = 0.0
    for sym in candidates:
        bars = symbol_bars.get(sym, [])
        if len(bars) <= lookback:
            continue
        closes = [b["close"] for b in bars]
        if any(math.isnan(c) or math.isinf(c) for c in closes):
            continue
        ret = (closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback]
        if ret > best_ret:
            best_ret = ret
            best_sym = sym
    return best_sym


def decide_baseline(
    symbol_bars: dict[str, list[dict]],
    current_positions: list[dict],
    equity_usd: float,
    config: BaselineConfig,
) -> dict:
    """Return a decision in the same schema the LLM uses, deterministically."""
    # If rotation_symbols is configured, choose the strongest-momentum symbol
    # as the primary for THIS decision; otherwise use config.primary_symbol.
    if config.rotation_symbols:
        rotation_winner = _strongest_momentum_symbol(
            symbol_bars, config.rotation_symbols, config.momentum_lookback_bars,
        )
        primary = rotation_winner if rotation_winner else config.primary_symbol
    else:
        primary = config.primary_symbol

    primary_bars = symbol_bars.get(primary, [])
    if not primary_bars or len(primary_bars) <= config.momentum_lookback_bars:
        return _hold()

    closes = [b["close"] for b in primary_bars]
    if _has_nan(closes):
        return _hold()

    momentum_up = _momentum_positive(closes, config.momentum_lookback_bars)
    momentum_down = _momentum_negative(closes, config.momentum_lookback_bars)
    if config.confluence_lookback_bars is not None:
        long_confluence = _momentum_positive(closes, config.confluence_lookback_bars)
        short_confluence = _momentum_negative(closes, config.confluence_lookback_bars)
        momentum_up = momentum_up and long_confluence
        momentum_down = momentum_down and short_confluence
    dd = _drawdown_pct(closes)
    needs_hedge = dd >= config.drawdown_hedge_threshold

    has_primary_long = any(
        p.get("symbol") == primary and p.get("side") == "long"
        for p in current_positions
    )
    has_primary_short = any(
        p.get("symbol") == primary and p.get("side") == "short"
        for p in current_positions
    )
    has_paxg_long = any(
        p.get("symbol") == config.hedge_symbol and p.get("side") == "long"
        for p in current_positions
    )

    actions: list[dict] = []
    long_pct = config.long_size_usd_pct
    short_pct = (config.short_size_usd_pct
                 if config.short_size_usd_pct is not None
                 else config.long_size_usd_pct)

    # Vol-targeting: scale size by target_vol / realized_vol (clamped)
    if config.vol_target_pct is not None and len(closes) > config.vol_lookback_bars + 1:
        recent = closes[-config.vol_lookback_bars - 1:]
        rets = [recent[i + 1] / recent[i] - 1.0 for i in range(len(recent) - 1)]
        if rets:
            mean_r = sum(rets) / len(rets)
            var_r = sum((r - mean_r) ** 2 for r in rets) / max(1, len(rets) - 1)
            sigma = math.sqrt(var_r)
            if sigma > 0:
                mult = config.vol_target_pct / sigma
                mult = max(config.vol_size_min_mult, min(config.vol_size_max_mult, mult))
                long_pct = config.long_size_usd_pct * mult
                short_pct = short_pct * mult

    target_long_usd = equity_usd * long_pct
    target_short_usd = equity_usd * short_pct
    btc_price = closes[-1]

    # Multi-asset rotation: if there are open positions in any symbol that is
    # NOT the current rotation winner (or the active hedge), close them.
    if config.rotation_symbols:
        keep_set = {primary, config.hedge_symbol}
        for p in current_positions:
            sym = p.get("symbol")
            if sym not in keep_set:
                actions.append({"type": "close", "symbol": sym})

    if needs_hedge:
        # Hedge regime: rotate to PAXG; close any directional primary position.
        if config.hedge_symbol in symbol_bars and not has_paxg_long:
            paxg_bars = symbol_bars.get(config.hedge_symbol, [])
            if paxg_bars:
                paxg_price = paxg_bars[-1]["close"]
                if paxg_price > 0 and not math.isnan(paxg_price):
                    actions.append({
                        "type": "open_long",
                        "symbol": config.hedge_symbol,
                        "size": target_long_usd / paxg_price,
                    })
        if has_primary_long or has_primary_short:
            actions.append({"type": "close", "symbol": primary})
    elif momentum_up:
        # Bullish: open long, close any short
        if has_primary_short:
            actions.append({"type": "close", "symbol": primary})
        elif not has_primary_long:
            action = {
                "type": "open_long",
                "symbol": primary,
                "size": target_long_usd / btc_price,
            }
            if config.long_stop_loss_pct is not None:
                action["stop_loss"] = btc_price * (1 - config.long_stop_loss_pct)
            actions.append(action)
    elif momentum_down and config.enable_shorts and dd >= config.short_min_drawdown:
        # Bearish + shorts enabled + regime gate (drawdown above threshold)
        if has_primary_long:
            actions.append({"type": "close", "symbol": primary})
        elif not has_primary_short:
            action = {
                "type": "open_short",
                "symbol": primary,
                "size": target_short_usd / btc_price,
                "stop_loss": btc_price * (1 + config.short_stop_loss_pct),
            }
            actions.append(action)
    else:
        # Momentum flat or short-disabled negative momentum → go to cash
        if has_primary_long or has_primary_short:
            actions.append({"type": "close", "symbol": primary})

    if not actions:
        actions = [{"type": "hold"}]

    regime = "elevated" if needs_hedge else "calm"
    if momentum_up:
        trend, conf = "bullish", 0.6
    elif momentum_down:
        trend, conf = "bearish", 0.6
    else:
        trend, conf = "neutral", 0.5
    return {
        "market_assessment": {
            "trend": trend,
            "confidence": conf,
            "regime": regime,
        },
        "actions": actions,
        "next_check_minutes": 10080,
        "risk_notes": (
            f"mom={'up' if momentum_up else ('down' if momentum_down else 'flat')}; "
            f"dd={dd:.1%}; hedge={'on' if needs_hedge else 'off'}; "
            f"shorts={'on' if config.enable_shorts else 'off'}"
        ),
    }
