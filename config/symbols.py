"""Trading symbol definitions and per-symbol parameters.

Symbols are written in CCXT unified format. Whether a symbol is actually
listed on the configured exchange is checked at startup; missing symbols
are silently skipped (with a clear log line) so the bot still runs on
exchanges with thinner asset coverage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolConfig:
    """Configuration for a single trading symbol."""

    symbol: str
    """CCXT unified symbol (e.g. ``"BTC/USDT:USDT"`` for a perpetual,
    ``"BTC/USDT"`` for spot)."""

    display_name: str
    asset_class: str  # "crypto" | "commodity" | "equity"
    max_leverage: int = 5
    enabled: bool = True
    priority: int = 1
    """1 = highest. Used to drop low-priority symbols if context budget is tight."""


# Default trading universe — mirrors the user's spec:
#   USDT (collateral, not a tradable symbol)
#   BTC, ETH, TRX, PAXG/XAUT (whichever is listed on the chosen exchange).
TRADING_SYMBOLS: list[SymbolConfig] = [
    SymbolConfig(
        symbol="BTC/USDT:USDT",
        display_name="Bitcoin",
        asset_class="crypto",
        max_leverage=5,
        priority=1,
    ),
    SymbolConfig(
        symbol="ETH/USDT:USDT",
        display_name="Ethereum",
        asset_class="crypto",
        max_leverage=5,
        priority=1,
    ),
    SymbolConfig(
        symbol="TRX/USDT:USDT",
        display_name="TRON",
        asset_class="crypto",
        max_leverage=5,
        priority=2,
    ),
    SymbolConfig(
        symbol="PAXG/USDT:USDT",
        display_name="PAX Gold",
        asset_class="commodity",
        max_leverage=3,
        priority=2,
    ),
]


# Symbols to attempt as fallbacks when the primary symbol isn't listed.
# Format: list of (replaces_symbol, fallback_config) pairs.
# At startup, if `replaces_symbol` is not on the exchange, the fallback is
# substituted in.
SYMBOL_FALLBACKS: list[tuple[str, SymbolConfig]] = [
    # Spot variants when no perpetual is listed.
    (
        "BTC/USDT:USDT",
        SymbolConfig("BTC/USDT", "Bitcoin (spot)", "crypto", 1, priority=1),
    ),
    (
        "ETH/USDT:USDT",
        SymbolConfig("ETH/USDT", "Ethereum (spot)", "crypto", 1, priority=1),
    ),
    (
        "TRX/USDT:USDT",
        SymbolConfig("TRX/USDT", "TRON (spot)", "crypto", 1, priority=2),
    ),
    # PAXG/XAUT cross-fallbacks (some exchanges list one but not the other).
    (
        "PAXG/USDT:USDT",
        SymbolConfig("XAUT/USDT:USDT", "Tether Gold", "commodity", 3, priority=2),
    ),
    (
        "PAXG/USDT:USDT",
        SymbolConfig("PAXG/USDT", "PAX Gold (spot)", "commodity", 1, priority=2),
    ),
    (
        "PAXG/USDT:USDT",
        SymbolConfig("XAUT/USDT", "Tether Gold (spot)", "commodity", 1, priority=2),
    ),
]


# Optional, opt-in extras (oil, exotic memes, etc.).
OPTIONAL_SYMBOLS: list[SymbolConfig] = [
    SymbolConfig(
        symbol="OIL/USDT:USDT",
        display_name="Crude Oil Token",
        asset_class="commodity",
        max_leverage=3,
        priority=3,
        enabled=False,
    ),
]


# Symbols to monitor for cross-exchange data only (not traded).
DATA_ONLY_SYMBOLS: list[str] = [
    "BTC/USDT",
    "ETH/USDT",
]


def get_enabled_symbols() -> list[SymbolConfig]:
    """Return all enabled trading + optional symbols."""
    return [s for s in TRADING_SYMBOLS + OPTIONAL_SYMBOLS if s.enabled]


def get_symbol_map() -> dict[str, SymbolConfig]:
    """Return ``{symbol: SymbolConfig}`` for enabled symbols."""
    return {s.symbol: s for s in get_enabled_symbols()}


def resolve_symbols_for_exchange(
    available: set[str],
    *,
    quote_currency: str = "USDT",
) -> list[SymbolConfig]:
    """Pick the best-available symbol from each preference group.

    Args:
        available: set of CCXT unified symbols listed on the chosen exchange
            (typically ``set(exchange.markets)`` after ``load_markets()``).
        quote_currency: collateral asset; used to rewrite symbols when
            an exchange uses a non-USDT quote (e.g. ``"USDC"``).

    Returns:
        A list of ``SymbolConfig`` objects, one per preference group, with
        unavailable groups dropped entirely. Always preserves the configured
        priority order.
    """
    resolved: list[SymbolConfig] = []
    for primary in TRADING_SYMBOLS:
        if not primary.enabled:
            continue
        candidate = _retarget_quote(primary, quote_currency)
        if candidate.symbol in available:
            resolved.append(candidate)
            continue

        # Try fallbacks in order.
        chosen: SymbolConfig | None = None
        for replaces, fb in SYMBOL_FALLBACKS:
            if replaces != primary.symbol:
                continue
            fb_retargeted = _retarget_quote(fb, quote_currency)
            if fb_retargeted.symbol in available:
                chosen = fb_retargeted
                break
        if chosen is not None:
            resolved.append(chosen)

    return resolved


def _retarget_quote(cfg: SymbolConfig, quote: str) -> SymbolConfig:
    """Rewrite a symbol's quote currency if the exchange uses something other
    than USDT (e.g. some venues quote in USDC).
    """
    if quote == "USDT":
        return cfg
    new_symbol = cfg.symbol.replace("/USDT", f"/{quote}").replace(
        ":USDT", f":{quote}"
    )
    if new_symbol == cfg.symbol:
        return cfg
    return SymbolConfig(
        symbol=new_symbol,
        display_name=cfg.display_name,
        asset_class=cfg.asset_class,
        max_leverage=cfg.max_leverage,
        enabled=cfg.enabled,
        priority=cfg.priority,
    )
