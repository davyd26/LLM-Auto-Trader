"""ExchangeProfile: per-exchange tunables that CCXT does not expose uniformly.

CCXT gives us a unified API surface (`fetch_ohlcv`, `create_order`, ...) but it
does NOT give us a unified, programmatic source of truth for things like:

  - Maker / taker fees at the user's actual VIP tier
  - Maintenance margin rate per market
  - Whether `set_leverage` is supported on this exchange
  - The collateral / quote currency used by perpetuals
  - Position mode (one-way vs hedge) defaults

These vary by exchange, sometimes by user, and sometimes by market. The bot
needs them at config time to size positions and project P&L correctly.

`ExchangeProfile` collects these into one frozen dataclass. We ship a small
registry of sensible defaults for common exchanges; users override any field
via env vars or YAML config.

The unified-API call sites (`fetch_ohlcv`, `create_order`, etc.) are NOT
touched — those keep working through `ccxt.async_support` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ExchangeProfile:
    """Per-exchange tunables resolved at config time."""

    name: str
    """CCXT id, e.g. ``"binance"``, ``"bybit"``, ``"okx"``, ``"mexc"``."""

    market_type: str = "swap"
    """One of ``"swap"`` (perpetuals), ``"future"`` (dated futures), ``"spot"``."""

    quote_currency: str = "USDT"
    """Collateral / margin asset. Most perpetuals are USDT-margined; some
    venues (Deribit, dYdX) use USDC, BTC, or other quotes."""

    maker_fee: float = 0.0002
    """Conservative default (2 bps). Override per VIP tier."""

    taker_fee: float = 0.0005
    """Conservative default (5 bps). Override per VIP tier."""

    maintenance_margin_rate: float = 0.005
    """Used to project liquidation prices. Real value varies by market and
    notional tier on most exchanges; this is a safe overestimate."""

    supports_set_leverage: bool = True
    """Most CCXT exchanges expose ``set_leverage``; a few do not."""

    position_mode: str = "one-way"
    """``"one-way"`` (net) or ``"hedge"`` (separate long/short books)."""

    extra_options: dict[str, Any] = field(default_factory=dict)
    """Passed through to the CCXT constructor's ``options`` dict."""

    def to_ccxt_options(self) -> dict[str, Any]:
        """Build the ``options`` dict for the CCXT exchange constructor."""
        opts: dict[str, Any] = {"defaultType": self.market_type}
        opts.update(self.extra_options)
        return opts

    def symbol_regex(self) -> str:
        """Regex matching unified perpetual + spot symbols for this profile.

        Examples:
            ``BTC/USDT`` (spot)
            ``BTC/USDT:USDT`` (USDT-margined perpetual)
        """
        q = self.quote_currency
        return rf"^[A-Z0-9]+/{q}(:.+)?$"

    def with_overrides(self, **kwargs: Any) -> ExchangeProfile:
        """Return a copy with selected fields replaced."""
        return replace(self, **kwargs)


PROFILE_REGISTRY: dict[str, ExchangeProfile] = {
    "binance": ExchangeProfile(
        name="binance",
        market_type="swap",
        quote_currency="USDT",
        maker_fee=0.0002,
        taker_fee=0.0004,
        maintenance_margin_rate=0.004,
    ),
    "binanceusdm": ExchangeProfile(
        name="binanceusdm",
        market_type="future",
        quote_currency="USDT",
        maker_fee=0.0002,
        taker_fee=0.0004,
        maintenance_margin_rate=0.004,
    ),
    "bybit": ExchangeProfile(
        name="bybit",
        market_type="swap",
        quote_currency="USDT",
        maker_fee=0.0001,
        taker_fee=0.0006,
        maintenance_margin_rate=0.005,
    ),
    "okx": ExchangeProfile(
        name="okx",
        market_type="swap",
        quote_currency="USDT",
        maker_fee=0.0002,
        taker_fee=0.0005,
        maintenance_margin_rate=0.005,
    ),
    "mexc": ExchangeProfile(
        name="mexc",
        market_type="swap",
        quote_currency="USDT",
        maker_fee=0.0,
        taker_fee=0.0001,
        maintenance_margin_rate=0.005,
    ),
    "gate": ExchangeProfile(
        name="gate",
        market_type="swap",
        quote_currency="USDT",
        maker_fee=0.0002,
        taker_fee=0.0005,
        maintenance_margin_rate=0.005,
    ),
    "kucoinfutures": ExchangeProfile(
        name="kucoinfutures",
        market_type="swap",
        quote_currency="USDT",
        maker_fee=0.0002,
        taker_fee=0.0006,
        maintenance_margin_rate=0.005,
    ),
    "bitget": ExchangeProfile(
        name="bitget",
        market_type="swap",
        quote_currency="USDT",
        maker_fee=0.0002,
        taker_fee=0.0006,
        maintenance_margin_rate=0.005,
    ),
}


def resolve_profile(
    exchange_name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> ExchangeProfile:
    """Look up a profile by exchange name, applying optional overrides.

    Falls back to a conservative generic profile (5 bps taker, 0.5% maint
    margin, USDT quote) for unknown exchanges. The bot will still run; just
    confirm the values match your venue and override if not.
    """
    base = PROFILE_REGISTRY.get(
        exchange_name.lower(),
        ExchangeProfile(name=exchange_name),
    )
    if overrides:
        clean = {k: v for k, v in overrides.items() if v is not None}
        if clean:
            base = base.with_overrides(**clean)
    return base
