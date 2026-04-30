"""ExchangeProfile registry + override + fallback tests."""

from __future__ import annotations

import pytest

from src.exchange.profile import (
    PROFILE_REGISTRY,
    ExchangeProfile,
    resolve_profile,
)


def test_known_exchanges_have_profiles():
    """Every CCXT exchange we ship a default for must have a profile."""
    for name in ("binance", "bybit", "okx", "mexc", "gate", "bitget"):
        assert name in PROFILE_REGISTRY
        prof = PROFILE_REGISTRY[name]
        assert prof.name == name
        assert prof.taker_fee >= 0
        assert prof.maintenance_margin_rate > 0


def test_unknown_exchange_falls_back_to_safe_defaults():
    """Unknown exchanges still produce a usable profile."""
    p = resolve_profile("some_exchange_we_have_never_heard_of")
    assert p.name == "some_exchange_we_have_never_heard_of"
    # Conservative defaults — overestimates of fees and margin requirements.
    assert p.taker_fee >= 0.0001
    assert p.maintenance_margin_rate >= 0.001


def test_overrides_replace_only_specified_fields():
    p = resolve_profile("binance", overrides={"taker_fee": 0.0001})
    assert p.taker_fee == 0.0001
    assert p.maker_fee == PROFILE_REGISTRY["binance"].maker_fee
    assert p.maintenance_margin_rate == PROFILE_REGISTRY["binance"].maintenance_margin_rate


def test_none_values_in_overrides_are_ignored():
    base = PROFILE_REGISTRY["binance"]
    p = resolve_profile(
        "binance",
        overrides={"taker_fee": None, "maker_fee": 0.001},
    )
    assert p.taker_fee == base.taker_fee
    assert p.maker_fee == 0.001


def test_symbol_regex_is_quote_currency_aware():
    """USDC-quoted exchanges should match USDC symbols, not USDT."""
    p = ExchangeProfile(name="custom", quote_currency="USDC")
    rx = p.symbol_regex()
    import re
    assert re.match(rx, "BTC/USDC")
    assert re.match(rx, "BTC/USDC:USDC")
    assert not re.match(rx, "BTC/USDT")


def test_ccxt_options_pass_through_extra():
    p = ExchangeProfile(
        name="custom",
        market_type="future",
        extra_options={"defaultSubType": "linear"},
    )
    opts = p.to_ccxt_options()
    assert opts["defaultType"] == "future"
    assert opts["defaultSubType"] == "linear"


def test_case_insensitive_lookup():
    """``EXCHANGE_NAME=Binance`` should still hit the registry."""
    p = resolve_profile("Binance")
    assert p.maker_fee == PROFILE_REGISTRY["binance"].maker_fee
