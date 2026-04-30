"""Symbol-resolution fallback tests.

These cover the non-obvious cases:
  - PAXG vs XAUT (some exchanges have one but not the other)
  - Perp-not-listed → spot fallback
  - Non-USDT quote currencies (USDC margin)
  - All preferences missing → empty list, no crash
"""

from __future__ import annotations

from config.symbols import resolve_symbols_for_exchange


def test_resolves_primary_perp_symbols():
    """Happy path: all perps listed."""
    available = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "TRX/USDT:USDT",
        "PAXG/USDT:USDT",
    }
    res = resolve_symbols_for_exchange(available)
    syms = [s.symbol for s in res]
    assert "BTC/USDT:USDT" in syms
    assert "ETH/USDT:USDT" in syms
    assert "TRX/USDT:USDT" in syms
    assert "PAXG/USDT:USDT" in syms


def test_falls_back_to_xaut_when_paxg_missing():
    """Some exchanges (e.g. MEXC) list XAUT but not PAXG perps."""
    available = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "TRX/USDT:USDT",
        "XAUT/USDT:USDT",  # PAXG missing
    }
    res = resolve_symbols_for_exchange(available)
    syms = [s.symbol for s in res]
    assert "PAXG/USDT:USDT" not in syms
    assert "XAUT/USDT:USDT" in syms


def test_falls_back_to_spot_when_no_perp():
    """Spot-only exchange should still produce a usable universe."""
    available = {"BTC/USDT", "ETH/USDT", "TRX/USDT"}
    res = resolve_symbols_for_exchange(available)
    syms = [s.symbol for s in res]
    assert "BTC/USDT" in syms
    assert "ETH/USDT" in syms
    assert "TRX/USDT" in syms


def test_drops_symbols_not_listed():
    """Missing groups are silently skipped — no crash, no None."""
    available = {"BTC/USDT:USDT"}  # only BTC
    res = resolve_symbols_for_exchange(available)
    syms = [s.symbol for s in res]
    assert syms == ["BTC/USDT:USDT"]


def test_empty_exchange_returns_empty():
    """Pathological case: exchange has zero markets."""
    res = resolve_symbols_for_exchange(set())
    assert res == []


def test_usdc_quoted_exchange():
    """When the exchange uses USDC, retarget all symbols."""
    available = {"BTC/USDC:USDC", "ETH/USDC:USDC"}
    res = resolve_symbols_for_exchange(available, quote_currency="USDC")
    syms = [s.symbol for s in res]
    assert "BTC/USDC:USDC" in syms
    assert "ETH/USDC:USDC" in syms


def test_resolver_preserves_priority_order():
    """Higher-priority symbols come first."""
    available = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "TRX/USDT:USDT",
        "PAXG/USDT:USDT",
    }
    res = resolve_symbols_for_exchange(available)
    # Priority 1 symbols (BTC, ETH) should appear before priority 2 (TRX, PAXG).
    p1_indices = [i for i, s in enumerate(res) if s.priority == 1]
    p2_indices = [i for i, s in enumerate(res) if s.priority == 2]
    if p1_indices and p2_indices:
        assert max(p1_indices) < min(p2_indices)
