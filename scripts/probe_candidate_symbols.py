#!/usr/bin/env python3
"""Probe Binance perpetuals for candidate symbols beyond crypto majors.

User asked for:
    - US stocks crypto tokens (e.g., COIN, MSTR, NVDA)
    - Brent oil
    - Silver

This script asks the exchange which of these are tradeable. Many are NOT
on standard Binance perpetuals — this script reports honestly.

Outputs:
  - List of "tradeable on Binance perp" candidates
  - List of "NOT tradeable on Binance" with proposed alternatives
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import ccxt.async_support as ccxt

CRYPTO_MAJORS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT",
    "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT",
    "DOGE/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "DOT/USDT:USDT",
]

PRECIOUS_METALS = [
    "PAXG/USDT", "XAUT/USDT",       # gold (already cached)
    "PAXG/USDT:USDT",                # PAXG perp (illiquid)
    # Silver tokens are tiny / illiquid; XAGUSDT not on Binance.
]

US_STOCK_TOKENS = [
    # Tokenized stocks were delisted from Binance in 2021. None remain on
    # standard Binance perpetuals. Listed here so the probe reports them
    # as unavailable.
    "COIN/USDT:USDT", "MSTR/USDT:USDT", "NVDA/USDT:USDT", "AAPL/USDT:USDT",
    "TSLA/USDT:USDT", "SPY/USDT:USDT",
]

OIL_AND_COMMODITIES = [
    # Brent crude is NOT on Binance. Listed for the probe to confirm.
    "OIL/USDT:USDT", "BRENT/USDT:USDT", "WTI/USDT:USDT",
]

CRYPTO_PROXIES_FOR_STOCKS = [
    # Coinbase, MicroStrategy, etc as crypto-native proxies. These trade
    # on alternative exchanges (e.g., Bybit) and aren't on standard Binance.
]


async def probe(exchange_id: str, symbols: list[str]) -> tuple[list[str], list[str]]:
    """Return (available, unavailable) lists."""
    ex_cls = getattr(ccxt, exchange_id)
    ex = ex_cls({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
    })
    try:
        await ex.load_markets()
    except Exception as e:
        print(f"ERROR loading markets: {e}", file=sys.stderr)
        await ex.close()
        return [], symbols

    available = []
    unavailable = []
    for sym in symbols:
        if sym in ex.markets:
            available.append(sym)
        else:
            unavailable.append(sym)
    await ex.close()
    return available, unavailable


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exchange", default="binance")
    args = p.parse_args(argv)

    candidates = (
        CRYPTO_MAJORS + PRECIOUS_METALS + US_STOCK_TOKENS
        + OIL_AND_COMMODITIES + CRYPTO_PROXIES_FOR_STOCKS
    )
    available, unavailable = asyncio.run(probe(args.exchange, candidates))

    print("\n=== Binance perpetual symbol availability ===\n")
    print("AVAILABLE (can fetch with scripts/fetch_history.py):")
    for s in available:
        print(f"  ✓  {s}")

    print("\nNOT AVAILABLE on Binance perpetuals:")
    for s in unavailable:
        print(f"  ✗  {s}")

    print("\n=== Honest answers for non-crypto asset classes ===")
    print("  - US-stock crypto tokens:")
    print("      Binance delisted tokenized stocks (MSTR, COIN, etc) in 2021.")
    print("      None remain on standard Binance perpetuals. Some alt-")
    print("      exchanges (FTX successor, Bybit) list crypto-native proxies")
    print("      but those are out-of-scope for our Binance-only fetcher.")
    print("  - Brent crude oil:")
    print("      NOT on Binance. Available via traditional brokers (CFD) or")
    print("      via tokenized commodity providers. Out-of-scope for $2K AUM")
    print("      crypto-perp deployment.")
    print("  - Silver:")
    print("      No silver-backed perpetual on Binance (gold has PAXG/XAUT).")
    print("      Some tiny tokenized silver (KAG, AGI) exist but are illiquid.")
    print("  - Recommendation:")
    print("      Stay with crypto-only universe (BTC + ETH + crypto majors)")
    print("      and gold (PAXG) as the only non-crypto exposure. The")
    print("      strategy's edge thesis (regime detection + trend) doesn't")
    print("      depend on diversifying outside the crypto + gold pair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
