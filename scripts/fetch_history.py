#!/usr/bin/env python3
"""Fetch historical OHLCV from a CCXT exchange and populate the OHLCV cache.

The CPCV backtest harness needs years of historical bars; this script
walks the exchange's paginated cursor and persists everything into the
SQLite cache. Idempotent: re-runs only fetch missing bars.

Usage:
    python scripts/fetch_history.py \\
        --exchange binance \\
        --symbols BTC/USDT,ETH/USDT,PAXG/USDT \\
        --timeframe 1w \\
        --since 2022-01-01 \\
        --until 2026-04-30 \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db

The exchange ID is anything CCXT supports. For perpetuals, use the swap
suffix (``BTC/USDT:USDT``); for spot, just ``BTC/USDT``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import ccxt.async_support as ccxt

from src.backtest.data_cache import OHLCVCache
from src.data.historical import fetch_ohlcv_range


def parse_iso_date(s: str) -> int:
    """YYYY-MM-DD UTC → millisecond timestamp."""
    return int(
        datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
    )


async def fetch_one(
    exchange,
    cache: OHLCVCache,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    batch_limit: int,
) -> int:
    """Fetch bars for ``symbol`` in [since, until] and store in cache.

    Returns the count of bars stored.
    """
    print(f"  fetching {symbol} {timeframe} from {since_ms} to {until_ms}...")
    try:
        bars = await fetch_ohlcv_range(
            exchange, symbol, timeframe,
            since_ms=since_ms, until_ms=until_ms, batch_limit=batch_limit,
        )
    except Exception as e:
        print(f"    ERROR fetching {symbol}: {e}", file=sys.stderr)
        return 0
    cache.store(symbol, timeframe, bars)
    print(f"    stored {len(bars)} bars")
    return len(bars)


async def main_async(args) -> int:
    cache_path = Path(args.cache_db).expanduser()
    cache = OHLCVCache(cache_path)
    print(f"OHLCV cache: {cache_path}")

    cls = getattr(ccxt, args.exchange)
    config = {
        "enableRateLimit": True,
        "options": {"defaultType": args.market_type},
    }
    exchange = cls(config)
    await exchange.load_markets()
    print(f"Exchange: {args.exchange} ({len(exchange.markets)} markets, type={args.market_type})")

    since_ms = parse_iso_date(args.since)
    until_ms = parse_iso_date(args.until)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    total = 0
    try:
        for symbol in symbols:
            if symbol not in exchange.markets:
                print(f"  SKIP {symbol}: not on {args.exchange}")
                continue
            n = await fetch_one(
                exchange, cache, symbol, args.timeframe,
                since_ms=since_ms, until_ms=until_ms,
                batch_limit=args.batch_limit,
            )
            total += n
    finally:
        await exchange.close()

    print(f"\nDone. {total} bars stored across {len(symbols)} symbols.")

    # Print cache range per symbol for visibility
    for symbol in symbols:
        r = cache.range(symbol, args.timeframe)
        if r:
            first_ms, last_ms = r
            first = datetime.fromtimestamp(first_ms / 1000, tz=timezone.utc).date()
            last = datetime.fromtimestamp(last_ms / 1000, tz=timezone.utc).date()
            print(f"  {symbol} {args.timeframe}: [{first} .. {last}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="swap",
                        help="swap|spot|future (CCXT defaultType)")
    parser.add_argument("--symbols", required=True, help="comma-separated")
    parser.add_argument("--timeframe", default="1w",
                        help="1m|5m|15m|1h|4h|1d|1w (exchange-supported)")
    parser.add_argument("--since", required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--until", required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--cache-db", required=True)
    parser.add_argument("--batch-limit", type=int, default=1000,
                        help="bars per CCXT call (CCXT enforces exchange max)")
    args = parser.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
