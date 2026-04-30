"""Paginated historical OHLCV fetch via CCXT.

CCXT's ``fetch_ohlcv`` returns up to ~1500 bars per call; covering 2 years of
daily data needs only a few calls but minute-resolution would need thousands.
This module walks the cursor with strict guards against infinite loops and
silent partial fetches.
"""

from __future__ import annotations

from typing import Protocol

from src.backtest.data_cache import OHLCVBar


class _ExchangeLike(Protocol):
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
    ) -> list[list]: ...


async def fetch_ohlcv_range(
    exchange: _ExchangeLike,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    batch_limit: int = 1500,
) -> list[OHLCVBar]:
    """Fetch all bars in [since_ms, until_ms] using paginated calls.

    The result is sorted, de-duplicated, and validated. Any exchange error
    propagates; partial results are not returned. If the exchange's cursor
    fails to advance (a pathological condition), the loop breaks defensively.
    """
    if until_ms < since_ms:
        return []

    seen: dict[int, OHLCVBar] = {}
    cursor = since_ms
    last_max_ts = -1

    while True:
        batch = await exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=batch_limit)
        if not batch:
            break

        new_in_batch = 0
        max_ts_in_batch = -1
        for row in batch:
            ts = int(row[0])
            if ts > until_ms:
                continue
            if ts < since_ms:
                continue
            if ts in seen:
                continue
            bar = OHLCVBar(
                ts_ms=ts,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            bar.validate()
            seen[ts] = bar
            new_in_batch += 1
            max_ts_in_batch = max(max_ts_in_batch, ts)

        if max_ts_in_batch <= last_max_ts:
            # Cursor failed to advance — exchange is stuck or out of data
            break
        last_max_ts = max_ts_in_batch

        if max_ts_in_batch >= until_ms:
            break

        # Advance cursor past the last bar we got.
        cursor = max_ts_in_batch + 1

        if new_in_batch == 0:
            break

    return [seen[ts] for ts in sorted(seen)]
