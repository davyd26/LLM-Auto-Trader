#!/usr/bin/env python3
"""Fetch historical OHLCV from Yahoo's v8 chart API directly.

The yfinance Python library is unreliable from non-US networks (rate-
limited even for first requests, "Too Many Requests" cascade). The
v8 chart endpoint at ``query2.finance.yahoo.com/v8/finance/chart/<sym>``
returns JSON without auth/crumb and is far more reliable.

Same OHLCVCache schema as the rest of the harness, so symbols fetched
here become available to ``sweep_symbols.py`` etc. without changes.

Usage:
    python scripts/fetch_yahoo_v8.py \\
        --symbols "MSFT,AAPL,SPY,QQQ,SLV,GLD,USO,BNO,NVDA,TSLA,COIN,MSTR" \\
        --since 2010-01-01 \\
        --timeframe 1wk \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.backtest.data_cache import OHLCVBar, OHLCVCache

_BASE = "https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
_HEADERS = {
    # Yahoo v8 CDN returns 429 when sent a browser-like Accept header. Send
    # just the User-Agent and let requests' default Accept header pass.
    "User-Agent": "Mozilla/5.0",
}

# Map our timeframe codes to Yahoo intervals
_TF_MAP = {"1d": "1d", "1w": "1wk", "1wk": "1wk", "1mo": "1mo"}


def fetch_yahoo_v8(
    symbol: str, since: str, until: str | None = None,
    interval: str = "1wk", timeout: int = 30, max_retries: int = 3,
) -> list[OHLCVBar]:
    """Fetch OHLCV from Yahoo v8 chart API. Returns OHLCVBar list."""
    yint = _TF_MAP.get(interval, interval)
    period1 = int(datetime.strptime(since, "%Y-%m-%d")
                  .replace(tzinfo=timezone.utc).timestamp())
    if until:
        period2 = int(datetime.strptime(until, "%Y-%m-%d")
                      .replace(tzinfo=timezone.utc).timestamp())
    else:
        period2 = int(datetime.now(timezone.utc).timestamp())

    url = _BASE.format(sym=symbol)
    # Note: NO 'events' param. The CDN routes 'events' requests through a
    # different path that's aggressively rate-limited from non-US IPs.
    params = {"interval": yint, "period1": period1, "period2": period2}

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(5.0 * (attempt + 1))
                continue
            if resp.status_code != 200:
                if attempt == max_retries - 1:
                    return []
                time.sleep(2.0 * (attempt + 1))
                continue
            data = resp.json()
            chart = data.get("chart", {})
            result_list = chart.get("result")
            if not result_list:
                err = chart.get("error", {})
                if err and attempt == max_retries - 1:
                    print(f"    error: {err.get('description', err)}", file=sys.stderr)
                return []
            result = result_list[0]
            ts = result.get("timestamp") or []
            quote = result.get("indicators", {}).get("quote", [{}])[0]
            opens = quote.get("open") or []
            highs = quote.get("high") or []
            lows = quote.get("low") or []
            closes = quote.get("close") or []
            volumes = quote.get("volume") or []
            bars: list[OHLCVBar] = []
            for i, t in enumerate(ts):
                try:
                    o = opens[i]
                    h = highs[i]
                    lo = lows[i]
                    c = closes[i]
                    v = volumes[i] if i < len(volumes) else None
                    if c is None or h is None or lo is None or o is None:
                        continue
                    bars.append(OHLCVBar(
                        ts_ms=int(t) * 1000,
                        open=float(o), high=float(h), low=float(lo),
                        close=float(c),
                        volume=float(v) if v is not None else 0.0,
                    ))
                except (TypeError, ValueError, IndexError):
                    continue
            return sorted(bars, key=lambda b: b.ts_ms)
        except requests.RequestException:
            if attempt == max_retries - 1:
                return []
            time.sleep(3.0 * (attempt + 1))
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True,
                   help="comma-separated Yahoo tickers (MSFT, SPY, GLD, etc.)")
    p.add_argument("--timeframe", default="1w", choices=["1d", "1w", "1wk", "1mo"])
    p.add_argument("--since", required=True, help="YYYY-MM-DD")
    p.add_argument("--until", default=None)
    p.add_argument("--cache-db", required=True)
    p.add_argument("--delay-seconds", type=float, default=1.5,
                   help="delay between symbol fetches")
    args = p.parse_args(argv)

    cache = OHLCVCache(Path(args.cache_db).expanduser())
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"Fetching {len(syms)} symbols from Yahoo v8 since {args.since}, tf={args.timeframe}")

    total = 0
    failed = []
    for i, sym in enumerate(syms):
        if i > 0:
            time.sleep(args.delay_seconds)
        try:
            bars = fetch_yahoo_v8(sym, args.since, args.until, args.timeframe)
        except Exception as e:
            print(f"  {sym:8s} ERROR — {e}", file=sys.stderr)
            failed.append(sym)
            continue
        if not bars:
            print(f"  {sym:8s} NO DATA", file=sys.stderr)
            failed.append(sym)
            continue
        # Store under the Yahoo ticker; same access path via cache.load_range.
        cache.store(sym, args.timeframe, bars)
        first = datetime.fromtimestamp(bars[0].ts_ms / 1000, tz=timezone.utc).date()
        last = datetime.fromtimestamp(bars[-1].ts_ms / 1000, tz=timezone.utc).date()
        print(f"  {sym:8s} {len(bars):4d} bars  [{first} .. {last}]")
        total += len(bars)
    cache.close()
    print(f"\nTotal: {total} bars across {len(syms) - len(failed)} symbols.")
    if failed:
        print(f"Failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
