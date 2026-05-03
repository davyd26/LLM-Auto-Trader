#!/usr/bin/env python3
"""Fetch historical OHLCV from Stooq (free CSV endpoint) into the same
OHLCVCache used by the rest of the harness.

Stooq is a Polish data provider with reliable US stocks / ETFs / futures
/ FX history in a simple CSV format. No rate limit (vs yfinance which
fails frequently from outside the US). Used to seed the cache with
non-crypto assets that aren't on Binance.

Symbol conventions (Stooq):
    US stocks/ETFs:     <ticker>.us       (lowercase, e.g., spy.us, aapl.us)
    US futures cont.:   <code>.f          (cl.f=WTI, co.f=Brent,
                                           si.f=Silver, gc.f=Gold)
    Indices:            ^<code>           (^spx, ^dji)
    FX:                 <pair>            (eurusd)

Usage:
    python scripts/fetch_stooq.py \\
        --symbols "spy.us,qqq.us,nvda.us,tsla.us,slv.us,gld.us,uso.us,bno.us" \\
        --since 2010-01-01 \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import requests

from src.backtest.data_cache import OHLCVBar, OHLCVCache

_STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i={interval}"


def _stooq_interval(timeframe: str) -> str:
    """Map our timeframe to Stooq's interval code: d=daily, w=weekly, m=monthly."""
    return {"1d": "d", "1w": "w", "1mo": "m"}.get(timeframe, "d")


def _parse_csv(text: str) -> list[OHLCVBar]:
    """Parse Stooq CSV (Date,Open,High,Low,Close,Volume) into OHLCVBar list."""
    bars: list[OHLCVBar] = []
    if not text or "No data" in text:
        return bars
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            d = row.get("Date", "").strip()
            if not d:
                continue
            ts = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            ts_ms = int(ts.timestamp() * 1000)
            o = float(row.get("Open") or 0.0)
            h = float(row.get("High") or 0.0)
            low = float(row.get("Low") or 0.0)
            c = float(row.get("Close") or 0.0)
            v = float(row.get("Volume") or 0.0)
            if c <= 0 or h <= 0 or low <= 0:
                continue
            bars.append(OHLCVBar(ts_ms=ts_ms, open=o, high=h, low=low, close=c, volume=v))
        except (ValueError, TypeError):
            continue
    return sorted(bars, key=lambda b: b.ts_ms)


def fetch_stooq_symbol(symbol: str, timeframe: str = "1w",
                       since: str | None = None,
                       max_retries: int = 3, timeout: int = 30,
                       ) -> list[OHLCVBar]:
    """Fetch OHLCV from Stooq CSV, optional since-date filter."""
    sid = symbol.strip().lower()
    interval = _stooq_interval(timeframe)
    url = _STOOQ_URL.format(sym=quote_plus(sid), interval=interval)
    headers = {"User-Agent": "llm-auto-trader/1.0"}
    text = ""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code in {429, 500, 502, 503, 504}:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            text = resp.text or ""
            break
        except Exception:
            if attempt == max_retries - 1:
                return []
            time.sleep(1.0 * (attempt + 1))
    bars = _parse_csv(text)
    if since:
        cutoff = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cutoff_ms = int(cutoff.timestamp() * 1000)
        bars = [b for b in bars if b.ts_ms >= cutoff_ms]
    return bars


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True,
                   help="comma-separated Stooq symbols (e.g., spy.us,nvda.us,co.f)")
    p.add_argument("--timeframe", default="1w", choices=["1d", "1w", "1mo"])
    p.add_argument("--since", default="2010-01-01", help="YYYY-MM-DD")
    p.add_argument("--cache-db", required=True)
    args = p.parse_args(argv)

    cache = OHLCVCache(Path(args.cache_db).expanduser())
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"Fetching {len(symbols)} symbols from Stooq, since {args.since}, tf={args.timeframe}")

    total = 0
    for i, sym in enumerate(symbols):
        if i > 0:
            time.sleep(0.5)  # polite throttling
        try:
            bars = fetch_stooq_symbol(sym, args.timeframe, args.since)
        except Exception as e:
            print(f"  {sym:14s} ERROR — {e}", file=sys.stderr)
            continue
        if not bars:
            print(f"  {sym:14s} NO DATA", file=sys.stderr)
            continue
        cache.store(sym, args.timeframe, bars)
        first = datetime.fromtimestamp(bars[0].ts_ms / 1000, tz=timezone.utc).date()
        last = datetime.fromtimestamp(bars[-1].ts_ms / 1000, tz=timezone.utc).date()
        print(f"  {sym:14s} {len(bars):4d} bars  [{first} .. {last}]")
        total += len(bars)
    cache.close()
    print(f"\nTotal: {total} bars across {len(symbols)} symbols.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
