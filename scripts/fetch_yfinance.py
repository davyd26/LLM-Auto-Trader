#!/usr/bin/env python3
"""Fetch historical OHLCV from Yahoo Finance into the same OHLCVCache used
by the rest of the backtest harness.

Used to seed the cache with non-crypto assets that aren't on Binance:
US stocks (long history), commodity ETFs (oil = USO/BNO, silver = SLV),
broad-market index ETFs (SPY, QQQ).

Symbols stored under their Yahoo ticker but tagged with timeframe '1w'
or '1d' identical to the CCXT path. The strategy code reads from
OHLCVCache irrespective of source — same schema, same access path.

Usage:
    python scripts/fetch_yfinance.py \\
        --symbols SPY,QQQ,MSFT,AAPL,GOOGL,NVDA,TSLA,COIN,MSTR,USO,BNO,SLV,GLD \\
        --timeframe 1w \\
        --since 2010-01-01 \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.backtest.data_cache import OHLCVBar, OHLCVCache

_TF_MAP = {"1d": "1d", "1w": "1wk", "1mo": "1mo"}


def fetch_symbol_yf(
    symbol: str, since: str, until: str | None, timeframe: str,
    max_retries: int = 4,
) -> list[OHLCVBar]:
    """Fetch OHLCV from yfinance and return as OHLCVBar list (UTC ms ts).

    Yahoo's weekly bars are aligned to Mondays; daily are calendar days.
    Retries with exponential backoff on rate-limit errors.
    """
    interval = _TF_MAP.get(timeframe, timeframe)
    df = None
    backoff = 5.0
    for attempt in range(max_retries):
        try:
            df = yf.download(
                symbol, start=since, end=until, interval=interval,
                auto_adjust=False, progress=False, threads=False,
            )
            if df is not None and not df.empty:
                break
        except Exception:
            pass
        time.sleep(backoff)
        backoff *= 2
    if df is None or df.empty:
        return []
    # MultiIndex columns when downloading single ticker can happen in newer
    # yfinance versions. Flatten if needed.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    bars: list[OHLCVBar] = []
    for _, row in df.iterrows():
        if pd.isna(row.get("Close")):
            continue
        ts = row.get("Date") or row.get("Datetime")
        if ts is None:
            continue
        if hasattr(ts, "tz_localize") and ts.tz is None:
            ts = ts.tz_localize("UTC")
        elif hasattr(ts, "tz_convert"):
            ts = ts.tz_convert("UTC")
        ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
        try:
            bars.append(OHLCVBar(
                ts_ms=ts_ms,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0) or 0.0),
            ))
        except (ValueError, TypeError):
            continue
    return bars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", required=True, help="comma-separated yfinance tickers")
    parser.add_argument("--timeframe", default="1w", choices=list(_TF_MAP))
    parser.add_argument("--since", required=True, help="YYYY-MM-DD")
    parser.add_argument("--until", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--cache-db", required=True)
    args = parser.parse_args(argv)

    cache_path = Path(args.cache_db).expanduser()
    cache = OHLCVCache(cache_path)
    until = args.until or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"Fetching {len(symbols)} symbols from yfinance: {symbols}")
    print(f"Range: {args.since} → {until}, timeframe: {args.timeframe}")

    total = 0
    for i, sym in enumerate(symbols):
        # Polite delay between requests to avoid Yahoo rate-limit
        if i > 0:
            time.sleep(2.0)
        try:
            bars = fetch_symbol_yf(sym, args.since, until, args.timeframe)
        except Exception as e:
            print(f"  {sym}: ERROR — {e}", file=sys.stderr)
            continue
        if not bars:
            print(f"  {sym}: NO DATA returned", file=sys.stderr)
            continue
        # Store under the yfinance ticker — strategy code can pass these
        # symbols directly to load_range with the same timeframe.
        cache.store(sym, args.timeframe, bars)
        r = cache.range(sym, args.timeframe)
        first = datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc).date() if r else "?"
        last = datetime.fromtimestamp(r[1] / 1000, tz=timezone.utc).date() if r else "?"
        print(f"  {sym:8s} stored {len(bars):4d} bars  [{first} .. {last}]")
        total += len(bars)
    cache.close()
    print(f"\nTotal: {total} bars across {len(symbols)} symbols.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
