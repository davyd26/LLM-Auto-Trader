#!/usr/bin/env python3
"""Generate snapshot manifest for a backtest run.

Reads bulk OHLCV from the cache (or fetches it via the configured exchange if
missing), partitions a date range into weekly observations, and writes one
JSON snapshot per observation to ``runs/<run_id>/snapshots/week_NNN.json``.

The Claude Code subagent dispatcher then iterates the manifest and, for each
pending snapshot, opens a fresh subagent that reads the snapshot JSON and
writes back a decision JSON keyed by the snapshot's content hash.

Usage:
    python scripts/prepare_snapshots.py \\
        --window 2024-01-01:2025-12-31 \\
        --universe BTC/USDT,ETH/USDT,PAXG/USDT \\
        --cache-db ~/.cache/llm-auto-trader/ohlcv.db \\
        --run-id low-freq-aggressive-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.backtest.data_cache import OHLCVCache
from src.backtest.snapshot_builder import SnapshotBuilder

WEEK_MS = 7 * 86_400_000


def parse_window(window: str) -> tuple[int, int]:
    """Parse 'YYYY-MM-DD:YYYY-MM-DD' into (start_ms, end_ms) UTC."""
    start_str, end_str = window.split(":")
    start = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def weekly_timestamps(start_ms: int, end_ms: int, day_of_week: int = 0, hour: int = 14) -> list[int]:
    """Generate Monday 14:00 UTC timestamps between start and end (inclusive)."""
    out = []
    cursor = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    # Roll forward to the next Monday at the configured hour
    days_ahead = (day_of_week - cursor.weekday()) % 7
    cursor = cursor.replace(hour=hour, minute=0, second=0, microsecond=0)
    if days_ahead == 0 and cursor.timestamp() * 1000 < start_ms:
        days_ahead = 7
    cursor = cursor.replace(day=cursor.day) if days_ahead == 0 else cursor
    if days_ahead > 0:
        from datetime import timedelta
        cursor = cursor + timedelta(days=days_ahead)
    while int(cursor.timestamp() * 1000) <= end_ms:
        out.append(int(cursor.timestamp() * 1000))
        from datetime import timedelta
        cursor = cursor + timedelta(days=7)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", required=True, help="YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--universe", required=True, help="comma-separated symbols")
    parser.add_argument("--cache-db", required=True, help="path to OHLCV SQLite cache")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs", help="parent dir for run outputs")
    parser.add_argument("--lookback-bars", type=int, default=12)
    parser.add_argument("--timeframes", default="1w", help="comma-separated timeframes")
    parser.add_argument("--initial-equity", type=float, default=2000.0)
    args = parser.parse_args(argv)

    start_ms, end_ms = parse_window(args.window)
    universe = [s.strip() for s in args.universe.split(",") if s.strip()]
    timeframes = tuple(s.strip() for s in args.timeframes.split(",") if s.strip())

    run_dir = Path(args.runs_dir) / args.run_id
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    cache = OHLCVCache(args.cache_db)
    builder = SnapshotBuilder(
        cache=cache,
        universe=universe,
        lookback_bars=args.lookback_bars,
        timeframes=timeframes,
    )

    timestamps = weekly_timestamps(start_ms, end_ms)
    if not timestamps:
        print(f"no weekly observations in window {args.window}", file=sys.stderr)
        return 2

    manifest_entries = []
    for week_idx, ts_ms in enumerate(timestamps):
        # At preparation time we don't know positions/equity yet (those evolve
        # through the simulation). The dispatcher will rebuild snapshots with
        # actual state during the run; for now we generate "initial-state"
        # snapshots so the user can review the time grid. The runner reuses
        # the SnapshotBuilder later — these on-disk snapshots are reference
        # only.
        snap = builder.build(t_ms=ts_ms, equity_usd=args.initial_equity, positions=[])
        snap_name = f"week_{week_idx:03d}.json"
        snap_path = snapshots_dir / snap_name
        snap_path.write_text(snap.to_json())
        manifest_entries.append({
            "week_idx": week_idx,
            "ts_ms": ts_ms,
            "snapshot_id_initial_state": snap.snapshot_id,
            "path": str(snap_path.relative_to(run_dir)),
        })

    manifest = {
        "run_id": args.run_id,
        "window": args.window,
        "universe": universe,
        "timeframes": list(timeframes),
        "lookback_bars": args.lookback_bars,
        "initial_equity": args.initial_equity,
        "n_observations": len(timestamps),
        "observations": manifest_entries,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(timestamps)} snapshots to {snapshots_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
