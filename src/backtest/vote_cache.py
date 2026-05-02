"""SQLite-backed cache for raw per-voter decisions.

Stores each voter's vote separately so the aggregator runs deterministically
on cached data — trying a new aggregation function is then free (just
re-aggregate over the same cached votes).

Schema:
    decisions_votes(
        snapshot_hash TEXT NOT NULL,
        voter_id      TEXT NOT NULL,
        decision_json TEXT NOT NULL,
        created_at    REAL NOT NULL,
        PRIMARY KEY (snapshot_hash, voter_id)
    )

The (snapshot_hash, voter_id) primary key enforces immutability per voter
per snapshot — putting different content for the same key raises, just
like DecisionCache.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from src.backtest.ensemble_decision import Vote

_HEX = set("0123456789abcdef")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions_votes (
    snapshot_hash TEXT NOT NULL,
    voter_id      TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    created_at    REAL NOT NULL,
    PRIMARY KEY (snapshot_hash, voter_id)
);
CREATE INDEX IF NOT EXISTS idx_votes_snapshot ON decisions_votes(snapshot_hash);
"""


def _validate_hash(h: str) -> None:
    if len(h) != 64 or not all(c in _HEX for c in h):
        raise ValueError(f"snapshot_hash must be 64-char hex, got {h!r}")


def _validate_vote(v: Vote) -> None:
    """Reject votes with malformed numeric fields."""
    for field, val in (("size", v.size), ("confidence", v.confidence)):
        if val is not None and (math.isnan(val) or math.isinf(val)):
            raise ValueError(f"vote {field} is NaN or inf")
    for field, val in (("stop_loss", v.stop_loss), ("take_profit", v.take_profit)):
        if val is not None and (math.isnan(val) or math.isinf(val)):
            raise ValueError(f"vote {field} is NaN or inf")


def _vote_to_blob(v: Vote) -> str:
    return json.dumps(asdict(v), sort_keys=True, separators=(",", ":"))


def _vote_from_blob(blob: str) -> Vote:
    d = json.loads(blob)
    return Vote(**d)


class VoteCache:
    """Per-voter decision cache. Immutable per (snapshot_hash, voter_id)."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def put(self, snapshot_hash: str, vote: Vote) -> None:
        _validate_hash(snapshot_hash)
        _validate_vote(vote)
        blob = _vote_to_blob(vote)

        existing = self._conn.execute(
            "SELECT decision_json FROM decisions_votes "
            "WHERE snapshot_hash = ? AND voter_id = ?",
            (snapshot_hash, vote.voter_id),
        ).fetchone()
        if existing is not None:
            if existing[0] != blob:
                raise ValueError(
                    f"vote for ({snapshot_hash[:12]}, {vote.voter_id!r}) already exists "
                    "with different content; cache is immutable"
                )
            return

        self._conn.execute(
            "INSERT INTO decisions_votes "
            "(snapshot_hash, voter_id, decision_json, created_at) "
            "VALUES (?, ?, ?, ?)",
            (snapshot_hash, vote.voter_id, blob, time.time()),
        )

    def get(self, snapshot_hash: str, voter_id: str) -> Vote | None:
        _validate_hash(snapshot_hash)
        row = self._conn.execute(
            "SELECT decision_json FROM decisions_votes "
            "WHERE snapshot_hash = ? AND voter_id = ?",
            (snapshot_hash, voter_id),
        ).fetchone()
        if row is None:
            return None
        return _vote_from_blob(row[0])

    def get_all(self, snapshot_hash: str) -> list[Vote]:
        _validate_hash(snapshot_hash)
        rows = self._conn.execute(
            "SELECT decision_json FROM decisions_votes "
            "WHERE snapshot_hash = ? ORDER BY voter_id ASC",
            (snapshot_hash,),
        ).fetchall()
        return [_vote_from_blob(r[0]) for r in rows]

    def has(self, snapshot_hash: str, voter_id: str) -> bool:
        _validate_hash(snapshot_hash)
        row = self._conn.execute(
            "SELECT 1 FROM decisions_votes "
            "WHERE snapshot_hash = ? AND voter_id = ?",
            (snapshot_hash, voter_id),
        ).fetchone()
        return row is not None

    def count(self, snapshot_hash: str) -> int:
        _validate_hash(snapshot_hash)
        row = self._conn.execute(
            "SELECT COUNT(*) FROM decisions_votes WHERE snapshot_hash = ?",
            (snapshot_hash,),
        ).fetchone()
        return int(row[0]) if row else 0

    def total_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM decisions_votes"
        ).fetchone()
        return int(row[0]) if row else 0
