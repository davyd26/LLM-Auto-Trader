"""SQLite-backed cache for LLM decisions, keyed by snapshot SHA256.

The cache lets CPCV replay the same decision across all combinatorial paths
without re-invoking subagents, and lets backtest re-runs (with different risk
params) reuse decisions for free. It is also the audit trail: every live
decision is hashable to its snapshot.

The store enforces immutability — once a key is set, putting different content
for that key raises. This catches the most insidious class of bug: silently
overwriting a previous decision with a "fresh re-run" that reflects different
LLM behavior, breaking the determinism CPCV depends on.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_HEX = set("0123456789abcdef")


def _validate_hash(h: str) -> None:
    if len(h) != 64 or not all(c in _HEX for c in h):
        raise ValueError(f"snapshot_hash must be 64-char hex, got {h!r}")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    snapshot_hash TEXT PRIMARY KEY,
    decision_json TEXT NOT NULL,
    created_at    REAL NOT NULL
);
"""


class DecisionCache:
    """Persistent decision cache. Content-addressed, immutable per key."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def put(self, snapshot_hash: str, decision: dict[str, Any]) -> None:
        _validate_hash(snapshot_hash)
        if not isinstance(decision, dict):
            raise TypeError("decision must be a dict")
        if not decision:
            raise ValueError("decision is empty")

        new_blob = _canonical_json(decision)

        existing = self._conn.execute(
            "SELECT decision_json FROM decisions WHERE snapshot_hash = ?",
            (snapshot_hash,),
        ).fetchone()
        if existing is not None:
            if existing[0] != new_blob:
                raise ValueError(
                    f"decision for snapshot {snapshot_hash[:12]}... already exists "
                    "with different content; cache is immutable"
                )
            return  # idempotent no-op

        self._conn.execute(
            "INSERT INTO decisions (snapshot_hash, decision_json, created_at) VALUES (?, ?, ?)",
            (snapshot_hash, new_blob, time.time()),
        )

    def get(self, snapshot_hash: str) -> dict[str, Any] | None:
        _validate_hash(snapshot_hash)
        row = self._conn.execute(
            "SELECT decision_json FROM decisions WHERE snapshot_hash = ?",
            (snapshot_hash,),
        ).fetchone()
        if row is None:
            return None
        return copy.deepcopy(json.loads(row[0]))

    def has(self, snapshot_hash: str) -> bool:
        _validate_hash(snapshot_hash)
        row = self._conn.execute(
            "SELECT 1 FROM decisions WHERE snapshot_hash = ?",
            (snapshot_hash,),
        ).fetchone()
        return row is not None

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return int(row[0]) if row else 0

    def keys(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT snapshot_hash FROM decisions ORDER BY created_at ASC"
        ).fetchall()
        return [r[0] for r in rows]


def _canonical_json(d: dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
