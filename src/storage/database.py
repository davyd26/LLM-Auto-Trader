"""SQLite database with WAL mode for persistent storage."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import aiosqlite

from src.utils.logger import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    prompt_tokens INTEGER,
    cached_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    prompt_compressed BLOB,
    response_json TEXT,
    execution_time_ms INTEGER
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_open REAL,
    timestamp_close REAL,
    symbol TEXT,
    side TEXT,
    entry_price REAL,
    exit_price REAL,
    size REAL,
    notional REAL,
    leverage INTEGER,
    pnl REAL,
    pnl_pct REAL,
    fees REAL,
    llm_call_id INTEGER,
    reasoning TEXT,
    -- Phase 1 §3.5 — triple-barrier labeling (filled on close)
    tb_label INTEGER,              -- +1 PT hit / -1 SL hit / 0 time-out / NULL not computed
    tb_barrier_touched TEXT,       -- "pt" / "sl" / "time" / NULL
    tb_return_at_touch REAL,       -- signed return at the touched barrier
    -- Triple-barrier configuration (set at open time, read at close)
    pt_pct REAL,
    sl_pct REAL,
    max_holding_s REAL,
    llm_confidence REAL,           -- market_assessment.confidence at entry
    FOREIGN KEY (llm_call_id) REFERENCES llm_calls(id)
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    llm_call_id INTEGER,
    action TEXT,
    symbol TEXT,
    market_trend TEXT,
    confidence REAL,
    reasoning TEXT,
    outcome TEXT,
    FOREIGN KEY (llm_call_id) REFERENCES llm_calls(id)
);

CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    breaker_type TEXT,
    action TEXT,
    reason TEXT,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    starting_equity REAL,
    ending_equity REAL,
    pnl REAL,
    pnl_pct REAL,
    trades_count INTEGER,
    wins INTEGER,
    losses INTEGER,
    llm_calls INTEGER,
    llm_cost REAL,
    max_drawdown_pct REAL
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv(symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp_close);
CREATE INDEX IF NOT EXISTS idx_llm_calls_timestamp ON llm_calls(timestamp);
"""


class Database:
    """Async SQLite database for bot data persistence."""

    def __init__(self, db_path: str = "data/trading_bot.db"):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database and tables."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        # Run additive-column migrations for pre-existing DBs (idempotent).
        await self._migrate()
        log.info("database_initialized", path=self._db_path)

    async def _migrate(self) -> None:
        """Apply additive-column migrations. Safe to run on every startup."""
        # trades: Phase 1 triple-barrier columns
        await self._add_column_if_missing("trades", "tb_label", "INTEGER")
        await self._add_column_if_missing("trades", "tb_barrier_touched", "TEXT")
        await self._add_column_if_missing("trades", "tb_return_at_touch", "REAL")
        await self._add_column_if_missing("trades", "pt_pct", "REAL")
        await self._add_column_if_missing("trades", "sl_pct", "REAL")
        await self._add_column_if_missing("trades", "max_holding_s", "REAL")
        await self._add_column_if_missing("trades", "llm_confidence", "REAL")
        await self.db.commit()

    async def _add_column_if_missing(
        self, table: str, column: str, coltype: str
    ) -> None:
        cursor = await self.db.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in await cursor.fetchall()]
        if column not in cols:
            await self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not initialized")
        return self._db

    # --- OHLCV ---

    async def save_ohlcv(
        self, symbol: str, timeframe: str, candles: list[list]
    ) -> int:
        """Save OHLCV candles, ignoring duplicates."""
        if not candles:
            return 0
        rows = [
            (symbol, timeframe, int(c[0]), c[1], c[2], c[3], c[4], c[5])
            for c in candles
        ]
        await self.db.executemany(
            "INSERT OR IGNORE INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        await self.db.commit()
        return len(rows)

    async def get_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[list]:
        """Retrieve OHLCV candles."""
        cursor = await self.db.execute(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol, timeframe, limit),
        )
        rows = await cursor.fetchall()
        return [list(r) for r in reversed(rows)]

    # --- LLM Calls ---

    async def save_llm_call(
        self,
        prompt: str,
        response_json: str,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        cost_usd: float,
        execution_time_ms: int,
        compress: bool = True,
    ) -> int:
        """Save LLM call with optionally compressed prompt."""
        prompt_data = gzip.compress(prompt.encode()) if compress else prompt.encode()
        cursor = await self.db.execute(
            "INSERT INTO llm_calls "
            "(timestamp, prompt_tokens, cached_tokens, output_tokens, cost_usd, "
            "prompt_compressed, response_json, execution_time_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), input_tokens, cached_tokens, output_tokens, cost_usd,
             prompt_data, response_json, execution_time_ms),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def get_llm_cost_summary(self, days: int = 30) -> dict:
        """Get LLM cost summary for last N days."""
        cutoff = time.time() - (days * 86400)
        cursor = await self.db.execute(
            "SELECT COUNT(*), SUM(cost_usd), SUM(prompt_tokens), SUM(output_tokens) "
            "FROM llm_calls WHERE timestamp > ?",
            (cutoff,),
        )
        row = await cursor.fetchone()
        return {
            "calls": row[0] or 0,
            "total_cost": round(row[1] or 0, 4),
            "total_input_tokens": row[2] or 0,
            "total_output_tokens": row[3] or 0,
        }

    # --- Trades ---

    async def save_trade(self, trade: dict, llm_call_id: int | None = None) -> int:
        """Save a completed trade (including AFML triple-barrier fields if present)."""
        cursor = await self.db.execute(
            "INSERT INTO trades "
            "(timestamp_open, timestamp_close, symbol, side, entry_price, exit_price, "
            "size, notional, leverage, pnl, pnl_pct, fees, llm_call_id, reasoning, "
            "tb_label, tb_barrier_touched, tb_return_at_touch, "
            "pt_pct, sl_pct, max_holding_s, llm_confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trade.get("timestamp_open"), trade.get("timestamp_close"),
                trade.get("symbol"), trade.get("side"),
                trade.get("entry_price"), trade.get("exit_price"),
                trade.get("size"), trade.get("notional"),
                trade.get("leverage"), trade.get("pnl"),
                trade.get("pnl_pct"), trade.get("fees"),
                llm_call_id, trade.get("reasoning"),
                trade.get("tb_label"), trade.get("tb_barrier_touched"),
                trade.get("tb_return_at_touch"),
                trade.get("pt_pct"), trade.get("sl_pct"),
                trade.get("max_holding_s"), trade.get("llm_confidence"),
            ),
        )
        await self.db.commit()
        return cursor.lastrowid

    async def get_recent_trades(self, limit: int = 20) -> list[dict]:
        """Get recent completed trades."""
        cursor = await self.db.execute(
            "SELECT symbol, side, entry_price, exit_price, pnl, pnl_pct, "
            "leverage, fees, timestamp_open, timestamp_close, reasoning "
            "FROM trades ORDER BY timestamp_close DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "symbol": r[0], "side": r[1], "entry_price": r[2],
                "exit_price": r[3], "pnl": r[4], "pnl_pct": r[5],
                "leverage": r[6], "fees": r[7],
                "timestamp_open": r[8], "timestamp_close": r[9],
                "reasoning": r[10],
            }
            for r in reversed(rows)
        ]

    async def get_trade_stats(self) -> dict:
        """Get aggregate trade statistics."""
        cursor = await self.db.execute(
            "SELECT COUNT(*), "
            "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), "
            "AVG(pnl), SUM(pnl), "
            "AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END), "
            "AVG(CASE WHEN pnl <= 0 THEN ABS(pnl) ELSE NULL END) "
            "FROM trades"
        )
        row = await cursor.fetchone()
        total = row[0] or 0
        wins = row[1] or 0
        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0,
            "avg_pnl": round(row[2] or 0, 4),
            "total_pnl": round(row[3] or 0, 4),
            "avg_win": round(row[4] or 0, 4),
            "avg_loss": round(row[5] or 0, 4),
        }

    # --- Decisions ---

    async def save_decision(
        self, action: str, symbol: str, trend: str, confidence: float,
        reasoning: str, llm_call_id: int | None = None,
    ) -> int:
        cursor = await self.db.execute(
            "INSERT INTO decisions "
            "(timestamp, llm_call_id, action, symbol, market_trend, confidence, reasoning) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), llm_call_id, action, symbol, trend, confidence, reasoning),
        )
        await self.db.commit()
        return cursor.lastrowid

    # --- Circuit Breaker Events ---

    async def save_circuit_breaker_event(
        self, breaker_type: str, action: str, reason: str, details: dict
    ) -> None:
        await self.db.execute(
            "INSERT INTO circuit_breaker_events "
            "(timestamp, breaker_type, action, reason, details_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), breaker_type, action, reason, json.dumps(details)),
        )
        await self.db.commit()

    # --- Daily Summary ---

    async def save_daily_summary(self, summary: dict) -> None:
        await self.db.execute(
            "INSERT OR REPLACE INTO daily_summaries "
            "(date, starting_equity, ending_equity, pnl, pnl_pct, "
            "trades_count, wins, losses, llm_calls, llm_cost, max_drawdown_pct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                summary["date"], summary.get("starting_equity"),
                summary.get("ending_equity"), summary.get("pnl"),
                summary.get("pnl_pct"), summary.get("trades_count"),
                summary.get("wins"), summary.get("losses"),
                summary.get("llm_calls"), summary.get("llm_cost"),
                summary.get("max_drawdown_pct"),
            ),
        )
        await self.db.commit()
