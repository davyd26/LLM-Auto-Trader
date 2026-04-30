"""Data retention and cleanup for storage management."""

from __future__ import annotations

import time

import aiosqlite

from src.utils.logger import get_logger

log = get_logger(__name__)


class RetentionManager:
    """Manages data retention policies to stay within storage budget."""

    def __init__(
        self,
        db: aiosqlite.Connection,
        ohlcv_1m_days: int = 2,
        ohlcv_5m_days: int = 30,
        ohlcv_15m_plus_days: int = 365,
        llm_prompt_full_days: int = 90,
        sentiment_days: int = 90,
        orderbook_days: int = 7,
    ):
        self._db = db
        self.ohlcv_1m_days = ohlcv_1m_days
        self.ohlcv_5m_days = ohlcv_5m_days
        self.ohlcv_15m_plus_days = ohlcv_15m_plus_days
        self.llm_prompt_full_days = llm_prompt_full_days
        self.sentiment_days = sentiment_days
        self.orderbook_days = orderbook_days

    async def run_cleanup(self) -> dict[str, int]:
        """Run all retention policies. Returns count of deleted rows per table."""
        results = {}

        # OHLCV cleanup by timeframe
        for tf, days in [("1m", self.ohlcv_1m_days), ("5m", self.ohlcv_5m_days)]:
            cutoff_ms = int((time.time() - days * 86400) * 1000)
            cursor = await self._db.execute(
                "DELETE FROM ohlcv WHERE timeframe = ? AND timestamp < ?",
                (tf, cutoff_ms),
            )
            results[f"ohlcv_{tf}"] = cursor.rowcount

        # LLM prompt cleanup: null out compressed prompts older than retention
        cutoff = time.time() - self.llm_prompt_full_days * 86400
        cursor = await self._db.execute(
            "UPDATE llm_calls SET prompt_compressed = NULL WHERE timestamp < ? AND prompt_compressed IS NOT NULL",
            (cutoff,),
        )
        results["llm_prompts_cleared"] = cursor.rowcount

        await self._db.commit()

        total = sum(results.values())
        if total > 0:
            log.info("retention_cleanup", **results)

        return results

    async def get_storage_stats(self) -> dict:
        """Get storage usage statistics."""
        stats = {}

        # Row counts
        for table in ["ohlcv", "llm_calls", "trades", "decisions", "circuit_breaker_events"]:
            cursor = await self._db.execute(f"SELECT COUNT(*) FROM {table}")
            row = await cursor.fetchone()
            stats[f"{table}_rows"] = row[0]

        # Database file size (approximate via page count)
        cursor = await self._db.execute("PRAGMA page_count")
        page_count = (await cursor.fetchone())[0]
        cursor = await self._db.execute("PRAGMA page_size")
        page_size = (await cursor.fetchone())[0]
        stats["db_size_mb"] = round(page_count * page_size / (1024 * 1024), 2)

        return stats
