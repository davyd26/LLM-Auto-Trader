"""Macro indicator collection via yfinance."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import asyncio
import yfinance as yf

from src.utils.logger import get_logger

log = get_logger(__name__)

MACRO_TICKERS = {
    "dxy": "DX-Y.NYB",
    "sp500": "^GSPC",
    "gold": "GC=F",
    "treasury_10y": "^TNX",
    "vix": "^VIX",
}


@dataclass
class MacroIndicator:
    """Single macro indicator value."""

    name: str
    value: float
    change_pct: float
    prev_value: float


@dataclass
class MacroSnapshot:
    """All macro indicators."""

    indicators: dict[str, MacroIndicator] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get(self, name: str) -> MacroIndicator | None:
        return self.indicators.get(name)


class MacroCollector:
    """Collects macro indicators via yfinance."""

    def __init__(self, tickers: dict[str, str] | None = None):
        self._tickers = tickers or MACRO_TICKERS
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def collect(self) -> MacroSnapshot:
        """Fetch all macro indicators.

        yfinance is synchronous, so we run it in a thread pool.
        """
        loop = asyncio.get_event_loop()
        try:
            snapshot = await loop.run_in_executor(self._executor, self._fetch_all)
            return snapshot
        except Exception as e:
            log.error("macro_fetch_failed", error=str(e))
            return MacroSnapshot()

    def _fetch_all(self) -> MacroSnapshot:
        """Synchronous fetch of all macro indicators."""
        snapshot = MacroSnapshot()

        for name, ticker_symbol in self._tickers.items():
            try:
                data = yf.download(
                    ticker_symbol,
                    period="5d",
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if data.empty or len(data) < 2:
                    continue

                current = float(data["Close"].iloc[-1])
                prev = float(data["Close"].iloc[-2])
                change_pct = ((current - prev) / prev * 100) if prev else 0

                snapshot.indicators[name] = MacroIndicator(
                    name=name,
                    value=round(current, 4),
                    change_pct=round(change_pct, 2),
                    prev_value=round(prev, 4),
                )
            except Exception as e:
                log.warning("macro_ticker_failed", ticker=name, error=str(e))

        return snapshot

    def close(self) -> None:
        self._executor.shutdown(wait=False)
