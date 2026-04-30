"""Aggregates all data sources into a unified snapshot for LLM consumption."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from src.data.macro import MacroCollector, MacroSnapshot
from src.data.market import FundingData, MarketDataCollector, OrderBookSummary, TickerData
from src.data.news import NewsCollector, NewsSnapshot
from src.data.onchain import OnChainCollector, OnChainSnapshot
from src.data.prediction_markets import PredictionMarketsCollector, PredictionMarketsSnapshot
from src.data.sentiment import SentimentCollector, SentimentSnapshot
from src.data.top_traders import TopTraderCollector, TopTradersSnapshot
from src.data.websocket import WebSocketManager
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SymbolSnapshot:
    """All data for a single symbol."""

    symbol: str
    ohlcv: dict[str, list[list]] = field(default_factory=dict)  # timeframe -> candles
    ticker: TickerData | None = None
    orderbook: OrderBookSummary | None = None
    funding: FundingData | None = None
    open_interest: dict | None = None
    cross_exchange_funding: dict[str, float] = field(default_factory=dict)


@dataclass
class MarketSnapshot:
    """Complete market data snapshot for one analysis cycle."""

    # Per-symbol data
    symbols: dict[str, SymbolSnapshot] = field(default_factory=dict)

    # Global data
    onchain: OnChainSnapshot | None = None
    sentiment: SentimentSnapshot | None = None
    macro: MacroSnapshot | None = None
    prediction_markets: PredictionMarketsSnapshot | None = None
    top_traders: TopTradersSnapshot | None = None
    news: NewsSnapshot | None = None

    # Account data
    balance: dict[str, float] = field(default_factory=dict)
    positions: list[dict] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)

    @property
    def has_critical_data(self) -> bool:
        """Check if we have minimum required data for LLM analysis."""
        return bool(self.symbols) and any(
            s.ticker is not None for s in self.symbols.values()
        )


class DataAggregator:
    """Coordinates all data collectors and produces unified snapshots."""

    def __init__(
        self,
        market: MarketDataCollector,
        websocket: WebSocketManager,
        onchain: OnChainCollector,
        sentiment: SentimentCollector,
        macro: MacroCollector,
        prediction_markets: PredictionMarketsCollector | None = None,
        top_traders: TopTraderCollector | None = None,
        news: NewsCollector | None = None,
        symbols: list[str] | None = None,
    ):
        self._market = market
        self._websocket = websocket
        self._onchain = onchain
        self._sentiment = sentiment
        self._macro = macro
        self._prediction_markets = prediction_markets
        self._top_traders = top_traders
        self._news = news
        self._symbols = symbols or []

        # Cache for slow-updating data
        self._cached_onchain: OnChainSnapshot | None = None
        self._cached_onchain_time: float = 0
        self._cached_macro: MacroSnapshot | None = None
        self._cached_macro_time: float = 0
        self._cached_sentiment: SentimentSnapshot | None = None
        self._cached_sentiment_time: float = 0
        self._cached_prediction_markets: PredictionMarketsSnapshot | None = None
        self._cached_prediction_markets_time: float = 0
        self._cached_top_traders: TopTradersSnapshot | None = None
        self._cached_top_traders_time: float = 0
        self._cached_news: NewsSnapshot | None = None
        self._cached_news_time: float = 0

    async def collect_snapshot(
        self,
        fetch_onchain: bool = False,
        fetch_sentiment: bool = True,
        fetch_macro: bool = False,
        fetch_prediction_markets: bool = False,
        fetch_top_traders: bool = False,
        fetch_news: bool = False,
    ) -> MarketSnapshot:
        """Collect a complete market snapshot from all sources.

        Slow-updating sources (onchain, macro) are cached and only
        refetched when their respective flags are True.
        """
        snapshot = MarketSnapshot()
        errors: list[str] = []

        # --- Parallel: Market data for all symbols ---
        symbol_tasks: dict[str, dict[str, Any]] = {}
        for sym in self._symbols:
            symbol_tasks[sym] = {
                "ohlcv": self._market.fetch_multi_timeframe_ohlcv(sym, ("15m", "1h", "4h", "1d")),
                "ticker": self._market.fetch_ticker(sym),
                "orderbook": self._market.fetch_orderbook_summary(sym),
                "funding": self._market.fetch_funding_rate(sym),
                "oi": self._market.fetch_open_interest(sym),
            }

        # Account data
        balance_task = self._market.fetch_balance()
        positions_task = self._market.fetch_positions()

        # Auxiliary data (conditional) — all run concurrently
        onchain_task = self._onchain.collect() if fetch_onchain else None
        sentiment_task = self._sentiment.collect() if fetch_sentiment else None
        macro_task = self._macro.collect() if fetch_macro else None
        pm_task = (self._prediction_markets.collect()
                   if fetch_prediction_markets and self._prediction_markets else None)
        tt_task = (self._top_traders.collect()
                   if fetch_top_traders and self._top_traders else None)
        news_task = (self._news.collect()
                     if fetch_news and self._news else None)

        # Gather all tasks
        all_coros = []
        coro_labels = []

        for sym, tasks in symbol_tasks.items():
            for label, coro in tasks.items():
                all_coros.append(coro)
                coro_labels.append((sym, label))

        all_coros.append(balance_task)
        coro_labels.append(("_account", "balance"))
        all_coros.append(positions_task)
        coro_labels.append(("_account", "positions"))

        if onchain_task:
            all_coros.append(onchain_task)
            coro_labels.append(("_global", "onchain"))
        if sentiment_task:
            all_coros.append(sentiment_task)
            coro_labels.append(("_global", "sentiment"))
        if macro_task:
            all_coros.append(macro_task)
            coro_labels.append(("_global", "macro"))
        if pm_task:
            all_coros.append(pm_task)
            coro_labels.append(("_global", "prediction_markets"))
        if tt_task:
            all_coros.append(tt_task)
            coro_labels.append(("_global", "top_traders"))
        if news_task:
            all_coros.append(news_task)
            coro_labels.append(("_global", "news"))

        results = await asyncio.gather(*all_coros, return_exceptions=True)

        # Process results
        for (owner, label), result in zip(coro_labels, results):
            if isinstance(result, Exception):
                errors.append(f"{owner}.{label}: {result}")
                log.warning("data_collect_error", source=f"{owner}.{label}", error=str(result))
                continue

            if owner == "_account":
                if label == "balance":
                    snapshot.balance = result
                elif label == "positions":
                    snapshot.positions = result
            elif owner == "_global":
                if label == "onchain":
                    self._cached_onchain = result
                    self._cached_onchain_time = time.time()
                elif label == "sentiment":
                    self._cached_sentiment = result
                    self._cached_sentiment_time = time.time()
                elif label == "macro":
                    self._cached_macro = result
                    self._cached_macro_time = time.time()
                elif label == "prediction_markets":
                    self._cached_prediction_markets = result
                    self._cached_prediction_markets_time = time.time()
                elif label == "top_traders":
                    self._cached_top_traders = result
                    self._cached_top_traders_time = time.time()
                elif label == "news":
                    self._cached_news = result
                    self._cached_news_time = time.time()
            else:
                # Per-symbol data
                if owner not in snapshot.symbols:
                    snapshot.symbols[owner] = SymbolSnapshot(symbol=owner)
                sym_snap = snapshot.symbols[owner]

                if label == "ohlcv":
                    sym_snap.ohlcv = result
                elif label == "ticker":
                    sym_snap.ticker = result
                elif label == "orderbook":
                    sym_snap.orderbook = result
                elif label == "funding":
                    sym_snap.funding = result
                elif label == "oi":
                    sym_snap.open_interest = result

        # Use cached values for unfetched slow data
        snapshot.onchain = self._cached_onchain
        snapshot.sentiment = self._cached_sentiment
        snapshot.macro = self._cached_macro
        snapshot.prediction_markets = self._cached_prediction_markets
        snapshot.top_traders = self._cached_top_traders
        snapshot.news = self._cached_news
        snapshot.errors = errors

        # Enrich with WS real-time prices where available
        for sym in self._symbols:
            ws_state = self._websocket.get_price_state(sym)
            if ws_state and ws_state.current_price > 0:
                if sym in snapshot.symbols and snapshot.symbols[sym].ticker:
                    # WS price is more current than REST ticker
                    snapshot.symbols[sym].ticker.last = ws_state.current_price

        log.info(
            "snapshot_collected",
            symbols=len(snapshot.symbols),
            has_onchain=snapshot.onchain is not None,
            has_sentiment=snapshot.sentiment is not None,
            has_macro=snapshot.macro is not None,
            errors=len(errors),
        )

        return snapshot
