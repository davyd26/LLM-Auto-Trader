"""Market data collection via CCXT from multiple exchanges."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import ccxt.async_support as ccxt

from src.utils.logger import get_logger

log = get_logger(__name__)

TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")


@dataclass
class OrderBookSummary:
    """Compact order book summary for LLM consumption."""

    spread: float
    spread_pct: float
    bid_ask_imbalance: float  # -1 (all asks) to +1 (all bids)
    top_bid: float
    top_ask: float
    bid_depth: float   # total bid volume in top N levels
    ask_depth: float   # total ask volume in top N levels
    timestamp: float = field(default_factory=time.time)


@dataclass
class TickerData:
    """24h ticker summary."""

    last: float
    high: float
    low: float
    volume: float
    change_pct: float
    vwap: float | None
    timestamp: float = field(default_factory=time.time)


@dataclass
class FundingData:
    """Funding rate information for perpetual futures."""

    symbol: str
    funding_rate: float
    next_funding_time: int | None
    timestamp: float = field(default_factory=time.time)


class MarketDataCollector:
    """Collects market data from one or more exchanges via CCXT."""

    def __init__(
        self,
        trading_exchange_id: str = "binance",
        trading_api_key: str = "",
        trading_secret: str = "",
        data_exchange_ids: tuple[str, ...] = (),
        market_type: str = "swap",
        quote_currency: str = "USDT",
    ):
        self._trading_exchange_id = trading_exchange_id
        self._trading_api_key = trading_api_key
        self._trading_secret = trading_secret
        self._data_exchange_ids = data_exchange_ids
        self._market_type = market_type
        self._quote_currency = quote_currency

        self._trading_exchange: ccxt.Exchange | None = None
        self._data_exchanges: dict[str, ccxt.Exchange] = {}

    async def initialize(self) -> None:
        """Initialize exchange connections and load markets."""
        self._trading_exchange = self._create_exchange(
            self._trading_exchange_id,
            self._trading_api_key,
            self._trading_secret,
        )
        await self._trading_exchange.load_markets()
        log.info("trading_exchange_ready", exchange=self._trading_exchange_id,
                 markets=len(self._trading_exchange.markets))

        for eid in self._data_exchange_ids:
            try:
                ex = self._create_exchange(eid)
                await ex.load_markets()
                self._data_exchanges[eid] = ex
                log.info("data_exchange_ready", exchange=eid, markets=len(ex.markets))
            except Exception as e:
                log.warning("data_exchange_init_failed", exchange=eid, error=str(e))

    def _create_exchange(
        self, exchange_id: str, api_key: str = "", secret: str = ""
    ) -> ccxt.Exchange:
        cls = getattr(ccxt, exchange_id)
        config: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": self._market_type},
        }
        if api_key:
            config["apiKey"] = api_key
            config["secret"] = secret
        return cls(config)

    @property
    def trading_exchange(self) -> ccxt.Exchange:
        if self._trading_exchange is None:
            raise RuntimeError("Call initialize() first")
        return self._trading_exchange

    def validate_symbol(self, symbol: str) -> bool:
        """Check if a symbol is available on the trading exchange."""
        return symbol in self.trading_exchange.markets

    def get_available_symbols(self, candidates: list[str]) -> list[str]:
        """Filter candidate symbols to those available on the trading exchange."""
        return [s for s in candidates if self.validate_symbol(s)]

    # --- OHLCV ---

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "15m", limit: int = 100
    ) -> list[list]:
        """Fetch OHLCV candles from trading exchange.

        Returns list of [timestamp, open, high, low, close, volume].
        """
        try:
            data = await self.trading_exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )
            return data
        except Exception as e:
            log.error("fetch_ohlcv_failed", symbol=symbol, timeframe=timeframe, error=str(e))
            return []

    async def fetch_multi_timeframe_ohlcv(
        self, symbol: str, timeframes: tuple[str, ...] = TIMEFRAMES, limit: int = 100
    ) -> dict[str, list[list]]:
        """Fetch OHLCV for multiple timeframes concurrently."""
        tasks = {tf: self.fetch_ohlcv(symbol, tf, limit) for tf in timeframes}
        results = {}
        for tf, coro in tasks.items():
            results[tf] = await coro
        return results

    # --- Order Book ---

    async def fetch_orderbook_summary(
        self, symbol: str, depth: int = 10
    ) -> OrderBookSummary | None:
        """Fetch order book and return compact summary."""
        try:
            ob = await self.trading_exchange.fetch_order_book(symbol, depth)
            if not ob["bids"] or not ob["asks"]:
                return None

            top_bid = ob["bids"][0][0]
            top_ask = ob["asks"][0][0]
            bid_depth = sum(b[1] for b in ob["bids"][:depth])
            ask_depth = sum(a[1] for a in ob["asks"][:depth])
            total = bid_depth + ask_depth

            return OrderBookSummary(
                spread=top_ask - top_bid,
                spread_pct=(top_ask - top_bid) / top_bid * 100 if top_bid else 0,
                bid_ask_imbalance=(bid_depth - ask_depth) / total if total else 0,
                top_bid=top_bid,
                top_ask=top_ask,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
            )
        except Exception as e:
            log.error("fetch_orderbook_failed", symbol=symbol, error=str(e))
            return None

    # --- Ticker ---

    async def fetch_ticker(self, symbol: str) -> TickerData | None:
        """Fetch 24h ticker stats."""
        try:
            t = await self.trading_exchange.fetch_ticker(symbol)
            return TickerData(
                last=t.get("last", 0) or 0,
                high=t.get("high", 0) or 0,
                low=t.get("low", 0) or 0,
                volume=t.get("quoteVolume", 0) or 0,
                change_pct=t.get("percentage", 0) or 0,
                vwap=t.get("vwap"),
            )
        except Exception as e:
            log.error("fetch_ticker_failed", symbol=symbol, error=str(e))
            return None

    # --- Funding Rate ---

    async def fetch_funding_rate(self, symbol: str) -> FundingData | None:
        """Fetch current funding rate for perpetual futures."""
        try:
            fr = await self.trading_exchange.fetch_funding_rate(symbol)
            return FundingData(
                symbol=symbol,
                funding_rate=fr.get("fundingRate", 0) or 0,
                next_funding_time=fr.get("fundingTimestamp"),
            )
        except Exception as e:
            log.warning("fetch_funding_failed", symbol=symbol, error=str(e))
            return None

    # --- Open Interest ---

    async def fetch_open_interest(self, symbol: str) -> dict | None:
        """Fetch open interest data."""
        try:
            oi = await self.trading_exchange.fetch_open_interest(symbol)
            return {
                "symbol": symbol,
                "open_interest": oi.get("openInterestAmount", 0),
                "open_interest_value": oi.get("openInterestValue", 0),
                "timestamp": time.time(),
            }
        except Exception as e:
            log.warning("fetch_oi_failed", symbol=symbol, error=str(e))
            return None

    # --- Cross-Exchange Data ---

    async def fetch_cross_exchange_funding(
        self, symbol: str
    ) -> dict[str, float]:
        """Fetch funding rates from multiple exchanges for comparison."""
        rates: dict[str, float] = {}

        async def _fetch(name: str, exchange: ccxt.Exchange) -> None:
            try:
                # Map symbol to exchange-specific format if needed
                if symbol in exchange.markets:
                    fr = await exchange.fetch_funding_rate(symbol)
                    rates[name] = fr.get("fundingRate", 0) or 0
            except Exception:
                pass

        tasks = []
        for name, ex in self._data_exchanges.items():
            tasks.append(_fetch(name, ex))
        if tasks:
            await asyncio.gather(*tasks)

        return rates

    # --- Account / Balance ---

    async def fetch_balance(self) -> dict:
        """Fetch account balance from trading exchange.

        Returns the configured quote-currency wallet (default USDT). Most
        perpetual exchanges margin in USDT; some (Coinbase, Deribit, dYdX,
        a few Bybit markets) use USDC, USD, or BTC instead — set
        ``EXCHANGE_QUOTE_CCY`` accordingly.
        """
        try:
            balance = await self.trading_exchange.fetch_balance()
            quote = balance.get(self._quote_currency, {})
            return {
                "total": quote.get("total", 0) or 0,
                "free": quote.get("free", 0) or 0,
                "used": quote.get("used", 0) or 0,
                "currency": self._quote_currency,
            }
        except Exception as e:
            log.error("fetch_balance_failed", error=str(e))
            return {"total": 0, "free": 0, "used": 0, "currency": self._quote_currency}

    # --- Positions ---

    async def fetch_positions(self) -> list[dict]:
        """Fetch open positions from trading exchange."""
        try:
            positions = await self.trading_exchange.fetch_positions()
            return [
                {
                    "symbol": p["symbol"],
                    "side": p["side"],
                    "size": abs(p.get("contracts", 0) or 0),
                    "notional": abs(p.get("notional", 0) or 0),
                    "entry_price": p.get("entryPrice", 0) or 0,
                    "mark_price": p.get("markPrice", 0) or 0,
                    "unrealized_pnl": p.get("unrealizedPnl", 0) or 0,
                    "leverage": p.get("leverage", 1) or 1,
                    "margin_ratio": p.get("marginRatio"),
                    "liquidation_price": p.get("liquidationPrice"),
                }
                for p in positions
                if abs(p.get("contracts", 0) or 0) > 0
            ]
        except Exception as e:
            log.error("fetch_positions_failed", error=str(e))
            return []

    # --- Cleanup ---

    async def close(self) -> None:
        """Close all exchange connections."""
        if self._trading_exchange:
            await self._trading_exchange.close()
        for ex in self._data_exchanges.values():
            await ex.close()
