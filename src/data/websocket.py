"""WebSocket streaming for real-time price updates and event detection."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import ccxt.pro as ccxtpro

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PriceUpdate:
    """Real-time price update from WebSocket."""

    symbol: str
    price: float
    timestamp: float
    volume_24h: float = 0
    change_pct_24h: float = 0


@dataclass
class PriceState:
    """Tracks price state for event detection."""

    symbol: str
    current_price: float = 0
    price_15m_ago: float = 0
    last_update: float = 0
    _price_history: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, price)

    def update(self, price: float, timestamp: float) -> None:
        self.current_price = price
        self.last_update = timestamp
        self._price_history.append((timestamp, price))

        # Keep only last 20 minutes of ticks
        cutoff = timestamp - 1200
        self._price_history = [
            (t, p) for t, p in self._price_history if t > cutoff
        ]

        # Update 15m-ago price
        target = timestamp - 900
        closest = min(self._price_history, key=lambda x: abs(x[0] - target), default=None)
        if closest and abs(closest[0] - target) < 120:  # within 2 min tolerance
            self.price_15m_ago = closest[1]

    @property
    def change_15m_pct(self) -> float:
        if not self.price_15m_ago or not self.current_price:
            return 0
        return (self.current_price - self.price_15m_ago) / self.price_15m_ago * 100


# Callback type: async function receiving a PriceUpdate
PriceCallback = Callable[[PriceUpdate], Awaitable[None]]


class WebSocketManager:
    """Manages WebSocket connections for real-time price streaming."""

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        secret: str = "",
        market_type: str = "swap",
    ):
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._secret = secret
        self._market_type = market_type
        self._exchange: ccxtpro.Exchange | None = None
        self._running = False
        self._price_states: dict[str, PriceState] = {}
        self._callbacks: list[PriceCallback] = []
        self._tasks: list[asyncio.Task] = []

    def on_price(self, callback: PriceCallback) -> None:
        """Register a callback for price updates."""
        self._callbacks.append(callback)

    async def initialize(self) -> None:
        """Create the exchange instance."""
        cls = getattr(ccxtpro, self._exchange_id)
        config = {
            "enableRateLimit": True,
            "options": {"defaultType": self._market_type},
        }
        if self._api_key:
            config["apiKey"] = self._api_key
            config["secret"] = self._secret
        self._exchange = cls(config)

    async def start(self, symbols: list[str]) -> None:
        """Start streaming prices for given symbols."""
        if not self._exchange:
            await self.initialize()

        self._running = True
        for symbol in symbols:
            self._price_states[symbol] = PriceState(symbol=symbol)
            task = asyncio.create_task(self._watch_ticker(symbol))
            self._tasks.append(task)

        log.info("websocket_started", symbols=symbols)

    async def _watch_ticker(self, symbol: str) -> None:
        """Watch ticker for a single symbol."""
        consecutive_errors = 0
        while self._running:
            try:
                ticker = await self._exchange.watch_ticker(symbol)
                consecutive_errors = 0

                update = PriceUpdate(
                    symbol=symbol,
                    price=ticker.get("last", 0) or 0,
                    timestamp=time.time(),
                    volume_24h=ticker.get("quoteVolume", 0) or 0,
                    change_pct_24h=ticker.get("percentage", 0) or 0,
                )

                # Update price state
                state = self._price_states[symbol]
                state.update(update.price, update.timestamp)

                # Notify callbacks
                for cb in self._callbacks:
                    try:
                        await cb(update)
                    except Exception as e:
                        log.error("price_callback_error", error=str(e))

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1
                log.error("ws_ticker_error", symbol=symbol, error=str(e),
                          consecutive=consecutive_errors)
                if consecutive_errors > 10:
                    log.critical("ws_too_many_errors", symbol=symbol)
                    break
                await asyncio.sleep(min(2 ** consecutive_errors, 30))

    def get_price_state(self, symbol: str) -> PriceState | None:
        """Get current price state for a symbol."""
        return self._price_states.get(symbol)

    def get_all_price_states(self) -> dict[str, PriceState]:
        """Get all price states."""
        return dict(self._price_states)

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is still receiving data."""
        if not self._price_states:
            return False
        now = time.time()
        return any(
            (now - s.last_update) < 30
            for s in self._price_states.values()
        )

    @property
    def seconds_since_last_update(self) -> float:
        """Seconds since last price update from any symbol."""
        if not self._price_states:
            return float("inf")
        latest = max(s.last_update for s in self._price_states.values())
        return time.time() - latest

    async def stop(self) -> None:
        """Stop all WebSocket streams."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._exchange:
            await self._exchange.close()
        log.info("websocket_stopped")
