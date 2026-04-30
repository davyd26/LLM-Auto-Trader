"""Sentiment and news data collection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import aiohttp

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FearGreedData:
    """Crypto Fear & Greed Index."""

    value: int            # 0-100
    classification: str   # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
    timestamp: float = field(default_factory=time.time)
    history_7d: list[int] = field(default_factory=list)  # last 7 days


@dataclass
class NewsItem:
    """Single news headline with sentiment."""

    title: str
    source: str
    sentiment: str  # bullish, bearish, neutral
    published_at: str
    votes_positive: int = 0
    votes_negative: int = 0


@dataclass
class SentimentSnapshot:
    """Combined sentiment data."""

    fear_greed: FearGreedData | None = None
    top_news: list[NewsItem] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SentimentCollector:
    """Collects sentiment data from free APIs."""

    def __init__(
        self,
        cryptopanic_api_key: str = "",
        session: aiohttp.ClientSession | None = None,
    ):
        self._cryptopanic_key = cryptopanic_api_key
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def collect(self) -> SentimentSnapshot:
        """Collect all sentiment data."""
        snapshot = SentimentSnapshot()

        try:
            snapshot.fear_greed = await self._fetch_fear_greed()
        except Exception as e:
            log.warning("fear_greed_fetch_failed", error=str(e))

        if self._cryptopanic_key:
            try:
                snapshot.top_news = await self._fetch_crypto_news()
            except Exception as e:
                log.warning("crypto_news_fetch_failed", error=str(e))

        return snapshot

    async def _fetch_fear_greed(self) -> FearGreedData:
        """Alternative.me Fear & Greed Index."""
        session = await self._get_session()
        url = "https://api.alternative.me/fng/?limit=7"
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()

        entries = data.get("data", [])
        if not entries:
            raise ValueError("No fear & greed data")

        current = entries[0]
        return FearGreedData(
            value=int(current["value"]),
            classification=current["value_classification"],
            history_7d=[int(e["value"]) for e in entries],
        )

    async def _fetch_crypto_news(
        self, currencies: str = "BTC,ETH", limit: int = 5
    ) -> list[NewsItem]:
        """CryptoPanic: Top crypto news with sentiment."""
        session = await self._get_session()
        url = (
            f"https://cryptopanic.com/api/v1/posts/"
            f"?auth_token={self._cryptopanic_key}"
            f"&currencies={currencies}"
            f"&kind=news"
            f"&filter=hot"
        )
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()

        items = []
        for post in data.get("results", [])[:limit]:
            votes = post.get("votes", {})
            positive = votes.get("positive", 0) + votes.get("liked", 0)
            negative = votes.get("negative", 0) + votes.get("disliked", 0)

            if positive > negative * 1.5:
                sentiment = "bullish"
            elif negative > positive * 1.5:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            items.append(NewsItem(
                title=post.get("title", ""),
                source=post.get("source", {}).get("title", "unknown"),
                sentiment=sentiment,
                published_at=post.get("published_at", ""),
                votes_positive=positive,
                votes_negative=negative,
            ))

        return items

    async def close(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
