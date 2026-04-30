"""Mass parallel news fetching from many free sources."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import aiohttp

from src.utils.logger import get_logger

log = get_logger(__name__)

# Relevance keywords with weights
_KEYWORDS = {
    # Tier 1 (weight 1.0)
    "bitcoin": 1.0, "btc": 1.0, "ethereum": 1.0, "eth": 1.0,
    "crypto": 1.0, "fed": 1.0, "fomc": 1.0, "rate cut": 1.0,
    "rate hike": 1.0, "federal reserve": 1.0,
    # Tier 2 (weight 0.7)
    "inflation": 0.7, "recession": 0.7, "tariff": 0.7, "sanctions": 0.7,
    "regulation": 0.7, "sec": 0.7, "etf": 0.7, "interest rate": 0.7,
    "cpi": 0.7, "jobs report": 0.7, "nonfarm": 0.7,
    # Tier 3 (weight 0.4)
    "economy": 0.4, "markets": 0.4, "stocks": 0.4, "bonds": 0.4,
    "dollar": 0.4, "gold": 0.4, "oil": 0.4, "geopolitical": 0.4,
    "trade war": 0.4, "election": 0.4, "treasury": 0.4, "yields": 0.4,
    "commodity": 0.4, "forex": 0.4,
}

# Sentiment keywords
_BULLISH = {"surge", "soar", "rally", "bull", "gain", "up", "rise", "high", "record", "boost",
            "approve", "adopt", "bullish", "breakout", "growth", "recovery"}
_BEARISH = {"crash", "plunge", "dump", "bear", "drop", "fall", "low", "decline", "selloff",
            "reject", "ban", "bearish", "breakdown", "recession", "warning", "risk"}

# RSS feed URLs (free, no auth)
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "CNBC Economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "The Block": "https://www.theblock.co/rss/all",
}

# Reddit subs (free JSON API)
REDDIT_SUBS = {
    "r/CryptoCurrency": "https://www.reddit.com/r/CryptoCurrency/hot.json?limit=10",
    "r/wallstreetbets": "https://www.reddit.com/r/wallstreetbets/hot.json?limit=10",
    "r/economics": "https://www.reddit.com/r/economics/hot.json?limit=8",
    "r/geopolitics": "https://www.reddit.com/r/geopolitics/hot.json?limit=5",
}


@dataclass
class NewsArticle:
    """Single news article with metadata."""

    title: str
    source: str
    category: str           # crypto, macro, politics, geopolitics, markets
    sentiment: str | None   # bullish, bearish, neutral
    published_at: str
    url: str | None = None
    summary: str | None = None
    relevance_score: float = 0.0


@dataclass
class NewsSnapshot:
    """Aggregated news from all sources."""

    articles: list[NewsArticle] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)
    sentiment_summary: dict[str, int] = field(default_factory=dict)
    top_headlines: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


def compute_relevance_score(text: str) -> float:
    """Score 0-1 based on keyword matches."""
    t = text.lower()
    score = 0.0
    for keyword, weight in _KEYWORDS.items():
        if keyword in t:
            score += weight
    return min(1.0, score / 3.0)


def classify_sentiment(text: str) -> str:
    """Simple keyword-based sentiment classification."""
    t = text.lower()
    bull_count = sum(1 for w in _BULLISH if w in t)
    bear_count = sum(1 for w in _BEARISH if w in t)
    if bull_count > bear_count:
        return "bullish"
    if bear_count > bull_count:
        return "bearish"
    return "neutral"


def categorize_news(text: str) -> str:
    """Classify news into a category."""
    t = text.lower()
    if any(k in t for k in ("bitcoin", "btc", "ethereum", "eth", "crypto", "token", "defi", "blockchain")):
        return "crypto"
    if any(k in t for k in ("war", "sanction", "nato", "military", "conflict", "geopolitical")):
        return "geopolitics"
    if any(k in t for k in ("election", "president", "congress", "vote", "bill", "policy")):
        return "politics"
    if any(k in t for k in ("fed", "inflation", "gdp", "unemployment", "cpi", "fomc", "rate")):
        return "macro"
    return "markets"


def deduplicate(articles: list[NewsArticle]) -> list[NewsArticle]:
    """Remove duplicate articles by URL and fuzzy title matching."""
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    unique: list[NewsArticle] = []

    for art in articles:
        # URL dedup
        if art.url and art.url in seen_urls:
            continue
        if art.url:
            seen_urls.add(art.url)

        # Fuzzy title dedup
        is_dup = False
        for seen_title in seen_titles:
            if SequenceMatcher(None, art.title.lower(), seen_title).ratio() > 0.80:
                is_dup = True
                break
        if is_dup:
            continue

        seen_titles.append(art.title.lower())
        unique.append(art)

    return unique


class NewsCollector:
    """Mass parallel news collector from many free sources."""

    def __init__(
        self,
        cryptopanic_api_key: str = "",
        newsdata_api_key: str = "",
        gnews_api_key: str = "",
        session: aiohttp.ClientSession | None = None,
    ):
        self._cryptopanic_key = cryptopanic_api_key
        self._newsdata_key = newsdata_api_key
        self._gnews_key = gnews_api_key
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8),
                headers={"User-Agent": "LLMAutoTrader/1.0"},
            )
        return self._session

    async def collect(self) -> NewsSnapshot:
        """Fetch from ALL sources concurrently with 5s timeout per source."""
        tasks = []

        # RSS feeds
        for name, url in RSS_FEEDS.items():
            tasks.append(self._safe_fetch(self._fetch_rss, name, url))

        # Reddit
        for name, url in REDDIT_SUBS.items():
            tasks.append(self._safe_fetch(self._fetch_reddit, name, url))

        # CryptoPanic
        if self._cryptopanic_key:
            tasks.append(self._safe_fetch(self._fetch_cryptopanic))

        # NewsData.io
        if self._newsdata_key:
            tasks.append(self._safe_fetch(self._fetch_newsdata))

        # GNews
        if self._gnews_key:
            tasks.append(self._safe_fetch(self._fetch_gnews))

        # Fire ALL concurrently
        results = await asyncio.gather(*tasks)

        # Flatten
        all_articles: list[NewsArticle] = []
        for result in results:
            if result:
                all_articles.extend(result)

        # Filter by relevance
        relevant = [a for a in all_articles if a.relevance_score > 0.15]

        # Deduplicate
        unique = deduplicate(relevant)

        # Sort by relevance
        unique.sort(key=lambda a: a.relevance_score, reverse=True)

        # Build snapshot
        cat_counts: dict[str, int] = {}
        sent_counts: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
        for a in unique:
            cat_counts[a.category] = cat_counts.get(a.category, 0) + 1
            if a.sentiment:
                sent_counts[a.sentiment] = sent_counts.get(a.sentiment, 0) + 1

        return NewsSnapshot(
            articles=unique[:50],
            category_counts=cat_counts,
            sentiment_summary=sent_counts,
            top_headlines=[a.title for a in unique[:10]],
        )

    async def _safe_fetch(self, fn, *args) -> list[NewsArticle] | None:
        """Wrap fetch in timeout + error handling."""
        try:
            return await asyncio.wait_for(fn(*args), timeout=5.0)
        except asyncio.TimeoutError:
            log.debug("news_fetch_timeout", source=str(args[:1]))
            return None
        except Exception as e:
            log.debug("news_fetch_error", source=str(args[:1]), error=str(e))
            return None

    async def _fetch_rss(self, source_name: str, url: str) -> list[NewsArticle]:
        """Parse RSS feed."""
        import feedparser

        session = await self._get_session()
        async with session.get(url) as resp:
            text = await resp.text()

        feed = feedparser.parse(text)
        articles: list[NewsArticle] = []

        for entry in feed.entries[:15]:
            title = entry.get("title", "")
            if not title:
                continue

            relevance = compute_relevance_score(title)
            summary_text = entry.get("summary", "")[:200] if entry.get("summary") else None

            articles.append(NewsArticle(
                title=title.strip(),
                source=source_name,
                category=categorize_news(title),
                sentiment=classify_sentiment(title + " " + (summary_text or "")),
                published_at=entry.get("published", ""),
                url=entry.get("link"),
                summary=summary_text,
                relevance_score=relevance,
            ))

        return articles

    async def _fetch_reddit(self, sub_name: str, url: str) -> list[NewsArticle]:
        """Fetch Reddit hot posts as news."""
        session = await self._get_session()
        headers = {"User-Agent": "LLMAutoTrader/1.0 (trading bot)"}
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        articles: list[NewsArticle] = []
        for post in data.get("data", {}).get("children", []):
            pdata = post.get("data", {})
            title = pdata.get("title", "")
            if not title or pdata.get("stickied"):
                continue

            relevance = compute_relevance_score(title)

            articles.append(NewsArticle(
                title=title.strip(),
                source=sub_name,
                category=categorize_news(title),
                sentiment=classify_sentiment(title),
                published_at=str(pdata.get("created_utc", "")),
                url=f"https://reddit.com{pdata.get('permalink', '')}",
                relevance_score=relevance,
            ))

        return articles

    async def _fetch_cryptopanic(self) -> list[NewsArticle]:
        """CryptoPanic API with sentiment."""
        session = await self._get_session()
        url = (
            f"https://cryptopanic.com/api/v1/posts/"
            f"?auth_token={self._cryptopanic_key}"
            f"&currencies=BTC,ETH"
            f"&kind=news&filter=hot"
        )
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()

        articles: list[NewsArticle] = []
        for post in data.get("results", [])[:10]:
            title = post.get("title", "")
            if not title:
                continue

            votes = post.get("votes", {})
            positive = votes.get("positive", 0) + votes.get("liked", 0)
            negative = votes.get("negative", 0) + votes.get("disliked", 0)
            if positive > negative * 1.5:
                sentiment = "bullish"
            elif negative > positive * 1.5:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            articles.append(NewsArticle(
                title=title.strip(),
                source="CryptoPanic",
                category="crypto",
                sentiment=sentiment,
                published_at=post.get("published_at", ""),
                url=post.get("url"),
                relevance_score=compute_relevance_score(title),
            ))

        return articles

    async def _fetch_newsdata(self) -> list[NewsArticle]:
        """NewsData.io free API."""
        session = await self._get_session()
        url = "https://newsdata.io/api/1/latest"
        params = {
            "apikey": self._newsdata_key,
            "q": "bitcoin OR crypto OR fed OR inflation OR tariff",
            "language": "en",
            "size": 10,
        }
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        articles: list[NewsArticle] = []
        for item in data.get("results", []):
            title = item.get("title", "")
            if not title:
                continue
            desc = (item.get("description") or "")[:200]
            articles.append(NewsArticle(
                title=title.strip(),
                source="NewsData",
                category=categorize_news(title),
                sentiment=classify_sentiment(title + " " + desc),
                published_at=item.get("pubDate", ""),
                url=item.get("link"),
                summary=desc or None,
                relevance_score=compute_relevance_score(title),
            ))

        return articles

    async def _fetch_gnews(self) -> list[NewsArticle]:
        """GNews free API."""
        session = await self._get_session()
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": "crypto finance economy",
            "token": self._gnews_key,
            "lang": "en",
            "max": 10,
        }
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        articles: list[NewsArticle] = []
        for item in data.get("articles", []):
            title = item.get("title", "")
            if not title:
                continue
            desc = (item.get("description") or "")[:200]
            articles.append(NewsArticle(
                title=title.strip(),
                source="GNews",
                category=categorize_news(title),
                sentiment=classify_sentiment(title + " " + desc),
                published_at=item.get("publishedAt", ""),
                url=item.get("url"),
                summary=desc or None,
                relevance_score=compute_relevance_score(title),
            ))

        return articles

    async def close(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
