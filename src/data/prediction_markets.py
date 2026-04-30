"""Prediction market data from Polymarket and Kalshi."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import aiohttp

from src.utils.logger import get_logger

log = get_logger(__name__)

# Keywords for filtering financial-market-relevant prediction markets
_KEYWORDS_T1 = {"bitcoin", "btc", "ethereum", "eth", "crypto", "fed", "fomc", "rate cut", "rate hike"}
_KEYWORDS_T2 = {"inflation", "recession", "tariff", "sanctions", "regulation", "sec", "etf", "interest rate"}
_KEYWORDS_T3 = {"economy", "markets", "stocks", "bonds", "dollar", "gold", "oil", "geopolitical", "war", "election"}

_ALL_KEYWORDS = _KEYWORDS_T1 | _KEYWORDS_T2 | _KEYWORDS_T3


@dataclass
class PredictionMarket:
    """Single prediction market with odds and volume."""

    source: str                     # "polymarket" | "kalshi"
    question: str
    category: str                   # crypto, economics, politics, geopolitics
    outcome_yes_price: float        # 0-1 (= probability)
    outcome_no_price: float
    volume_total: float             # lifetime USD volume
    volume_24h: float
    volume_change_pct: float | None = None
    liquidity: float | None = None
    end_date: str | None = None
    url: str | None = None
    relevance_score: float = 0.0    # 0-1


@dataclass
class PredictionMarketAnomaly:
    """Detected anomaly in a prediction market."""

    market: PredictionMarket
    anomaly_type: str               # volume_spike, odds_shift, new_high_volume
    magnitude: float                # Z-score or multiplier
    description: str


@dataclass
class PredictionMarketsSnapshot:
    """All prediction market data for one collection cycle."""

    markets: list[PredictionMarket] = field(default_factory=list)
    anomalies: list[PredictionMarketAnomaly] = field(default_factory=list)
    top_movers: list[PredictionMarket] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


def _categorize(text: str) -> str:
    """Classify a market question into a category."""
    t = text.lower()
    if any(k in t for k in ("bitcoin", "btc", "ethereum", "eth", "crypto", "token", "defi")):
        return "crypto"
    if any(k in t for k in ("fed", "fomc", "inflation", "rate", "recession", "gdp", "unemployment")):
        return "economics"
    if any(k in t for k in ("war", "sanction", "nato", "military", "conflict", "invasion")):
        return "geopolitics"
    if any(k in t for k in ("election", "president", "congress", "vote", "bill", "policy", "regulation")):
        return "politics"
    return "other"


def compute_relevance(text: str) -> float:
    """Score 0-1 based on keyword matches."""
    t = text.lower()
    score = 0.0
    for kw in _KEYWORDS_T1:
        if kw in t:
            score += 1.0
    for kw in _KEYWORDS_T2:
        if kw in t:
            score += 0.7
    for kw in _KEYWORDS_T3:
        if kw in t:
            score += 0.4
    return min(1.0, score / 3.0)  # normalize


def detect_anomalies(
    markets: list[PredictionMarket],
) -> list[PredictionMarketAnomaly]:
    """Detect statistical anomalies in prediction market data."""
    anomalies: list[PredictionMarketAnomaly] = []
    if not markets:
        return anomalies

    # Volume Z-scores by category
    cats: dict[str, list[float]] = {}
    for m in markets:
        cats.setdefault(m.category, []).append(m.volume_24h)

    cat_stats: dict[str, tuple[float, float]] = {}
    for cat, vols in cats.items():
        if len(vols) >= 3:
            mean = sum(vols) / len(vols)
            std = math.sqrt(sum((v - mean) ** 2 for v in vols) / len(vols)) or 1.0
            cat_stats[cat] = (mean, std)

    for m in markets:
        # Volume spike: Z-score > 2.0
        if m.category in cat_stats:
            mean, std = cat_stats[m.category]
            z = (m.volume_24h - mean) / std
            if z > 2.0:
                anomalies.append(PredictionMarketAnomaly(
                    market=m,
                    anomaly_type="volume_spike",
                    magnitude=round(z, 2),
                    description=f"Volume Z-score {z:.1f} in {m.category} ({m.source}): {m.question[:80]}",
                ))

        # Extreme odds (very close to resolution)
        if m.outcome_yes_price > 0.90 or m.outcome_yes_price < 0.10:
            if m.volume_24h > 10_000:
                anomalies.append(PredictionMarketAnomaly(
                    market=m,
                    anomaly_type="odds_shift",
                    magnitude=round(max(m.outcome_yes_price, 1 - m.outcome_yes_price), 2),
                    description=(
                        f"Extreme odds {m.outcome_yes_price:.0%} YES with "
                        f"${m.volume_24h:,.0f} 24h vol: {m.question[:80]}"
                    ),
                ))

    # Sort by magnitude descending
    anomalies.sort(key=lambda a: a.magnitude, reverse=True)
    return anomalies[:10]


class PredictionMarketsCollector:
    """Collects prediction market data from Polymarket and Kalshi."""

    POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
    KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session

    async def collect(self) -> PredictionMarketsSnapshot:
        """Collect from all prediction market sources concurrently."""
        import asyncio

        poly_task = self._fetch_polymarket()
        kalshi_task = self._fetch_kalshi()

        results = await asyncio.gather(poly_task, kalshi_task, return_exceptions=True)

        all_markets: list[PredictionMarket] = []
        for result in results:
            if isinstance(result, Exception):
                log.warning("prediction_market_fetch_failed", error=str(result))
                continue
            all_markets.extend(result)

        # Filter to relevant markets only
        relevant = [m for m in all_markets if m.relevance_score > 0.1]
        relevant.sort(key=lambda m: m.relevance_score, reverse=True)

        # Top movers: highest volume markets
        top_movers = sorted(relevant, key=lambda m: m.volume_24h, reverse=True)[:5]

        # Anomaly detection
        anomalies = detect_anomalies(relevant)

        return PredictionMarketsSnapshot(
            markets=relevant[:20],  # top 20 most relevant
            anomalies=anomalies,
            top_movers=top_movers,
        )

    async def _fetch_polymarket(self) -> list[PredictionMarket]:
        """Fetch markets from Polymarket Gamma API."""
        session = await self._get_session()
        markets: list[PredictionMarket] = []

        try:
            url = f"{self.POLYMARKET_GAMMA}/markets"
            params = {
                "limit": 100,
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
            }
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            for item in data if isinstance(data, list) else data.get("data", data.get("markets", [])):
                question = item.get("question", "") or item.get("title", "")
                if not question:
                    continue

                rel = compute_relevance(question)
                if rel < 0.1:
                    continue

                yes_price = 0.5
                no_price = 0.5
                # Polymarket may return outcomes as list or as fields
                outcomes = item.get("outcomePrices") or item.get("outcomes", [])
                if isinstance(outcomes, str):
                    try:
                        import json
                        prices = json.loads(outcomes)
                        if len(prices) >= 2:
                            yes_price = float(prices[0])
                            no_price = float(prices[1])
                    except (ValueError, TypeError):
                        pass
                elif isinstance(outcomes, list) and len(outcomes) >= 2:
                    try:
                        yes_price = float(outcomes[0])
                        no_price = float(outcomes[1])
                    except (ValueError, TypeError):
                        pass

                volume_total = float(item.get("volume", 0) or 0)
                volume_24h = float(item.get("volume24hr", 0) or 0)

                markets.append(PredictionMarket(
                    source="polymarket",
                    question=question,
                    category=_categorize(question),
                    outcome_yes_price=yes_price,
                    outcome_no_price=no_price,
                    volume_total=volume_total,
                    volume_24h=volume_24h,
                    liquidity=float(item.get("liquidity", 0) or 0),
                    end_date=item.get("endDate") or item.get("end_date_iso"),
                    url=f"https://polymarket.com/event/{item.get('slug', '')}",
                    relevance_score=rel,
                ))

        except Exception as e:
            log.warning("polymarket_fetch_error", error=str(e))

        return markets

    async def _fetch_kalshi(self) -> list[PredictionMarket]:
        """Fetch markets from Kalshi public API."""
        session = await self._get_session()
        markets: list[PredictionMarket] = []

        try:
            url = f"{self.KALSHI_API}/markets"
            params = {"limit": 100, "status": "open"}
            headers = {"Accept": "application/json"}
            async with session.get(url, params=params, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json()

            items = data.get("markets", [])
            for item in items:
                title = item.get("title", "") or item.get("subtitle", "")
                if not title:
                    continue

                rel = compute_relevance(title)
                if rel < 0.1:
                    continue

                yes_price = float(item.get("yes_ask", 0) or item.get("last_price", 0.5) or 0.5)
                # Kalshi prices are in cents (0-100), normalize to 0-1
                if yes_price > 1:
                    yes_price = yes_price / 100
                no_price = 1 - yes_price

                volume = float(item.get("volume", 0) or 0)
                volume_24h = float(item.get("volume_24h", 0) or 0)

                markets.append(PredictionMarket(
                    source="kalshi",
                    question=title,
                    category=_categorize(title),
                    outcome_yes_price=round(yes_price, 4),
                    outcome_no_price=round(no_price, 4),
                    volume_total=volume,
                    volume_24h=volume_24h,
                    end_date=item.get("close_time") or item.get("expiration_time"),
                    url=f"https://kalshi.com/markets/{item.get('ticker', '')}",
                    relevance_score=rel,
                ))

        except Exception as e:
            log.warning("kalshi_fetch_error", error=str(e))

        return markets

    async def close(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
