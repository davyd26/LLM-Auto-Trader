"""Top trader analysis from Polymarket leaderboard."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import aiohttp

from src.utils.logger import get_logger

log = get_logger(__name__)

# Minimum thresholds for a "qualified" trader
MIN_WIN_RATE = 0.55
MIN_PROFIT = 10_000  # $10K total profit
MIN_TRADES = 50


@dataclass
class TraderPosition:
    """A single position held by a top trader."""

    market_question: str
    side: str           # "yes" | "no"
    size_usd: float | None = None
    current_odds: float = 0.5


@dataclass
class TopTrader:
    """A high-performing trader with their stats and positions."""

    username: str
    profit_total: float
    win_rate: float     # 0-1
    total_trades: int
    active_positions: list[TraderPosition] = field(default_factory=list)

    @property
    def weight(self) -> float:
        """Composite weight based on performance metrics."""
        return compute_trader_weight(self.win_rate, self.profit_total, self.total_trades)


@dataclass
class TraderConsensus:
    """Aggregated signal from multiple top traders on the same market."""

    market_question: str
    category: str
    bullish_count: int      # traders betting YES
    bearish_count: int      # traders betting NO
    total_tracked: int
    avg_win_rate: float     # weighted avg win rate
    confidence_score: float  # 0-1 weighted by win rate and profit
    description: str


@dataclass
class TopTradersSnapshot:
    """Collection of top trader data and consensus signals."""

    traders: list[TopTrader] = field(default_factory=list)
    consensus: list[TraderConsensus] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


def compute_trader_weight(win_rate: float, profit: float, trades: int) -> float:
    """Weight = win_rate * log(1 + profit) * sqrt(trades)."""
    if profit <= 0 or trades <= 0 or win_rate <= 0:
        return 0.0
    return win_rate * math.log(1 + profit) * math.sqrt(trades)


def compute_consensus(traders: list[TopTrader]) -> list[TraderConsensus]:
    """Compute consensus signals across qualified traders' positions."""
    if not traders:
        return []

    # Group positions by market question
    market_positions: dict[str, list[tuple[TopTrader, TraderPosition]]] = {}
    for trader in traders:
        for pos in trader.active_positions:
            key = pos.market_question.strip().lower()
            market_positions.setdefault(key, []).append((trader, pos))

    consensus_list: list[TraderConsensus] = []
    for _, entries in market_positions.items():
        if len(entries) < 2:
            continue

        bullish = [(t, p) for t, p in entries if p.side == "yes"]
        bearish = [(t, p) for t, p in entries if p.side == "no"]

        # Weighted averages
        total_weight = sum(t.weight for t, _ in entries)
        if total_weight == 0:
            continue

        bullish_weight = sum(t.weight for t, _ in bullish)
        bearish_weight = sum(t.weight for t, _ in bearish)

        weighted_wr = sum(t.win_rate * t.weight for t, _ in entries) / total_weight
        confidence = max(bullish_weight, bearish_weight) / total_weight

        question = entries[0][1].market_question
        from src.data.prediction_markets import _categorize
        category = _categorize(question)

        direction = "YES" if bullish_weight > bearish_weight else "NO"
        consensus_list.append(TraderConsensus(
            market_question=question,
            category=category,
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            total_tracked=len(entries),
            avg_win_rate=round(weighted_wr, 3),
            confidence_score=round(confidence, 3),
            description=(
                f"{len(entries)} top traders on '{question[:60]}': "
                f"{len(bullish)} YES vs {len(bearish)} NO "
                f"(consensus: {direction}, confidence: {confidence:.0%})"
            ),
        ))

    # Sort by number of traders, then confidence
    consensus_list.sort(key=lambda c: (c.total_tracked, c.confidence_score), reverse=True)
    return consensus_list[:10]


class TopTraderCollector:
    """Fetches top trader data from Polymarket analytics."""

    # Polymarket profile API (public)
    POLYMARKET_PROFILE_API = "https://gamma-api.polymarket.com/leaderboard"

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def collect(self) -> TopTradersSnapshot:
        """Fetch top trader data and compute consensus."""
        traders = await self._fetch_top_traders()
        qualified = [
            t for t in traders
            if t.win_rate >= MIN_WIN_RATE
            and t.profit_total >= MIN_PROFIT
            and t.total_trades >= MIN_TRADES
        ]
        consensus = compute_consensus(qualified)

        return TopTradersSnapshot(
            traders=qualified[:20],
            consensus=consensus,
        )

    async def _fetch_top_traders(self) -> list[TopTrader]:
        """Fetch top traders from Polymarket leaderboard API."""
        session = await self._get_session()
        traders: list[TopTrader] = []

        try:
            # Polymarket Gamma API leaderboard endpoint
            url = self.POLYMARKET_PROFILE_API
            params = {"limit": 50, "window": "all"}
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    log.warning("top_traders_api_error", status=resp.status)
                    return []
                data = await resp.json()

            items = data if isinstance(data, list) else data.get("leaderboard", data.get("data", []))
            for item in items:
                username = item.get("username", "") or item.get("name", "") or item.get("address", "")[:8]
                profit = float(item.get("pnl", 0) or item.get("profit", 0) or 0)
                trades_count = int(item.get("numTrades", 0) or item.get("trades_count", 0) or 0)
                markets_won = int(item.get("marketsWon", 0) or 0)
                markets_traded = int(item.get("marketsTraded", 0) or item.get("markets_count", 0) or 0)

                win_rate = markets_won / markets_traded if markets_traded > 0 else 0.0

                # Positions may or may not be included
                positions: list[TraderPosition] = []
                for pos in item.get("positions", item.get("active_positions", [])):
                    positions.append(TraderPosition(
                        market_question=pos.get("title", "") or pos.get("question", ""),
                        side=pos.get("side", "yes").lower(),
                        size_usd=float(pos.get("size", 0) or 0),
                        current_odds=float(pos.get("price", 0.5) or 0.5),
                    ))

                traders.append(TopTrader(
                    username=username,
                    profit_total=profit,
                    win_rate=round(win_rate, 4),
                    total_trades=trades_count or markets_traded,
                    active_positions=positions,
                ))

        except Exception as e:
            log.warning("top_traders_fetch_error", error=str(e))

        return traders

    async def close(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
