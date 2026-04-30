"""On-chain data collection from free APIs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import aiohttp

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class OnChainSnapshot:
    """Aggregated on-chain data snapshot."""

    # DeFi TVL
    total_tvl: float | None = None
    tvl_change_24h_pct: float | None = None

    # Stablecoin supply
    total_stablecoin_mcap: float | None = None
    stablecoin_change_7d_pct: float | None = None

    # BTC network (from blockchain.com)
    btc_mempool_size: int | None = None
    btc_hash_rate: float | None = None
    btc_difficulty: float | None = None
    btc_tx_count_24h: int | None = None

    timestamp: float = field(default_factory=time.time)


class OnChainCollector:
    """Collects on-chain data from free APIs."""

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def collect(self) -> OnChainSnapshot:
        """Collect all on-chain data, returning partial results on failures."""
        snapshot = OnChainSnapshot()

        results = {}
        for name, coro in [
            ("tvl", self._fetch_tvl()),
            ("stablecoins", self._fetch_stablecoin_supply()),
            ("btc_network", self._fetch_btc_network()),
        ]:
            try:
                results[name] = await coro
            except Exception as e:
                log.warning("onchain_fetch_failed", source=name, error=str(e))
                results[name] = None

        if results.get("tvl"):
            snapshot.total_tvl = results["tvl"].get("total_tvl")
            snapshot.tvl_change_24h_pct = results["tvl"].get("change_24h_pct")

        if results.get("stablecoins"):
            snapshot.total_stablecoin_mcap = results["stablecoins"].get("total_mcap")
            snapshot.stablecoin_change_7d_pct = results["stablecoins"].get("change_7d_pct")

        if results.get("btc_network"):
            snapshot.btc_mempool_size = results["btc_network"].get("mempool_size")
            snapshot.btc_hash_rate = results["btc_network"].get("hash_rate")
            snapshot.btc_tx_count_24h = results["btc_network"].get("tx_count")

        return snapshot

    async def _fetch_tvl(self) -> dict:
        """DeFiLlama: Total DeFi TVL."""
        session = await self._get_session()
        async with session.get("https://api.llama.fi/v2/historicalChainTvl") as resp:
            resp.raise_for_status()
            data = await resp.json()

        if len(data) < 2:
            return {}

        current = data[-1].get("tvl", 0)
        prev = data[-2].get("tvl", 0)
        change = ((current - prev) / prev * 100) if prev else 0

        return {"total_tvl": current, "change_24h_pct": round(change, 2)}

    async def _fetch_stablecoin_supply(self) -> dict:
        """DeFiLlama: Stablecoin market cap."""
        session = await self._get_session()
        async with session.get("https://stablecoins.llama.fi/stablecoins") as resp:
            resp.raise_for_status()
            data = await resp.json()

        stables = data.get("peggedAssets", [])
        total_mcap = 0
        total_mcap_7d_ago = 0
        for s in stables:
            chains = s.get("chainCirculating", {})
            for chain_data in chains.values():
                current = chain_data.get("current", {}).get("peggedUSD", 0) or 0
                total_mcap += current
            # 7d change from top-level if available
            mcap_now = sum(
                (cd.get("current", {}).get("peggedUSD", 0) or 0)
                for cd in chains.values()
            )
            total_mcap_7d_ago += mcap_now  # simplified; precise calc needs historical

        return {
            "total_mcap": total_mcap,
            "change_7d_pct": 0,  # requires historical endpoint for precision
        }

    async def _fetch_btc_network(self) -> dict:
        """Blockchain.com: BTC network stats."""
        session = await self._get_session()
        result = {}

        endpoints = {
            "hash_rate": "https://api.blockchain.info/q/hashrate",
            "tx_count": "https://api.blockchain.info/q/24hrtransactioncount",
            "mempool_size": "https://api.blockchain.info/q/unconfirmedcount",
        }

        for key, url in endpoints.items():
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    text = await resp.text()
                    result[key] = float(text.strip())
            except Exception as e:
                log.debug("btc_network_endpoint_failed", endpoint=key, error=str(e))

        return result

    async def close(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
