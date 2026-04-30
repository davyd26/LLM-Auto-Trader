"""Tests for prediction market data collection and anomaly detection."""

import pytest
from src.data.prediction_markets import (
    PredictionMarket,
    PredictionMarketAnomaly,
    PredictionMarketsSnapshot,
    compute_relevance,
    detect_anomalies,
    _categorize,
)


def _make_market(
    question: str = "Will BTC exceed $100K?",
    source: str = "polymarket",
    yes_price: float = 0.65,
    volume_24h: float = 50_000,
    volume_total: float = 500_000,
    category: str | None = None,
    relevance: float | None = None,
) -> PredictionMarket:
    cat = category or _categorize(question)
    rel = relevance if relevance is not None else compute_relevance(question)
    return PredictionMarket(
        source=source,
        question=question,
        category=cat,
        outcome_yes_price=yes_price,
        outcome_no_price=round(1 - yes_price, 4),
        volume_total=volume_total,
        volume_24h=volume_24h,
        relevance_score=rel,
    )


class TestCategorize:
    def test_crypto_keywords(self):
        assert _categorize("Will Bitcoin hit 100K?") == "crypto"
        assert _categorize("ETH merge success") == "crypto"

    def test_economics_keywords(self):
        assert _categorize("Will the Fed cut rates in March?") == "economics"
        assert _categorize("US inflation above 3%?") == "economics"

    def test_geopolitics_keywords(self):
        assert _categorize("Will NATO intervene in the conflict?") == "geopolitics"
        assert _categorize("New sanctions on Russia") == "geopolitics"

    def test_politics_keywords(self):
        assert _categorize("Who wins the 2026 election?") == "politics"
        assert _categorize("New regulation bill passed") == "politics"

    def test_other_fallback(self):
        assert _categorize("Will it rain tomorrow?") == "other"


class TestComputeRelevance:
    def test_high_relevance_crypto(self):
        score = compute_relevance("Bitcoin BTC price prediction crypto")
        assert score > 0.5

    def test_medium_relevance_macro(self):
        score = compute_relevance("Fed interest rate decision inflation")
        assert score > 0.3

    def test_low_relevance_unrelated(self):
        score = compute_relevance("Will it snow in Alaska?")
        assert score < 0.15

    def test_zero_for_empty(self):
        assert compute_relevance("") == 0.0

    def test_capped_at_one(self):
        # Stuff all keywords in one string
        score = compute_relevance(
            "bitcoin btc ethereum eth crypto fed fomc rate cut rate hike "
            "inflation recession tariff sanctions regulation sec etf"
        )
        assert score <= 1.0


class TestDetectAnomalies:
    def test_no_anomalies_normal_data(self):
        markets = [
            _make_market(question=f"Market {i}", volume_24h=10_000, category="crypto")
            for i in range(5)
        ]
        anomalies = detect_anomalies(markets)
        assert len(anomalies) == 0

    def test_volume_spike_detected(self):
        # Many normal markets + 1 extreme outlier to get Z > 2.0
        markets = [
            _make_market(question="Crypto market A", volume_24h=1_000, category="crypto"),
            _make_market(question="Crypto market B", volume_24h=1_200, category="crypto"),
            _make_market(question="Crypto market C", volume_24h=900, category="crypto"),
            _make_market(question="Crypto market D", volume_24h=1_100, category="crypto"),
            _make_market(question="Crypto market E", volume_24h=1_050, category="crypto"),
            _make_market(question="Crypto market F", volume_24h=950, category="crypto"),
            _make_market(question="Crypto market G", volume_24h=1_000, category="crypto"),
            _make_market(question="BTC spike market", volume_24h=500_000, category="crypto"),
        ]
        anomalies = detect_anomalies(markets)
        assert len(anomalies) >= 1
        assert any(a.anomaly_type == "volume_spike" for a in anomalies)

    def test_extreme_odds_detected(self):
        markets = [
            _make_market(question="BTC above 50K", yes_price=0.95, volume_24h=20_000, category="crypto"),
            _make_market(question="BTC above 200K", yes_price=0.05, volume_24h=15_000, category="crypto"),
            _make_market(question="ETH above 3K", yes_price=0.50, volume_24h=10_000, category="crypto"),
        ]
        anomalies = detect_anomalies(markets)
        odds_anomalies = [a for a in anomalies if a.anomaly_type == "odds_shift"]
        assert len(odds_anomalies) >= 1

    def test_empty_markets(self):
        assert detect_anomalies([]) == []

    def test_anomalies_sorted_by_magnitude(self):
        markets = [
            _make_market(question="Crypto A", volume_24h=100, category="crypto"),
            _make_market(question="Crypto B", volume_24h=100, category="crypto"),
            _make_market(question="Crypto C", volume_24h=100, category="crypto"),
            _make_market(question="BTC huge spike", volume_24h=100_000, category="crypto"),
            _make_market(question="ETH big spike", volume_24h=50_000, category="crypto"),
        ]
        anomalies = detect_anomalies(markets)
        if len(anomalies) >= 2:
            assert anomalies[0].magnitude >= anomalies[1].magnitude


class TestPredictionMarketsSnapshot:
    def test_empty_snapshot(self):
        snap = PredictionMarketsSnapshot()
        assert snap.markets == []
        assert snap.anomalies == []
        assert snap.top_movers == []

    def test_snapshot_with_data(self):
        m = _make_market()
        snap = PredictionMarketsSnapshot(
            markets=[m],
            anomalies=[],
            top_movers=[m],
        )
        assert len(snap.markets) == 1
        assert snap.markets[0].source == "polymarket"
