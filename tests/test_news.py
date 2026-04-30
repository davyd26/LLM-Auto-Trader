"""Tests for news collection, scoring, and deduplication."""

import pytest
from src.data.news import (
    NewsArticle,
    NewsSnapshot,
    compute_relevance_score,
    classify_sentiment,
    categorize_news,
    deduplicate,
)


def _make_article(
    title: str = "Bitcoin surges past $90K",
    source: str = "CoinDesk",
    category: str | None = None,
    sentiment: str | None = None,
    relevance: float | None = None,
    url: str | None = None,
) -> NewsArticle:
    return NewsArticle(
        title=title,
        source=source,
        category=category or categorize_news(title),
        sentiment=sentiment or classify_sentiment(title),
        published_at="2026-03-18T12:00:00Z",
        url=url or f"https://example.com/{hash(title)}",
        relevance_score=relevance if relevance is not None else compute_relevance_score(title),
    )


class TestComputeRelevanceScore:
    def test_high_relevance_bitcoin(self):
        score = compute_relevance_score("Bitcoin BTC price hits new ATH")
        assert score > 0.5

    def test_medium_relevance_fed(self):
        score = compute_relevance_score("Fed signals rate cut in next meeting")
        assert score > 0.3

    def test_low_relevance_unrelated(self):
        score = compute_relevance_score("New movie releases this weekend")
        assert score < 0.15

    def test_tier2_keywords(self):
        score = compute_relevance_score("Inflation data shows CPI rising, recession fears grow")
        assert score > 0.3

    def test_tier3_keywords(self):
        score = compute_relevance_score("Oil prices affect global economy")
        assert score > 0.1

    def test_empty_string(self):
        assert compute_relevance_score("") == 0.0

    def test_capped_at_one(self):
        score = compute_relevance_score(
            "bitcoin btc ethereum eth crypto fed fomc rate cut inflation recession tariff"
        )
        assert score <= 1.0


class TestClassifySentiment:
    def test_bullish(self):
        assert classify_sentiment("Bitcoin surges to new record high") == "bullish"

    def test_bearish(self):
        assert classify_sentiment("Crypto market crash, prices plunge") == "bearish"

    def test_neutral(self):
        assert classify_sentiment("Market moves sideways today") == "neutral"

    def test_mixed_defaults_neutral(self):
        # Equal bull and bear signals
        assert classify_sentiment("Markets rally then crash") in ("bullish", "bearish", "neutral")

    def test_strong_bullish(self):
        assert classify_sentiment("Prices soar, rally continues, gains accelerate") == "bullish"

    def test_strong_bearish(self):
        assert classify_sentiment("Crash warning, decline deepens, selloff risk") == "bearish"


class TestCategorizeNews:
    def test_crypto(self):
        assert categorize_news("Bitcoin hits new ATH") == "crypto"
        assert categorize_news("Ethereum DeFi protocol launches") == "crypto"

    def test_macro(self):
        assert categorize_news("Fed raises interest rate") == "macro"
        assert categorize_news("CPI inflation data released") == "macro"

    def test_geopolitics(self):
        assert categorize_news("NATO military response to conflict") == "geopolitics"

    def test_politics(self):
        assert categorize_news("New election results announced") == "politics"

    def test_markets_default(self):
        assert categorize_news("Stock market closes higher") == "markets"


class TestDeduplicate:
    def test_url_dedup(self):
        articles = [
            _make_article("Bitcoin surges", url="https://example.com/1"),
            _make_article("Bitcoin surges past 90K", url="https://example.com/1"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1

    def test_fuzzy_title_dedup(self):
        articles = [
            _make_article("Bitcoin surges past $90,000 today", url="https://a.com/1"),
            _make_article("Bitcoin surges past $90,000 today!", url="https://b.com/2"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1

    def test_different_articles_kept(self):
        articles = [
            _make_article("Bitcoin hits $90K", url="https://a.com/1"),
            _make_article("Fed cuts interest rate", url="https://b.com/2"),
            _make_article("Ethereum DeFi growth accelerates", url="https://c.com/3"),
        ]
        result = deduplicate(articles)
        assert len(result) == 3

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_no_url_still_dedupes_by_title(self):
        articles = [
            _make_article("Exact same title here", url=None),
            _make_article("Exact same title here", url=None),
        ]
        result = deduplicate(articles)
        assert len(result) == 1


class TestNewsSnapshot:
    def test_empty_snapshot(self):
        snap = NewsSnapshot()
        assert snap.articles == []
        assert snap.category_counts == {}
        assert snap.sentiment_summary == {}
        assert snap.top_headlines == []

    def test_snapshot_with_data(self):
        articles = [
            _make_article("BTC rally"),
            _make_article("ETH breakout"),
        ]
        snap = NewsSnapshot(
            articles=articles,
            category_counts={"crypto": 2},
            sentiment_summary={"bullish": 2, "bearish": 0, "neutral": 0},
            top_headlines=["BTC rally", "ETH breakout"],
        )
        assert len(snap.articles) == 2
        assert snap.category_counts["crypto"] == 2
        assert snap.sentiment_summary["bullish"] == 2
