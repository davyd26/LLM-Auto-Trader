"""Tests for top trader analysis and consensus computation."""

from src.data.top_traders import (
    TopTrader,
    TopTradersSnapshot,
    TraderPosition,
    compute_consensus,
    compute_trader_weight,
)


def _make_trader(
    username: str = "trader1",
    profit: float = 100_000,
    win_rate: float = 0.70,
    trades: int = 200,
    positions: list[TraderPosition] | None = None,
) -> TopTrader:
    return TopTrader(
        username=username,
        profit_total=profit,
        win_rate=win_rate,
        total_trades=trades,
        active_positions=positions or [],
    )


def _make_position(
    question: str = "Will BTC exceed $100K?",
    side: str = "yes",
    size: float = 5000,
    odds: float = 0.65,
) -> TraderPosition:
    return TraderPosition(
        market_question=question,
        side=side,
        size_usd=size,
        current_odds=odds,
    )


class TestComputeTraderWeight:
    def test_positive_weight(self):
        w = compute_trader_weight(0.70, 100_000, 200)
        assert w > 0

    def test_higher_winrate_higher_weight(self):
        w1 = compute_trader_weight(0.60, 100_000, 200)
        w2 = compute_trader_weight(0.80, 100_000, 200)
        assert w2 > w1

    def test_higher_profit_higher_weight(self):
        w1 = compute_trader_weight(0.70, 10_000, 200)
        w2 = compute_trader_weight(0.70, 1_000_000, 200)
        assert w2 > w1

    def test_more_trades_higher_weight(self):
        w1 = compute_trader_weight(0.70, 100_000, 50)
        w2 = compute_trader_weight(0.70, 100_000, 500)
        assert w2 > w1

    def test_zero_profit_zero_weight(self):
        assert compute_trader_weight(0.70, 0, 200) == 0.0

    def test_zero_trades_zero_weight(self):
        assert compute_trader_weight(0.70, 100_000, 0) == 0.0

    def test_negative_profit_zero_weight(self):
        assert compute_trader_weight(0.70, -5000, 200) == 0.0


class TestComputeConsensus:
    def test_consensus_with_agreement(self):
        question = "Will BTC exceed $100K by June 2026?"
        traders = [
            _make_trader("t1", positions=[_make_position(question, "yes")]),
            _make_trader("t2", positions=[_make_position(question, "yes")]),
            _make_trader("t3", positions=[_make_position(question, "no")]),
        ]
        consensus = compute_consensus(traders)
        assert len(consensus) == 1
        c = consensus[0]
        assert c.bullish_count == 2
        assert c.bearish_count == 1
        assert c.total_tracked == 3
        assert c.confidence_score > 0.5

    def test_no_consensus_single_trader(self):
        traders = [
            _make_trader("t1", positions=[_make_position("BTC question", "yes")]),
        ]
        consensus = compute_consensus(traders)
        assert len(consensus) == 0  # Need >= 2 traders per market

    def test_empty_traders(self):
        assert compute_consensus([]) == []

    def test_multiple_markets(self):
        traders = [
            _make_trader("t1", positions=[
                _make_position("Will BTC hit 100K?", "yes"),
                _make_position("Will ETH hit 5K?", "no"),
            ]),
            _make_trader("t2", positions=[
                _make_position("Will BTC hit 100K?", "yes"),
                _make_position("Will ETH hit 5K?", "yes"),
            ]),
        ]
        consensus = compute_consensus(traders)
        assert len(consensus) == 2

    def test_weighted_by_performance(self):
        question = "Fed rate cut?"
        # Strong trader bets YES, weak trader bets NO
        traders = [
            _make_trader("strong", profit=500_000, win_rate=0.80, trades=1000,
                         positions=[_make_position(question, "yes")]),
            _make_trader("weak", profit=10_000, win_rate=0.55, trades=50,
                         positions=[_make_position(question, "no")]),
        ]
        consensus = compute_consensus(traders)
        assert len(consensus) == 1
        c = consensus[0]
        # Strong trader's signal should dominate
        assert c.confidence_score > 0.7


class TestTopTrader:
    def test_weight_property(self):
        t = _make_trader(profit=100_000, win_rate=0.70, trades=200)
        assert t.weight > 0
        assert t.weight == compute_trader_weight(0.70, 100_000, 200)

    def test_snapshot_defaults(self):
        snap = TopTradersSnapshot()
        assert snap.traders == []
        assert snap.consensus == []
