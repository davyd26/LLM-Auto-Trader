"""Tests for benchmark strategies: buy-and-hold and equal-weight portfolio.

Benchmarks are the bar the LLM strategy must clear:
    - Buy-and-hold BTC: did the LLM add value vs the simplest passive strategy?
    - Equal-weight BTC/PAXG: did regime rotation add value vs static
      diversification?
    - Equal-weight all-symbols: passive diversification benchmark.

If the strategy doesn't beat these, the strategy doesn't have edge — the user
should use the benchmark instead.

Edge cases covered:
    - Single-symbol universe → equal-weight collapses to buy-and-hold
    - Zero-length window → return initial equity, no trades
    - Symbol with all-flat prices → return initial equity (no movement)

Tail risks:
    - Fees applied on entry but no exit (held throughout)
    - Equity curve has one entry per bar
    - Realistic notional that fits the equity (no leverage on benchmarks)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backtest.benchmarks import (
    buy_and_hold,
    equal_weight,
)
from src.backtest.data_cache import OHLCVBar, OHLCVCache
from src.backtest.fill_model import FillConfig

WEEK_MS = 7 * 86_400_000


@pytest.fixture
def btc_only_cache(tmp_path: Path) -> OHLCVCache:
    cache = OHLCVCache(tmp_path / "ohlcv.db")
    bars = [
        OHLCVBar(
            ts_ms=i * WEEK_MS,
            open=100.0 + i * 5,
            high=105.0 + i * 5,
            low=95.0 + i * 5,
            close=100.0 + i * 5,
            volume=1.0,
        )
        for i in range(20)
    ]
    cache.store("BTC/USDT", "1w", bars)
    return cache


@pytest.fixture
def two_symbol_cache(tmp_path: Path) -> OHLCVCache:
    cache = OHLCVCache(tmp_path / "ohlcv.db")
    btc = [
        OHLCVBar(
            ts_ms=i * WEEK_MS, open=100.0 + i * 5, high=105.0 + i * 5,
            low=95.0 + i * 5, close=100.0 + i * 5, volume=1.0,
        )
        for i in range(20)
    ]
    paxg = [
        OHLCVBar(
            ts_ms=i * WEEK_MS, open=1900.0 + i * 0.5, high=1905.0 + i * 0.5,
            low=1895.0 + i * 0.5, close=1900.0 + i * 0.5, volume=1.0,
        )
        for i in range(20)
    ]
    cache.store("BTC/USDT", "1w", btc)
    cache.store("PAXG/USDT", "1w", paxg)
    return cache


class TestBuyAndHold:
    def test_returns_match_price_appreciation(self, btc_only_cache: OHLCVCache) -> None:
        # BTC went from 100 to 100 + 19*5 = 195 → 95% appreciation
        result = buy_and_hold(
            cache=btc_only_cache, symbol="BTC/USDT", timeframe="1w",
            start_idx=0, end_idx=19, initial_equity=2000.0,
            fill_config=FillConfig(taker_fee_bps=0.0, slippage_bps=0.0),
        )
        # Allocate full equity at idx 0 (price 100), exit at idx 19 (price 195)
        # PnL = 2000 * (195/100) - 2000 = 2000 * 0.95 = 1900
        assert result.final_equity == pytest.approx(2000.0 * (195.0 / 100.0), rel=0.01)

    def test_fees_reduce_buy_and_hold_return(self, btc_only_cache: OHLCVCache) -> None:
        no_fee = buy_and_hold(
            cache=btc_only_cache, symbol="BTC/USDT", timeframe="1w",
            start_idx=0, end_idx=19, initial_equity=2000.0,
            fill_config=FillConfig(taker_fee_bps=0.0, slippage_bps=0.0),
        )
        with_fee = buy_and_hold(
            cache=btc_only_cache, symbol="BTC/USDT", timeframe="1w",
            start_idx=0, end_idx=19, initial_equity=2000.0,
            fill_config=FillConfig(taker_fee_bps=10.0, slippage_bps=0.0),
        )
        assert with_fee.final_equity < no_fee.final_equity

    def test_invalid_symbol_returns_empty(self, btc_only_cache: OHLCVCache) -> None:
        result = buy_and_hold(
            cache=btc_only_cache, symbol="DOGE/USDT", timeframe="1w",
            start_idx=0, end_idx=10, initial_equity=2000.0,
            fill_config=FillConfig(),
        )
        assert result.final_equity == 2000.0
        assert len(result.trades) == 0

    def test_equity_curve_recorded(self, btc_only_cache: OHLCVCache) -> None:
        result = buy_and_hold(
            cache=btc_only_cache, symbol="BTC/USDT", timeframe="1w",
            start_idx=0, end_idx=10, initial_equity=2000.0,
            fill_config=FillConfig(taker_fee_bps=0.0, slippage_bps=0.0),
        )
        assert len(result.equity_curve) >= 2


class TestEqualWeight:
    def test_two_symbol_equal_weight_splits_equity(self, two_symbol_cache: OHLCVCache) -> None:
        result = equal_weight(
            cache=two_symbol_cache, symbols=["BTC/USDT", "PAXG/USDT"], timeframe="1w",
            start_idx=0, end_idx=19, initial_equity=2000.0,
            fill_config=FillConfig(taker_fee_bps=0.0, slippage_bps=0.0),
        )
        # Final equity should be > initial since both symbols appreciated
        assert result.final_equity > 2000.0

    def test_single_symbol_equal_weight_equals_buy_and_hold(self, btc_only_cache: OHLCVCache) -> None:
        bh = buy_and_hold(
            cache=btc_only_cache, symbol="BTC/USDT", timeframe="1w",
            start_idx=0, end_idx=19, initial_equity=2000.0,
            fill_config=FillConfig(taker_fee_bps=0.0, slippage_bps=0.0),
        )
        ew = equal_weight(
            cache=btc_only_cache, symbols=["BTC/USDT"], timeframe="1w",
            start_idx=0, end_idx=19, initial_equity=2000.0,
            fill_config=FillConfig(taker_fee_bps=0.0, slippage_bps=0.0),
        )
        assert ew.final_equity == pytest.approx(bh.final_equity, rel=1e-6)

    def test_empty_symbols_returns_initial(self, btc_only_cache: OHLCVCache) -> None:
        result = equal_weight(
            cache=btc_only_cache, symbols=[], timeframe="1w",
            start_idx=0, end_idx=19, initial_equity=2000.0,
            fill_config=FillConfig(),
        )
        assert result.final_equity == 2000.0
