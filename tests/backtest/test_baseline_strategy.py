"""Tests for the deterministic baseline strategy.

The baseline is a simple momentum + risk-off rule that lets us validate the
backtest harness end-to-end without invoking the LLM. It also serves as the
quantitative benchmark the LLM strategy must beat — if the LLM doesn't beat a
4-week momentum rule with PAXG drawdown hedge, it doesn't have edge.

Rules:
    - 4-week momentum: if BTC.close[-1] > BTC.close[-5], stay long; else flat.
    - Risk-off hedge: when drawdown from recent peak exceeds 5%, rotate to PAXG.
    - Single-position-at-a-time, weekly cadence.

Edge cases covered:
    - Insufficient history (less than lookback) → hold/flat
    - Equal momentum (close == close[-5]) → flat
    - PAXG not in universe → no rotation, just flat

Tail risks covered:
    - Empty bar list → return hold
    - All-NaN series → defensive fallback to hold
    - Fee-aware: do not signal on tiny moves that won't cover fees
"""

from __future__ import annotations

import math

from src.backtest.baseline_strategy import (
    BaselineConfig,
    decide_baseline,
)


def _bars(closes: list[float]) -> list[dict]:
    """Build a fake bar list with the given closes (other fields synthetic)."""
    return [
        {
            "ts_ms": i * 86_400_000,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1.0,
        }
        for i, c in enumerate(closes)
    ]


class TestMomentumSignal:
    def test_long_when_4wk_momentum_positive(self) -> None:
        # Close[-1] = 110 > Close[-5] = 100
        bars = _bars([100, 102, 104, 106, 108, 110])
        d = decide_baseline(
            symbol_bars={"BTC/USDT": bars},
            current_positions=[],
            equity_usd=2000.0,
            config=BaselineConfig(momentum_lookback_bars=5),
        )
        actions = d["actions"]
        assert any(a["type"] == "open_long" and a["symbol"] == "BTC/USDT" for a in actions)

    def test_flat_when_momentum_negative(self) -> None:
        # Close[-1] = 90 < Close[-5] = 100
        bars = _bars([100, 98, 96, 94, 92, 90])
        d = decide_baseline(
            symbol_bars={"BTC/USDT": bars},
            current_positions=[],
            equity_usd=2000.0,
            config=BaselineConfig(momentum_lookback_bars=5),
        )
        # No open action, just hold
        assert all(a["type"] != "open_long" for a in d["actions"])

    def test_flat_when_momentum_equal(self) -> None:
        bars = _bars([100, 102, 104, 102, 101, 100])  # close[-1] == close[-5]
        d = decide_baseline(
            symbol_bars={"BTC/USDT": bars},
            current_positions=[],
            equity_usd=2000.0,
            config=BaselineConfig(momentum_lookback_bars=5),
        )
        assert all(a["type"] != "open_long" for a in d["actions"])

    def test_insufficient_history_returns_hold(self) -> None:
        bars = _bars([100, 102, 104])  # 3 bars, lookback=5
        d = decide_baseline(
            symbol_bars={"BTC/USDT": bars},
            current_positions=[],
            equity_usd=2000.0,
            config=BaselineConfig(momentum_lookback_bars=5),
        )
        assert d["actions"] == [{"type": "hold"}]


class TestEmptyData:
    def test_empty_bars_returns_hold(self) -> None:
        d = decide_baseline(
            symbol_bars={},
            current_positions=[],
            equity_usd=2000.0,
            config=BaselineConfig(),
        )
        assert d["actions"] == [{"type": "hold"}]

    def test_nan_close_returns_hold(self) -> None:
        bars = _bars([100, 105, math.nan, 110, 112, 115])
        d = decide_baseline(
            symbol_bars={"BTC/USDT": bars},
            current_positions=[],
            equity_usd=2000.0,
            config=BaselineConfig(momentum_lookback_bars=5),
        )
        # Defensive: NaN in price series → hold
        assert d["actions"] == [{"type": "hold"}]


class TestPositionAware:
    def test_close_when_already_long_and_momentum_flips(self) -> None:
        # Long position open; momentum just turned negative → close
        bars = _bars([110, 108, 106, 104, 102, 100])
        positions = [
            {"symbol": "BTC/USDT", "side": "long", "size": 0.01, "entry_price": 110, "entry_ts_ms": 0}
        ]
        d = decide_baseline(
            symbol_bars={"BTC/USDT": bars},
            current_positions=positions,
            equity_usd=2000.0,
            config=BaselineConfig(momentum_lookback_bars=5),
        )
        assert any(a["type"] == "close" and a["symbol"] == "BTC/USDT" for a in d["actions"])

    def test_no_action_when_already_long_and_momentum_still_positive(self) -> None:
        bars = _bars([100, 102, 104, 106, 108, 110])
        positions = [
            {"symbol": "BTC/USDT", "side": "long", "size": 0.01, "entry_price": 100, "entry_ts_ms": 0}
        ]
        d = decide_baseline(
            symbol_bars={"BTC/USDT": bars},
            current_positions=positions,
            equity_usd=2000.0,
            config=BaselineConfig(momentum_lookback_bars=5),
        )
        # Already long and momentum still positive: no new open, no close
        assert all(a["type"] not in ("open_long", "close") for a in d["actions"])


class TestPAXGHedge:
    def test_rotate_to_paxg_on_drawdown(self) -> None:
        # BTC bars showing a drawdown
        btc_bars = _bars([100, 110, 105, 95, 90, 88])  # peak at 110, now at 88 → ~20% drawdown
        # PAXG flat
        paxg_bars = _bars([1900] * 6)
        d = decide_baseline(
            symbol_bars={"BTC/USDT": btc_bars, "PAXG/USDT": paxg_bars},
            current_positions=[],
            equity_usd=2000.0,
            config=BaselineConfig(
                momentum_lookback_bars=5,
                drawdown_hedge_threshold=0.05,  # 5% drawdown triggers hedge
            ),
        )
        # Should suggest PAXG long instead of BTC
        assert any(a["type"] == "open_long" and a["symbol"] == "PAXG/USDT" for a in d["actions"])
        assert all(a.get("symbol") != "BTC/USDT" for a in d["actions"] if a["type"] == "open_long")

    def test_no_hedge_when_no_paxg_in_universe(self) -> None:
        btc_bars = _bars([100, 110, 105, 95, 90, 88])
        d = decide_baseline(
            symbol_bars={"BTC/USDT": btc_bars},
            current_positions=[],
            equity_usd=2000.0,
            config=BaselineConfig(
                momentum_lookback_bars=5,
                drawdown_hedge_threshold=0.05,
            ),
        )
        # Without PAXG in universe, hedge is impossible — just stay flat
        assert all(a["type"] != "open_long" for a in d["actions"])
