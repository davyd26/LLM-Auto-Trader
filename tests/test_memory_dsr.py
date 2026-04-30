"""Tests for AFML metrics surfaced via TradingMemory.get_performance_summary."""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.memory import TradingMemory


@dataclass
class _AFMLStub:
    enable_all: bool = True
    enable_dsr: bool = True
    enable_cscv: bool = True
    fallback_trials: int = 50
    cscv_min_trades: int = 20
    cscv_min_p_grid: tuple = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def _trade(pnl_pct: float, conf: float = 0.6, t: float = 1000.0) -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "pnl": pnl_pct * 10,
        "pnl_pct": pnl_pct,
        "llm_confidence": conf,
        "timestamp_open": t - 60,
        "timestamp_close": t,
    }


class TestDSRBasic:
    def test_no_afml_gives_legacy_fields_only(self) -> None:
        mem = TradingMemory()
        for i in range(5):
            mem.record_trade(_trade(pnl_pct=1.0, t=1000.0 + i))
        summary = mem.get_performance_summary(current_equity=10_000.0)
        assert "dsr" not in summary
        assert "sharpe" in summary

    def test_afml_enabled_adds_dsr_fields_with_enough_trades(self) -> None:
        mem = TradingMemory()
        for i in range(5):
            mem.record_trade(_trade(pnl_pct=1.0 if i % 2 == 0 else -0.5, t=1000.0 + i))
        afml = _AFMLStub()
        summary = mem.get_performance_summary(current_equity=10_000.0, afml_settings=afml)
        assert "dsr" in summary
        assert "psr_vs_zero" in summary
        assert "pnl_skew" in summary
        assert "pnl_kurtosis" in summary
        assert "num_trials_for_dsr" in summary
        assert 0.0 <= summary["dsr"] <= 1.0

    def test_cscv_kicks_in_with_enough_trades(self) -> None:
        mem = TradingMemory()
        import random
        rng = random.Random(0)
        for i in range(25):
            pnl = rng.uniform(-2.0, 2.0)
            conf = 0.5 + 0.3 * rng.random()
            mem.record_trade(_trade(pnl_pct=pnl, conf=conf, t=1000.0 + i))
        afml = _AFMLStub(cscv_min_trades=20)
        summary = mem.get_performance_summary(current_equity=10_000.0, afml_settings=afml)
        assert "dsr" in summary
        # CSCV should have kicked in → pbo present
        assert "pbo" in summary
        assert 0.0 <= summary["pbo"] <= 1.0

    def test_too_few_trades_skips_dsr(self) -> None:
        mem = TradingMemory()
        mem.record_trade(_trade(pnl_pct=1.0))  # only 1 trade
        afml = _AFMLStub()
        summary = mem.get_performance_summary(current_equity=10_000.0, afml_settings=afml)
        assert "dsr" not in summary
