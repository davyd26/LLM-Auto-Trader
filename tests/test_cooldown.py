"""Tests for symbol cooldown / chop detector (Phase 3 §3B.9 + §3B.15)."""

from __future__ import annotations

from src.risk.cooldown import SymbolCooldown


class TestSymbolCooldown:
    def test_new_symbol_not_locked(self) -> None:
        c = SymbolCooldown(base_minutes=30)
        locked, _ = c.is_locked("BTC/USDT:USDT", now=0)
        assert locked is False

    def test_after_sl_symbol_locked(self) -> None:
        c = SymbolCooldown(base_minutes=30)
        c.on_stop_out("BTC/USDT:USDT", now=0)
        locked, unlock_ts = c.is_locked("BTC/USDT:USDT", now=60)
        assert locked is True
        assert unlock_ts > 60

    def test_cooldown_expires(self) -> None:
        c = SymbolCooldown(base_minutes=30)
        c.on_stop_out("BTC/USDT:USDT", now=0)
        # 30 min later - should be unlocked
        locked, _ = c.is_locked("BTC/USDT:USDT", now=30 * 60 + 1)
        assert locked is False

    def test_consecutive_sl_extends_cooldown(self) -> None:
        c = SymbolCooldown(base_minutes=30)
        c.on_stop_out("BTC/USDT:USDT", now=0)
        # 2nd SL within the first cooldown
        c.on_stop_out("BTC/USDT:USDT", now=10 * 60)
        locked, unlock = c.is_locked("BTC/USDT:USDT", now=31 * 60)
        # Second SL extends: 30 * (1 + 1) = 60 min from the 2nd event = 70 min
        assert locked is True

    def test_chop_detection_after_n_sls_in_window(self) -> None:
        c = SymbolCooldown(base_minutes=30, chop_count=3, chop_window_hours=4)
        for i in range(3):
            c.on_stop_out("SOL/USDT:USDT", now=i * 1800)  # 30 min apart
        assert c.is_choppy("SOL/USDT:USDT", now=3 * 1800) is True

    def test_chop_clears_when_old_sls_fall_out_of_window(self) -> None:
        c = SymbolCooldown(base_minutes=30, chop_count=3, chop_window_hours=4)
        for i in range(3):
            c.on_stop_out("SOL/USDT:USDT", now=i * 1800)
        # 5 hours later all SLs out of window
        assert c.is_choppy("SOL/USDT:USDT", now=3 * 1800 + 5 * 3600) is False

    def test_different_symbols_independent(self) -> None:
        c = SymbolCooldown(base_minutes=30)
        c.on_stop_out("BTC/USDT:USDT", now=0)
        locked, _ = c.is_locked("ETH/USDT:USDT", now=60)
        assert locked is False
