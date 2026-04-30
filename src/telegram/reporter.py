"""Telegram notification system for the trading bot."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from telegram import Bot
from telegram.constants import ParseMode

from src.utils.logger import get_logger

log = get_logger(__name__)


class TelegramReporter:
    """Sends formatted trade notifications and summaries to Telegram."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        min_interval_seconds: int = 300,
    ):
        self._bot = Bot(token=bot_token) if bot_token else None
        self._chat_id = chat_id
        self._min_interval = min_interval_seconds
        self._last_trade_notif: float = 0
        self._queued_messages: list[str] = []
        self._enabled = bool(bot_token and chat_id)

    async def send(self, text: str, urgent: bool = False) -> None:
        """Send a message to the Telegram channel."""
        if not self._enabled:
            log.debug("telegram_disabled", message=text[:100])
            return

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            log.error("telegram_send_failed", error=str(e))

    # --- Trade Notifications ---

    async def notify_trade_open(
        self,
        symbol: str,
        side: str,
        size: float,
        leverage: int,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        reasoning: str,
    ) -> None:
        """Notify about a new position opening."""
        now = time.time()
        if now - self._last_trade_notif < self._min_interval:
            return  # anti-spam

        emoji = "🟢" if side == "long" else "🔴"
        msg = (
            f"{emoji} <b>OPEN {side.upper()}</b>\n"
            f"<b>{symbol}</b>\n"
            f"Entry: <code>${entry_price:,.2f}</code> | {leverage}x\n"
            f"Size: <code>${size:,.2f}</code>\n"
        )
        if stop_loss:
            msg += f"SL: <code>${stop_loss:,.2f}</code>\n"
        if take_profit:
            msg += f"TP: <code>${take_profit:,.2f}</code>\n"
        msg += f"\n<i>{reasoning[:200]}</i>"

        await self.send(msg)
        self._last_trade_notif = now

    async def notify_trade_close(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        duration: str,
        reasoning: str,
    ) -> None:
        """Notify about a position closing."""
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} <b>CLOSE {side.upper()}</b>\n"
            f"<b>{symbol}</b>\n"
            f"Entry: <code>${entry_price:,.2f}</code> → "
            f"Exit: <code>${exit_price:,.2f}</code>\n"
            f"PnL: <code>${pnl:+,.2f}</code> ({pnl_pct:+.1f}%)\n"
            f"Duration: {duration}\n"
            f"\n<i>{reasoning[:200]}</i>"
        )
        await self.send(msg)
        self._last_trade_notif = time.time()

    # --- Summary Notifications ---

    async def send_hourly_summary(
        self,
        equity: float,
        positions: list[dict],
        unrealized_pnl: float,
    ) -> None:
        """Hourly summary when positions are open."""
        msg = "📊 <b>HOURLY UPDATE</b>\n\n"
        msg += f"Equity: <code>${equity:,.2f}</code>\n"
        msg += f"Unrealized PnL: <code>${unrealized_pnl:+,.2f}</code>\n\n"

        if positions:
            msg += "<b>Open Positions:</b>\n"
            for p in positions:
                pnl = p.get("unrealized_pnl", 0)
                msg += (
                    f"  {'🟢' if p.get('side') == 'long' else '🔴'} "
                    f"{p.get('symbol', '?')} {p.get('side', '?').upper()} "
                    f"${pnl:+,.2f}\n"
                )

        await self.send(msg)

    async def send_daily_summary(
        self,
        date: str,
        starting_equity: float,
        ending_equity: float,
        pnl: float,
        pnl_pct: float,
        trades: int,
        wins: int,
        losses: int,
        max_drawdown_pct: float,
        llm_cost: float,
    ) -> None:
        """End-of-day summary."""
        emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>DAILY SUMMARY — {date}</b>\n\n"
            f"Start: <code>${starting_equity:,.2f}</code>\n"
            f"End: <code>${ending_equity:,.2f}</code>\n"
            f"PnL: <code>${pnl:+,.2f}</code> ({pnl_pct:+.1f}%)\n\n"
            f"Trades: {trades} (W:{wins} / L:{losses})\n"
            f"Win Rate: {wins/trades*100:.0f}%\n" if trades > 0 else ""
            f"Max Drawdown: {max_drawdown_pct*100:.1f}%\n"
            f"LLM Cost: <code>${llm_cost:.2f}</code>\n"
            f"Net after LLM: <code>${pnl - llm_cost:+,.2f}</code>"
        )
        await self.send(msg)

    # --- Alert Notifications ---

    async def send_circuit_breaker_alert(
        self,
        breaker_type: str,
        action: str,
        reason: str,
    ) -> None:
        """Urgent circuit breaker alert."""
        msg = (
            f"🚨 <b>CIRCUIT BREAKER</b>\n\n"
            f"Type: <code>{breaker_type}</code>\n"
            f"Action: <code>{action}</code>\n"
            f"Reason: {reason}"
        )
        await self.send(msg, urgent=True)

    async def send_startup_message(
        self,
        mode: str,
        equity: float,
        symbols: list[str],
        positions: int,
    ) -> None:
        """Bot startup notification."""
        msg = (
            f"🤖 <b>BOT STARTED</b>\n\n"
            f"Mode: <code>{mode}</code>\n"
            f"Equity: <code>${equity:,.2f}</code>\n"
            f"Symbols: {', '.join(symbols)}\n"
            f"Open Positions: {positions}"
        )
        await self.send(msg)

    async def send_shutdown_message(self, reason: str = "manual") -> None:
        """Bot shutdown notification."""
        msg = f"⏹️ <b>BOT STOPPED</b>\nReason: {reason}"
        await self.send(msg)

    async def send_error_alert(self, error: str) -> None:
        """Non-critical error alert."""
        msg = f"⚠️ <b>ERROR</b>\n<code>{error[:500]}</code>"
        await self.send(msg)
