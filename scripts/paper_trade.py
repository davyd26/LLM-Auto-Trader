#!/usr/bin/env python3
"""Paper trading entry point.

Runs the bot in paper (simulated) mode using real market data
but without executing real trades.

Usage:
    python -m scripts.paper_trade
    # or
    python scripts/paper_trade.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_settings
from src.bot import TradingBot


def main() -> None:
    """Run the bot in paper trading mode."""
    # Force paper mode regardless of .env
    os.environ["BOT_MODE"] = "paper"

    settings = load_settings()
    errors = settings.validate()

    if any("LLM API key" in e for e in errors):
        print("ERROR: An LLM API key is required even for paper trading.")
        print(
            "Set LLM_API_KEY in .env (or the provider-specific equivalent: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)."
        )
        sys.exit(1)

    print("Starting paper trading bot...")
    print(f"  Exchange:       {settings.exchange.name}")
    print(f"  Quote currency: {settings.exchange.quote_currency}")
    print(f"  LLM provider:   {settings.llm.provider}")
    print(f"  LLM model:      {settings.llm.model}")
    print(f"  Cadence:        {settings.scheduler.cadence}")
    print(f"  Daily budget:   ${settings.llm.daily_budget_usd:.2f}")
    print(f"  Max drawdown:   {settings.risk.max_drawdown_pct * 100:.0f}%")
    print(f"  Max leverage:   {settings.risk.max_leverage}x")
    print()

    bot = TradingBot(settings)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
