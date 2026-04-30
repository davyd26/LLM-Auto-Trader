"""Pydantic models for structured LLM response validation.

The trade-action symbol regex is built dynamically from the configured quote
currency at module import time, so the bot supports USDT, USDC, BUSD, USD, etc.
without code edits.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.utils.errors import LLMParseError


def _quote_ccy() -> str:
    """Quote currency, read at validator-build time from env.

    We intentionally read the env var rather than importing ``Settings`` here
    to avoid a circular import (``Settings`` may import LLM modules).
    """
    return (os.getenv("EXCHANGE_QUOTE_CCY") or "USDT").upper()


_SYMBOL_PATTERN = rf"^[A-Z0-9]+/{_quote_ccy()}(:.+)?$"


class ActionType(str, Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    ADJUST_SL = "adjust_sl"
    ADJUST_TP = "adjust_tp"
    ADD_TO_POSITION = "add_to_position"
    REDUCE_POSITION = "reduce_position"
    HOLD = "hold"


class TradeAction(BaseModel):
    """A single trading action from the LLM."""

    action: ActionType
    symbol: str = Field(pattern=_SYMBOL_PATTERN)
    size_pct: float | None = Field(default=None, ge=1, le=25)
    leverage: int | None = Field(default=None, ge=1, le=5)
    order_type: Literal["market", "limit"] = "market"
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    reasoning: str = Field(min_length=5, max_length=500)

    @model_validator(mode="after")
    def validate_new_position_requirements(self) -> "TradeAction":
        if self.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT):
            if self.stop_loss is None:
                raise ValueError("stop_loss is required for new positions")
            if self.size_pct is None:
                raise ValueError("size_pct is required for new positions")
            if self.leverage is None:
                raise ValueError("leverage is required for new positions")
            if self.take_profit is not None and self.stop_loss is not None:
                if self.action == ActionType.OPEN_LONG:
                    if self.entry_price and self.take_profit <= self.entry_price:
                        raise ValueError("take_profit must be above entry for long")
                    if self.entry_price and self.stop_loss >= self.entry_price:
                        raise ValueError("stop_loss must be below entry for long")
                elif self.action == ActionType.OPEN_SHORT:
                    if self.entry_price and self.take_profit >= self.entry_price:
                        raise ValueError("take_profit must be below entry for short")
                    if self.entry_price and self.stop_loss <= self.entry_price:
                        raise ValueError("stop_loss must be above entry for short")
        return self


class MarketAssessment(BaseModel):
    """LLM's market assessment."""

    trend: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    regime: Literal["trending_up", "trending_down", "ranging", "volatile"]
    key_factors: list[str] = Field(min_length=1, max_length=5)


class LLMResponse(BaseModel):
    """Complete validated LLM response."""

    market_assessment: MarketAssessment
    actions: list[TradeAction] = Field(max_length=4)
    next_check_minutes: int = Field(ge=5, le=120)
    risk_notes: str = ""


def parse_llm_response(raw_text: str) -> LLMResponse:
    """Parse and validate raw LLM text into structured response.

    Tolerant of common formatting issues (markdown fences, extra whitespace,
    leading prose) that some models like to emit despite explicit instructions.
    """
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    if not cleaned.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group()
        else:
            raise LLMParseError(f"No JSON object found in response: {raw_text[:200]}")

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Invalid JSON: {e}") from e

    try:
        return LLMResponse.model_validate(data)
    except Exception as e:
        raise LLMParseError(f"Validation error: {e}") from e


def format_retry_prompt(error: LLMParseError) -> str:
    """Compose an instruction for the LLM telling it what went wrong."""
    return (
        f"Your previous response had a validation error:\n"
        f"{error}\n\n"
        f"Please fix the issue and respond with ONLY valid JSON matching the schema. "
        f"Remember: stop_loss is REQUIRED for open_long/open_short actions, "
        f"and all numeric fields must have valid values."
    )
