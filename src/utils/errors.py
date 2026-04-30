"""Custom exception hierarchy for the trading bot."""


class BotError(Exception):
    """Base exception for all bot errors."""


# --- Exchange Errors ---

class ExchangeError(BotError):
    """Error communicating with exchange."""


class OrderError(ExchangeError):
    """Error placing or managing orders."""


class InsufficientBalanceError(OrderError):
    """Not enough balance to place order."""


class OrderNotFoundError(OrderError):
    """Order not found on exchange."""


# --- Risk Errors ---

class RiskError(BotError):
    """Risk management violation."""


class DrawdownLimitError(RiskError):
    """Maximum drawdown limit reached."""


class PositionLimitError(RiskError):
    """Position size or count limit exceeded."""


class CircuitBreakerError(RiskError):
    """Circuit breaker triggered."""


# --- LLM Errors ---

class LLMError(BotError):
    """Error with LLM interaction."""


class LLMParseError(LLMError):
    """Failed to parse LLM response."""


class LLMBudgetExceededError(LLMError):
    """LLM API budget exceeded."""


class LLMTimeoutError(LLMError):
    """LLM API call timed out."""


# --- Data Errors ---

class DataError(BotError):
    """Error fetching or processing data."""


class StaleDataError(DataError):
    """Data is too old to be trusted."""


class DataSourceUnavailableError(DataError):
    """A data source is temporarily unavailable."""


# --- Config Errors ---

class ConfigError(BotError):
    """Configuration error."""
