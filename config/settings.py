"""Centralized configuration for the trading bot.

Default-friendly: out of the box, the bot runs in **paper mode** against
**Binance** with **Claude Opus 4.7 (extended thinking, 1M context)**. Every
field is overridable via env var, and the legacy ``MEXC_*`` / ``ANTHROPIC_*``
env names are still honored so existing setups don't break.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from src.exchange.profile import ExchangeProfile, resolve_profile

load_dotenv()


def _env(*names: str, default: str = "") -> str:
    """First non-empty env var from ``names``, else ``default``."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def _env_float(*names: str, default: float | None = None) -> float | None:
    raw = _env(*names)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ExchangeSettings:
    """Exchange connection and trading parameters.

    Defaults: Binance perpetual swaps (USDT-margined). Override via env:

        EXCHANGE_NAME=bybit
        EXCHANGE_API_KEY=...
        EXCHANGE_API_SECRET=...
        EXCHANGE_QUOTE_CCY=USDC
        EXCHANGE_MAKER_FEE=0.0001
        EXCHANGE_TAKER_FEE=0.0004
        EXCHANGE_MAINT_MARGIN=0.004

    Legacy ``MEXC_API_KEY`` / ``MEXC_SECRET`` are still read for backward
    compat.
    """

    name: str = field(
        default_factory=lambda: _env("EXCHANGE_NAME", default="binance").lower()
    )
    api_key: str = field(
        default_factory=lambda: _env(
            "EXCHANGE_API_KEY", "MEXC_API_KEY", "BINANCE_API_KEY"
        )
    )
    secret: str = field(
        default_factory=lambda: _env(
            "EXCHANGE_API_SECRET", "MEXC_SECRET", "BINANCE_SECRET"
        )
    )

    # Cross-exchange data sources (no trading; just for funding/price triangulation).
    data_exchanges: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            x.strip()
            for x in _env(
                "DATA_EXCHANGES", default="binance,bybit,okx"
            ).split(",")
            if x.strip()
        )
    )

    # Tunables — when set, override the registry's defaults for the chosen exchange.
    market_type_override: str = field(
        default_factory=lambda: _env("EXCHANGE_MARKET_TYPE", default="")
    )
    quote_currency_override: str = field(
        default_factory=lambda: _env("EXCHANGE_QUOTE_CCY", default="")
    )
    maker_fee_override: float | None = field(
        default_factory=lambda: _env_float("EXCHANGE_MAKER_FEE")
    )
    taker_fee_override: float | None = field(
        default_factory=lambda: _env_float("EXCHANGE_TAKER_FEE")
    )
    maintenance_margin_override: float | None = field(
        default_factory=lambda: _env_float("EXCHANGE_MAINT_MARGIN")
    )

    # Rate limiting safety margin (ms above CCXT's built-in throttle).
    rate_limit_buffer_ms: int = 100

    @property
    def profile(self) -> ExchangeProfile:
        """Resolve the ExchangeProfile for the configured exchange."""
        overrides: dict[str, object] = {}
        if self.market_type_override:
            overrides["market_type"] = self.market_type_override
        if self.quote_currency_override:
            overrides["quote_currency"] = self.quote_currency_override
        if self.maker_fee_override is not None:
            overrides["maker_fee"] = self.maker_fee_override
        if self.taker_fee_override is not None:
            overrides["taker_fee"] = self.taker_fee_override
        if self.maintenance_margin_override is not None:
            overrides["maintenance_margin_rate"] = self.maintenance_margin_override
        return resolve_profile(self.name, overrides=overrides)

    # Convenience accessors so legacy code that read settings.exchange.taker_fee
    # / .market_type still works.
    @property
    def taker_fee(self) -> float:
        return self.profile.taker_fee

    @property
    def maker_fee(self) -> float:
        return self.profile.maker_fee

    @property
    def market_type(self) -> str:
        return self.profile.market_type

    @property
    def quote_currency(self) -> str:
        return self.profile.quote_currency


@dataclass(frozen=True)
class LLMSettings:
    """LLM provider configuration.

    Default: Anthropic, Claude Opus 4.7 with extended thinking and the 1M
    context-window beta. Switch providers by setting ``LLM_PROVIDER=openai``
    (or ``google`` / ``openai-compat``) and the matching ``LLM_API_KEY`` /
    ``LLM_MODEL`` / ``LLM_BASE_URL`` env vars.
    """

    provider: str = field(
        default_factory=lambda: _env("LLM_PROVIDER", default="anthropic").lower()
    )
    api_key: str = field(
        default_factory=lambda: _env(
            "LLM_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        )
    )
    model: str = field(
        default_factory=lambda: _env("LLM_MODEL", default="claude-opus-4-7")
    )

    # OpenAI-compatible providers (Ollama, vLLM, OpenRouter, Together, ...).
    base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", default="")
    )

    # Anthropic extended thinking (ignored by other providers).
    extended_thinking: bool = field(
        default_factory=lambda: _env(
            "LLM_EXTENDED_THINKING", default="true"
        ).lower()
        == "true"
    )
    thinking_budget_tokens: int = field(
        default_factory=lambda: int(_env("LLM_THINKING_BUDGET", default="8000"))
    )

    # Anthropic 1M-context beta gate.
    request_long_context: bool = field(
        default_factory=lambda: _env(
            "LLM_LONG_CONTEXT", default="true"
        ).lower()
        == "true"
    )

    max_output_tokens: int = field(
        default_factory=lambda: int(_env("LLM_MAX_OUTPUT_TOKENS", default="2048"))
    )
    temperature: float = field(
        default_factory=lambda: float(_env("LLM_TEMPERATURE", default="0.0"))
    )

    # Call cadence (seconds).
    base_interval: int = 900  # 15 min
    min_interval: int = 300
    max_interval: int = 3600
    default_no_position: int = 1800

    # Budget guards.
    daily_budget_usd: float = field(
        default_factory=lambda: float(_env("LLM_DAILY_BUDGET_USD", default="5.00"))
    )
    monthly_budget_usd: float = field(
        default_factory=lambda: float(_env("LLM_MONTHLY_BUDGET_USD", default="100.0"))
    )

    # Pricing override (per million tokens). When 0/None, the provider's
    # built-in pricing table is used.
    input_price_per_m: float = field(
        default_factory=lambda: float(_env("LLM_INPUT_PRICE_PER_M", default="0"))
    )
    cached_input_price_per_m: float = field(
        default_factory=lambda: float(_env("LLM_CACHED_INPUT_PRICE_PER_M", default="0"))
    )
    output_price_per_m: float = field(
        default_factory=lambda: float(_env("LLM_OUTPUT_PRICE_PER_M", default="0"))
    )

    max_retries: int = 2
    timeout_seconds: int = 60


@dataclass(frozen=True)
class SchedulerSettings:
    """How often the LLM is consulted."""

    cadence: str = field(
        default_factory=lambda: _env("REBALANCE_CADENCE", default="adaptive").lower()
    )
    """One of ``"daily"``, ``"hourly"``, ``"adaptive"``, or a 5-field cron
    expression. ``"adaptive"`` lets the LLM choose its own next-check interval
    within ``[adaptive_min_minutes, adaptive_max_minutes]``."""

    adaptive_min_minutes: int = 15
    adaptive_max_minutes: int = 60


@dataclass(frozen=True)
class RiskSettings:
    """Risk management parameters."""

    # Leverage
    max_leverage: int = 5
    default_leverage: int = 2

    # Position limits
    max_position_pct: float = 0.25
    max_total_exposure_pct: float = 0.60
    max_concurrent_positions: int = 3

    # Drawdown
    max_drawdown_pct: float = 0.05
    drawdown_warning_pct: float = 0.045
    daily_loss_limit_pct: float = 0.03

    # Per-trade risk
    max_risk_per_trade_pct: float = 0.02
    min_reward_risk_ratio: float = 1.5

    # Circuit breakers
    rapid_loss_count: int = 3
    rapid_loss_pct: float = 0.02
    rapid_loss_window_seconds: int = 3600
    volatility_spike_multiplier: float = 2.0
    max_spread_pct: float = 0.001
    max_funding_rate_8h: float = 0.001
    ws_disconnect_close_seconds: int = 60
    llm_failure_threshold: int = 3

    # Position sizing
    kelly_fraction_under_2k: float = 0.125
    kelly_fraction_2k_5k: float = 0.20
    kelly_fraction_5k_10k: float = 0.25
    kelly_fraction_10k_plus: float = 0.33
    min_trade_fee_multiple: float = 3.0

    # Stop loss
    sl_atr_multiplier: float = 1.5
    trailing_sl_activation_atr: float = 1.0
    trailing_sl_distance_atr: float = 1.0
    partial_tp_pct: float = 0.50
    partial_tp_rr_ratio: float = 2.0
    time_stop_hours: float = 6.0
    time_stop_min_progress_atr: float = 0.5


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: str = field(
        default_factory=lambda: _env("TELEGRAM_BOT_TOKEN")
    )
    chat_id: str = field(
        default_factory=lambda: _env("TELEGRAM_CHAT_ID")
    )

    min_notification_interval_seconds: int = 300
    hourly_summary_enabled: bool = True
    daily_summary_enabled: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass(frozen=True)
class StorageSettings:
    db_path: str = field(
        default_factory=lambda: _env("DB_PATH", default="data/trading_bot.db")
    )

    ohlcv_1m_retention_days: int = 2
    ohlcv_5m_retention_days: int = 30
    ohlcv_15m_plus_retention_days: int = 365
    orderbook_retention_days: int = 7
    llm_prompt_full_retention_days: int = 90
    sentiment_retention_days: int = 90

    compress_llm_prompts: bool = True


@dataclass(frozen=True)
class AFMLSettings:
    """Advanced data-hygiene + risk-overlay feature flags.

    All features are safe to enable in paper mode. ``enable_all`` is the
    kill switch; individual flags allow bisecting if a regression appears.
    """

    enable_all: bool = True
    live_override: bool = field(
        default_factory=lambda: _env("AFML_LIVE_OVERRIDE", default="") == "1"
    )

    enable_ffd: bool = True
    enable_cusum: bool = True
    enable_microstructure: bool = True
    enable_bet_sizing: bool = True
    enable_triple_barrier: bool = True
    enable_dsr: bool = True
    enable_cscv: bool = True

    min_p: float = 0.55
    max_holding_s: int = 259_200
    fallback_trials: int = 50
    cusum_k: float = 2.5
    cusum_halflife: float = 20.0
    cusum_min_periods: int = 20

    ffd_d_max: float = 1.0
    ffd_d_step: float = 0.1
    ffd_adf_pvalue: float = 0.05
    ffd_threshold: float = 1e-4
    ffd_warmup_bars: int = 500

    cscv_min_trades: int = 20
    cscv_min_p_grid: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

    enable_risk_regime: bool = True
    enable_baselines: bool = True
    enable_cost_model: bool = True
    enable_garch: bool = True
    enable_scenario_kelly: bool = True

    regime_elevated_threshold: float = 1.5
    regime_crisis_threshold: float = 2.5
    regime_min_dwell_minutes: float = 30.0
    crisis_gate_min_confidence: float = 0.80
    baseline_refresh_hours: int = 24
    baseline_extreme_threshold: float = 0.5
    regime_calm_size_mult: float = 1.00
    regime_elevated_size_mult: float = 0.50
    regime_crisis_size_mult: float = 0.25
    regime_elevated_min_p: float = 0.60
    regime_crisis_min_p: float = 0.70

    enable_sl_cooldown: bool = True
    enable_margin_monitor: bool = True
    enable_slippage_sizing: bool = True
    enable_stale_data_guard: bool = True
    enable_clock_drift_check: bool = True
    enable_correlation_cap: bool = True
    enable_funding_autoclose: bool = True
    enable_partial_fill_reconciliation: bool = True
    enable_tail_risk: bool = True
    enable_vol_target: bool = True
    enable_daily_peak_anchor: bool = True
    enable_usdt_depeg_monitor: bool = True
    enable_event_calendar: bool = True
    enable_revenge_trade_detector: bool = True
    enable_protective_orders: bool = False

    sl_cooldown_base_minutes: int = 30
    chop_sl_count: int = 3
    chop_window_hours: float = 4.0
    chop_lockout_hours: float = 4.0
    chop_min_confidence: float = 0.80
    chop_min_adx: float = 25.0
    margin_force_close_buffer: float = 0.15
    margin_halt_buffer: float = 0.25
    slippage_budget_pct: float = 0.0015
    stale_data_rest_fallback_s: int = 10
    stale_data_halt_s: int = 60
    clock_drift_warn_s: float = 2.0
    clock_drift_halt_s: float = 10.0
    correlation_window_days: int = 30
    correlation_effective_count_cap: float = 2.0
    funding_autoclose_cumulative_pct: float = 0.03
    funding_autoclose_rate_per_8h: float = 0.003
    var_per_trade_cap_pct: float = 0.0125
    portfolio_daily_vol_target: float = 0.015
    daily_drawdown_halt_pct: float = 0.03
    session_drawdown_close_all_pct: float = 0.05
    usdt_depeg_threshold: float = 0.005
    usdt_depeg_sustain_count: int = 3
    event_pre_halt_minutes: int = 30
    event_warning_hours: int = 4


@dataclass(frozen=True)
class Settings:
    """Master configuration — passed once into ``TradingBot``."""

    exchange: ExchangeSettings = field(default_factory=ExchangeSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    afml: AFMLSettings = field(default_factory=AFMLSettings)

    bot_mode: str = field(default_factory=lambda: _env("BOT_MODE", default="paper"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", default="INFO"))
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", default="logs"))

    onchain_interval: int = 21600
    sentiment_interval: int = 900
    macro_interval: int = 14400

    @property
    def is_live(self) -> bool:
        return self.bot_mode == "live"

    def validate(self) -> list[str]:
        """Validate critical settings; return list of human-readable errors."""
        errors: list[str] = []
        if self.is_live and not self.exchange.api_key:
            errors.append(
                "Exchange API key required for live trading. Set "
                "EXCHANGE_API_KEY (or the legacy MEXC_API_KEY) in your .env."
            )
        if not self.llm.api_key:
            errors.append(
                f"LLM API key required for provider '{self.llm.provider}'. "
                f"Set LLM_API_KEY (or the provider-specific equivalent: "
                f"ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY)."
            )
        if self.risk.max_drawdown_pct < self.risk.drawdown_warning_pct:
            errors.append("drawdown_warning_pct must be < max_drawdown_pct")
        if self.is_live and self.afml.enable_all and not self.afml.live_override:
            errors.append(
                "Advanced overlays enabled while BOT_MODE=live; set "
                "AFML_LIVE_OVERRIDE=1 to acknowledge the paper-only rollout gate."
            )
        return errors


def load_settings() -> Settings:
    """Load settings from environment and print any validation errors."""
    settings = Settings()
    errors = settings.validate()
    if errors:
        for e in errors:
            print(f"CONFIG ERROR: {e}")
    return settings
