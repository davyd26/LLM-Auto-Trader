"""Builds prioritized, token-budgeted prompts from market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.analysis.indicators import IndicatorSummary, compute_indicators, ohlcv_list_to_df
from src.analysis.statistics import compute_statistics, StatisticalFeatures
from src.analysis.regime import classify_regime, MarketRegime
from src.analysis.microstructure import MicrostructureFeatures, compute_microstructure
from src.analysis.frac_diff import frac_diff_ffd, find_min_d
from src.data.aggregator import MarketSnapshot, SymbolSnapshot


@dataclass
class AnalyzedSymbol:
    """Symbol data enriched with computed indicators."""

    symbol: str
    indicators_15m: IndicatorSummary | None = None
    indicators_1h: IndicatorSummary | None = None
    indicators_4h: IndicatorSummary | None = None
    statistics: StatisticalFeatures | None = None
    regime: MarketRegime | None = None
    microstructure: MicrostructureFeatures | None = None


class ContextBuilder:
    """Builds token-budgeted prompts from market snapshots."""

    # Approximate token budget per section
    MAX_USER_TOKENS = 4000

    def __init__(
        self, afml_settings=None, market_state=None,
        symbol_cooldown=None, event_calendar=None,
        usdt_monitor=None, daily_peak=None, margin_monitor=None,
    ) -> None:
        """Phase 2/3 aggregators plugged in by the bot."""
        self._afml = afml_settings
        self._market_state = market_state
        self._cooldown = symbol_cooldown
        self._event_calendar = event_calendar
        self._usdt_monitor = usdt_monitor
        self._daily_peak = daily_peak
        self._margin_state_provider = margin_monitor
        self._ffd_d_cache: dict[str, float] = {}

    def _get_ffd_d(self, symbol: str, close_series) -> float | None:
        if self._afml is not None and not getattr(self._afml, "enable_ffd", True):
            return None
        if not getattr(self._afml, "enable_all", True) if self._afml else False:
            return None
        if symbol in self._ffd_d_cache:
            return self._ffd_d_cache[symbol]
        if len(close_series) < 200:
            return None  # wait for enough warm-up data; retry next tick
        try:
            max_d = float(getattr(self._afml, "ffd_d_max", 1.0)) if self._afml else 1.0
            step = float(getattr(self._afml, "ffd_d_step", 0.1)) if self._afml else 0.1
            pval = float(getattr(self._afml, "ffd_adf_pvalue", 0.05)) if self._afml else 0.05
            d = find_min_d(close_series, max_d=max_d, step=step, adf_pvalue=pval)
            self._ffd_d_cache[symbol] = d
            return d
        except Exception:
            return None

    def build(
        self,
        snapshot: MarketSnapshot,
        trade_history: list[dict] | None = None,
        performance: dict | None = None,
        previous_decision: dict | None = None,
    ) -> dict[str, str]:
        """Build all prompt sections from a market snapshot.

        Returns dict of section_name -> formatted text.
        Sections are already prioritized and trimmed to token budget.
        """
        sections: dict[str, str] = {}
        analyzed: dict[str, AnalyzedSymbol] = {}
        micro_enabled = (
            self._afml is None
            or (getattr(self._afml, "enable_all", True)
                and getattr(self._afml, "enable_microstructure", True))
        )

        # Compute indicators for each symbol
        for sym, sym_snap in snapshot.symbols.items():
            a = AnalyzedSymbol(symbol=sym)

            for tf, attr in [("15m", "indicators_15m"), ("1h", "indicators_1h"), ("4h", "indicators_4h")]:
                if tf in sym_snap.ohlcv and sym_snap.ohlcv[tf]:
                    df = ohlcv_list_to_df(sym_snap.ohlcv[tf])
                    setattr(a, attr, compute_indicators(df))
                    if tf == "4h" and self._market_state is not None:
                        # Feed daily-ish close to MarketState baseline cache.
                        # We use the 4h close series (reasonable daily proxy
                        # when no explicit 1d bars are in the snapshot).
                        try:
                            self._market_state.refresh_symbol(
                                symbol=sym,
                                daily_close=df["close"].astype(float),
                                current_price=float(df["close"].iloc[-1]),
                                returns_15m=None,
                            )
                        except Exception:
                            pass
                    if tf == "15m":
                        # Feed 15m returns into MarketState for GARCH updates.
                        if self._market_state is not None and len(df) >= 100:
                            try:
                                import numpy as _np
                                returns = df["close"].pct_change().dropna().to_numpy(dtype=float)
                                self._market_state.refresh_symbol(
                                    symbol=sym,
                                    daily_close=df["close"].astype(float),
                                    current_price=float(df["close"].iloc[-1]),
                                    returns_15m=returns,
                                )
                            except Exception:
                                pass
                        # Phase 1 §3.1 — FFD on 15m close (preferred tf for stats)
                        ffd_close = None
                        ffd_d = self._get_ffd_d(sym, df["close"].astype(float))
                        if ffd_d is not None and ffd_d > 0:
                            try:
                                ffd_close = frac_diff_ffd(
                                    df["close"].astype(float), d=ffd_d,
                                )
                            except Exception:
                                ffd_close = None
                        a.statistics = compute_statistics(df, ffd_close=ffd_close, ffd_d=ffd_d)
                        a.regime = classify_regime(df)
                        if micro_enabled:
                            a.microstructure = compute_microstructure(df)

            analyzed[sym] = a

        # --- Priority 1: Always included ---
        sections["price_action"] = self._format_price_action(snapshot)
        sections["technicals"] = self._format_technicals(analyzed)
        sections["positions"] = self._format_positions(snapshot)
        sections["previous_decision"] = self._format_previous_decision(previous_decision)
        # Phase 2 sections (Priority 1 — always shown when enabled)
        sections["risk_regime"] = self._format_risk_regime()
        sections["baselines"] = self._format_baselines(snapshot)
        # Phase 3 risk sections
        sections["upcoming_events"] = self._format_upcoming_events(snapshot)
        sections["system_health"] = self._format_system_health()
        sections["account_margin"] = self._format_account_margin()
        sections["symbol_cooldowns"] = self._format_symbol_cooldowns()

        # --- Priority 2: High ---
        sections["multi_timeframe"] = self._format_multi_timeframe(analyzed)
        sections["support_resistance"] = self._format_support_resistance(analyzed)
        sections["derivatives"] = self._format_derivatives(snapshot)
        sections["orderbook"] = self._format_orderbook(snapshot)
        sections["microstructure"] = self._format_microstructure(analyzed)
        sections["recent_trades"] = self._format_trade_history(trade_history)

        # --- Priority 2 continued: New data sources ---
        sections["prediction_markets"] = self._format_prediction_markets(snapshot)
        sections["news"] = self._format_news(snapshot)

        # --- Priority 3: Medium ---
        sections["top_traders"] = self._format_top_traders(snapshot)
        sections["performance"] = self._format_performance(performance)
        sections["sentiment"] = self._format_sentiment(snapshot)
        sections["onchain"] = self._format_onchain(snapshot)
        sections["macro"] = self._format_macro(snapshot)

        # Remove empty sections
        sections = {k: v for k, v in sections.items() if v}

        # Token budget enforcement: estimate and trim if needed
        total_est = sum(len(v) // 4 for v in sections.values())
        if total_est > self.MAX_USER_TOKENS:
            # Drop lowest priority sections first
            drop_order = ["macro", "onchain", "top_traders", "sentiment", "performance", "microstructure", "orderbook"]
            for section in drop_order:
                if section in sections and total_est > self.MAX_USER_TOKENS:
                    total_est -= len(sections[section]) // 4
                    del sections[section]

        return sections

    def _format_price_action(self, snapshot: MarketSnapshot) -> str:
        lines = ["=== PRICE ACTION ==="]
        for sym, s in snapshot.symbols.items():
            if s.ticker:
                lines.append(
                    f"{sym}: ${s.ticker.last:,.2f} | "
                    f"24h: {s.ticker.change_pct:+.1f}% | "
                    f"H: ${s.ticker.high:,.2f} L: ${s.ticker.low:,.2f} | "
                    f"Vol: ${s.ticker.volume:,.0f}"
                )
        if snapshot.balance:
            lines.append(
                f"\nAccount: ${snapshot.balance.get('total', 0):,.2f} total, "
                f"${snapshot.balance.get('free', 0):,.2f} free, "
                f"${snapshot.balance.get('used', 0):,.2f} in use"
            )
        return "\n".join(lines)

    def _format_technicals(self, analyzed: dict[str, AnalyzedSymbol]) -> str:
        lines = ["=== TECHNICALS (15m) ==="]
        for sym, a in analyzed.items():
            ind = a.indicators_15m
            if not ind:
                continue
            lines.append(f"--- {sym} ---")
            lines.append(f"RSI(14): {ind.rsi_14} ({ind.rsi_zone})")
            lines.append(f"MACD: {ind.macd_signal}, histogram {ind.macd_histogram_trend}")
            lines.append(f"EMA: {ind.ema_position}")
            lines.append(f"ADX(14): {ind.adx_14} ({ind.adx_trend})")
            lines.append(f"ATR: {ind.atr_pct}% of price")
            lines.append(f"Bollinger: position={ind.bb_position}, bandwidth={ind.bb_bandwidth}")
            lines.append(f"Volume: {ind.volume_ratio}x average")
            if ind.stoch_rsi is not None:
                lines.append(f"Stoch RSI: {ind.stoch_rsi} ({ind.stoch_rsi_zone})")
            if ind.obv_divergence:
                lines.append(f"OBV: {ind.obv_divergence}")

            if a.statistics:
                st = a.statistics
                lines.append(f"Volatility regime: {st.vol_regime} (realized vol {st.realized_vol_20:.0f}%)")
                lines.append(f"Hurst: {st.hurst_exponent} ({st.trend_classification})")
                lines.append(f"Price Z-score: {st.price_z_score}")
                if st.ffd_d is not None and st.ffd_last is not None:
                    hurst_str = f", hurst={st.ffd_hurst}" if st.ffd_hurst is not None else ""
                    lines.append(
                        f"FFD(d={st.ffd_d:.2f}): last={st.ffd_last}"
                        f" z={st.ffd_z_score}{hurst_str} [stationary-memory series]"
                    )

            if a.regime:
                lines.append(f"Regime: {a.regime.description}")

        return "\n".join(lines)

    def _format_multi_timeframe(self, analyzed: dict[str, AnalyzedSymbol]) -> str:
        lines = ["=== MULTI-TIMEFRAME ==="]
        for sym, a in analyzed.items():
            parts = [f"{sym}:"]
            for tf, ind in [("1h", a.indicators_1h), ("4h", a.indicators_4h)]:
                if ind:
                    parts.append(
                        f"  {tf}: RSI={ind.rsi_14}({ind.rsi_zone}) "
                        f"MACD={ind.macd_signal} ADX={ind.adx_14}({ind.adx_trend}) "
                        f"EMA={ind.ema_position}"
                    )
            if len(parts) > 1:
                lines.extend(parts)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_support_resistance(self, analyzed: dict[str, AnalyzedSymbol]) -> str:
        lines = ["=== SUPPORT/RESISTANCE ==="]
        for sym, a in analyzed.items():
            ind = a.indicators_15m
            if ind and (ind.nearest_support or ind.nearest_resistance):
                s = f"${ind.nearest_support:,.2f}" if ind.nearest_support else "N/A"
                r = f"${ind.nearest_resistance:,.2f}" if ind.nearest_resistance else "N/A"
                lines.append(f"{sym}: Support={s}, Resistance={r}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_risk_regime(self) -> str:
        """Phase 2 §3A.1 — cross-domain regime state with policy hint."""
        if self._market_state is None:
            return ""
        regime_enabled = (
            self._afml is not None
            and getattr(self._afml, "enable_all", True)
            and getattr(self._afml, "enable_risk_regime", True)
        )
        if not regime_enabled:
            return ""
        ms = self._market_state
        state = ms.state
        conf = ms.regime.confirmations
        lines = ["=== RISK REGIME ==="]
        lines.append(
            f"State: {state.upper()}"
            + (f" | Confirmations: {', '.join(conf)}" if conf else "")
        )
        if self._afml is not None:
            if state == "elevated":
                lines.append(
                    f"Policy: size mult {getattr(self._afml, 'regime_elevated_size_mult', 0.50):.2f}, "
                    f"min confidence {getattr(self._afml, 'regime_elevated_min_p', 0.60):.2f}"
                )
            elif state == "crisis":
                lines.append(
                    f"Policy: size mult {getattr(self._afml, 'regime_crisis_size_mult', 0.25):.2f}, "
                    f"min confidence {getattr(self._afml, 'regime_crisis_min_p', 0.70):.2f}, "
                    f"crisis gate ACTIVE (requires confidence >= "
                    f"{getattr(self._afml, 'crisis_gate_min_confidence', 0.80):.2f})"
                )
        return "\n".join(lines)

    def _format_upcoming_events(self, snapshot: MarketSnapshot) -> str:
        if self._event_calendar is None:
            return ""
        enabled = (
            self._afml is not None
            and getattr(self._afml, "enable_all", True)
            and getattr(self._afml, "enable_event_calendar", True)
        )
        if not enabled:
            return ""
        upcoming = self._event_calendar.upcoming()[:5]
        if not upcoming:
            return ""
        lines = ["=== UPCOMING EVENTS ==="]
        import time as _t
        now = _t.time()
        for e in upcoming:
            mins = (e.ts - now) / 60.0
            scope = ("all symbols" if not e.symbols else ",".join(e.symbols))
            lines.append(
                f"{e.name} ({e.impact}, {e.category}) in "
                f"{mins:.0f}m | scope: {scope}"
            )
        # Highlight if any currently within pre-halt window
        halt_hits = [e for e in upcoming
                     if self._event_calendar.state(symbol=None).pre_halt_active
                     and e.ts - now <= self._event_calendar.pre_halt_minutes * 60]
        if halt_hits:
            lines.append("⚠ Pre-event halt ACTIVE — no new entries until event passes")
        return "\n".join(lines)

    def _format_system_health(self) -> str:
        """USDT peg + clock drift + stale data in one compact section."""
        enabled = (
            self._afml is not None
            and getattr(self._afml, "enable_all", True)
        )
        if not enabled:
            return ""
        parts: list[str] = []
        if self._usdt_monitor is not None and self._usdt_monitor.last_status is not None:
            s = self._usdt_monitor.last_status
            if s.off_peg or s.sustained:
                parts.append(
                    f"USDT: ${s.price} ({'sustained off-peg' if s.sustained else 'off-peg'})"
                )
        if not parts:
            return ""
        return "=== SYSTEM HEALTH ===\n" + "\n".join(parts)

    def _format_account_margin(self) -> str:
        state = None
        if self._margin_state_provider is not None:
            # Either a MarginState directly, or a callable that returns one.
            if callable(self._margin_state_provider):
                try:
                    state = self._margin_state_provider()
                except Exception:
                    state = None
            else:
                state = self._margin_state_provider
        enabled = (
            self._afml is not None
            and getattr(self._afml, "enable_all", True)
            and getattr(self._afml, "enable_margin_monitor", True)
        )
        if not enabled or state is None or not getattr(state, "positions", []):
            return ""
        lines = ["=== ACCOUNT MARGIN ==="]
        lines.append(
            f"Used: ${state.used_margin:,.2f} ({state.margin_ratio*100:.1f}%) | "
            f"Free: ${state.free_margin:,.2f} | Equity: ${state.equity:,.2f}"
        )
        for p in state.positions:
            lines.append(
                f"  {p.symbol} {p.side}: mark ${p.mark_price:,.2f} | "
                f"liq ${p.liquidation_price:,.2f} | "
                f"buffer {p.distance_to_liquidation_pct*100:.1f}%"
            )
        return "\n".join(lines)

    def _format_symbol_cooldowns(self) -> str:
        if self._cooldown is None:
            return ""
        enabled = (
            self._afml is not None
            and getattr(self._afml, "enable_all", True)
            and getattr(self._afml, "enable_sl_cooldown", True)
        )
        if not enabled:
            return ""
        items = self._cooldown.lockouts_summary()
        if not items:
            return ""
        lines = ["=== SYMBOL COOLDOWNS ==="]
        for it in items:
            status = []
            if it["locked"]:
                status.append(f"locked for {it['seconds_until_unlock']/60:.0f}m")
            if it["choppy"]:
                status.append("CHOPPY")
            status.append(f"{it['sl_count_in_window']} SLs in 4h window")
            lines.append(f"{it['symbol']}: " + ", ".join(status))
        return "\n".join(lines)

    def _format_baselines(self, snapshot: MarketSnapshot) -> str:
        """Phase 2 §3A.2 — log-relative dynamic baselines per symbol."""
        if self._market_state is None:
            return ""
        baselines_enabled = (
            self._afml is not None
            and getattr(self._afml, "enable_all", True)
            and getattr(self._afml, "enable_baselines", True)
        )
        if not baselines_enabled:
            return ""
        lines = ["=== BASELINES ==="]
        any_set = False
        for sym in snapshot.symbols:
            st = self._market_state.get_symbol(sym)
            if st is None or st.baselines is None:
                continue
            any_set = True
            b = st.baselines
            lines.append(
                f"{sym}: 1Y MA ${b.ma_365d:,.2f} | "
                f"log_rel_1y={b.log_rel_1y:+.3f} ({b.log_rel_1y*100:+.1f}%) | "
                f"90d MA ${b.ma_90d:,.2f} ({b.log_rel_90d*100:+.1f}%) | "
                f"1Y percentile={b.percentile_1y:.2f}"
            )
        return "\n".join(lines) if any_set else ""

    def _format_microstructure(self, analyzed: dict[str, AnalyzedSymbol]) -> str:
        lines = ["=== MICROSTRUCTURE ==="]
        any_set = False
        for sym, a in analyzed.items():
            m = a.microstructure
            if m is None:
                continue
            any_set = True
            lines.append(
                f"{sym}: VPIN-lite={m.vpin_lite:.2f} ({m.buy_pressure}) | "
                f"Amihud z={m.amihud_z:+.2f} | Kyle λ z={m.kyle_lambda_z:+.2f}"
            )
        return "\n".join(lines) if any_set else ""

    def _format_orderbook(self, snapshot: MarketSnapshot) -> str:
        lines = ["=== ORDERBOOK ==="]
        for sym, s in snapshot.symbols.items():
            if s.orderbook:
                ob = s.orderbook
                lines.append(
                    f"{sym}: spread={ob.spread_pct:.3f}% | "
                    f"imbalance={ob.bid_ask_imbalance:+.2f} | "
                    f"bid_depth={ob.bid_depth:.2f} ask_depth={ob.ask_depth:.2f}"
                )
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_derivatives(self, snapshot: MarketSnapshot) -> str:
        lines = ["=== DERIVATIVES ==="]
        for sym, s in snapshot.symbols.items():
            parts = []
            if s.funding:
                rate = s.funding.funding_rate
                ann = rate * 3 * 365 * 100  # annualized
                direction = "longs pay" if rate > 0 else "shorts pay"
                parts.append(f"Funding: {rate*100:.4f}% ({direction}, ~{ann:.1f}% ann)")
            if s.open_interest:
                oi = s.open_interest.get("open_interest_value", 0)
                parts.append(f"OI: ${oi:,.0f}")
            if s.cross_exchange_funding:
                rates_str = ", ".join(
                    f"{ex}={r*100:.4f}%" for ex, r in s.cross_exchange_funding.items()
                )
                parts.append(f"Cross-exchange funding: {rates_str}")
            if parts:
                lines.append(f"{sym}: " + " | ".join(parts))
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_onchain(self, snapshot: MarketSnapshot) -> str:
        if not snapshot.onchain:
            return ""
        oc = snapshot.onchain
        lines = ["=== ON-CHAIN (daily) ==="]
        if oc.total_tvl:
            lines.append(f"DeFi TVL: ${oc.total_tvl/1e9:.1f}B ({oc.tvl_change_24h_pct:+.1f}% 24h)")
        if oc.total_stablecoin_mcap:
            lines.append(f"Stablecoin supply: ${oc.total_stablecoin_mcap/1e9:.1f}B")
        if oc.btc_hash_rate:
            lines.append(f"BTC hash rate: {oc.btc_hash_rate/1e9:.1f} GH/s")
        if oc.btc_mempool_size:
            lines.append(f"BTC mempool: {oc.btc_mempool_size:,} unconfirmed tx")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_sentiment(self, snapshot: MarketSnapshot) -> str:
        if not snapshot.sentiment:
            return ""
        s = snapshot.sentiment
        lines = ["=== SENTIMENT ==="]
        if s.fear_greed:
            fg = s.fear_greed
            lines.append(f"Fear & Greed: {fg.value} ({fg.classification})")
            if fg.history_7d:
                lines.append(f"  7-day trend: {fg.history_7d}")
        if s.top_news:
            lines.append("Top headlines:")
            for i, news in enumerate(s.top_news[:3], 1):
                lines.append(f"  {i}. [{news.sentiment}] {news.title} ({news.source})")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_macro(self, snapshot: MarketSnapshot) -> str:
        if not snapshot.macro or not snapshot.macro.indicators:
            return ""
        lines = ["=== MACRO ==="]
        for name, ind in snapshot.macro.indicators.items():
            lines.append(f"{name.upper()}: {ind.value} ({ind.change_pct:+.1f}%)")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_positions(self, snapshot: MarketSnapshot) -> str:
        lines = ["=== CURRENT POSITIONS ==="]
        if not snapshot.positions:
            lines.append("No open positions")
        else:
            for p in snapshot.positions:
                pnl = p.get("unrealized_pnl", 0)
                pnl_pct = (pnl / (p.get("notional", 1) / p.get("leverage", 1))) * 100 if p.get("notional") else 0
                lines.append(
                    f"{p['symbol']}: {p['side'].upper()} | "
                    f"Size: ${p.get('notional', 0):,.2f} ({p.get('leverage', 1)}x) | "
                    f"Entry: ${p.get('entry_price', 0):,.2f} | "
                    f"Mark: ${p.get('mark_price', 0):,.2f} | "
                    f"PnL: ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                )
                if p.get("liquidation_price"):
                    lines.append(f"  Liquidation: ${p['liquidation_price']:,.2f}")
        return "\n".join(lines)

    def _format_trade_history(self, trades: list[dict] | None) -> str:
        if not trades:
            return ""
        lines = ["=== RECENT TRADES (last 5) ==="]
        for t in trades[-5:]:
            lines.append(
                f"{t.get('side', '').upper()} {t.get('symbol', '')} | "
                f"Entry: ${t.get('entry_price', 0):,.2f} -> "
                f"Exit: ${t.get('exit_price', 0):,.2f} | "
                f"PnL: ${t.get('pnl', 0):+,.2f} ({t.get('pnl_pct', 0):+.1f}%) | "
                f"Duration: {t.get('duration', 'N/A')}"
            )
        return "\n".join(lines)

    def _format_performance(self, perf: dict | None) -> str:
        if not perf:
            return ""
        lines = ["=== PERFORMANCE ==="]
        lines.append(
            f"Win rate: {perf.get('win_rate', 0)*100:.0f}% | "
            f"Trades: {perf.get('total_trades', 0)} | "
            f"Avg PnL: ${perf.get('avg_pnl', 0):+.2f}"
        )
        lines.append(
            f"Max DD: {perf.get('max_drawdown_pct', 0)*100:.1f}% | "
            f"Current DD: {perf.get('current_drawdown_pct', 0)*100:.1f}% | "
            f"Sharpe: {perf.get('sharpe', 0):.2f}"
        )
        # Phase 1 §3.6 + §3.7 — AFML-aware metrics (only shown when computed)
        if "dsr" in perf:
            pbo_str = f" | PBO: {perf['pbo']:.2f}" if "pbo" in perf else ""
            lines.append(
                f"Sharpe (ann): {perf.get('sharpe_annualized', 0):+.2f} | "
                f"DSR (overfit-adj): {perf.get('dsr', 0):.2f} | "
                f"PSR: {perf.get('psr_vs_zero', 0):.2f}{pbo_str}"
            )
            lines.append(
                f"Return skew: {perf.get('pnl_skew', 0):+.2f} | "
                f"Kurtosis: {perf.get('pnl_kurtosis', 0):.2f} | "
                f"Trials deflated: {perf.get('num_trials_for_dsr', 0)}"
            )
        lines.append(
            f"PnL today: ${perf.get('pnl_today', 0):+.2f} | "
            f"PnL total: ${perf.get('pnl_total', 0):+.2f} | "
            f"Streak: {perf.get('streak', 'N/A')}"
        )
        return "\n".join(lines)

    def _format_prediction_markets(self, snapshot: MarketSnapshot) -> str:
        if not snapshot.prediction_markets:
            return ""
        pm = snapshot.prediction_markets
        lines = ["=== PREDICTION MARKETS ==="]

        # Top markets by relevance
        for m in pm.markets[:5]:
            lines.append(
                f"[{m.source}] {m.question[:80]} | "
                f"YES: {m.outcome_yes_price:.0%} | "
                f"Vol 24h: ${m.volume_24h:,.0f} | "
                f"Cat: {m.category}"
            )

        # Anomalies
        if pm.anomalies:
            lines.append("Anomalies:")
            for a in pm.anomalies[:3]:
                lines.append(f"  ! {a.description}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_news(self, snapshot: MarketSnapshot) -> str:
        if not snapshot.news or not snapshot.news.articles:
            return ""
        news = snapshot.news
        lines = ["=== NEWS & HEADLINES ==="]

        # Sentiment summary
        s = news.sentiment_summary
        lines.append(
            f"Sentiment: {s.get('bullish', 0)} bullish, "
            f"{s.get('bearish', 0)} bearish, "
            f"{s.get('neutral', 0)} neutral"
        )

        # Top headlines
        for art in news.articles[:8]:
            sent_tag = f"[{art.sentiment}]" if art.sentiment else ""
            lines.append(f"  {sent_tag} {art.title[:90]} ({art.source})")

        return "\n".join(lines)

    def _format_top_traders(self, snapshot: MarketSnapshot) -> str:
        if not snapshot.top_traders or not snapshot.top_traders.consensus:
            return ""
        lines = ["=== TOP TRADER CONSENSUS ==="]
        for c in snapshot.top_traders.consensus[:5]:
            lines.append(
                f"{c.description} | "
                f"Avg WR: {c.avg_win_rate:.0%}"
            )
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_previous_decision(self, decision: dict | None) -> str:
        if not decision:
            return ""
        lines = ["=== PREVIOUS DECISION ==="]
        lines.append(
            f"Action: {decision.get('action', 'N/A')} | "
            f"Time: {decision.get('time_ago', 'N/A')} ago"
        )
        lines.append(f"Reasoning: {decision.get('reasoning', 'N/A')}")
        if decision.get("outcome"):
            lines.append(f"Outcome: {decision['outcome']}")
        return "\n".join(lines)
