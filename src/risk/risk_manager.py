"""Master risk gate: validates all LLM actions before execution."""

from __future__ import annotations

from dataclasses import dataclass

from src.llm.response_parser import ActionType, LLMResponse, TradeAction
from src.risk.circuit_breakers import CircuitBreakers
from src.risk.position_sizer import PositionSizer
from src.risk.stop_loss import validate_sl_tp
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of risk validation for a single action."""

    approved: bool
    action: TradeAction
    adjusted_size_pct: float | None = None  # may be reduced by risk manager
    rejection_reason: str = ""
    warnings: list[str] | None = None


class RiskManager:
    """Master risk gate — ALL LLM actions pass through here before execution.

    The risk manager can:
    - Approve actions as-is
    - Reduce position sizes
    - Reject actions entirely
    - Never INCREASE position sizes or leverage
    """

    def __init__(
        self,
        max_leverage: int = 5,
        max_position_pct: float = 0.25,
        max_total_exposure_pct: float = 0.60,
        max_risk_per_trade_pct: float = 0.02,
        min_reward_risk_ratio: float = 1.5,
        max_concurrent_positions: int = 3,
        allowed_symbols: list[str] | None = None,
        position_sizer: PositionSizer | None = None,
        circuit_breakers: CircuitBreakers | None = None,
        afml_settings: object | None = None,
        market_state: object | None = None,
        symbol_cooldown: object | None = None,
        event_calendar: object | None = None,
    ):
        self.max_leverage = max_leverage
        self.max_position_pct = max_position_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.min_reward_risk_ratio = min_reward_risk_ratio
        self.max_concurrent_positions = max_concurrent_positions
        self.allowed_symbols = allowed_symbols or []
        self.sizer = position_sizer or PositionSizer()
        self.breakers = circuit_breakers or CircuitBreakers()
        self._afml = afml_settings
        self._market_state = market_state
        self._cooldown = symbol_cooldown
        self._event_calendar = event_calendar

    def _afml_bet_sizing_enabled(self) -> bool:
        if self._afml is None:
            return False
        return bool(
            getattr(self._afml, "enable_all", False)
            and getattr(self._afml, "enable_bet_sizing", False)
        )

    def _afml_regime_enabled(self) -> bool:
        if self._afml is None or self._market_state is None:
            return False
        return bool(
            getattr(self._afml, "enable_all", False)
            and getattr(self._afml, "enable_risk_regime", False)
        )

    def _afml_min_p(self) -> float:
        """Dynamic min_p: base AFML value unless regime elevates it."""
        if self._afml is None:
            return 0.55
        base = float(getattr(self._afml, "min_p", 0.55))
        if self._afml_regime_enabled():
            state = getattr(self._market_state, "state", "calm")
            if state == "elevated":
                return max(base, float(getattr(self._afml, "regime_elevated_min_p", 0.60)))
            if state == "crisis":
                return max(base, float(getattr(self._afml, "regime_crisis_min_p", 0.70)))
        return base

    def _regime_size_multiplier(self) -> float:
        if not self._afml_regime_enabled():
            return 1.0
        state = getattr(self._market_state, "state", "calm")
        if state == "crisis":
            return float(getattr(self._afml, "regime_crisis_size_mult", 0.25))
        if state == "elevated":
            return float(getattr(self._afml, "regime_elevated_size_mult", 0.50))
        return float(getattr(self._afml, "regime_calm_size_mult", 1.00))

    def _crisis_gate_blocks(self, confidence: float | None) -> bool:
        """Return True when crisis gate should REJECT the trade."""
        if not self._afml_regime_enabled():
            return False
        state = getattr(self._market_state, "state", "calm")
        if state != "crisis":
            return False
        min_conf = float(getattr(self._afml, "crisis_gate_min_confidence", 0.80))
        return confidence is None or float(confidence) < min_conf

    def validate_response(
        self,
        response: LLMResponse,
        equity: float,
        open_positions: list[dict],
        current_prices: dict[str, float],
        orderbook_depths: dict[str, float] | None = None,
        spread_pcts: dict[str, float] | None = None,
    ) -> list[ValidationResult]:
        """Validate all actions in an LLM response.

        `orderbook_depths` / `spread_pcts` — per-symbol depth (USDT) and spread
        (fraction, not %). When provided, slippage-aware sizing gates apply.
        """
        results = []

        confidence = response.market_assessment.confidence

        for action in response.actions:
            sym_depth = orderbook_depths.get(action.symbol) if orderbook_depths else None
            sym_spread = spread_pcts.get(action.symbol) if spread_pcts else None
            result = self.validate_action(
                action=action,
                equity=equity,
                open_positions=open_positions,
                current_prices=current_prices,
                confidence=confidence,
                orderbook_depth_usdt=sym_depth,
                spread_pct=sym_spread,
            )
            results.append(result)

            log.info(
                "risk_validation",
                action=action.action.value,
                symbol=action.symbol,
                approved=result.approved,
                reason=result.rejection_reason or "OK",
            )

        return results

    def validate_action(
        self,
        action: TradeAction,
        equity: float,
        open_positions: list[dict],
        current_prices: dict[str, float],
        confidence: float | None = None,
        orderbook_depth_usdt: float | None = None,
        spread_pct: float | None = None,
    ) -> ValidationResult:
        """Validate a single trade action.

        `orderbook_depth_usdt` / `spread_pct`: enable slippage-aware sizing
        (Phase 3 §3B.3) when provided. Otherwise that check is skipped.
        """
        warnings: list[str] = []

        # Hold action always passes
        if action.action == ActionType.HOLD:
            return ValidationResult(approved=True, action=action)

        # Close/reduce always allowed (reducing risk is always OK)
        if action.action in (ActionType.CLOSE, ActionType.REDUCE_POSITION):
            return ValidationResult(approved=True, action=action)

        # Circuit breaker check
        trade_allowed, reason = self.breakers.check_trade_allowed()
        if not trade_allowed and action.action in (
            ActionType.OPEN_LONG, ActionType.OPEN_SHORT, ActionType.ADD_TO_POSITION
        ):
            return ValidationResult(
                approved=False, action=action,
                rejection_reason=f"Circuit breaker: {reason}",
            )

        # --- Symbol validation ---
        if self.allowed_symbols and action.symbol not in self.allowed_symbols:
            return ValidationResult(
                approved=False, action=action,
                rejection_reason=f"Symbol {action.symbol} not in allowed list",
            )

        # --- Leverage validation ---
        if action.leverage is not None and action.leverage > self.max_leverage:
            return ValidationResult(
                approved=False, action=action,
                rejection_reason=f"Leverage {action.leverage}x exceeds max {self.max_leverage}x",
            )

        # --- New position checks ---
        if action.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT):
            return self._validate_new_position(
                action, equity, open_positions, current_prices,
                warnings, confidence,
                orderbook_depth_usdt=orderbook_depth_usdt,
                spread_pct=spread_pct,
            )

        # --- Adjust SL/TP checks ---
        if action.action in (ActionType.ADJUST_SL, ActionType.ADJUST_TP):
            return self._validate_adjustment(action, open_positions, current_prices, warnings)

        # --- Add to position ---
        if action.action == ActionType.ADD_TO_POSITION:
            return self._validate_new_position(
                action, equity, open_positions, current_prices,
                warnings, confidence,
                orderbook_depth_usdt=orderbook_depth_usdt,
                spread_pct=spread_pct,
            )

        return ValidationResult(approved=True, action=action, warnings=warnings)

    def _validate_new_position(
        self,
        action: TradeAction,
        equity: float,
        open_positions: list[dict],
        current_prices: dict[str, float],
        warnings: list[str],
        confidence: float | None = None,
        orderbook_depth_usdt: float | None = None,
        spread_pct: float | None = None,
    ) -> ValidationResult:
        """Validate a new position (open_long/open_short/add_to_position)."""
        # Position count limit
        if len(open_positions) >= self.max_concurrent_positions:
            return ValidationResult(
                approved=False, action=action,
                rejection_reason=f"Already {len(open_positions)} positions (max {self.max_concurrent_positions})",
            )

        # Phase 3 §3B.9 — post-SL cooldown per symbol
        if (self._cooldown is not None
                and self._afml is not None
                and getattr(self._afml, "enable_all", False)
                and getattr(self._afml, "enable_sl_cooldown", False)):
            locked, unlock_ts = self._cooldown.is_locked(action.symbol)
            if locked:
                remaining = self._cooldown.seconds_until_unlock(action.symbol) / 60.0
                return ValidationResult(
                    approved=False, action=action,
                    rejection_reason=(
                        f"Symbol {action.symbol} on post-SL cooldown "
                        f"({remaining:.0f} min remaining)"
                    ),
                )
            # Chop check (Phase 3 §3B.15)
            if (getattr(self._afml, "enable_revenge_trade_detector", False)
                    and self._cooldown.is_choppy(action.symbol)):
                min_conf = float(getattr(self._afml, "chop_min_confidence", 0.80))
                if confidence is None or float(confidence) < min_conf:
                    return ValidationResult(
                        approved=False, action=action,
                        rejection_reason=(
                            f"Symbol {action.symbol} classified CHOPPY — "
                            f"confidence {confidence} below chop threshold {min_conf}"
                        ),
                    )

        # Phase 3 §3B.14 — pre-event halt
        if (self._event_calendar is not None
                and self._afml is not None
                and getattr(self._afml, "enable_all", False)
                and getattr(self._afml, "enable_event_calendar", False)):
            state = self._event_calendar.state(symbol=action.symbol)
            if state.pre_halt_active and state.next_event is not None:
                return ValidationResult(
                    approved=False, action=action,
                    rejection_reason=(
                        f"Pre-event halt: '{state.next_event.name}' in "
                        f"{state.minutes_until:.0f} min (impact: {state.next_event.impact})"
                    ),
                )

        # Position size cap
        size_pct = action.size_pct or 10
        if size_pct > self.max_position_pct * 100:
            size_pct = self.max_position_pct * 100
            warnings.append(f"Size reduced from {action.size_pct}% to {size_pct}%")

        # Apply circuit breaker size reduction
        size_pct *= self.breakers.size_reduction
        if self.breakers.size_reduction < 1.0:
            warnings.append(f"Volatility reduction applied: {self.breakers.size_reduction:.0%}")

        # Phase 2 §3A.1 — crisis gate + regime-policy size multiplier
        if self._crisis_gate_blocks(confidence):
            return ValidationResult(
                approved=False, action=action,
                rejection_reason=(
                    "Crisis regime active and confidence below crisis-gate threshold "
                    f"(required >= {getattr(self._afml, 'crisis_gate_min_confidence', 0.80):.2f})"
                ),
            )
        regime_mult = self._regime_size_multiplier()
        if regime_mult < 1.0:
            warnings.append(
                f"Regime size reduction applied: {regime_mult:.2f}x "
                f"(state={getattr(self._market_state, 'state', 'calm')})"
            )
            size_pct *= regime_mult

        # Phase 1 §3.4 — AFML bet sizing from LLM confidence (strictly size-reducing
        # multiplier ∈ [0, 1]; only applied to new-position openings).
        if confidence is not None and self._afml_bet_sizing_enabled():
            from src.risk.bet_sizing import bet_size_from_probability

            mult = bet_size_from_probability(
                p=float(confidence),
                num_classes=2,
                min_p=float(self._afml_min_p()),
                max_scale=1.0,
            )
            if mult <= 0.0:
                return ValidationResult(
                    approved=False, action=action,
                    rejection_reason=(
                        f"AFML bet-sizing rejected trade: confidence {confidence:.2f} "
                        f"below min_p {self._afml_min_p():.2f}"
                    ),
                )
            if mult < 1.0:
                warnings.append(
                    f"AFML bet-sizing: size scaled by {mult:.2f} "
                    f"(confidence {confidence:.2f})"
                )
            size_pct *= mult

        # Phase 3 §3B.3 — slippage-aware shrink
        if (self._afml is not None
                and getattr(self._afml, "enable_all", False)
                and getattr(self._afml, "enable_slippage_sizing", False)
                and orderbook_depth_usdt is not None
                and spread_pct is not None):
            from src.execution.order_utils import estimate_slippage

            side = ("buy" if action.action in (ActionType.OPEN_LONG, ActionType.ADD_TO_POSITION)
                    else "sell")
            budget_pct = float(getattr(self._afml, "slippage_budget_pct", 0.0015))
            proposed_notional = equity * (size_pct / 100)
            depth_half = max(1.0, orderbook_depth_usdt / 2.0)
            slip = estimate_slippage(
                side=side, size_usd=proposed_notional,
                bid_depth=depth_half, ask_depth=depth_half,
                spread_pct=spread_pct,
            )
            if slip > budget_pct:
                # Shrink notional so that predicted slippage ≈ budget_pct
                # (slippage scales ~ linearly with size beyond spread).
                # Conservative: multiplicative shrink by budget/slip ratio.
                shrink = budget_pct / slip if slip > 0 else 1.0
                new_size_pct = max(0.0, size_pct * shrink)
                warnings.append(
                    f"Slippage-aware shrink: size_pct {size_pct:.1f}% → "
                    f"{new_size_pct:.1f}% (est slip {slip*100:.3f}% > budget {budget_pct*100:.3f}%)"
                )
                size_pct = new_size_pct

        # Total exposure check
        current_exposure = sum(
            abs(p.get("notional", 0)) for p in open_positions
        )
        new_position_notional = equity * (size_pct / 100)
        total_after = current_exposure + new_position_notional
        if total_after > equity * self.max_total_exposure_pct:
            available = max(0, equity * self.max_total_exposure_pct - current_exposure)
            if available <= 0:
                return ValidationResult(
                    approved=False, action=action,
                    rejection_reason="Maximum total exposure reached",
                )
            size_pct = (available / equity) * 100
            warnings.append(f"Size reduced to {size_pct:.1f}% due to exposure limit")

        # Stop-loss validation
        entry_price = action.entry_price or current_prices.get(action.symbol, 0)
        if entry_price <= 0:
            return ValidationResult(
                approved=False, action=action,
                rejection_reason="Cannot determine entry price",
            )

        if action.stop_loss is not None:
            side = "long" if action.action in (ActionType.OPEN_LONG, ActionType.ADD_TO_POSITION) else "short"
            sl_valid, sl_reason = validate_sl_tp(
                entry_price=entry_price,
                side=side,
                stop_loss=action.stop_loss,
                take_profit=action.take_profit,
                max_risk_pct=self.max_risk_per_trade_pct * (action.leverage or 1),
                min_rr_ratio=self.min_reward_risk_ratio,
            )
            if not sl_valid:
                return ValidationResult(
                    approved=False, action=action,
                    rejection_reason=f"SL/TP invalid: {sl_reason}",
                )

        return ValidationResult(
            approved=True,
            action=action,
            adjusted_size_pct=size_pct,
            warnings=warnings if warnings else None,
        )

    def _validate_adjustment(
        self,
        action: TradeAction,
        open_positions: list[dict],
        current_prices: dict[str, float],
        warnings: list[str],
    ) -> ValidationResult:
        """Validate SL/TP adjustments — must not increase risk."""
        # Find the existing position
        position = None
        for p in open_positions:
            if p.get("symbol") == action.symbol:
                position = p
                break

        if position is None:
            return ValidationResult(
                approved=False, action=action,
                rejection_reason=f"No open position for {action.symbol}",
            )

        # For SL adjustments: new SL must not be further from entry than current SL
        # (i.e., cannot increase risk)
        if action.action == ActionType.ADJUST_SL and action.stop_loss is not None:
            entry = position.get("entry_price", 0)
            side = position.get("side", "long")

            if side == "long":
                # Moving SL down = increasing risk = NOT allowed
                current_sl_distance = entry - (position.get("stop_loss") or entry)
                new_sl_distance = entry - action.stop_loss
                if new_sl_distance > current_sl_distance and current_sl_distance > 0:
                    return ValidationResult(
                        approved=False, action=action,
                        rejection_reason="Cannot move SL further from entry (increases risk)",
                    )

        return ValidationResult(approved=True, action=action, warnings=warnings if warnings else None)
