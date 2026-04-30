"""Position sizing: Fractional Kelly + ATR-based."""

from __future__ import annotations

import math


class PositionSizer:
    """Calculates position sizes using the minimum of Kelly and ATR methods."""

    def __init__(
        self,
        kelly_fraction_under_2k: float = 0.125,
        kelly_fraction_2k_5k: float = 0.20,
        kelly_fraction_5k_10k: float = 0.25,
        kelly_fraction_10k_plus: float = 0.33,
        max_position_pct: float = 0.25,
        max_risk_per_trade_pct: float = 0.02,
        min_trade_fee_multiple: float = 3.0,
        taker_fee: float = 0.0001,
    ):
        self._kelly_tiers = [
            (2000, kelly_fraction_under_2k),
            (5000, kelly_fraction_2k_5k),
            (10000, kelly_fraction_5k_10k),
            (float("inf"), kelly_fraction_10k_plus),
        ]
        self.max_position_pct = max_position_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.min_trade_fee_multiple = min_trade_fee_multiple
        self.taker_fee = taker_fee

    def calculate_size(
        self,
        equity: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        atr_pct: float,
        leverage: int,
        current_price: float,
        existing_exposure_pct: float = 0,
        max_total_exposure_pct: float = 0.60,
        correlated_positions: int = 0,
    ) -> float:
        """Calculate optimal position size in USDT.

        Returns the position size (notional, before leverage) in USDT.
        The actual margin used = position_size / leverage.

        Args:
            equity: Current account equity in USDT
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade profit in USDT
            avg_loss: Average losing trade loss in USDT (positive number)
            atr_pct: Current ATR as % of price
            leverage: Intended leverage for this trade
            current_price: Current price of the asset
            existing_exposure_pct: Current total exposure as % of equity
            max_total_exposure_pct: Maximum allowed total exposure %
            correlated_positions: Number of existing correlated positions
        """
        if equity <= 0 or atr_pct <= 0:
            return 0

        # --- Method 1: Fractional Kelly ---
        kelly_size = self._kelly_size(equity, win_rate, avg_win, avg_loss)

        # --- Method 2: ATR-based risk sizing ---
        atr_size = self._atr_size(equity, atr_pct, leverage)

        # --- Method 3: Hard cap ---
        cap_size = equity * self.max_position_pct

        # Take the MINIMUM (most conservative)
        size = min(kelly_size, atr_size, cap_size)

        # Adjust for remaining exposure room
        room = max(0, max_total_exposure_pct - existing_exposure_pct)
        size = min(size, equity * room)

        # Correlation discount: reduce by 25% per correlated position
        if correlated_positions > 0:
            discount = 1 - (0.25 * min(correlated_positions, 3))
            size *= max(0.25, discount)

        # Minimum profitability check: trade must exceed 3x round-trip fees
        round_trip_fees = size * self.taker_fee * 2  # entry + exit
        min_expected_profit = round_trip_fees * self.min_trade_fee_multiple
        if size * (atr_pct / 100) < min_expected_profit:
            return 0  # Not worth trading

        # Round down to reasonable precision
        return max(0, math.floor(size * 100) / 100)

    def _kelly_size(
        self, equity: float, win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Calculate position size using fractional Kelly criterion."""
        # Need minimum data for Kelly
        if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
            # No historical data: use conservative fixed fraction
            return equity * 0.05  # 5% of equity

        b = avg_win / avg_loss  # win/loss ratio
        q = 1 - win_rate

        full_kelly = (win_rate * b - q) / b
        if full_kelly <= 0:
            return 0  # Negative edge: don't trade

        # Apply fractional Kelly based on capital tier
        fraction = self._get_kelly_fraction(equity)
        return equity * full_kelly * fraction

    def _atr_size(self, equity: float, atr_pct: float, leverage: int) -> float:
        """Calculate position size based on ATR risk.

        Logic: risk_per_trade = equity * max_risk_pct
        If SL is at 1.5 * ATR, then:
        position_size = risk_per_trade / (1.5 * atr_pct / 100)
        """
        risk_amount = equity * self.max_risk_per_trade_pct
        sl_distance_pct = 1.5 * atr_pct / 100  # SL at 1.5 ATR

        if sl_distance_pct <= 0:
            return 0

        # Position size that risks exactly max_risk_per_trade at the SL distance
        size = risk_amount / sl_distance_pct
        return size

    def _get_kelly_fraction(self, equity: float) -> float:
        """Get Kelly fraction based on capital tier."""
        for threshold, fraction in self._kelly_tiers:
            if equity < threshold:
                return fraction
        return self._kelly_tiers[-1][1]


def minimum_trade_size(equity: float, taker_fee: float = 0.0001) -> float:
    """Calculate minimum trade size worth executing given fees."""
    # A trade needs to cover round-trip fees * 3
    # round_trip = size * fee * 2
    # profit = size * expected_move
    # Need: expected_move > fee * 2 * 3 = fee * 6
    # For 0.01% taker: minimum move of 0.06% needed
    # Practical minimum: $10 or 1% of equity, whichever is larger
    return max(10, equity * 0.01)
