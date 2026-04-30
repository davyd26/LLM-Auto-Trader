"""Market regime classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange


@dataclass
class MarketRegime:
    """Market regime classification result."""

    regime: str          # trending_up, trending_down, ranging, volatile
    confidence: float    # 0-1
    adx: float
    atr_pct: float
    vol_percentile: float  # current volatility percentile (0-100)
    description: str       # human-readable summary for LLM


def classify_regime(df: pd.DataFrame) -> MarketRegime | None:
    """Classify current market regime from OHLCV data.

    Uses ADX for trend strength, ATR for volatility, and
    price position relative to EMAs for direction.

    Args:
        df: OHLCV DataFrame with at least 50 rows.
    """
    if df is None or len(df) < 50:
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ADX for trend strength
    adx_ind = ADXIndicator(high, low, close, window=14)
    adx = float(adx_ind.adx().iloc[-1])
    plus_di = float(adx_ind.adx_pos().iloc[-1])
    minus_di = float(adx_ind.adx_neg().iloc[-1])

    if np.isnan(adx):
        return None

    # ATR for volatility
    atr_ind = AverageTrueRange(high, low, close, window=14)
    current_atr = float(atr_ind.average_true_range().iloc[-1])
    if np.isnan(current_atr):
        return None
    atr_pct = current_atr / float(close.iloc[-1]) * 100

    # Volatility percentile (where is current vol vs historical)
    returns = close.pct_change().dropna()
    current_vol = float(returns.tail(14).std())
    historical_vols = returns.rolling(14).std().dropna()
    if len(historical_vols) > 0:
        vol_percentile = float(
            (historical_vols < current_vol).sum() / len(historical_vols) * 100
        )
    else:
        vol_percentile = 50.0

    # EMA direction
    ema_21 = EMAIndicator(close, window=21).ema_indicator()
    ema_50 = EMAIndicator(close, window=50).ema_indicator() if len(close) >= 50 else None

    price_above_ema21 = float(close.iloc[-1]) > float(ema_21.iloc[-1])
    price_above_ema50 = float(close.iloc[-1]) > float(ema_50.iloc[-1]) if ema_50 is not None else True

    # Classify
    if adx > 25:
        if plus_di > minus_di and price_above_ema21:
            regime = "trending_up"
            description = f"Strong uptrend (ADX={adx:.0f}, +DI>{minus_di:.0f})"
        elif minus_di > plus_di and not price_above_ema21:
            regime = "trending_down"
            description = f"Strong downtrend (ADX={adx:.0f}, -DI>{plus_di:.0f})"
        else:
            regime = "trending_up" if price_above_ema50 else "trending_down"
            description = f"Trending ({'up' if price_above_ema50 else 'down'}) with mixed signals (ADX={adx:.0f})"
    elif vol_percentile > 80:
        regime = "volatile"
        description = f"High volatility regime (ATR={atr_pct:.1f}%, vol percentile={vol_percentile:.0f})"
    else:
        regime = "ranging"
        description = f"Range-bound market (ADX={adx:.0f}, low directional movement)"

    # Confidence based on signal clarity
    if regime.startswith("trending"):
        confidence = min(1.0, adx / 40)
    elif regime == "volatile":
        confidence = min(1.0, vol_percentile / 100)
    else:
        confidence = 1.0 - (adx / 50)

    confidence = max(0.1, min(1.0, confidence))

    return MarketRegime(
        regime=regime,
        confidence=round(confidence, 2),
        adx=round(adx, 1),
        atr_pct=round(atr_pct, 2),
        vol_percentile=round(vol_percentile, 1),
        description=description,
    )
