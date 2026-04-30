"""Technical indicator computation and LLM-friendly summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


@dataclass
class IndicatorSummary:
    """LLM-friendly summary of all technical indicators for one symbol/timeframe."""

    # Trend
    rsi_14: float
    rsi_zone: str  # oversold / neutral / overbought
    macd_signal: str  # bullish / bearish
    macd_histogram_trend: str  # expanding / contracting
    ema_position: str  # description of price vs EMAs
    adx_14: float
    adx_trend: str  # strong_trend / weak_trend / no_trend

    # Volatility
    atr_14: float
    atr_pct: float  # ATR as % of price
    bb_position: float  # 0=lower band, 0.5=middle, 1=upper band
    bb_bandwidth: float

    # Volume
    volume_ratio: float  # current vs 20-period average

    # Support/Resistance
    nearest_support: float | None
    nearest_resistance: float | None

    # Extra (Tier 2 — included conditionally)
    stoch_rsi: float | None = None
    stoch_rsi_zone: str | None = None
    obv_divergence: str | None = None  # bullish_div / bearish_div / none
    vwap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if v is not None:
                d[k] = v
        return d


def compute_indicators(df: pd.DataFrame) -> IndicatorSummary | None:
    """Compute all indicators on OHLCV DataFrame and return LLM summary.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
            (or timestamp, open, high, low, close, volume from CCXT)

    Returns:
        IndicatorSummary or None if insufficient data
    """
    if df is None or len(df) < 50:
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    current_price = float(close.iloc[-1])

    # --- RSI ---
    rsi_ind = RSIIndicator(close, window=14)
    rsi = float(rsi_ind.rsi().iloc[-1])
    if np.isnan(rsi):
        rsi = 50.0
    if rsi < 30:
        rsi_zone = "oversold"
    elif rsi > 70:
        rsi_zone = "overbought"
    else:
        rsi_zone = "neutral"

    # --- MACD ---
    macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()
    hist = macd_ind.macd_diff()
    if not hist.empty and len(hist) > 1:
        macd_val = float(macd_line.iloc[-1])
        sig_val = float(signal_line.iloc[-1])
        hist_val = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2])
        macd_signal_str = "bullish" if macd_val > sig_val else "bearish"
        macd_histogram_trend = "expanding" if abs(hist_val) > abs(prev_hist) else "contracting"
    else:
        macd_signal_str = "neutral"
        macd_histogram_trend = "flat"

    # --- EMA crossovers ---
    ema_parts = []
    for length, label in [(9, "EMA9"), (21, "EMA21"), (50, "EMA50"), (200, "EMA200")]:
        if len(close) >= length:
            ema = EMAIndicator(close, window=length).ema_indicator()
            ema_val = float(ema.iloc[-1])
            if not np.isnan(ema_val):
                if current_price > ema_val:
                    ema_parts.append(f"above {label}")
                else:
                    ema_parts.append(f"below {label}")
    ema_position = ", ".join(ema_parts) if ema_parts else "N/A"

    # --- ADX ---
    adx_ind = ADXIndicator(high, low, close, window=14)
    adx_val = float(adx_ind.adx().iloc[-1])
    if np.isnan(adx_val):
        adx_val = 0
    if adx_val > 25:
        adx_trend = "strong_trend"
    elif adx_val > 20:
        adx_trend = "weak_trend"
    else:
        adx_trend = "no_trend"

    # --- ATR ---
    atr_ind = AverageTrueRange(high, low, close, window=14)
    atr_val = float(atr_ind.average_true_range().iloc[-1])
    if np.isnan(atr_val):
        atr_val = 0
    atr_pct = (atr_val / current_price * 100) if current_price else 0

    # --- Bollinger Bands ---
    bb_ind = BollingerBands(close, window=20, window_dev=2)
    bb_upper = float(bb_ind.bollinger_hband().iloc[-1])
    bb_lower = float(bb_ind.bollinger_lband().iloc[-1])
    bb_middle = float(bb_ind.bollinger_mavg().iloc[-1])
    bb_range = bb_upper - bb_lower
    bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
    bb_bandwidth = bb_range / bb_middle if bb_middle > 0 else 0

    # --- Volume ---
    vol_sma = volume.rolling(20).mean()
    current_vol = float(volume.iloc[-1])
    avg_vol = float(vol_sma.iloc[-1]) if not vol_sma.empty and not np.isnan(vol_sma.iloc[-1]) else current_vol
    volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    # --- Support/Resistance (pivot points) ---
    nearest_support, nearest_resistance = _compute_pivots(high, low, close, current_price)

    # --- Tier 2: Stochastic RSI ---
    stoch_rsi_val = None
    stoch_rsi_zone = None
    if rsi < 35 or rsi > 65:  # only compute near extremes
        try:
            stoch_ind = StochRSIIndicator(close, window=14)
            k_val = float(stoch_ind.stochrsi_k().iloc[-1])
            if not np.isnan(k_val):
                stoch_rsi_val = k_val
                if stoch_rsi_val < 0.2:
                    stoch_rsi_zone = "oversold"
                elif stoch_rsi_val > 0.8:
                    stoch_rsi_zone = "overbought"
                else:
                    stoch_rsi_zone = "neutral"
        except Exception:
            pass

    # --- Tier 2: OBV Divergence ---
    obv_div = _detect_obv_divergence(close, volume)

    return IndicatorSummary(
        rsi_14=round(rsi, 1),
        rsi_zone=rsi_zone,
        macd_signal=macd_signal_str,
        macd_histogram_trend=macd_histogram_trend,
        ema_position=ema_position,
        adx_14=round(adx_val, 1),
        adx_trend=adx_trend,
        atr_14=round(atr_val, 4),
        atr_pct=round(atr_pct, 2),
        bb_position=round(bb_position, 2),
        bb_bandwidth=round(bb_bandwidth, 4),
        volume_ratio=round(volume_ratio, 2),
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        stoch_rsi=round(stoch_rsi_val, 2) if stoch_rsi_val is not None else None,
        stoch_rsi_zone=stoch_rsi_zone,
        obv_divergence=obv_div,
    )


def ohlcv_list_to_df(ohlcv: list[list]) -> pd.DataFrame:
    """Convert CCXT OHLCV list to DataFrame."""
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def _compute_pivots(
    high: pd.Series, low: pd.Series, close: pd.Series, current_price: float
) -> tuple[float | None, float | None]:
    """Compute nearest support/resistance from pivot points."""
    h = high.tail(20)
    l = low.tail(20)

    supports = []
    resistances = []

    for i in range(2, len(l) - 2):
        if float(l.iloc[i]) <= float(l.iloc[i - 1]) and float(l.iloc[i]) <= float(l.iloc[i + 1]):
            supports.append(float(l.iloc[i]))
        if float(h.iloc[i]) >= float(h.iloc[i - 1]) and float(h.iloc[i]) >= float(h.iloc[i + 1]):
            resistances.append(float(h.iloc[i]))

    nearest_support = None
    nearest_resistance = None

    below = [s for s in supports if s < current_price]
    above = [r for r in resistances if r > current_price]

    if below:
        nearest_support = round(max(below), 2)
    if above:
        nearest_resistance = round(min(above), 2)

    return nearest_support, nearest_resistance


def _detect_obv_divergence(close: pd.Series, volume: pd.Series) -> str | None:
    """Detect OBV divergence with price."""
    if len(close) < 20:
        return None

    obv_ind = OnBalanceVolumeIndicator(close, volume)
    obv = obv_ind.on_balance_volume()
    if obv is None or obv.empty:
        return None

    price_trend = float(close.iloc[-1]) - float(close.iloc[-10])
    obv_trend = float(obv.iloc[-1]) - float(obv.iloc[-10])

    if price_trend > 0 and obv_trend < 0:
        return "bearish_div"
    elif price_trend < 0 and obv_trend > 0:
        return "bullish_div"
    return None
