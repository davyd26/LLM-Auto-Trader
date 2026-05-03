"""Quantitative overlays on top of the deterministic baseline strategy.

The baseline picks DIRECTION (open_long / hold / hedge); the overlay
adjusts SIZING and applies hard tail-risk overrides. Same architectural
split that a production LLM-driven system uses: the LLM (or a signal
process) picks direction, the overlay handles the math.

Components:

    GARCH-vol-targeted sizing:
        Replace realized-vol denominator in vol-targeting with a
        GARCH(1,1) one-step-ahead forecast. GARCH gives advance warning
        of crisis tape (vol z-score > 2.5).

    Adaptive Kelly fraction:
        From the strategy's own realized PnL distribution over the most
        recent N trades, compute the empirical Kelly fraction
        (clamped to [floor, cap]). Multiplies the size_pct.

    Z-scores (1w / 4w / 12w / 26w returns):
        Pure signal description for the LLM context — not used in
        deterministic sizing. The LLM voter prompts can read these to
        gauge stretch/exhaustion.

    Emergency overrides:
        Hard rules that fire BEFORE the LLM sees the snapshot, forcing
        the strategy flat regardless of opinion:
          - Drawdown halt   (live cap)
          - GARCH vol z-score spike (regime shock)

These overlays are OPT-IN via ``QuantOverlayConfig``. Behavior is byte-
identical to the prior engine when ``quant_overlay=None``.

GARCH state is computed once per backtest from a warmup window so the
parameters are FROZEN — no in-sample re-fitting on realized PnL,
preventing data-snooping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.analysis.garch import fit_garch_11


@dataclass
class QuantOverlayConfig:
    """Opt-in overlay configuration. None disables all overlays."""

    # GARCH vol-targeting: replace realized-vol denominator
    use_garch_vol: bool = False
    garch_warmup_bars: int = 52       # min bars to fit GARCH parameters
    garch_crisis_z: float = 2.5       # sigma_z above this → crisis flag

    # Adaptive Kelly sizing: scale size by empirical Kelly fraction
    kelly_adaptive: bool = False
    kelly_lookback_trades: int = 20    # rolling-window of recent trades
    kelly_fraction_multiplier: float = 0.25  # 1/4 Kelly (safe)
    kelly_floor: float = 0.20         # don't shrink size below this fraction
    kelly_cap: float = 1.0            # don't expand above this

    # Emergency overrides (deterministic; fire before sizing/decision)
    emergency_dd_halt: float | None = None     # if running DD ≥ this → flat
    emergency_garch_z: float | None = None     # if GARCH sigma_z ≥ this → flat
    emergency_disable_hedge: bool = True       # in emergency, also stay out of PAXG


MA_HORIZONS: tuple[int, ...] = (4, 8, 12, 26, 52)
RSI_PERIODS: tuple[int, ...] = (7, 14, 21)
MACD_PARAMS: tuple[tuple[int, int, int], ...] = ((12, 26, 9), (5, 13, 5))


@dataclass
class QuantSignals:
    """Per-bar signal context produced by ``compute_quant_signals``."""

    garch_sigma_t: np.ndarray            # current conditional sigma per bar
    garch_sigma_t_plus_1: np.ndarray     # 1-step-ahead forecast per bar
    garch_sigma_z: np.ndarray            # z-score of sigma vs trailing distribution
    garch_crisis_flag: np.ndarray        # bool array; True when sigma_z ≥ crisis_z
    z_4w: np.ndarray                     # z-score of 4-week return vs 26w distribution
    z_12w: np.ndarray                    # z-score of 12-week return
    z_26w: np.ndarray                    # z-score of 26-week return
    sma: dict[int, np.ndarray] = field(default_factory=dict)   # SMA at each horizon
    wma: dict[int, np.ndarray] = field(default_factory=dict)   # WMA at each horizon
    ema: dict[int, np.ndarray] = field(default_factory=dict)   # EMA at each horizon
    rsi: dict[int, np.ndarray] = field(default_factory=dict)   # RSI(period) per horizon
    macd: dict[tuple[int, int, int], dict] = field(default_factory=dict)
    # macd[(f, s, sig)] = {"line", "signal", "hist"} → np.ndarray each
    volume_recent: np.ndarray = field(default_factory=lambda: np.array([]))
    volume_z: np.ndarray = field(default_factory=lambda: np.array([]))
    volume_profile: list[dict] = field(default_factory=list)


def _sma(values: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average over `n` bars (NaN until window fills)."""
    out = np.full(len(values), np.nan)
    if n <= 0 or len(values) < n:
        return out
    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    out[n - 1:] = (cumsum[n:] - cumsum[:-n]) / n
    return out


def _wma(values: np.ndarray, n: int) -> np.ndarray:
    """Linearly-weighted moving average: weight i = i (1..n) / sum."""
    out = np.full(len(values), np.nan)
    if n <= 0 or len(values) < n:
        return out
    weights = np.arange(1, n + 1, dtype=float)
    weight_sum = weights.sum()
    for i in range(n - 1, len(values)):
        window = values[i - n + 1: i + 1]
        out[i] = float(np.dot(window, weights) / weight_sum)
    return out


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI(period). NaN until period+1 bars available."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # Seed averages with the first `period` bars
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    out[period] = 100.0 - 100.0 / (1.0 + (avg_gain / avg_loss if avg_loss > 0 else float("inf")))
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _macd(
    closes: np.ndarray, fast: int, slow: int, signal_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (macd_line, signal_line, histogram). Each NaN-padded until ready."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    # Signal line = EMA of macd_line; we EMA the non-NaN portion to align
    valid_idx = np.where(~np.isnan(macd_line))[0]
    signal_line = np.full(len(closes), np.nan)
    if len(valid_idx) >= signal_n:
        sub = macd_line[valid_idx]
        sub_ema = _ema(sub, signal_n)
        signal_line[valid_idx] = sub_ema
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _ema(values: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average with smoothing 2/(n+1)."""
    out = np.full(len(values), np.nan)
    if n <= 0 or len(values) == 0:
        return out
    alpha = 2.0 / (n + 1.0)
    # Seed with the first SMA(n) bar (so EMA aligns with SMA at warmup)
    if len(values) < n:
        return out
    seed = float(np.mean(values[:n]))
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(values)):
        prev = alpha * float(values[i]) + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _rolling_z(values: np.ndarray, lookback: int) -> np.ndarray:
    """Trailing z-score over `lookback` bars (NaN until window fills)."""
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(lookback, n):
        win = values[i - lookback:i]
        win = win[np.isfinite(win)]
        if len(win) < 4:
            continue
        mu = float(np.mean(win))
        sd = float(np.std(win, ddof=1))
        if sd <= 0:
            out[i] = 0.0
        else:
            out[i] = (values[i] - mu) / sd
    return out


def _volume_profile(
    closes: np.ndarray,
    volumes: np.ndarray,
    n_bins: int = 12,
    lookback: int = 26,
) -> list[dict]:
    """Build a recent-history volume profile (TPO-style) over the last
    `lookback` bars: bins of [low, high, total_volume, n_bars_in_bin].

    Returns the bin list sorted high→low price (display convention).
    """
    if len(closes) < 4 or lookback < 4:
        return []
    n = len(closes)
    start = max(0, n - lookback)
    win_close = closes[start:]
    win_vol = volumes[start:] if len(volumes) >= n else np.full(len(win_close), 1.0)
    if win_close.size == 0:
        return []
    pmin, pmax = float(np.min(win_close)), float(np.max(win_close))
    if pmax <= pmin:
        return [{"low": pmin, "high": pmax, "volume": float(np.sum(win_vol)),
                 "n_bars": int(len(win_close))}]
    edges = np.linspace(pmin, pmax, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (win_close >= lo) & (win_close < hi if i < n_bins - 1 else win_close <= hi)
        bins.append({
            "low": lo, "high": hi,
            "volume": float(np.sum(win_vol[mask])),
            "n_bars": int(np.sum(mask)),
        })
    bins.sort(key=lambda b: -b["high"])  # high price first
    return bins


def compute_quant_signals(
    closes: np.ndarray,
    cfg: QuantOverlayConfig,
    volumes: np.ndarray | None = None,
) -> QuantSignals:
    """Compute GARCH(1,1) state series + multi-horizon z-scores once.

    GARCH parameters are fit on the warmup bars and FROZEN. Per-bar
    sigma series is the recurrence:
        var_t = omega + alpha * r_{t-1}^2 + beta * var_{t-1}

    No future bars are used (purely backward-looking).
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    sigma_t = np.full(n, np.nan)
    sigma_fwd = np.full(n, np.nan)
    crisis = np.zeros(n, dtype=bool)
    sigma_z = np.full(n, np.nan)

    if n >= 2:
        rets = np.zeros(n)
        rets[1:] = np.diff(np.log(np.maximum(closes, 1e-12)))
    else:
        rets = np.zeros(n)

    warmup = max(20, cfg.garch_warmup_bars)
    if n >= warmup:
        warmup_rets = rets[1: warmup + 1]  # exclude rets[0]=0
        omega, alpha, beta = fit_garch_11(warmup_rets)
        # Initialize variance at warmup boundary; fit_garch_11 uses var(returns)
        # for the first bin. We just step forward bar-by-bar from warmup.
        var_prev = float(np.var(warmup_rets[np.isfinite(warmup_rets)])) if len(warmup_rets) > 0 else 1e-6
        var_prev = max(var_prev, 1e-12)
        # Walk from warmup onward
        for t in range(warmup, n):
            var_t = omega + alpha * (rets[t - 1] ** 2) + beta * var_prev
            var_t = max(var_t, 1e-12)
            sigma_t[t] = math.sqrt(var_t)
            # 1-step-ahead forecast (using the new return)
            fwd_var = omega + alpha * (rets[t] ** 2) + beta * var_t
            sigma_fwd[t] = math.sqrt(max(fwd_var, 1e-12))
            var_prev = var_t

        # Compute sigma_z = (sigma_t - trailing_mean) / trailing_std with
        # trailing window of `warmup` bars
        for t in range(warmup * 2, n):
            window = sigma_t[t - warmup: t]
            window = window[np.isfinite(window)]
            if len(window) >= 4:
                mu = float(np.mean(window))
                sd = float(np.std(window, ddof=1))
                if sd > 0:
                    sigma_z[t] = (sigma_t[t] - mu) / sd
                    crisis[t] = sigma_z[t] >= cfg.garch_crisis_z

    # Multi-horizon return z-scores
    log_close = np.log(np.maximum(closes, 1e-12))

    def _ret_horizon(h: int) -> np.ndarray:
        out = np.full(n, np.nan)
        for i in range(h, n):
            out[i] = float(log_close[i] - log_close[i - h])
        return out

    z_4 = _rolling_z(_ret_horizon(4), 26)
    z_12 = _rolling_z(_ret_horizon(12), 52)
    z_26 = _rolling_z(_ret_horizon(26), 78)

    # Moving-average panel (per LLM-context request): SMA / WMA / EMA at
    # each MA_HORIZONS entry. Pure backward-looking, no future bars.
    sma_dict = {h: _sma(closes, h) for h in MA_HORIZONS}
    wma_dict = {h: _wma(closes, h) for h in MA_HORIZONS}
    ema_dict = {h: _ema(closes, h) for h in MA_HORIZONS}

    # RSI panel
    rsi_dict = {p: _rsi(closes, p) for p in RSI_PERIODS}

    # MACD panel
    macd_dict: dict[tuple[int, int, int], dict] = {}
    for fast, slow, sig in MACD_PARAMS:
        line, sigl, hist = _macd(closes, fast, slow, sig)
        macd_dict[(fast, slow, sig)] = {"line": line, "signal": sigl, "hist": hist}

    # Volume features (last bar's volume + z-score + 26-bar profile)
    if volumes is None:
        volumes_arr = np.zeros(n)
    else:
        volumes_arr = np.asarray(volumes, dtype=float)
        if len(volumes_arr) != n:
            volumes_arr = np.zeros(n)
    vol_z = _rolling_z(volumes_arr, 26)
    vol_profile_last = _volume_profile(closes, volumes_arr, n_bins=12, lookback=26)

    return QuantSignals(
        garch_sigma_t=sigma_t,
        garch_sigma_t_plus_1=sigma_fwd,
        garch_sigma_z=sigma_z,
        garch_crisis_flag=crisis,
        z_4w=z_4,
        z_12w=z_12,
        z_26w=z_26,
        sma=sma_dict,
        wma=wma_dict,
        ema=ema_dict,
        rsi=rsi_dict,
        macd=macd_dict,
        volume_recent=volumes_arr,
        volume_z=vol_z,
        volume_profile=vol_profile_last,
    )


@dataclass
class KellyState:
    """Rolling Kelly fraction from realized strategy trades."""

    fraction: float = 1.0  # current adaptive multiplier on size_pct
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    n_trades_used: int = 0


def update_kelly_state(
    pnls: list[float],
    overlay: QuantOverlayConfig,
) -> KellyState:
    """Compute fractional-Kelly multiplier from a list of recent trade PnLs.

    Classic Kelly for asymmetric trades:
        f* = (p × b − q) / b   where b = avg_win / avg_loss, q = 1 − p

    We apply ``kelly_fraction_multiplier`` (e.g., 0.25 = 1/4 Kelly) to
    avoid blowing up on small samples, then clamp to [floor, cap].

    Returns a KellyState with `fraction` ∈ [floor, cap] suitable for
    multiplying ``long_size_usd_pct``.
    """
    if not pnls or len(pnls) < 4:
        return KellyState(fraction=1.0)
    arr = np.asarray(pnls, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return KellyState(fraction=1.0)

    wins = arr[arr > 0]
    losses = -arr[arr < 0]
    if len(wins) == 0 or len(losses) == 0:
        # All wins or all losses — degenerate; default neutral fraction
        return KellyState(fraction=1.0, n_trades_used=len(arr))

    p = float(len(wins) / len(arr))
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(losses))
    if avg_loss <= 0:
        return KellyState(fraction=1.0, win_rate=p, avg_win=avg_win, avg_loss=avg_loss,
                          n_trades_used=len(arr))
    b = avg_win / avg_loss
    if b <= 0:
        return KellyState(fraction=overlay.kelly_floor, win_rate=p, avg_win=avg_win,
                          avg_loss=avg_loss, n_trades_used=len(arr))
    q = 1.0 - p
    f_star = (p * b - q) / b
    f_used = max(0.0, f_star) * overlay.kelly_fraction_multiplier
    f_used = max(overlay.kelly_floor, min(overlay.kelly_cap, f_used))
    return KellyState(
        fraction=float(f_used), win_rate=p, avg_win=avg_win, avg_loss=avg_loss,
        n_trades_used=len(arr),
    )


@dataclass
class EmergencyState:
    """Emergency-override decision for the current bar."""

    force_flat: bool = False
    reason: str = ""


def portfolio_state_to_llm_context(
    equity_usd: float,
    positions: list[dict] | None,
    current_prices: dict[str, float] | None,
    recent_pnls: list[float] | None,
    equity_curve_recent: list[float] | None,
) -> dict:
    """Render the FULL portfolio state for the LLM.

    Inputs:
        equity_usd        : total account equity in USD
        positions         : list of {symbol, side, size, entry_price, entry_ts_ms}
        current_prices    : map symbol → latest close (for unrealized PnL marks)
        recent_pnls       : list of recent closed-trade PnLs (for Kelly stats)
        equity_curve_recent : recent equity points (last 12-26 weeks) for trend

    Output: JSON-friendly dict with cash, per-position MTM, exposure totals,
    leverage, running max-DD, recent realized return.
    """
    positions = positions or []
    current_prices = current_prices or {}
    recent_pnls = recent_pnls or []
    eq_curve = equity_curve_recent or []

    long_notional = 0.0
    short_notional = 0.0
    pos_block = []
    total_unreal_pnl = 0.0
    for p in positions:
        sym = p.get("symbol")
        side = p.get("side")
        size = float(p.get("size", 0.0) or 0.0)
        entry = float(p.get("entry_price", 0.0) or 0.0)
        last = float(current_prices.get(sym, entry) or entry)
        notional = entry * size
        if side == "long":
            unreal = (last - entry) * size
            long_notional += notional
        else:  # short
            unreal = (entry - last) * size
            short_notional += notional
        total_unreal_pnl += unreal
        pos_block.append({
            "symbol": sym, "side": side, "size": size,
            "entry_price": entry, "last_price": last,
            "notional_usd": notional,
            "unrealized_pnl_usd": unreal,
            "unrealized_pnl_pct": (unreal / notional) if notional > 0 else 0.0,
        })
    net_exposure_usd = long_notional - short_notional
    gross_exposure_usd = long_notional + short_notional
    cash_usd = max(0.0, equity_usd - gross_exposure_usd)
    leverage = gross_exposure_usd / equity_usd if equity_usd > 0 else 0.0

    # Running max-DD over the recent equity curve
    max_dd_recent = 0.0
    if len(eq_curve) >= 2:
        peak = eq_curve[0]
        for v in eq_curve[1:]:
            peak = max(peak, v)
            if peak > 0:
                max_dd_recent = max(max_dd_recent, (peak - v) / peak)

    # Recent realized return (4 / 12 weeks if available)
    def _ret(curve: list[float], n: int) -> float | None:
        if len(curve) < n + 1 or curve[-n - 1] <= 0:
            return None
        return curve[-1] / curve[-n - 1] - 1.0

    realized_4w = _ret(eq_curve, 4)
    realized_12w = _ret(eq_curve, 12)

    # Win/loss summary from recent_pnls
    wins = [p for p in recent_pnls if p > 0]
    losses = [-p for p in recent_pnls if p < 0]
    n_total = len(recent_pnls)
    win_rate = len(wins) / n_total if n_total > 0 else None
    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(losses)) if losses else None
    profit_factor = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else None

    return {
        "equity_usd": float(equity_usd),
        "cash_usd": float(cash_usd),
        "positions": pos_block,
        "exposure": {
            "long_notional_usd": long_notional,
            "short_notional_usd": short_notional,
            "gross_notional_usd": gross_exposure_usd,
            "net_notional_usd": net_exposure_usd,
            "leverage_x": leverage,
        },
        "unrealized_pnl_usd": total_unreal_pnl,
        "running_max_dd_recent": max_dd_recent,
        "realized_return": {
            "4w": realized_4w,
            "12w": realized_12w,
        },
        "trade_history_summary": {
            "n_trades": n_total,
            "win_rate": win_rate,
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "profit_factor": profit_factor,
        },
    }


def signals_to_llm_context(signals: QuantSignals, idx: int) -> dict:
    """Render bar-`idx` quant signals as a JSON-friendly dict for LLM context.

    Includes:
      - GARCH sigma_t, 1-step forecast, sigma_z, crisis_flag
      - 4w / 12w / 26w return z-scores
      - SMA / WMA / EMA at horizons 4/8/12/26/52
      - RSI(7) / RSI(14) / RSI(21)
      - MACD(12,26,9) and MACD(5,13,5)
      - Volume current / z-score / recent-5 / 26w profile
    """
    def _f(arr: np.ndarray) -> float | None:
        if idx >= len(arr):
            return None
        v = arr[idx]
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return None
        return float(v)

    out = {
        "garch": {
            "sigma_t": _f(signals.garch_sigma_t),
            "sigma_t_plus_1_forecast": _f(signals.garch_sigma_t_plus_1),
            "sigma_z": _f(signals.garch_sigma_z),
            "crisis_flag": bool(signals.garch_crisis_flag[idx])
                if idx < len(signals.garch_crisis_flag) else False,
        },
        "z_scores": {
            "4w_return": _f(signals.z_4w),
            "12w_return": _f(signals.z_12w),
            "26w_return": _f(signals.z_26w),
        },
        "moving_averages": {
            "sma": {str(h): _f(signals.sma.get(h, np.array([]))) for h in MA_HORIZONS},
            "wma": {str(h): _f(signals.wma.get(h, np.array([]))) for h in MA_HORIZONS},
            "ema": {str(h): _f(signals.ema.get(h, np.array([]))) for h in MA_HORIZONS},
        },
        "rsi": {str(p): _f(signals.rsi.get(p, np.array([]))) for p in RSI_PERIODS},
        "macd": {
            f"{f}_{s}_{sig}": {
                "line": _f(signals.macd.get((f, s, sig), {}).get("line", np.array([]))),
                "signal": _f(signals.macd.get((f, s, sig), {}).get("signal", np.array([]))),
                "hist": _f(signals.macd.get((f, s, sig), {}).get("hist", np.array([]))),
            }
            for (f, s, sig) in MACD_PARAMS
        },
        "volume": {
            "current": _f(signals.volume_recent),
            "z_score": _f(signals.volume_z),
            "recent_5": [
                _f(signals.volume_recent[max(0, idx - k):idx - k + 1])
                if 0 <= idx - k < len(signals.volume_recent) else None
                for k in range(5, 0, -1)
            ],
            "profile_recent_26w": signals.volume_profile,
        },
    }
    return out


def portfolio_optimization_context(
    symbol_log_returns: dict[str, np.ndarray],
    cash_cap: float = 0.20,
) -> dict:
    """Compute portfolio-optimization weights for the LLM's context.

    Returns three weighting recommendations the LLM can consider:
      - equal_weight       : 1/N for each asset (no optimization)
      - inverse_vol        : risk-parity (1/sigma) — defensive proxy
      - markowitz_min_var  : analytic minimum-variance (long-only,
                              cash capped at `cash_cap`)

    The LLM is NOT required to follow these — they're informational
    portfolio context. Sizing decisions still go through Kelly +
    GARCH-vol-target in the deterministic pipeline.

    Inputs: dict of symbol → np.ndarray of recent log-returns.
    """
    syms = list(symbol_log_returns.keys())
    if not syms:
        return {"equal_weight": {}, "inverse_vol": {}, "markowitz_min_var": {}}

    # Align to common length
    min_len = min(len(r) for r in symbol_log_returns.values())
    if min_len < 8:
        # Insufficient history for Markowitz; fall back to equal-weight
        eq = {s: 1.0 / len(syms) for s in syms}
        return {"equal_weight": eq, "inverse_vol": eq, "markowitz_min_var": eq}

    cols = [np.asarray(symbol_log_returns[s])[-min_len:] for s in syms]
    ret_mat = np.vstack(cols).T  # (T, N)

    # Equal-weight
    n = len(syms)
    eq_w = {s: 1.0 / n for s in syms}

    # Inverse-vol
    sigmas = np.std(ret_mat, axis=0, ddof=1)
    sigmas = np.where(sigmas > 1e-9, sigmas, np.median(sigmas) if sigmas.size else 1.0)
    inv_w = 1.0 / sigmas
    inv_w = inv_w / inv_w.sum()
    iv_dict = {s: float(inv_w[i]) for i, s in enumerate(syms)}

    # Markowitz min-var, long-only, with cash leg if "USD_CASH" present
    cash_idx = [i for i, s in enumerate(syms) if s == "USD_CASH"]
    risk_idx = [i for i, s in enumerate(syms) if s != "USD_CASH"]
    mv_dict = {}
    if not risk_idx:
        mv_dict = eq_w.copy()
    else:
        risk_mat = ret_mat[:, risk_idx]
        cov = np.cov(risk_mat, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov) + 1e-8]])
        cov = np.atleast_2d(cov) + np.eye(len(risk_idx)) * 1e-8
        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(len(risk_idx))
            w_risk = inv_cov @ ones
            denom = float(ones @ inv_cov @ ones)
            if denom > 0:
                w_risk = w_risk / denom
                w_risk = np.maximum(w_risk, 0.0)
                if w_risk.sum() > 0:
                    w_risk = w_risk / w_risk.sum()
                else:
                    w_risk = np.full(len(risk_idx), 1.0 / len(risk_idx))
            else:
                w_risk = np.full(len(risk_idx), 1.0 / len(risk_idx))
        except np.linalg.LinAlgError:
            w_risk = np.full(len(risk_idx), 1.0 / len(risk_idx))
        risk_share = 1.0 - cash_cap if cash_idx else 1.0
        for j, idx in enumerate(risk_idx):
            mv_dict[syms[idx]] = float(w_risk[j] * risk_share)
        if cash_idx:
            for idx in cash_idx:
                mv_dict[syms[idx]] = float(cash_cap / len(cash_idx))

    return {
        "equal_weight": eq_w,
        "inverse_vol_risk_parity": iv_dict,
        "markowitz_min_var": mv_dict,
        "_meta": {
            "n_symbols": n,
            "lookback_bars": int(min_len),
            "cash_cap": float(cash_cap) if cash_idx else None,
        },
    }


def emergency_check(
    overlay: QuantOverlayConfig,
    drawdown_now: float,
    garch_sigma_z: float,
) -> EmergencyState:
    """Return EmergencyState.force_flat=True when any hard rule fires.

    Rules (any one triggers):
      - drawdown_now >= overlay.emergency_dd_halt
      - garch_sigma_z >= overlay.emergency_garch_z
    """
    if overlay.emergency_dd_halt is not None and drawdown_now >= overlay.emergency_dd_halt:
        return EmergencyState(force_flat=True, reason=f"dd_halt_{drawdown_now:.3f}")
    if (overlay.emergency_garch_z is not None
            and not math.isnan(garch_sigma_z)
            and garch_sigma_z >= overlay.emergency_garch_z):
        return EmergencyState(force_flat=True, reason=f"garch_z_{garch_sigma_z:.2f}")
    return EmergencyState(force_flat=False, reason="")
