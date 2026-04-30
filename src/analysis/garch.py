"""GARCH(1,1) volatility model (Phase 2 §3A.4).

Fit once at startup on a warm-up window; update incrementally each new bar.
Parameters are frozen after the initial fit — no re-fitting mid-session to
prevent data-snooping on realized PnL.

Model: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
with stationarity constraint alpha + beta < 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class GARCHState:
    sigma_t: float           # current conditional stddev (per-period)
    sigma_t_plus_1: float    # 1-step-ahead forecast
    sigma_z: float           # (sigma_t - trailing_mean_sigma) / trailing_std_sigma
    crisis_flag: bool        # sigma_z > 2.5

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _garch_neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e20

    n = len(returns)
    variance = np.zeros(n)
    variance[0] = float(np.var(returns))
    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
    variance = np.maximum(variance, 1e-12)
    ll = -0.5 * np.sum(np.log(variance) + returns ** 2 / variance)
    return -ll


def fit_garch_11(returns: np.ndarray) -> tuple[float, float, float]:
    """Fit GARCH(1,1) via maximum likelihood. Returns (omega, alpha, beta).

    Silently falls back to (var(returns), 0.05, 0.90) if optimization fails.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 50:
        return (float(np.var(arr)) if len(arr) else 1e-6, 0.05, 0.90)

    x0 = np.array([float(np.var(arr)) * 0.1, 0.05, 0.90])
    try:
        res = minimize(
            _garch_neg_log_likelihood,
            x0,
            args=(arr,),
            method="L-BFGS-B",
            bounds=[(1e-12, None), (0.0, 1.0), (0.0, 1.0)],
            options={"maxiter": 200},
        )
        if res.success:
            omega, alpha, beta = map(float, res.x)
            if alpha + beta >= 0.999:
                # clamp to stationarity
                scale = 0.99 / (alpha + beta)
                alpha *= scale
                beta *= scale
            return omega, alpha, beta
    except Exception:
        pass
    return (float(np.var(arr)) * 0.1, 0.05, 0.90)


def update_garch(
    state: GARCHState,
    r_t: float,
    omega: float,
    alpha: float,
    beta: float,
) -> GARCHState:
    """Incremental one-step GARCH update.

    Uses the new return `r_t` to produce the NEXT-period forecast.
    """
    prev_var = max(1e-12, state.sigma_t ** 2)
    new_var = omega + alpha * r_t ** 2 + beta * prev_var
    new_var = max(1e-12, new_var)
    new_sigma = new_var ** 0.5
    # One-step-ahead forecast from the new state
    fwd_var = omega + alpha * r_t ** 2 + beta * new_var
    fwd_sigma = max(1e-12, fwd_var) ** 0.5
    return GARCHState(
        sigma_t=new_sigma,
        sigma_t_plus_1=fwd_sigma,
        sigma_z=state.sigma_z,
        crisis_flag=state.crisis_flag,
    )


def garch_state_from_returns(
    returns: np.ndarray,
    crisis_z_threshold: float = 2.5,
) -> GARCHState:
    """Fit GARCH on the given returns window and produce the full state with
    z-score vs the trailing distribution of sigma values.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1e-6
        return GARCHState(
            sigma_t=sigma, sigma_t_plus_1=sigma, sigma_z=0.0, crisis_flag=False,
        )

    omega, alpha, beta = fit_garch_11(arr)

    n = len(arr)
    variance = np.zeros(n)
    variance[0] = float(np.var(arr))
    for t in range(1, n):
        variance[t] = omega + alpha * arr[t - 1] ** 2 + beta * variance[t - 1]
    variance = np.maximum(variance, 1e-12)
    sigma_series = variance ** 0.5

    sigma_t = float(sigma_series[-1])
    fwd_var = omega + alpha * arr[-1] ** 2 + beta * variance[-1]
    sigma_t_plus_1 = float(max(1e-12, fwd_var) ** 0.5)

    mean_sigma = float(np.mean(sigma_series[:-1])) if n > 1 else sigma_t
    std_sigma = float(np.std(sigma_series[:-1], ddof=1)) if n > 2 else 0.0
    sigma_z = (sigma_t - mean_sigma) / std_sigma if std_sigma > 0 else 0.0

    return GARCHState(
        sigma_t=sigma_t,
        sigma_t_plus_1=sigma_t_plus_1,
        sigma_z=sigma_z,
        crisis_flag=sigma_z >= crisis_z_threshold,
    )
