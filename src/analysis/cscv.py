"""Combinatorially Symmetric Cross-Validation → PBO (AFML Ch. 11.4).

Given a matrix `M` of per-period returns for `S` strategy variants, CSCV:
1. Partitions periods into `S` equal-size sub-groups.
2. For every unordered split of sub-groups into an in-sample half and its
   complement, ranks strategies by IS Sharpe.
3. Checks the relative rank of the IS-winner when evaluated OOS.
4. Aggregates over all splits to compute PBO = P(OOS rank of IS-winner is in
   the bottom half).

For our LLM-led bot, we synthesize the `S` variants by re-scoring the same
realized trade log under different `min_p` confidence cutoffs.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _sharpe(col: np.ndarray) -> float:
    if len(col) < 2:
        return 0.0
    std = float(np.std(col, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(col)) / std


def cscv_pbo(
    returns_matrix: np.ndarray,
    n_groups: int | None = None,
    max_n_groups: int = 16,
) -> dict:
    """AFML Ch. 11.4 CSCV → PBO.

    Args:
        returns_matrix: shape `(n_periods, n_strategies)`, per-period returns.
        n_groups: Number of row sub-groups S. Must be even. Defaults to
                  min(max_n_groups, largest-even ≤ n_periods).
        max_n_groups: Hard cap — C(n, n/2) grows combinatorially; `16` yields
                      12,870 splits (fast); `20` yields 184,756 (slow).
                      AFML Ch. 11.4 recommends S=16.
    """
    arr = np.asarray(returns_matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return {"pbo": 0.5, "logit_lambdas": np.array([]), "num_trials": 0,
                "sr_is_oos": []}

    n_periods, n_strats = arr.shape
    if n_groups is None:
        largest_even = n_periods if n_periods % 2 == 0 else n_periods - 1
        n_groups = min(max_n_groups, largest_even)
    n_groups = min(n_groups, n_periods - (n_periods % 2))
    # Always cap for tractability even if caller requested more
    if n_groups > max_n_groups:
        n_groups = max_n_groups - (max_n_groups % 2)
    if n_groups < 2 or n_strats < 2:
        return {"pbo": 0.5, "logit_lambdas": np.array([]), "num_trials": n_strats,
                "sr_is_oos": []}

    # Even partition of rows into n_groups sub-arrays. Trim extra rows at the
    # tail rather than unevenly assigning.
    rows_per_group = n_periods // n_groups
    trimmed_n = rows_per_group * n_groups
    trimmed = arr[:trimmed_n, :]
    group_idx = np.arange(n_groups)

    half = n_groups // 2
    logit_lambdas: list[float] = []
    sr_pairs: list[tuple[float, float]] = []

    for is_groups in combinations(group_idx, half):
        is_mask = np.zeros(n_groups, dtype=bool)
        is_mask[list(is_groups)] = True
        is_rows = np.concatenate(
            [trimmed[g * rows_per_group : (g + 1) * rows_per_group]
             for g in range(n_groups) if is_mask[g]]
        )
        oos_rows = np.concatenate(
            [trimmed[g * rows_per_group : (g + 1) * rows_per_group]
             for g in range(n_groups) if not is_mask[g]]
        )
        sr_is = np.array([_sharpe(is_rows[:, s]) for s in range(n_strats)])
        sr_oos = np.array([_sharpe(oos_rows[:, s]) for s in range(n_strats)])

        is_winner = int(np.argmax(sr_is))
        sr_pairs.append((float(sr_is[is_winner]), float(sr_oos[is_winner])))

        # Rank of IS-winner in OOS Sharpes (0 = worst, n_strats-1 = best)
        ranks = sr_oos.argsort().argsort()
        rank_of_winner = int(ranks[is_winner])
        # Relative rank in (0, 1) interval, strictly bounded for logit stability.
        w_bar = (rank_of_winner + 1) / (n_strats + 1)
        logit = float(np.log(w_bar / (1.0 - w_bar)))
        logit_lambdas.append(logit)

    logit_arr = np.array(logit_lambdas, dtype=float)
    if logit_arr.size == 0:
        pbo = 0.5
    else:
        # PBO = P(logit <= 0) = P(OOS rank of IS winner is in the bottom half).
        pbo = float(np.mean(logit_arr <= 0.0))

    return {
        "pbo": pbo,
        "logit_lambdas": logit_arr,
        "num_trials": n_strats,
        "sr_is_oos": sr_pairs,
    }


def variants_from_realized_trades(
    trades: list[dict],
    min_p_grid: list[float],
    confidence_field: str = "llm_confidence",
    pnl_field: str = "pnl_pct",
    timestamp_field: str = "timestamp_close",
) -> np.ndarray:
    """Build a returns matrix by re-scoring trades under each `min_p` cutoff.

    Each row is a trade ordered by close time; each column is a variant. A
    variant "takes" a trade only if its `confidence >= min_p`; otherwise
    contributes 0 for that row.
    """
    if not trades:
        return np.zeros((0, len(min_p_grid)), dtype=float)

    # Sort trades by close timestamp (ascending) so rows are time-ordered.
    sorted_trades = sorted(trades, key=lambda t: t.get(timestamp_field, 0))
    n_rows = len(sorted_trades)
    n_cols = len(min_p_grid)
    out = np.zeros((n_rows, n_cols), dtype=float)
    for i, tr in enumerate(sorted_trades):
        pnl = float(tr.get(pnl_field, 0.0))
        conf = float(tr.get(confidence_field, 0.0))
        for j, cutoff in enumerate(min_p_grid):
            if conf >= cutoff:
                out[i, j] = pnl
    return out
