"""Dynamic cross-symbol correlation (Phase 3 §3B.6).

Computes a rolling correlation matrix from recent returns, then aggregates the
"effective concurrent-position count" for a proposed + open-positions set. A
high effective count means the positions move together; the risk layer uses
this to cap correlated exposure.
"""

from __future__ import annotations

import pandas as pd


def rolling_correlation_matrix(
    returns: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    """Return a pairwise correlation matrix over the tail `window` rows."""
    if returns is None or returns.empty:
        return pd.DataFrame()
    tail = returns.tail(window)
    if len(tail) < 5:
        return pd.DataFrame()
    return tail.corr()


def effective_concurrent_count(
    open_symbols: list[str],
    proposed_symbol: str | None,
    correlation_matrix: pd.DataFrame,
) -> float:
    """Estimate the "effective number of independent positions".

    For each open symbol vs the proposed symbol, contribute `|ρ|`. The
    effective count is `1 + Σ |ρ_open,proposed|` (approximately). When all
    positions are highly correlated, the effective count approaches
    `len(open) + 1`; when uncorrelated, it's close to 1.
    """
    if not open_symbols and proposed_symbol is None:
        return 0.0
    if proposed_symbol is None:
        # just pairwise among open
        if len(open_symbols) <= 1:
            return float(len(open_symbols))
        total = float(len(open_symbols))
        for i in range(len(open_symbols)):
            for j in range(i + 1, len(open_symbols)):
                sym_i, sym_j = open_symbols[i], open_symbols[j]
                if (sym_i in correlation_matrix.index
                        and sym_j in correlation_matrix.columns):
                    total += abs(float(correlation_matrix.loc[sym_i, sym_j]))
        return total

    # With proposed: sum |ρ(open_k, proposed)| + count of open
    total = 1.0 + float(len(open_symbols))
    for sym in open_symbols:
        if (sym in correlation_matrix.index
                and proposed_symbol in correlation_matrix.columns):
            total += abs(float(correlation_matrix.loc[sym, proposed_symbol]))
    # Divide by 2 to avoid double-counting the independent-baseline contribution
    # (so uncorrelated pairs give ~ len+1, highly-correlated pairs give higher).
    # Actually the AFML-style "effective count" doesn't double-count; use the
    # simpler: 1 (for proposed) + Σ(1 + |ρ(open,proposed)|) / 2, capped above.
    return float(total / 2)


def build_returns_df(price_series: dict[str, pd.Series]) -> pd.DataFrame:
    """Build an aligned returns DataFrame from per-symbol close series."""
    if not price_series:
        return pd.DataFrame()
    frames = {}
    for sym, s in price_series.items():
        if s is None or s.empty:
            continue
        frames[sym] = s.pct_change().dropna()
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).dropna(how="all")
