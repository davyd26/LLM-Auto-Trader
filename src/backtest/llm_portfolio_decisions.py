"""Decision sources for ``run_llm_portfolio``.

Each callable takes a snapshot dict (built by ``build_rich_snapshot``) and
returns a target-weight dict.

  - ``llm_decision_source``: dispatches ``claude -p`` per snapshot, parses
    the JSON response, caches results in SQLite by snapshot hash.
  - ``strawman_decision_source``: deterministic rule-based weights from
    momentum + volatility signals. Used for fast harness validation and
    as a comparison baseline against the real LLM.
  - ``equal_weight_decision_source``: just 1/N. Reference benchmark.
  - ``inverse_vol_decision_source``: weight ∝ 1/σ. Reference benchmark.
  - ``markowitz_min_var_decision_source``: long-only min-var with cash cap.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import subprocess
import time
from pathlib import Path

from src.backtest.llm_portfolio_strategy import USD_CASH, snapshot_hash

# ----------------------------------------------------------------------------
# Reference (deterministic) decision sources
# ----------------------------------------------------------------------------

def equal_weight_decision_source(snapshot: dict) -> dict[str, float]:
    universe = snapshot.get("universe", [])
    if not universe:
        return {}
    w = 1.0 / len(universe)
    return {s: w for s in universe}


def inverse_vol_decision_source(snapshot: dict) -> dict[str, float]:
    """Use the snapshot's advisory inv-vol scheme directly."""
    advisory = snapshot.get("reference_weight_schemes_advisory", {})
    iv = advisory.get("inverse_vol_risk_parity")
    if iv:
        return dict(iv)
    return equal_weight_decision_source(snapshot)


def markowitz_min_var_decision_source(snapshot: dict) -> dict[str, float]:
    advisory = snapshot.get("reference_weight_schemes_advisory", {})
    mv = advisory.get("markowitz_min_var")
    if mv:
        return dict(mv)
    return equal_weight_decision_source(snapshot)


def strawman_decision_source(snapshot: dict) -> dict[str, float]:
    """Deterministic rule-based portfolio rebalancer.

    Goal: a sensible "what an LLM might do" baseline, useful for fast
    sweep dev. Logic:

    1. Per non-cash symbol: build a momentum score from MA crosses
       (close > SMA(8) and close > SMA(26)) and the GARCH crisis flag.
    2. Symbols with positive score get inv-vol weights; symbols with
       negative score get zero.
    3. Cash leg fills the residual; never below 5%.
    4. Anchor symbols (BTC, GLD, USD_CASH) get a minimum 5% floor.
    """
    universe = snapshot.get("universe", [])
    sigs = snapshot.get("quant_signals", {})
    advisory = snapshot.get("reference_weight_schemes_advisory", {})

    iv_advisory = advisory.get("inverse_vol_risk_parity") or {}
    base_iv = {s: iv_advisory.get(s, 0.0) for s in universe}

    raw_scores: dict[str, float] = {}
    for sym in universe:
        if sym == USD_CASH:
            raw_scores[sym] = 0.0
            continue
        s = sigs.get(sym, {})
        ma = s.get("moving_averages", {}).get("sma", {})
        ema = s.get("moving_averages", {}).get("ema", {})
        rsi = s.get("rsi", {}).get("14")
        macd = s.get("macd", {}).get("12_26_9", {})
        macd_hist = macd.get("hist")
        garch = s.get("garch", {})
        crisis = bool(garch.get("crisis_flag"))

        sma_8 = ma.get("8")
        sma_26 = ma.get("26")
        ema_4 = ema.get("4")
        ema_12 = ema.get("12")

        # Build a 0-2 score: trend + momentum + non-crisis
        trend_up = (sma_8 is not None and sma_26 is not None and sma_8 > sma_26)
        ema_cross = (ema_4 is not None and ema_12 is not None and ema_4 > ema_12)
        macd_pos = (macd_hist is not None and macd_hist > 0)
        rsi_ok = (rsi is None) or (30 < rsi < 80)

        score = 0.0
        if trend_up:
            score += 1.0
        if ema_cross:
            score += 0.5
        if macd_pos:
            score += 0.5
        if not rsi_ok:
            score *= 0.5
        if crisis:
            score = 0.0
        raw_scores[sym] = score

    # Combine with inv-vol base; symbols with score=0 get zero allocation
    weights: dict[str, float] = {}
    for sym in universe:
        if sym == USD_CASH:
            continue
        sc = raw_scores.get(sym, 0.0)
        weights[sym] = base_iv.get(sym, 0.0) * (sc / 2.0)

    # Anchor floor: BTC + GLD always at least 5% if in universe
    for anchor_sym in ("BTC/USDT:USDT", "GLD"):
        if anchor_sym in universe and weights.get(anchor_sym, 0.0) < 0.05:
            weights[anchor_sym] = 0.05

    # Total risk-asset weight; cash takes residual but at least 5%
    total_risk = sum(weights.values())
    cash_w = max(0.05, 1.0 - total_risk)
    if USD_CASH in universe:
        weights[USD_CASH] = cash_w

    # Renormalize
    total = sum(weights.values())
    if total <= 0:
        n = len(universe)
        return {s: 1.0 / n for s in universe}
    return {s: v / total for s, v in weights.items()}


# ----------------------------------------------------------------------------
# Real LLM decision source: claude -p with caching
# ----------------------------------------------------------------------------

_LLM_PROMPT_TEMPLATE = """\
You are a disciplined cross-asset portfolio manager. Your job is to set
target weights for the next week given the data below.

Universe: {universe}
Initial equity (USD): ${initial_eq:,.0f}
Current equity (USD): ${cur_eq:,.0f}
Current weights: {cur_weights}

Hard constraints:
  - Long-only (no shorts).
  - Weights must sum to 1.0.
  - The anchor symbols MUST each have weight > 0: BTC/USDT:USDT, GLD, USD_CASH.
  - All decisions are based on USD value. CAGR / DD / Sharpe in your reasoning
    are USD-denominated.

Inputs (advisory — use as one signal among many; you are NOT bound to follow):
  - reference_weight_schemes_advisory: equal-weight, inverse-vol risk parity,
    Markowitz min-variance long-only.
  - per-symbol quant signals: GARCH(1,1) sigma + 1-step forecast + sigma_z
    + crisis flag, return z-scores at 4w/12w/26w, SMA/WMA/EMA at horizons
    4/8/12/26/52, RSI(7/14/21), MACD(12,26,9) and (5,13,5), recent volume +
    z-score + 26w volume profile.
  - portfolio state: cash, leverage, gross/net notional, position MTM,
    running max-DD, recent realized 4w/12w return, win rate, profit factor.
  - recent_bars: last 12 weekly OHLCV bars per symbol.

OUTPUT FORMAT — return ONE JSON object, no markdown fences, no commentary
outside the JSON:

{{
  "weights": {{
    "<symbol>": <float in [0, 1]>,
    ...
  }},
  "reasoning": "<one short sentence about the dominant signals you weighted>"
}}

The weights object MUST cover every symbol in the universe. If you don't
want exposure to a symbol, set its weight to 0.

SNAPSHOT DATA:
{snapshot_json}
"""


def _strip_md_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # Sometimes the response has prose before the JSON; grab the first {…}
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        return m.group(0)
    return s


class LLMPortfolioCache:
    """SQLite cache of (snapshot_hash → weights_json)."""

    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio_decisions (
                snapshot_hash TEXT PRIMARY KEY,
                weights_json  TEXT NOT NULL,
                created_at    REAL NOT NULL
            );
        """)

    def get(self, h: str) -> dict[str, float] | None:
        row = self._conn.execute(
            "SELECT weights_json FROM portfolio_decisions WHERE snapshot_hash = ?",
            (h,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def put(self, h: str, weights: dict[str, float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO portfolio_decisions "
            "(snapshot_hash, weights_json, created_at) VALUES (?, ?, ?)",
            (h, json.dumps(weights, sort_keys=True), time.time()),
        )

    def total(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM portfolio_decisions"
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()


def _invoke_claude(prompt: str, model: str, timeout_s: int) -> str | None:
    """Run ``claude -p`` and return the assistant text or None on error."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--model", model],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        envelope = json.loads(result.stdout)
        return envelope.get("result") or envelope.get("text") or ""
    except json.JSONDecodeError:
        return None


def make_llm_decision_source(
    cache: LLMPortfolioCache | None = None,
    model: str = "haiku",
    timeout_s: int = 120,
    fallback_source: callable | None = None,
) -> callable:
    """Build an LLM-backed decision source that caches by snapshot hash.

    Args:
        cache: optional LLMPortfolioCache; if provided, hits/misses are
            tracked and successful responses are stored.
        model: Claude Code model name passed to ``claude -p``.
        timeout_s: subprocess timeout per call.
        fallback_source: if the LLM call fails / can't be parsed, fall back
            to this decision source (default: equal-weight). The fallback
            keeps the simulation moving even on transient failures.

    Returns:
        Callable suitable for ``run_llm_portfolio(decision_source=...)``.
    """
    fallback = fallback_source or equal_weight_decision_source

    def _src(snapshot: dict) -> dict[str, float]:
        sh = snapshot_hash(snapshot)
        if cache is not None:
            cached = cache.get(sh)
            if cached is not None:
                return cached
        # Build prompt
        snap_json = json.dumps(snapshot, separators=(",", ":"), default=float)
        prompt = _LLM_PROMPT_TEMPLATE.format(
            universe=snapshot.get("universe", []),
            initial_eq=float(snapshot.get("initial_equity_usd", 0.0)),
            cur_eq=float(snapshot.get("portfolio_state", {}).get("equity_usd", 0.0)),
            cur_weights=json.dumps(snapshot.get("current_weights", {})),
            snapshot_json=snap_json,
        )
        text = _invoke_claude(prompt, model=model, timeout_s=timeout_s)
        if text is None:
            return fallback(snapshot)
        # Parse JSON from the response
        cleaned = _strip_md_fences(text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return fallback(snapshot)
        weights = parsed.get("weights")
        if not isinstance(weights, dict):
            return fallback(snapshot)
        # Coerce to float; non-numeric → 0
        out: dict[str, float] = {}
        for s, v in weights.items():
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out[s] = fv
            except (TypeError, ValueError):
                continue
        if not out:
            return fallback(snapshot)
        if cache is not None:
            cache.put(sh, out)
        return out

    return _src
