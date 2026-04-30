"""Build a point-in-time snapshot of what a trader could see at simulated time t.

The snapshot is the unit a Claude Code subagent receives. The two non-negotiable
invariants:
    1. NO future data: every field is derived from cache rows with ts_ms <= t.
    2. NO absolute dates in the prompt-facing text: relative phrasing only,
       so the LLM cannot anchor decisions to "this was 2024-05" memorization.

Symbol names and price levels are retained — they are weaker memorization
channels than dates and stripping them would destroy the prompt's information
content. The defense is layered: snapshot stripping is one layer, CPCV
combinatorial paths are another, and the post-cutoff forward paper-trade is
the final clean check.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from src.backtest.data_cache import OHLCVCache


@dataclass(frozen=True)
class PositionRecord:
    symbol: str
    side: str  # "long" or "short"
    size: float  # in base currency units
    entry_price: float
    entry_ts_ms: int


@dataclass
class Snapshot:
    t_ms: int
    universe: list[str]
    bars: dict[str, dict[str, list[dict]]]  # symbol -> timeframe -> [bar_dicts]
    positions: list[PositionRecord]
    equity_usd: float
    unrealized_pnl_usd: float = 0.0
    funding_history: dict[str, list[dict]] = field(default_factory=dict)
    open_interest_history: dict[str, list[dict]] = field(default_factory=dict)

    def to_json(self) -> str:
        d = {
            "t_ms": self.t_ms,
            "universe": list(self.universe),
            "bars": self.bars,
            "positions": [asdict(p) for p in self.positions],
            "equity_usd": self.equity_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "funding_history": self.funding_history,
            "open_interest_history": self.open_interest_history,
        }
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @property
    def snapshot_id(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def to_prompt_text(self) -> str:
        """Return the LLM-facing string. Contains no absolute dates or years."""
        lines: list[str] = []
        lines.append(f"Account equity: ${self.equity_usd:,.2f}")
        if self.unrealized_pnl_usd:
            lines.append(f"Unrealized P&L: ${self.unrealized_pnl_usd:,.2f}")

        if self.positions:
            lines.append("Current positions:")
            for p in self.positions:
                lines.append(
                    f"  - {p.symbol}: {p.side} {p.size} @ entry {p.entry_price:.2f}"
                )
        else:
            lines.append("Current positions: none")

        for symbol in self.universe:
            sym_bars = self.bars.get(symbol, {})
            if not any(sym_bars.values()):
                lines.append(f"\n{symbol}: no recent data")
                continue
            lines.append(f"\n{symbol} (last bars, most-recent last):")
            for tf, bar_list in sym_bars.items():
                if not bar_list:
                    continue
                n = len(bar_list)
                lines.append(f"  {tf} timeframe ({n} bars):")
                # Index from the most-recent bar backwards to keep "N bars ago" semantics
                for i, b in enumerate(bar_list):
                    bars_ago = n - 1 - i
                    label = "current" if bars_ago == 0 else f"{bars_ago} bars ago"
                    lines.append(
                        f"    [{label}] O={b['open']:.2f} H={b['high']:.2f} "
                        f"L={b['low']:.2f} C={b['close']:.2f} V={b['volume']:.4f}"
                    )

        return "\n".join(lines)


class SnapshotBuilder:
    """Builds a Snapshot from a cache + position state at simulated time t."""

    def __init__(
        self,
        cache: OHLCVCache,
        universe: list[str],
        lookback_bars: int,
        timeframes: tuple[str, ...] = ("1d",),
    ):
        if lookback_bars < 0:
            raise ValueError("lookback_bars must be non-negative")
        self._cache = cache
        self._universe = list(universe)
        self._lookback = lookback_bars
        self._timeframes = timeframes

    def build(
        self,
        t_ms: int,
        equity_usd: float,
        positions: list[PositionRecord],
    ) -> Snapshot:
        bars: dict[str, dict[str, list[dict]]] = {}
        for symbol in self._universe:
            bars[symbol] = {}
            for tf in self._timeframes:
                bars[symbol][tf] = self._load_bars(symbol, tf, t_ms)

        unrealized = self._compute_unrealized_pnl(positions, bars, t_ms)

        return Snapshot(
            t_ms=t_ms,
            universe=list(self._universe),
            bars=bars,
            positions=list(positions),
            equity_usd=equity_usd,
            unrealized_pnl_usd=unrealized,
        )

    def _load_bars(self, symbol: str, timeframe: str, t_ms: int) -> list[dict]:
        if self._lookback == 0:
            return []
        df = self._cache.load_range(symbol, timeframe, start_ms=0, end_ms=t_ms)
        if df.empty:
            return []
        df = df.tail(self._lookback)
        return [
            {
                "ts_ms": int(row.ts_ms),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
            for row in df.itertuples()
        ]

    def _compute_unrealized_pnl(
        self,
        positions: list[PositionRecord],
        bars: dict[str, dict[str, list[dict]]],
        t_ms: int,
    ) -> float:
        total = 0.0
        for p in positions:
            sym_bars = bars.get(p.symbol, {})
            # Use the highest-resolution timeframe available
            latest_close: float | None = None
            for tf in self._timeframes:
                tf_bars = sym_bars.get(tf, [])
                if tf_bars:
                    latest_close = tf_bars[-1]["close"]
                    break
            if latest_close is None:
                continue
            if p.side == "long":
                total += (latest_close - p.entry_price) * p.size
            elif p.side == "short":
                total += (p.entry_price - latest_close) * p.size
        return total
