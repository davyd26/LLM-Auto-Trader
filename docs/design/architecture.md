# Architecture

A short note on how the bot is structured. For setup and configuration, see
[../../README.md](../../README.md).

## Two layers

The bot is deliberately split into two layers:

1. **LLM decision layer** — picks direction (`open_long` / `open_short` /
   `hold` / ...), proposes size, sets stop-loss/take-profit, and tells the
   scheduler when to check next.
2. **Deterministic risk layer** — accepts or rejects the LLM's proposal
   based on hard caps: leverage ≤ 5×, position size ≤ 25% equity, total
   exposure ≤ 60%, drawdown ≤ 5%, regime-adjusted size multipliers,
   correlation caps, fee + slippage budgets, etc.

This separation matters because the LLM is the *only* component capable of
reasoning over heterogeneous, mixed-signal data, but it's also the only
component that can hallucinate. The risk layer is dumb on purpose — it just
enforces invariants the LLM cannot violate.

## Data flow per cycle

1. **Pull** — REST + WebSocket via CCXT. OHLCV across multiple timeframes,
   order book, funding, OI. Cross-exchange data from a configurable list of
   read-only exchanges (default: Binance, Bybit, OKX) for triangulation.
2. **Compute** — indicators (`ta` library), market regime (3-state: calm /
   elevated / crisis), log-relative baselines, GARCH volatility forecast.
3. **Build context** — the `ContextBuilder` assembles ~3-4k tokens of
   prioritized snippets. When the budget is tight, the lowest-priority
   sections (sentiment, macro) are dropped first.
4. **LLM call** — single call per cycle through `BaseLLMProvider`. The
   system prompt is cached; the per-call user context is not.
5. **Parse + validate** — Pydantic schema; retry on invalid JSON with
   feedback to the LLM.
6. **Risk gate** — circuit breakers (drawdown, daily loss, rapid loss,
   volatility spike, spread blowout, funding spike, WS disconnect, LLM
   failure streak), regime gate, correlation cap, fee floor, slippage
   budget.
7. **Execute** — paper-mode simulation or real CCXT order placement.
8. **Record** — SQLite database, structured log, optional Telegram.

## Key abstractions

- **`ExchangeProfile`** ([`src/exchange/profile.py`](../../src/exchange/profile.py))
  — per-exchange tunables CCXT does not expose uniformly (fees,
  maintenance margin, quote currency, position mode).
- **`BaseLLMProvider`** ([`src/llm/providers/base.py`](../../src/llm/providers/base.py))
  — one method (`complete()`), one return type (`LLMCallResult`).
  Concrete providers: Anthropic, OpenAI, Google, OpenAI-compat (covers
  Ollama, vLLM, OpenRouter, Together, Groq, LM Studio).
- **`ContextBuilder`** — token-budgeted, priority-ordered snippet
  assembly. Drops low-priority sections when the budget is tight.
- **`CostTracker`** — per-million-token pricing pulled from the active
  provider; hard daily/monthly budget gates.
- **`MarketState`** — cross-symbol regime aggregator. Feeds the regime
  gate and the LLM's risk-context section.

## Why these choices

**Why CCXT over native exchange SDKs?** Avoids per-exchange code paths
for the 80% of operations that are uniform. Per-exchange differences
(fees, margin) are pushed into config.

**Why one LLM call per cycle, not many?** Cost, latency, simplicity. The
context window on Opus 4.7 (1M with the beta) is large enough to fit
everything we need in one shot. Splitting into specialist sub-prompts
multiplies cost without obvious gain in quality on this task.

**Why not let the LLM execute directly?** Because LLMs hallucinate, and
trading bots that send malformed orders lose money. The deterministic
risk layer is non-negotiable.

**Why a JSON schema, not function-calling / tool use?** Portability. The
JSON contract works identically across Claude, GPT, Gemini, and any
local model. Tool-use APIs differ between providers and add a parsing
indirection we don't need.
