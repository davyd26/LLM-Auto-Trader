# LLM-Auto-Trader

An automated crypto trading bot that asks a language model what to do, and a
deterministic risk layer makes sure the model can't do anything reckless.

It runs on **any CCXT-supported exchange** (Binance, Bybit, OKX, MEXC, Gate,
KuCoin Futures, Bitget, and many more), and it can drive **any major LLM**
(Claude, GPT, Gemini, or any local model behind an OpenAI-compatible API).
Out of the box it runs in **paper-trading mode** — no real money, no
exchange keys required for the LLM-only loop.

> ⚠️  **Read this once before you do anything else.**
> This software is provided for research and educational use. It is **not
> financial advice**. Trading crypto futures is high-risk and you can lose
> more than your initial deposit. The bot has hard risk limits, but markets
> can still gap, exchanges can go down, language models can hallucinate, and
> bugs happen. **Paper-trade first, then live-trade with money you can afford
> to lose.** No warranty, express or implied. You are 100% responsible for
> what your bot does with your account.

---

## Table of contents

1. [What it actually does](#what-it-actually-does)
2. [Quick start](#quick-start)
3. [How a single decision happens](#how-a-single-decision-happens)
4. [Choosing an exchange](#choosing-an-exchange)
5. [Choosing an LLM](#choosing-an-llm)
6. [Token cost — read this carefully](#token-cost--read-this-carefully)
7. [Configuration reference](#configuration-reference)
8. [Trade-offs you should know about](#trade-offs-you-should-know-about)
9. [Project layout](#project-layout)
10. [Running tests](#running-tests)
11. [Troubleshooting](#troubleshooting)
12. [License](#license)

---

## What it actually does

Every few minutes (configurable: daily, hourly, or adaptive), the bot:

1. **Pulls fresh market data** — OHLCV bars across multiple timeframes,
   the order book, funding rates, open interest, on-chain metrics, news,
   sentiment, prediction-market odds, top-trader consensus — from the
   exchange you trade on plus a few extra exchanges for cross-exchange
   triangulation.
2. **Computes a context blob** for the LLM — technicals, market regime,
   open positions, recent trades, what the bot decided last time and how
   that turned out.
3. **Asks the LLM**: "given all this, what should I do right now?" The LLM
   replies with structured JSON: a market view, zero or more trade actions,
   and a recommended next-check time.
4. **Validates the response** — if the JSON is malformed or violates risk
   constraints, the bot retries with feedback or skips the cycle.
5. **Runs the trade through a deterministic risk layer** — leverage caps,
   position-size caps, drawdown circuit breakers, fee/slippage budgets,
   correlation caps, regime-adjusted sizing, etc. The LLM proposes; the
   risk layer disposes.
6. **Executes** the surviving actions on the exchange (or simulates them in
   paper mode).
7. **Reports** — to the console, to a SQLite database, and optionally to a
   Telegram bot.

Two-layer architecture: the **LLM decides**, the **rules enforce**. If the
LLM misbehaves, you don't blow up; the worst case is the LLM does nothing
useful and you sit in cash.

---

## Quick start

```bash
# 1. Clone.
git clone https://github.com/Erfaniaa/llm-auto-trader.git
cd llm-auto-trader

# 2. Install (with the Anthropic extra; see "Choosing an LLM" for others).
python -m venv .venv && source .venv/bin/activate
pip install -e ".[anthropic]"

# 3. Configure.
cp .env.example .env
$EDITOR .env  # paste at minimum your LLM_API_KEY

# 4. Paper-trade. No real money moves.
python -m scripts.paper_trade
```

That's it. With nothing else configured, you get:

| Thing            | Default                                                                 |
|------------------|-------------------------------------------------------------------------|
| Mode             | Paper trading                                                           |
| Exchange         | Binance perpetual swaps                                                 |
| Symbols          | BTC/USDT, ETH/USDT, TRX/USDT, PAXG/USDT (or XAUT/USDT if PAXG isn't listed) |
| Quote / margin   | USDT                                                                    |
| Rebalance        | Adaptive (LLM picks, 15–60 min)                                         |
| LLM              | Claude Opus 4.7 with extended thinking + 1M context                     |
| Max leverage     | 5×                                                                      |
| Max drawdown     | 5% (hard halt)                                                          |
| Daily LLM budget | $5.00 (USD)                                                             |

Every default is overridable. See [Configuration reference](#configuration-reference).

---

## How a single decision happens

```
┌──────────────┐   ┌────────────┐   ┌──────────────┐   ┌────────────┐
│ market data  │   │ derivatives│   │  on-chain /  │   │   news /   │
│  (CCXT, WS)  │   │ (funding,  │   │   sentiment  │   │ prediction │
│              │   │   OI, ...) │   │              │   │  markets   │
└──────┬───────┘   └─────┬──────┘   └──────┬───────┘   └─────┬──────┘
       └────────┬────────┴─────────┬───────┴─────────────────┘
                │                  │
       ┌────────▼────────┐  ┌──────▼─────────┐
       │  indicators &   │  │  market regime │
       │   statistics    │  │  (calm / vol / │
       │                 │  │   crisis)      │
       └────────┬────────┘  └────────┬───────┘
                └────────┬───────────┘
                         │
              ┌──────────▼──────────┐
              │   ContextBuilder    │
              │  (token-budgeted)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │       LLM call      │
              │  (cached system +   │
              │  per-call user ctx) │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  parse & validate   │
              │   (Pydantic JSON)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   risk layer:       │
              │  caps, breakers,    │
              │  regime gate, ...   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  execute (CCXT) or  │
              │   simulate (paper)  │
              └─────────────────────┘
```

---

## Choosing an exchange

Anything `ccxt` supports works. The bot has been smoke-tested with:

- **Binance** (`EXCHANGE_NAME=binance`) — broadest asset coverage, deep
  liquidity, default. Account-tier fees override the conservative defaults.
- **Bybit** (`EXCHANGE_NAME=bybit`) — also very deep; requires explicit
  one-way / hedge mode setup on your account.
- **OKX** (`EXCHANGE_NAME=okx`) — solid liquidity, broad token coverage.
- **MEXC** (`EXCHANGE_NAME=mexc`) — broadest exotic-token coverage,
  competitive 0% maker / 0.01% taker fees.
- **Gate**, **KuCoin Futures**, **Bitget** — all wired with sensible
  defaults in `src/exchange/profile.py`.

For any other CCXT exchange, it'll still run — you may just need to set
`EXCHANGE_MAKER_FEE`, `EXCHANGE_TAKER_FEE`, and `EXCHANGE_MAINT_MARGIN`
manually so the cost model and liquidation projections match reality.
Check your exchange's actual fee schedule and margin tier table.

**Live trading vs. paper.** The default is paper mode. To go live, set
`BOT_MODE=live` AND provide `EXCHANGE_API_KEY` / `EXCHANGE_API_SECRET`.
For paper mode you don't need exchange keys at all — only an LLM key.

---

## Choosing an LLM

The bot calls one LLM per decision cycle. You pick the provider with
`LLM_PROVIDER`, the model with `LLM_MODEL`, and the key with `LLM_API_KEY`.

| Provider          | `LLM_PROVIDER`   | Recommended models                               | When to use                                   |
|-------------------|------------------|--------------------------------------------------|-----------------------------------------------|
| Anthropic Claude  | `anthropic`      | `claude-opus-4-7`, `claude-sonnet-4-6`           | Best reasoning over noisy, mixed-signal data. Default. |
| OpenAI            | `openai`         | `gpt-4.1`, `o4-mini`                             | Strong, well-documented, reasoning models bill separately. |
| Google Gemini     | `google`         | `gemini-2.5-pro`, `gemini-2.5-flash`             | Cheap-and-fast Flash is great for ablation runs. |
| OpenAI-compatible | `openai-compat`  | anything served via `/v1/chat/completions`        | Local: Ollama, vLLM, LM Studio. Hosted: OpenRouter, Together, Groq. |

Install only the extras you need:

```bash
pip install -e ".[anthropic]"        # Claude
pip install -e ".[openai]"           # GPT / o-series
pip install -e ".[google]"           # Gemini
pip install -e ".[local]"            # OpenAI-compatible (Ollama, vLLM, ...)
```

Pluralism is good for two reasons: (a) you can A/B test model quality on
your own decision log, and (b) you avoid being captive to one provider's
rate limits or pricing changes.

---

## Token cost — read this carefully

This is the single most underrated factor in whether the bot is profitable.

**Every cycle costs you tokens.** A typical cycle looks like:

| Component                | Tokens    | Notes                                       |
|--------------------------|-----------|---------------------------------------------|
| System prompt            | ~1,500    | Cached after the first call (10× cheaper). |
| Per-call user context    | 3,000–4,000 | Market data, indicators, positions, etc. |
| Output (the JSON)        | 300–800   | More with extended thinking enabled.       |
| Reasoning (Opus 4.7)     | up to 8,000 | Hidden tokens; billed at output rate.    |

**Worked example, default settings.**
Claude Opus 4.7 + extended thinking, 4 calls/hour × 24h = 96 calls/day.

- ~1,500 system tokens × 96 = 144k input tokens, ~85% cached on average.
- ~3,500 user tokens × 96 = 336k input tokens (never cached).
- ~5,000 output+reasoning tokens × 96 = 480k output tokens.

At Anthropic's published rates (Opus 4.7: $15/M input, $1.50/M cached input,
$75/M output) that comes out roughly $40–$50/day. **Roughly $1,200–$1,500/month.**

**Now apply common sense.**
If your trading capital is $1,000 and your edge over fees is +0.5%/day
($5/day in expectation), and your LLM bill is $40/day, you have a
*guaranteed* −$35/day expected return. The model can be brilliant and you
will still bleed money. **The bot only makes economic sense once your
trading capital is large enough that alpha clearly exceeds LLM costs +
exchange fees + slippage + funding.**

**Knobs to turn it down.**

- `LLM_PROVIDER=google` `LLM_MODEL=gemini-2.5-flash` → roughly **30–50× cheaper**.
- `LLM_EXTENDED_THINKING=false` → cuts ~5,000 hidden output tokens per call.
- `LLM_THINKING_BUDGET=2000` → keeps thinking, but bounded.
- `REBALANCE_CADENCE=hourly` (or `daily`) → fewer cycles per day.
- `LLM_PROVIDER=openai-compat` + a local model → free at the marginal cost
  of electricity. Quality drops a lot — measure on paper first.
- `LLM_DAILY_BUDGET_USD=2.00` → the bot stops calling the LLM and reports
  to Telegram once you hit the cap. No surprise bills.

**Cache discipline.** The bot uses one ephemeral cache breakpoint on the
system prompt. The user context changes every call, so it's never cached.
Don't try to pre-cache market data — it's stale within minutes.

---

## Configuration reference

Everything is configurable via environment variables (paste into `.env`).
Defaults are conservative on purpose.

### Exchange

| Env var                        | Default       | What it does                                          |
|--------------------------------|---------------|-------------------------------------------------------|
| `EXCHANGE_NAME`                | `binance`     | CCXT exchange id                                       |
| `EXCHANGE_API_KEY`             |               | Required if `BOT_MODE=live`                            |
| `EXCHANGE_API_SECRET`          |               | Required if `BOT_MODE=live`                            |
| `EXCHANGE_MARKET_TYPE`         | `swap`        | `swap`, `future`, or `spot`                            |
| `EXCHANGE_QUOTE_CCY`           | `USDT`        | Margin / collateral asset                              |
| `EXCHANGE_MAKER_FEE`           | venue default | Override your VIP-tier maker fee                       |
| `EXCHANGE_TAKER_FEE`           | venue default | Override your VIP-tier taker fee                       |
| `EXCHANGE_MAINT_MARGIN`        | venue default | Maintenance margin rate (used for liquidation projections) |
| `DATA_EXCHANGES`               | `binance,bybit,okx` | Comma-separated extra exchanges for cross-exchange data only |
| `BOT_MODE`                     | `paper`       | `paper` (simulated) or `live` (real orders)            |

### LLM

| Env var                       | Default                | What it does                            |
|-------------------------------|------------------------|-----------------------------------------|
| `LLM_PROVIDER`                | `anthropic`            | `anthropic`, `openai`, `google`, `openai-compat` |
| `LLM_API_KEY`                 |                        | Provider key (or set `ANTHROPIC_API_KEY` etc.) |
| `LLM_MODEL`                   | `claude-opus-4-7`      | Model id                                 |
| `LLM_BASE_URL`                |                        | OpenAI-compat only (Ollama, vLLM, ...)   |
| `LLM_EXTENDED_THINKING`       | `true`                 | Anthropic Opus 4.7+ thinking on/off      |
| `LLM_THINKING_BUDGET`         | `8000`                 | Hidden-thinking token budget             |
| `LLM_LONG_CONTEXT`            | `true`                 | Request the 1M-context-window beta header |
| `LLM_MAX_OUTPUT_TOKENS`       | `2048`                 | Cap on visible response tokens           |
| `LLM_TEMPERATURE`             | `0.0`                  | Sampling temperature (forced to 1.0 with thinking on Opus 4.7) |
| `LLM_DAILY_BUDGET_USD`        | `5.00`                 | Daily hard stop                           |
| `LLM_MONTHLY_BUDGET_USD`      | `100.0`                | Monthly hard stop                         |
| `LLM_INPUT_PRICE_PER_M`       | provider default       | Manual price override ($/M tokens)       |
| `LLM_CACHED_INPUT_PRICE_PER_M`| provider default       | Manual price override                    |
| `LLM_OUTPUT_PRICE_PER_M`      | provider default       | Manual price override                    |

### Scheduling

| Env var               | Default     | What it does                                                                |
|-----------------------|-------------|------------------------------------------------------------------------------|
| `REBALANCE_CADENCE`   | `adaptive`  | `daily`, `hourly`, `adaptive`, or any 5-field cron expression               |

`adaptive` lets the LLM pick its own next-check interval (constrained to
[15, 60] minutes by default). `daily` and `hourly` are simpler and cheaper.

### Telegram (optional)

| Env var               | Default | What it does                                |
|-----------------------|---------|---------------------------------------------|
| `TELEGRAM_BOT_TOKEN`  |         | From `@BotFather`                            |
| `TELEGRAM_CHAT_ID`    |         | Your chat or group id                        |

When both are set, you'll get trade notifications, daily summaries, and
budget alerts. Leave empty to silence.

### Storage / logs

| Env var      | Default                  |
|--------------|--------------------------|
| `DB_PATH`    | `data/trading_bot.db`    |
| `LOG_LEVEL`  | `INFO`                   |
| `LOG_DIR`    | `logs`                   |

---

## Trade-offs you should know about

**Paper vs. live.** Paper mode simulates fills at the current mid-price
plus a slippage estimate. It is *optimistic* — real fills slip more in
fast markets, partial fills happen, and exchanges go down. Treat
paper-mode P&L as an upper bound, not a forecast.

**Fixed vs. adaptive cadence.**
- *Fixed* (`daily`, `hourly`, cron): predictable cost, predictable
  responsiveness, can miss fast-moving setups.
- *Adaptive*: the LLM picks `next_check_minutes`. More expensive when
  active, cheaper when sleeping. Can spike if the LLM systematically
  asks for short intervals.

**One model vs. ensemble.** The bot uses one LLM call per cycle. Ensembles
(majority vote across models) reduce variance but multiply cost and add
disagreement-handling complexity. Out of scope here.

**Leverage cap (default 5×).** Higher leverage shrinks the distance to
liquidation. The default is conservative; raising it past 10× makes the
bot dangerously sensitive to gap moves. The risk layer enforces whatever
you set, but be honest about your risk tolerance.

**SQLite vs. Postgres.** Single-user local SQLite is the default. It's
fast enough, simple, no service to manage. If you run multiple bots on the
same data, switch.

**Retention.** OHLCV-1m bars are kept for 2 days, 5m for 30 days, 15m+ for
a year. LLM prompts are stored compressed for 90 days. Tune in
`config/settings.py` under `StorageSettings`.

---

## Project layout

```
llm-auto-trader/
├── config/                # Settings + symbol definitions
│   ├── settings.py        # Layered config; reads env vars
│   └── symbols.py         # Trading universe + fallback resolver
├── src/
│   ├── analysis/          # Indicators, statistics, regime detection
│   ├── data/              # CCXT REST + WS, on-chain, sentiment, news, ...
│   ├── exchange/
│   │   └── profile.py     # ExchangeProfile (fees, margin, quote ccy)
│   ├── execution/         # Order placement, position tracking, paper sim
│   ├── llm/
│   │   ├── client.py      # Provider-agnostic LLM driver
│   │   ├── context_builder.py
│   │   ├── cost_tracker.py
│   │   ├── prompts.py     # System/user prompt templates
│   │   ├── response_parser.py
│   │   └── providers/     # Anthropic, OpenAI, Google, OpenAI-compat
│   ├── risk/              # Position sizing, circuit breakers, margin monitor
│   ├── storage/           # SQLite (WAL mode) + retention
│   ├── telegram/          # Notifications
│   └── bot.py             # Main orchestrator
├── scripts/
│   └── paper_trade.py     # Entry point: `python -m scripts.paper_trade`
├── tests/
└── docs/design/           # Design notes
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest -xvs
```

The test suite is deterministic — no live network calls, no real exchange,
no real LLM. CCXT is mocked at the boundary; LLM responses are pre-recorded.

---

## Backtest harness

The repo includes a CPCV (Combinatorial Purged Cross-Validation, López de
Prado AFML Ch. 12) backtest harness for honest, distribution-aware
evaluation of strategies before live deployment.

### Why CPCV instead of a single backtest

A single backtest gives a single Sharpe number — that number is almost
always optimistic relative to forward performance. CPCV partitions the
timeline into N groups, takes all C(N, K) test-group combinations, and
produces a *distribution* of Sharpe ratios across paths. The user reads
median/p5/p95, not just the optimistic point estimate, plus PBO
(Probability of Backtest Overfitting).

### Lookahead defense

The harness ships with two complementary defenses:

1. **`SnapshotBuilder`** — every snapshot the LLM sees contains only data a
   real human trader could observe at the simulated time `t`. Absolute
   dates and years are stripped from the prompt text; only relative
   phrasing remains ("4 weeks ago", "current week").
2. **CPCV with purging and embargo** — train observations near test-group
   boundaries are removed; observations immediately after each test group
   are embargoed. Together these prevent serial-correlation leakage.

### Decision-making backend: Claude Code subscription, not API

To avoid metered API spend on a multi-year backtest, the harness sources
LLM decisions from a Claude Code subscription:

- **Backtest mode**: Claude Code subagents read each snapshot in isolation
  and produce a JSON decision. Decisions are cached by snapshot SHA-256;
  re-running CPCV across all 28 paths replays from cache for free.
- **Live mode**: a cron job runs `scripts/live_cycle.sh` Mondays 14:00 UTC.
  The script invokes `claude -p` headlessly with the current snapshot,
  parses the JSON decision, validates it through `decision_validator`, and
  hands it to the executor. Each invocation is a fresh session — no
  conversational state leaks across weeks.

### Running a backtest

```bash
# 1. Prepare the snapshot manifest for a window.
python scripts/prepare_snapshots.py \
    --window 2024-01-01:2025-12-31 \
    --universe BTC/USDT,ETH/USDT,PAXG/USDT \
    --cache-db ~/.cache/llm-auto-trader/ohlcv.db \
    --run-id low-freq-aggressive-v1

# 2. Dispatch decisions. Either:
#    a) From a Claude Code session, invoke subagents per snapshot.
#    b) Headless:
python scripts/dispatch_decisions.py \
    --run-id low-freq-aggressive-v1 \
    --decisions-db runs/low-freq-aggressive-v1/decisions.db \
    --mode headless --model opus

# 3. Run CPCV simulation across the prepared decisions.
python scripts/run_cpcv.py \
    --run-id low-freq-aggressive-v1 \
    --cache-db ~/.cache/llm-auto-trader/ohlcv.db \
    --decisions-db runs/low-freq-aggressive-v1/decisions.db \
    --cpcv-n 8 --cpcv-k 2 --purge 2 --embargo 2

# Outputs: runs/<id>/summary.json, equity_curves.csv, trades.csv
```

To validate the harness without spending any decisions, use the
deterministic baseline strategy:

```bash
python scripts/run_cpcv.py ... --decisions-source baseline-stub
```

### Statistics produced

`summary.json` contains:

- `median_sharpe`, `p5_sharpe`, `p95_sharpe` — distribution across CPCV paths
- `sharpe_ci_lo_95`, `sharpe_ci_hi_95` — block-bootstrap CI (preserves
  serial correlation, unlike parametric PSR/DSR)
- `psr_on_median_sharpe` — Bailey & López de Prado's PSR
- `median_max_drawdown`, `p95_max_drawdown`
- `aggregate_hit_rate`, `aggregate_profit_factor`, `aggregate_expectancy`
- For multi-config sweeps: `compute_pbo(sharpes_matrix)` produces a
  Bailey/López de Prado PBO

### Comparing the LLM against deterministic baselines

`scripts/compare_llm_vs_baselines.py` scores the cached Opus decisions
against the deterministic rebalancers (EqualWeight, InverseVol,
MinVariance, Strawman) over the same window, so the LLM's edge is measured
net of what a mechanical strategy would have earned for free.

```bash
python scripts/compare_llm_vs_baselines.py \
    --llm-cache-db runs/llm-portfolio/llm_decisions_opus.db \
    --out-dir runs/llm-portfolio
```

Snapshots with no cached LLM decision fall back to equal-weight, and the
output reports cache coverage per subset (`llm_cache_hits` /
`llm_cache_misses`) so partial coverage can't be mistaken for a real
result. Outputs: `comparison_with_llm.csv` and `comparison_with_llm_meta.json`.

### Pass criteria for "honest super profitability"

For a strategy to be deemed safe to deploy at $2K AUM with weekly cadence:

- Median path Sharpe ≥ 1.0
- p5 path Sharpe ≥ 0 (5th-percentile path at least breaks even)
- Median max drawdown ≤ 12% (well within a 15% live ceiling)
- PSR > 0.95 on the median Sharpe
- PBO < 0.4
- Hit rate ≥ 45%, profit factor ≥ 1.5

These are necessary, not sufficient. A forward paper-trade window post
training-cutoff is the final clean check.

---

## Troubleshooting

**`LLM API key required for provider 'anthropic'`**
You haven't set `LLM_API_KEY` (or `ANTHROPIC_API_KEY`). Even paper mode
needs the LLM key — that's what makes the decisions.

**`set_leverage_unsupported` warning**
Some CCXT exchanges don't expose `set_leverage` (or your account doesn't
have margin enabled). The bot logs the warning and proceeds at the
account-default leverage. Set leverage on the exchange's web UI if needed.

**Symbols not appearing**
The bot silently drops symbols not listed on your exchange. Look for
`symbol_dropped` log lines. For example, MEXC lists `XAUT` but not `PAXG`;
the symbol resolver will pick `XAUT/USDT:USDT` automatically.

**LLM budget exceeded mid-day**
By design — the bot stops calling the LLM rather than running up a bill.
Either increase `LLM_DAILY_BUDGET_USD`, switch to a cheaper model, or wait
for the next UTC day.

**Paper-mode P&L looks too good**
It probably is. Paper mode is optimistic about slippage and never
experiences exchange downtime, fills against you on bad news, or
funding-rate surges. Compare paper P&L to real-mode P&L on tiny capital
(e.g., $50) to calibrate.

---

## License

MIT — see [LICENSE](LICENSE).
