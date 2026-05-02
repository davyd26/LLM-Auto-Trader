# Ideas / future work

## LLM ensemble voting for trade decisions

**Status:** dormant. Implement after the LLM strategy itself is benchmarked.

**Goal:** instead of one LLM call per decision, generate **N prompts** per
snapshot (different framings, role personas, time horizons, technical
emphases), dispatch all of them in parallel, and aggregate to one
decision via majority vote / median / weighted consensus.

**Why this is high ROI:**
- Variance reduction — random forests vs decision trees, model averaging
  in ML. Single LLM calls can be confidently wrong; ensembles average out
  idiosyncratic errors.
- The current strategy's p5-path Sharpe is around −0.22. Ensembles
  should pull p5 toward zero (the worst-regime paths still break even).
- For weekly cadence with $2K AUM, each decision matters disproportionately;
  ensemble robustness directly translates to risk-adjusted return.

### Plug-in points (the harness is already structured for this)

- `src/backtest/snapshot_builder.py` — produces the canonical snapshot
  (already content-addressed by SHA-256).
- `src/backtest/decision_cache.py` — keys decisions by snapshot hash.
- An ensemble layer sits between snapshot and cache: `(snapshot, N prompts)`
  → `N raw votes` → `aggregate()` → `1 decision` → store under snapshot
  hash.

### Aggregation functions to try

- Majority vote on action type (open_long / open_short / close / hold)
- Median on numeric fields (size_pct, leverage, stop_loss_pct)
- Weighted by self-reported confidence
- Veto rule: any "crisis" regime call from any voter overrides bullish
  action types
- Confidence-thresholded: only act if ≥M of N agree

### Fast-backtest constraints

The user explicitly required: keep it **very fast in backtest without
losing functionality, accuracy, or precision**.

1. **Decision cache by snapshot hash** — already exists. Aggregated
   decisions cached the same way as single decisions. Re-runs of CPCV
   across 28 paths get cached aggregated decisions for free.
2. **Cache by (snapshot_hash, prompt_template_id)** — store each voter's
   raw decision separately. New aggregation function = re-aggregate
   over cached votes; no re-dispatching needed.
3. **Parallel dispatch** — N subagents run concurrently via the Task
   tool (in-session) or via a multiprocessing pool of `claude -p` calls
   (headless / cron).
4. **Best-of-N early termination** — if first M dispatched calls agree
   above a confidence threshold, skip the remaining. Variance gain
   plateaus around N=5–9 for most ensembles.
5. **Persistent intermediate state** — partial dispatch progress survives
   interruption (already supported in current decision_cache).

### Implementation order when picked up

1. New module: `src/backtest/ensemble_decision.py` with
   `aggregate(votes: list[dict]) -> dict`.
2. New CLI: `scripts/dispatch_ensemble.py` that for each pending snapshot
   generates N prompts and dispatches them.
3. Extend `prompts/decide_system_prompt.md` with N personas:
   `decide_quant.md`, `decide_macro.md`, `decide_technical.md`,
   `decide_contrarian.md`, `decide_riskaverse.md`.
4. New SQLite table `decisions_votes(snapshot_hash, voter_id,
   decision_json, ts)`. Keep `decision_cache.put(snapshot_hash,
   aggregated)` for the canonical aggregated decision.
5. Sweep over aggregation functions on the same vote set (cheap — just
   re-aggregating cached votes).

### Why this is dormant

The deterministic baseline already hit 44% CAGR / 14% DD with vol-targeting
(the realistic ceiling for naive momentum). The next big jump is the LLM
strategy itself — ensemble voting is the upgrade *on top of* a working
single-LLM strategy, not a replacement for building the LLM strategy.
