#!/usr/bin/env python3
"""Dispatch LLM decisions for pending snapshots in a run.

Modes:
    print: print instructions for manually dispatching subagents from a
           Claude Code session.
    headless: invoke ``claude -p`` once per pending snapshot, parse stdout
              JSON into a Pydantic decision and store it under the snapshot's
              hash. This is the path used by the live cron job and by
              fully automated backtest runs.

Usage:
    python scripts/dispatch_decisions.py \\
        --run-id low-freq-aggressive-v1 --runs-dir runs \\
        --decisions-db runs/.../decisions.db \\
        --mode headless --model opus

Notes:
    - Each ``claude -p`` call gets a fresh session — no conversational state
      leaks across snapshots. This is structurally cleaner than even an API
      call (no shared client / no shared cache).
    - If a snapshot already has a cached decision, it is skipped.
    - The system prompt is read from prompts/decide_system_prompt.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.backtest.decision_cache import DecisionCache


def _decide_prompt() -> str:
    return (
        "You are a disciplined cryptocurrency portfolio manager.\n"
        "Below is a snapshot of market state and your current portfolio.\n"
        "Decide what to do this week.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "Return a single JSON object matching this schema (no markdown, "
        "no preamble, no explanation outside the JSON):\n\n"
        "{\n"
        '  "market_assessment": {\n'
        '    "trend": "bullish" | "bearish" | "neutral",\n'
        '    "confidence": <float 0-1>,\n'
        '    "regime": "calm" | "elevated" | "crisis"\n'
        "  },\n"
        '  "actions": [\n'
        "    {\n"
        '      "type": "open_long" | "open_short" | "close" | "hold",\n'
        '      "symbol": "<symbol or omitted for hold>",\n'
        '      "size": <float, base-currency units>,\n'
        '      "stop_loss": <optional float price>,\n'
        '      "take_profit": <optional float price>\n'
        "    }\n"
        "  ],\n"
        '  "next_check_minutes": <integer>,\n'
        '  "risk_notes": "<one-sentence reasoning>"\n'
        "}\n\n"
        "RULES:\n"
        "- Max one position per symbol at a time.\n"
        "- Size in base-currency units (e.g., 0.05 for 0.05 BTC).\n"
        "- Stop_loss is mandatory for open_long/open_short.\n"
        "- For long, stop_loss < entry; for short, stop_loss > entry.\n"
        "- Output JSON only.\n"
    )


def _print_mode(run_dir: Path, manifest: dict, decision_cache: DecisionCache) -> int:
    print(f"Run: {manifest['run_id']}")
    print(f"Universe: {', '.join(manifest['universe'])}")
    print(f"Observations: {manifest['n_observations']}")
    n_done = decision_cache.count()
    print(f"Decisions cached: {n_done}")
    print()
    print("To dispatch decisions from a Claude Code session, run subagents")
    print("via the Task tool. Each subagent must be given:")
    print("  - The system prompt above (from prompts/decide_system_prompt.md)")
    print("  - The snapshot JSON for that observation")
    print("  - Restriction: read only the snapshot file; no WebSearch/Bash for")
    print("    market data; output the strict JSON decision schema.")
    print()
    print(f"Snapshots are in: {run_dir / 'snapshots'}")
    return 0


def _headless_mode(
    run_dir: Path,
    manifest: dict,
    decision_cache: DecisionCache,
    model: str,
    claude_bin: str,
) -> int:
    import hashlib
    pending = []
    for entry in manifest["observations"]:
        snap_path = run_dir / entry["path"]
        snap_hash = hashlib.sha256(snap_path.read_text().encode("utf-8")).hexdigest()
        if not decision_cache.has(snap_hash):
            pending.append((entry["week_idx"], snap_path, snap_hash))

    print(f"Pending: {len(pending)} of {manifest['n_observations']} snapshots.")
    if not pending:
        return 0

    decide_prompt = _decide_prompt()
    for week_idx, snap_path, snap_hash in pending:
        prompt_body = (
            decide_prompt
            + "\n\nSNAPSHOT JSON:\n"
            + snap_path.read_text()
            + "\n\nREMEMBER: output JSON only, no markdown, no explanation outside the JSON."
        )
        cmd = [claude_bin, "-p", prompt_body, "--output-format", "json", "--model", model]
        print(f"[{week_idx:03d}] dispatching headless claude -p ({len(prompt_body)} chars)")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except FileNotFoundError:
            print(f"ERROR: '{claude_bin}' not found in PATH. Install Claude Code CLI.", file=sys.stderr)
            return 2
        except subprocess.TimeoutExpired:
            print(f"[{week_idx:03d}] TIMEOUT after 5 minutes; skipping", file=sys.stderr)
            continue
        if result.returncode != 0:
            print(f"[{week_idx:03d}] non-zero exit ({result.returncode})", file=sys.stderr)
            print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
            continue
        # claude -p with --output-format json wraps the assistant text in a JSON envelope.
        # The assistant text itself is expected to be JSON (the decision).
        try:
            envelope = json.loads(result.stdout)
            assistant_text = envelope.get("result", envelope.get("text", ""))
            decision = json.loads(assistant_text)
        except json.JSONDecodeError as e:
            print(f"[{week_idx:03d}] failed to parse JSON: {e}", file=sys.stderr)
            print(f"  stdout: {result.stdout[:500]}", file=sys.stderr)
            continue
        try:
            decision_cache.put(snap_hash, decision)
            print(f"[{week_idx:03d}] stored decision: {decision.get('actions', [])}")
        except Exception as e:
            print(f"[{week_idx:03d}] cache.put failed: {e}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--decisions-db", required=True)
    parser.add_argument("--mode", choices=("print", "headless"), default="print")
    parser.add_argument("--model", default="opus")
    parser.add_argument("--claude-bin", default="claude")
    args = parser.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    decision_cache = DecisionCache(args.decisions_db)

    if args.mode == "print":
        return _print_mode(run_dir, manifest, decision_cache)
    return _headless_mode(run_dir, manifest, decision_cache, args.model, args.claude_bin)


if __name__ == "__main__":
    sys.exit(main())
