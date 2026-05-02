#!/usr/bin/env python3
"""Dispatch one Claude Code (`claude -p`) call per (snapshot, voter) pair.

For each snapshot in a run's manifest, this script invokes ``claude -p`` once
per persona prompt in ``prompts/voters/`` and stores each persona's parsed
decision into a ``VoteCache`` keyed by (snapshot_hash, voter_id).

Calls run in parallel up to ``--max-concurrent`` to respect Claude Code rate
limits (default 4). Each invocation gets a fresh Claude Code session — no
state leaks across snapshots or across voters.

After this script finishes, ``scripts/run_ensemble_cached.py`` simulates the
strategy off the cached votes (deterministic, fast, free of API spend).

Usage:
    python scripts/dispatch_ensemble.py \\
        --run-id ensemble-real-llm-v1 --runs-dir runs \\
        --voters-dir prompts/voters --votes-db runs/.../votes.db \\
        --max-concurrent 4 --model opus
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from src.backtest.ensemble_decision import Vote
from src.backtest.vote_cache import VoteCache


def _decision_schema_block() -> str:
    return (
        "OUTPUT REQUIREMENTS:\n"
        "Return a single JSON object matching this exact schema "
        "(no markdown, no preamble, no explanation outside the JSON):\n\n"
        "{\n"
        '  "market_assessment": {\n'
        '    "trend": "bullish" | "bearish" | "neutral",\n'
        '    "confidence": <float 0-1>,\n'
        '    "regime": "calm" | "elevated" | "crisis"\n'
        "  },\n"
        '  "actions": [\n'
        "    {\n"
        '      "type": "open_long" | "open_short" | "close" | "hold",\n'
        '      "symbol": "<symbol or null for hold>",\n'
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
        "- Size is in base-currency units (e.g., 0.05 means 0.05 BTC).\n"
        "- stop_loss is mandatory for open_long / open_short.\n"
        "- For long, stop_loss must be below entry; for short, above.\n"
        "- Output JSON only, no markdown fences.\n"
    )


def _build_voter_prompt(persona_md: str, snapshot_json: str) -> str:
    return (
        persona_md.strip()
        + "\n\n"
        + _decision_schema_block()
        + "\n\nSNAPSHOT JSON:\n"
        + snapshot_json
        + "\n\nREMEMBER: output JSON only, no markdown, no commentary outside the JSON."
    )


def _hash_snapshot(snapshot_path: Path) -> str:
    return hashlib.sha256(snapshot_path.read_text().encode("utf-8")).hexdigest()


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` fences from a model response.

    Models often add fences even when told not to. Locate the first {...}
    JSON object span; if not found, return the original text.
    """
    s = text.strip()
    if s.startswith("```"):
        # drop opening fence (with optional language tag)
        first_nl = s.find("\n")
        if first_nl > 0:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()
    return s


def _decision_to_vote(voter_id: str, decision: dict) -> Vote:
    actions = decision.get("actions") or [{"type": "hold"}]
    first = actions[0] if actions else {"type": "hold"}
    ma = decision.get("market_assessment") or {}
    return Vote(
        voter_id=voter_id,
        action_type=str(first.get("type", "hold")),
        symbol=first.get("symbol"),
        size=float(first.get("size", 0.0) or 0.0),
        stop_loss=float(first["stop_loss"]) if first.get("stop_loss") is not None else None,
        take_profit=float(first["take_profit"]) if first.get("take_profit") is not None else None,
        confidence=float(ma.get("confidence", 0.5)),
        regime=str(ma.get("regime", "calm")),
    )


def _invoke_one(
    voter_id: str,
    persona_md: str,
    snapshot_path: Path,
    snapshot_hash: str,
    claude_bin: str,
    model: str,
    timeout_s: int,
) -> tuple[str, str, Vote | None, str | None]:
    """Invoke claude -p once. Returns (voter_id, snapshot_hash, vote, error)."""
    snapshot_json = snapshot_path.read_text()
    prompt = _build_voter_prompt(persona_md, snapshot_json)
    cmd = [claude_bin, "-p", prompt, "--output-format", "json", "--model", model]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except FileNotFoundError:
        return voter_id, snapshot_hash, None, f"'{claude_bin}' not found"
    except subprocess.TimeoutExpired:
        return voter_id, snapshot_hash, None, f"timeout after {timeout_s}s"
    if result.returncode != 0:
        return voter_id, snapshot_hash, None, f"exit {result.returncode}: {result.stderr[:200]}"
    try:
        envelope = json.loads(result.stdout)
        assistant_text = envelope.get("result") or envelope.get("text") or ""
        decision = json.loads(_strip_markdown_fences(assistant_text))
    except json.JSONDecodeError as e:
        return voter_id, snapshot_hash, None, f"json parse: {e}; text={assistant_text[:200]!r}"
    try:
        vote = _decision_to_vote(voter_id, decision)
    except (KeyError, ValueError, TypeError) as e:
        return voter_id, snapshot_hash, None, f"decision shape: {e}"
    return voter_id, snapshot_hash, vote, None


def _load_voter_personas(voters_dir: Path) -> list[tuple[str, str]]:
    """Return list of (voter_id, persona_markdown)."""
    out = []
    for path in sorted(voters_dir.glob("*.md")):
        out.append((path.stem, path.read_text()))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--voters-dir", default="prompts/voters")
    p.add_argument("--votes-db", required=True)
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--model", default="opus")
    p.add_argument("--claude-bin", default="claude")
    p.add_argument("--timeout-seconds", type=int, default=300)
    p.add_argument("--limit", type=int, default=0,
                   help="if > 0, dispatch at most this many tasks (for smoke tests)")
    args = p.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    voters = _load_voter_personas(Path(args.voters_dir))
    if not voters:
        print(f"ERROR: no persona prompts found in {args.voters_dir}", file=sys.stderr)
        return 2
    print(f"Loaded {len(voters)} voter personas: {[v[0] for v in voters]}")

    cache = VoteCache(args.votes_db)

    # Build task list: (voter_id, snapshot_path, snapshot_hash) for missing votes only.
    tasks = []
    for entry in manifest["observations"]:
        snap_path = run_dir / entry["path"]
        snap_hash = _hash_snapshot(snap_path)
        for voter_id, persona_md in voters:
            if not cache.has(snap_hash, voter_id):
                tasks.append((voter_id, persona_md, snap_path, snap_hash))

    total_obs = len(manifest["observations"])
    n_voters = len(voters)
    expected = total_obs * n_voters
    print(f"Snapshots: {total_obs}, voters: {n_voters}, expected votes: {expected}")
    print(f"Already cached: {expected - len(tasks)}")
    print(f"Pending: {len(tasks)}")

    if args.limit > 0:
        tasks = tasks[: args.limit]
        print(f"--limit set: dispatching {len(tasks)} tasks")

    if not tasks:
        print("Nothing to do.")
        return 0

    n_done = 0
    n_failed = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futs = {
            ex.submit(
                _invoke_one,
                voter_id, persona_md, snap_path, snap_hash,
                args.claude_bin, args.model, args.timeout_seconds,
            ): (voter_id, snap_path.name)
            for voter_id, persona_md, snap_path, snap_hash in tasks
        }
        for fut in cf.as_completed(futs):
            voter_id, snap_name = futs[fut]
            try:
                _vid, snap_hash, vote, err = fut.result()
            except Exception as e:
                print(f"  [{voter_id} {snap_name}] worker error: {e}", file=sys.stderr)
                n_failed += 1
                continue
            if err is not None:
                print(f"  [{voter_id} {snap_name}] FAIL: {err}", file=sys.stderr)
                n_failed += 1
                continue
            try:
                cache.put(snap_hash, vote)
            except ValueError as e:
                print(f"  [{voter_id} {snap_name}] cache.put failed: {e}", file=sys.stderr)
                n_failed += 1
                continue
            n_done += 1
            if n_done % 10 == 0:
                print(f"  progress: {n_done} ok, {n_failed} fail / {len(tasks)}")
    cache.close()

    print()
    print(f"Done: {n_done} stored, {n_failed} failed.")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
