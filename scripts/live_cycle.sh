#!/usr/bin/env bash
# Weekly live decision cycle (intended for cron).
#
# 1. Build the current live snapshot (Python; no LLM).
# 2. Invoke Claude Code headlessly to produce a decision (subscription, no API).
# 3. Hand the decision to the executor (Python) which validates against the
#    risk gate and places the order on the live exchange.
#
# Cron entry (Mondays 14:00 UTC):
#   0 14 * * 1 cd /opt/llm-auto-trader && bash scripts/live_cycle.sh \
#       >> logs/live_cycle.log 2>&1

set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
LIVE_DIR="${REPO_ROOT}/live"
mkdir -p "${LIVE_DIR}"

SNAPSHOT_FILE="${LIVE_DIR}/current_snapshot.json"
DECISION_FILE="${LIVE_DIR}/current_decision.json"
LOG_FILE="${REPO_ROOT}/logs/live_cycle.log"
mkdir -p "${REPO_ROOT}/logs"

echo "[$(date -u +%FT%TZ)] live_cycle: starting"

# 1. Build snapshot
python -m src.live.prepare_snapshot \
    --output "${SNAPSHOT_FILE}" \
    || {
        echo "[$(date -u +%FT%TZ)] live_cycle: snapshot prep FAILED" >&2
        exit 1
    }

# 2. Decide via headless claude -p
SYSTEM_PROMPT_FILE="${REPO_ROOT}/prompts/decide_system_prompt.md"
if [[ ! -f "${SYSTEM_PROMPT_FILE}" ]]; then
    echo "[$(date -u +%FT%TZ)] live_cycle: missing ${SYSTEM_PROMPT_FILE}" >&2
    exit 1
fi

PROMPT_BODY="$(cat "${SYSTEM_PROMPT_FILE}")
SNAPSHOT JSON:
$(cat "${SNAPSHOT_FILE}")
REMEMBER: output JSON only."

claude -p "${PROMPT_BODY}" \
    --output-format json \
    --model "${CLAUDE_MODEL:-opus}" \
    > "${DECISION_FILE}.envelope" || {
        echo "[$(date -u +%FT%TZ)] live_cycle: claude -p FAILED" >&2
        exit 1
    }

# Extract the assistant's response text (the decision JSON)
python -c "
import json, sys
envelope = json.load(open('${DECISION_FILE}.envelope'))
text = envelope.get('result', envelope.get('text', ''))
decision = json.loads(text)
json.dump(decision, open('${DECISION_FILE}', 'w'), indent=2)
" || {
    echo "[$(date -u +%FT%TZ)] live_cycle: decision parsing FAILED" >&2
    exit 1
}

# 3. Execute via the live executor
python -m src.live.execute_decision \
    --snapshot "${SNAPSHOT_FILE}" \
    --decision "${DECISION_FILE}" \
    || {
        echo "[$(date -u +%FT%TZ)] live_cycle: executor FAILED" >&2
        exit 1
    }

echo "[$(date -u +%FT%TZ)] live_cycle: complete"
