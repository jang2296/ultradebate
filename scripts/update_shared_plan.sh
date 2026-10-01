#!/usr/bin/env bash
# ultradebate v2.3 — shared_plan.json updater (bash wrapper around Python helper)
#
# Usage:
#   update_shared_plan.sh heartbeat <claude|codex> [action_summary]
#   update_shared_plan.sh status <PASS|FAIL|RUNNING|BLOCKED> <updated_by> [evidence_json]
#   update_shared_plan.sh transition <new_step_id> <status> <owner> <updated_by>
#   update_shared_plan.sh summary
#   update_shared_plan.sh peer_stale <claude|codex>
#   update_shared_plan.sh codex_pending
#
# All commands return exit 0 on success, non-zero on failure.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MCP_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${MCP_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python3}"

cmd="${1:-summary}"

case "$cmd" in
  heartbeat)
    actor="${2:-claude}"
    action="${3:-heartbeat tick}"
    "$PYTHON" -m ultradebate.shared_plan heartbeat "$actor" "$action"
    ;;
  status)
    new_status="${2:-RUNNING}"
    updated_by="${3:-claude}"
    evidence_json="${4:-}"
    "$PYTHON" -c "
import json, sys
sys.path.insert(0, '${MCP_ROOT}/src')
from ultradebate.shared_plan import mark_step_status
ev = json.loads('${evidence_json}') if '${evidence_json}' else None
ok = mark_step_status('${new_status}', '${updated_by}', ev)
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
"
    ;;
  transition)
    new_step_id="${2:?need new step id}"
    status="${3:-pending}"
    owner="${4:-tbd}"
    updated_by="${5:-claude}"
    "$PYTHON" -c "
import sys
sys.path.insert(0, '${MCP_ROOT}/src')
from ultradebate.shared_plan import transition_step
ok = transition_step('${new_step_id}', '${status}', '${owner}', '${updated_by}')
print('OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)
"
    ;;
  summary)
    "$PYTHON" -m ultradebate.shared_plan summary
    ;;
  read)
    "$PYTHON" -m ultradebate.shared_plan read
    ;;
  peer_stale)
    actor="${2:-claude}"
    "$PYTHON" -m ultradebate.shared_plan peer_stale "$actor"
    ;;
  codex_pending)
    "$PYTHON" -m ultradebate.shared_plan codex_pending
    ;;
  *)
    echo "usage: $0 {heartbeat|status|transition|summary|read|peer_stale|codex_pending} ..." >&2
    exit 2
    ;;
esac
