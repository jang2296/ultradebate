#!/usr/bin/env bash
# ultradebate v2.3 — Codex auto-pull daemon (B11 rule)
#
# Polls /tmp/ultradebate_v2/shared_plan.json every 60s. When current step is
# owned by codex_* AND status is pending/running/awaiting_b3, emits a single
# ping to Codex via inbox file (idempotent — same step_id won't ping twice).
#
# Usage (in Codex tmux session):
#   nohup bash codex_auto_pull_daemon.sh > /tmp/ultradebate_v2/codex_daemon.log 2>&1 &
#
# Stops by deleting /tmp/ultradebate_v2/codex_daemon.stop

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
INTERVAL="${INTERVAL:-60}"
STOP_FLAG="/tmp/ultradebate_v2/codex_daemon.stop"
INBOX_DIR="/tmp/ultradebate_v2/codex/inbox"
SEEN_FILE="/tmp/ultradebate_v2/codex_daemon.seen.txt"

mkdir -p "$INBOX_DIR"
touch "$SEEN_FILE"

echo "[codex_auto_pull_daemon] starting pid=$$ interval=${INTERVAL}s"

while [ ! -f "$STOP_FLAG" ]; do
  pending=$("$SCRIPT_DIR/update_shared_plan.sh" codex_pending 2>/dev/null || echo "NONE")
  if [ "$pending" != "NONE" ]; then
    step_id=$(echo "$pending" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('step_id',''))")
    status=$(echo "$pending" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))")
    key="${step_id}:${status}"
    if [ -n "$step_id" ] && ! grep -qxF "$key" "$SEEN_FILE"; then
      ts=$(date -u +%Y%m%dT%H%M%S)
      out="$INBOX_DIR/${ts}-daemon-codex-pending-${step_id}.json"
      cat > "$out" <<JSON
{
  "id": "${ts}-daemon-codex-pending-${step_id}",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sender": "shared_plan_daemon",
  "recipient": "codex",
  "type": "auto_pull_notification",
  "payload": {
    "tag": "[B11 CODEX AUTO-PULL — STEP ${step_id} OWNED BY CODEX]",
    "step_id": "${step_id}",
    "status": "${status}",
    "details": ${pending},
    "action_required": "Codex should read shared_plan.json + perform owner action without waiting for explicit Claude message."
  }
}
JSON
      echo "$key" >> "$SEEN_FILE"
      echo "[codex_auto_pull_daemon] $(date +%H:%M:%S) ping codex for step=$step_id status=$status"
    fi
  fi
  sleep "$INTERVAL"
done

echo "[codex_auto_pull_daemon] stopped via STOP_FLAG=$STOP_FLAG"
