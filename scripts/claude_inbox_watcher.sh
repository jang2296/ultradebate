#!/bin/bash
# Inbox watcher — emits one JSON stdout event per new inbox message.
# Combine with the Claude Code `Monitor` tool for instant conversation re-entry
# (each stdout line becomes a notification that wakes Claude).
#
# 2026-05-24 FIX (user mandate "코덱스에서 온 메시지는 반드시 클로드를 깨워서
# 묻히지 않게"):
#   - PATH BUG: previously watched /tmp/ultradebate_v2/$PEER/inbox/*.json
#     (a per-message directory) which priority_send NEVER writes to.
#     priority_send appends to /tmp/ultradebate_v2/inbox/$PEER.jsonl (a single
#     JSONL file) — so the old watcher saw nothing. Path corrected.
#   - B34 compliance: switched from a `for *.json + sleep 2` poll loop to
#     `tail -F` so appends are followed regardless of timing (no wc-l/line-count
#     snapshot race that could miss a message that landed before arming).
#   - `-n 0`: start at end-of-file → only NEW messages fire (no re-announce of
#     history on startup).
#
# Usage under the Monitor tool (persistent, recommended):
#   Monitor(command="bash .../claude_inbox_watcher.sh claude", persistent=true)
# Standalone (background) + separate Monitor on the log:
#   nohup bash claude_inbox_watcher.sh claude > /tmp/claude_inbox_watcher.log 2>&1 &
#   Monitor(command="tail -F /tmp/claude_inbox_watcher.log | grep --line-buffered V2_INBOX_NEW")

set -uo pipefail

PEER=${1:-claude}
case "$PEER" in
    claude|codex) ;;
    *) echo 'peer must be claude or codex' >&2; exit 2 ;;
esac
INBOX_JSONL=/tmp/ultradebate_v2/inbox/${PEER}.jsonl

mkdir -p "$(dirname "$INBOX_JSONL")"
touch "$INBOX_JSONL"

# tail -F follows appends + survives truncation/rotation (B34 mandate). Pipe is
# line-buffered via python -u so each new message fires an event immediately.
tail -F -n 0 "$INBOX_JSONL" 2>/dev/null | PEER="$PEER" python3 -u -c "
import sys, os, json
peer = os.environ.get('PEER', 'claude')
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    frm = d.get('from', '?')
    pri = d.get('priority', '?')
    typ = str(d.get('type', '?'))[:55]
    mid = str(d.get('msg_id', '?'))[:16]
    # Emit for every inbound message; downstream grep can filter by priority.
    event = {'event': 'V2_INBOX_NEW', 'peer': peer, 'from': frm,
             'priority': pri, 'type': typ, 'msg_id': mid}
    print(json.dumps(event, ensure_ascii=True), flush=True)
"
