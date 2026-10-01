#!/bin/bash
# Presence auto-emitter (v2.5+ 2026-05-16) — bridges TUI ephemeral state to
# /tmp/ultradebate_v2/heartbeat/{agent}.json so peers can see each other's
# working/thinking state without relying on manual presence_update calls.
#
# Polls each active_agents pane registered in session.json every ~3s, runs
# `tmux capture-pane -p` on its tmux_session, classifies the visible TUI state
# (working/thinking/idle), and writes the agent's heartbeat.json + appends a
# status_transition event when the classification changes.
#
# Scope: codex only by default (claude is stdio, no tmux pane). Extendable.
#
# Usage:
#   nohup bash presence_auto_emitter.sh > /tmp/presence_auto_emitter.log 2>&1 &
#
# Detection rules for Codex 0.130 TUI:
#   - "Working (.*s • esc to interrupt)" or "esc to interrupt" → status=working
#   - "tab to queue message" alone (no Working line) → status=idle
#   - "Thinking" (codex deep reasoning indicator) → status=thinking
#   - last 12 lines empty of all markers AND prompt visible "› " → status=idle
#
# B25 + v2.4 §2.5 alignment: this daemon makes B13 auto-triggers actually
# usable. Without it the heartbeat enum is decorative.

set -uo pipefail

V25_SESSIONS=/tmp/ultradebate_v2/sessions
HEARTBEAT=/tmp/ultradebate_v2/heartbeat
EVENTS=/tmp/ultradebate_v2/events.jsonl
LOCK=/tmp/presence_auto_emitter.lock
POLL_INTERVAL=${POLL_INTERVAL:-3}

mkdir -p "$HEARTBEAT"
touch "$EVENTS"

ts() { date '+%H:%M:%S'; }
iso() { date -u +"%Y-%m-%dT%H:%M:%S+09:00" -d "+9 hours"; }

# Single instance lock
if [ -f "$LOCK" ]; then
    pid=$(cat "$LOCK" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[$(ts)] PRESENCE_EMITTER_DUP pid=$pid — exiting"
        exit 1
    fi
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK" EXIT

echo "[$(ts)] PRESENCE_EMITTER_START interval=${POLL_INTERVAL}s heartbeat=$HEARTBEAT"

# Resolve {agent: tmux_session} map from session.json scan
resolve_panes() {
    [ -d "$V25_SESSIONS" ] || return
    for sj in "$V25_SESSIONS"/*/session.json; do
        [ -f "$sj" ] || continue
        python3 -c "
import json
try:
    d = json.load(open('$sj'))
    for agent, ag in (d.get('active_agents') or {}).items():
        if ag and not ag.get('unattached_at') and ag.get('tmux_session'):
            print(f\"{agent} {ag['tmux_session']}\")
except Exception:
    pass
" 2>/dev/null
    done | sort -u
}

# Classify pane content into one of: working | thinking | idle | offline
classify_pane() {
    local sess="$1"
    if ! tmux has-session -t "$sess" 2>/dev/null; then
        echo "offline"
        return
    fi
    local content
    # capture-pane -J + iconv -c: see codex_inbox_watcher.sh comment for the
    # 2026-05-22 text_safety rationale. Strips invalid UTF-8 byte sequences
    # (lone surrogates surface here when a multi-byte char gets cut at the
    # pane wrap column) so the grep matchers below are deterministic.
    content=$(tmux capture-pane -J -t "$sess" -p 2>/dev/null \
        | tail -25 \
        | iconv -f utf-8 -t utf-8 -c 2>/dev/null \
        || tmux capture-pane -J -t "$sess" -p 2>/dev/null \
           | tail -25 \
           | tr -d '\000-\010\013\014\016-\037')
    if [ -z "$content" ]; then
        echo "offline"
        return
    fi
    # Codex 0.130 markers
    if echo "$content" | grep -qE "Working \([0-9]+s|esc to interrupt"; then
        echo "working"
        return
    fi
    if echo "$content" | grep -qiE "thinking|reasoning"; then
        echo "thinking"
        return
    fi
    # Idle = visible prompt with no working markers
    if echo "$content" | grep -qE "tab to queue message|^› "; then
        echo "idle"
        return
    fi
    # Unknown — treat as idle to avoid false working
    echo "idle"
}

write_heartbeat() {
    local agent="$1"
    local status="$2"
    local sess="$3"
    local now
    now=$(iso)
    python3 - "$agent" "$status" "$sess" "$now" "$HEARTBEAT" "$EVENTS" <<'PYEOF'
import json, sys, pathlib, os
agent, status, sess, now, hb_dir, events_path = sys.argv[1:7]
hb = pathlib.Path(hb_dir) / f"{agent}.json"
prev = {}
if hb.exists():
    try:
        prev = json.loads(hb.read_text())
    except Exception:
        prev = {}
prev_status = prev.get("status")
data = {
    "agent": agent,
    "status": status,
    "current_task": {
        "tmux_session": sess,
        "since": now,
        "note": f"auto-emitter detected {status}",
    },
    "last_ping": now,
    "queue_depth": prev.get("queue_depth", 0),
    "next_available": None,
    "schema_version": "2.4.0",
    "auto_emitter": True,
}
hb.write_text(json.dumps(data, indent=2, ensure_ascii=False))
if prev_status != status:
    ev = {
        "ts": now,
        "agent": agent,
        "event": "status_transition",
        "from": prev_status,
        "to": status,
        "context": {"tmux_session": sess, "source": "auto_emitter"},
    }
    with open(events_path, "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"TRANSITION {agent}: {prev_status} → {status}")
PYEOF
}

declare -A LAST_STATUS

while true; do
    panes=$(resolve_panes)
    if [ -z "$panes" ]; then
        sleep "$POLL_INTERVAL"
        continue
    fi
    while IFS=' ' read -r agent sess; do
        [ -n "$agent" ] || continue
        [ -n "$sess" ] || continue
        new_status=$(classify_pane "$sess")
        old="${LAST_STATUS[$agent]:-}"
        if [ "$new_status" != "$old" ]; then
            echo "[$(ts)] PRESENCE_TRANSITION agent=$agent sess=$sess $old → $new_status"
        fi
        write_heartbeat "$agent" "$new_status" "$sess" 2>&1 | sed -e "s/^/[$(ts)] /"
        LAST_STATUS[$agent]="$new_status"
    done <<< "$panes"
    sleep "$POLL_INTERVAL"
done
