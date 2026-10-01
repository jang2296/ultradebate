#!/bin/bash
# Codex pane auto-detect daemon (v2.5+ 2026-05-16) — scans tmux periodically
# for a pane running `node codex` / `codex` binary, then writes/updates the
# active session.json's active_agents.codex entry so the watcher + presence
# emitter follow automatically.
#
# User flow this enables:
#   1. User runs: tmux new-session -s codex-work 'codex -c disable_paste_burst=true'
#      (or any other tmux session/pane name they like)
#   2. This daemon detects pid + tmux pane within ~5s
#   3. session.json updates → codex_inbox_watcher.sh + presence_auto_emitter.sh
#      both re-bind on their next iteration (~3s later)
#   4. Claude → Codex live bridge ready, no manual session_attach needed
#
# Scope (CWD): tied to a single working directory; resolves session_hash from
# that dir and updates exactly ONE session.json. To support multi-folder, run
# multiple instances with different CWD args.
#
# Usage:
#   nohup bash codex_pane_autodetect.sh /path/to/project \
#     > /tmp/codex_pane_autodetect.log 2>&1 &

set -uo pipefail

CWD=${1:-$PWD}
POLL_INTERVAL=${POLL_INTERVAL:-5}
V25_SESSIONS=/tmp/ultradebate_v2/sessions
LOCK=/tmp/codex_pane_autodetect.lock

ts() { date '+%H:%M:%S'; }
iso() { date -u +"%Y-%m-%dT%H:%M:%S+09:00" -d "+9 hours"; }

# Compute session_hash exactly the same way MCP server.py does (sha256(cwd)[:12])
SESSION_HASH=$(printf '%s' "$CWD" | sha256sum | cut -c1-12)
SESSION_DIR="$V25_SESSIONS/$SESSION_HASH"
SESSION_JSON="$SESSION_DIR/session.json"

mkdir -p "$SESSION_DIR"

if [ -f "$LOCK" ]; then
    pid=$(cat "$LOCK" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[$(ts)] AUTODETECT_DUP pid=$pid — exiting"
        exit 1
    fi
fi
echo $$ > "$LOCK"
trap "rm -f $LOCK" EXIT

echo "[$(ts)] AUTODETECT_START cwd=$CWD hash=$SESSION_HASH interval=${POLL_INTERVAL}s"

# Find a tmux pane currently running a codex CLI binary
find_codex_pane() {
    if ! tmux info >/dev/null 2>&1; then
        return  # no tmux server
    fi
    # List every pane: session:window.pane pane_pid pane_current_command
    tmux list-panes -a -F "#{session_name}|#{window_index}|#{pane_index}|#{pane_pid}|#{pane_current_command}|#{pane_current_path}" 2>/dev/null | while IFS='|' read -r sess win pane pid cmd cpath; do
        # Heuristic: pane runs `node` AND ancestor tree contains `codex` binary
        # OR pane command is directly `codex`
        if [ "$cmd" = "codex" ]; then
            echo "$sess|$pane|$pid|$cpath"
            return
        fi
        if [ "$cmd" = "node" ]; then
            # Check descendants for codex binary
            if pgrep -P "$pid" -a 2>/dev/null | grep -qE "codex/codex|codex-linux-x64.*codex"; then
                echo "$sess|$pane|$pid|$cpath"
                return
            fi
        fi
    done
}

# Look up codex CLI version if available (graceful fallback)
get_codex_version() {
    local v
    v=$(codex --version 2>/dev/null | head -1 || echo "")
    [ -z "$v" ] && v="codex-cli"
    echo "$v"
}

update_session_json() {
    local sess="$1"
    local pane="$2"
    local pid="$3"
    local now
    now=$(iso)
    local cli_version
    cli_version=$(get_codex_version)
    python3 - "$SESSION_JSON" "$CWD" "$SESSION_HASH" "$sess" "$pane" "$pid" "$now" "$cli_version" <<'PYEOF'
import json, sys, pathlib
sj_path, cwd, h, sess, pane, pid, now, ver = sys.argv[1:9]
p = pathlib.Path(sj_path)
data = {}
if p.exists():
    try:
        data = json.loads(p.read_text())
    except Exception:
        data = {}
if not data:
    data = {
        "session_hash": h,
        "cwd": cwd,
        "schema_version": "2.5.0",
        "created_at": now,
        "active_agents": {},
        "current_cycle_id": None,
        "binding_status": "idle",
    }
ag = data.setdefault("active_agents", {})
prev = ag.get("codex") or {}
new = {
    "joined_at": prev.get("joined_at", now),
    "pid": int(pid),
    "cli_version": ver,
    "transport": "tmux",
    "tmux_session": sess,
    "tmux_pane": f"{sess}:{pane}",
    "last_autodetect": now,
}
# unchanged? skip write
relevant_prev = {k: prev.get(k) for k in ("pid", "tmux_session", "tmux_pane") if k in prev}
relevant_new = {k: new.get(k) for k in ("pid", "tmux_session", "tmux_pane")}
if relevant_prev == relevant_new and not prev.get("unattached_at") is None:
    pass  # no-op when nothing changed
ag["codex"] = new
alive = [a for a, m in ag.items() if not m.get("unattached_at")]
data["binding_status"] = "single_active" if len(alive) == 1 else ("dual_active" if len(alive) >= 2 else "idle")
data["cwd"] = cwd
data["schema_version"] = "2.5.0"
p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
print(f"updated binding_status={data['binding_status']} codex={sess} pid={pid}")
PYEOF
}

mark_codex_unattached() {
    local now
    now=$(iso)
    python3 - "$SESSION_JSON" "$now" <<'PYEOF'
import json, sys, pathlib
sj_path, now = sys.argv[1:3]
p = pathlib.Path(sj_path)
if not p.exists():
    sys.exit(0)
try:
    data = json.loads(p.read_text())
except Exception:
    sys.exit(0)
ag = data.get("active_agents", {})
codex = ag.get("codex")
if codex and not codex.get("unattached_at"):
    codex["unattached_at"] = now
    codex["unattach_reason"] = "autodetect_pane_gone"
    alive = [a for a, m in ag.items() if not m.get("unattached_at")]
    data["binding_status"] = "single_active" if len(alive) == 1 else ("dual_active" if len(alive) >= 2 else "idle")
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"codex marked unattached")
PYEOF
}

LAST_FINGERPRINT=""

while true; do
    found=$(find_codex_pane | head -1)
    if [ -n "$found" ]; then
        IFS='|' read -r sess pane pid cpath <<< "$found"
        fingerprint="$sess|$pane|$pid"
        if [ "$fingerprint" != "$LAST_FINGERPRINT" ]; then
            echo "[$(ts)] AUTODETECT_FOUND sess=$sess pane=$pane pid=$pid cwd=$cpath"
            update_session_json "$sess" "$pane" "$pid" 2>&1 | sed -e "s/^/[$(ts)] /"
            LAST_FINGERPRINT="$fingerprint"
        fi
    else
        if [ -n "$LAST_FINGERPRINT" ]; then
            echo "[$(ts)] AUTODETECT_LOST prev=$LAST_FINGERPRINT"
            mark_codex_unattached 2>&1 | sed -e "s/^/[$(ts)] /"
            LAST_FINGERPRINT=""
        fi
    fi
    sleep "$POLL_INTERVAL"
done
