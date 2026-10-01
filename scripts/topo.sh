#!/usr/bin/env bash
# =============================================================================
# topo.sh — Process / session topology visualizer for ultradebate stack
# =============================================================================
# Purpose : Single-glance answer to "which claude / codex / tmux pane / watcher
#           is actually live right now, and which are ghosts?" Replaces the
#           ad-hoc ps|grep recipe in design doc §4.4.
#
# Usage   : topo.sh [--json] [--watch SECONDS] [-h|--help]
#           --json          : machine-readable JSON output
#           --watch N       : prints suggested `watch -n N topo.sh` command
#
# Sections: 1) claude instances (ghost-labeled)
#           2) codex instances  (wrapper + binary structure noted)
#           3) tmux server / sessions / panes
#           4) /tmp/ultradebate_v2 inbox + sessions + jobs health
#           5) watcher daemon lockfile status (codex + claude)
# =============================================================================

set -euo pipefail

JSON=0
WATCH=""

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage ;;
        --json) JSON=1; shift ;;
        --watch) WATCH="${2:-2}"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -n "$WATCH" ]; then
    echo "Hint: run -> watch -n ${WATCH} bash $0"
    exit 0
fi

V2_ROOT=/tmp/ultradebate_v2

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
classify_ghost() {
    # args: ppid tty -> echoes "GHOST(reason)" or "LIVE"
    local ppid="$1" tty="$2" reasons=""
    [ "$tty" = "?" ] && reasons="no-tty"
    [ "$ppid" = "1" ] && reasons="${reasons:+$reasons,}orphan"
    if [ -n "$reasons" ]; then echo "GHOST($reasons)"; else echo "LIVE"; fi
}

count_files() { ls -1 "$1" 2>/dev/null | wc -l | tr -d ' '; }

lock_status() {
    # args: lockfile -> echoes "ALIVE pid=N" / "STALE pid=N" / "ABSENT"
    local lf="$1"
    if [ ! -f "$lf" ]; then echo "ABSENT"; return; fi
    local pid
    pid=$(cat "$lf" 2>/dev/null | tr -d '[:space:]')
    if [ -z "$pid" ]; then echo "EMPTY"; return; fi
    if [ -d "/proc/$pid" ]; then echo "ALIVE pid=$pid"; else echo "STALE pid=$pid"; fi
}

# -----------------------------------------------------------------------------
# Plain output
# -----------------------------------------------------------------------------
if [ "$JSON" = "0" ]; then
    echo "===== claude instances ====="
    ps -eo pid=,ppid=,tty=,etime=,cmd= 2>/dev/null \
        | grep -E '[c]laude --dangerously' \
        | while read -r pid ppid tty etime cmd; do
            tag=$(classify_ghost "$ppid" "$tty")
            printf "  pid=%-7s ppid=%-7s tty=%-8s up=%-10s %s\n" "$pid" "$ppid" "$tty" "$etime" "$tag"
          done || true

    echo ""
    echo "===== codex instances ====="
    echo "  (note: 'codex' npm wrapper + native binary is NORMAL — not a ghost)"
    ps -eo pid=,ppid=,tty=,etime=,cmd= 2>/dev/null \
        | grep -E '([[:space:]]|/)codex([[:space:]]|$)' \
        | grep -v 'grep' \
        | while read -r pid ppid tty etime cmd; do
            role="binary"
            echo "$cmd" | grep -q 'node.*codex' && role="npm-wrapper"
            tag=$(classify_ghost "$ppid" "$tty")
            printf "  pid=%-7s ppid=%-7s tty=%-8s up=%-10s role=%-11s %s\n" \
                "$pid" "$ppid" "$tty" "$etime" "$role" "$tag"
          done || true

    echo ""
    echo "===== tmux ====="
    if command -v tmux >/dev/null 2>&1; then
        if tmux list-sessions >/dev/null 2>&1; then
            echo "  sessions:"
            tmux list-sessions 2>/dev/null | sed 's/^/    /'
            echo "  panes:"
            tmux list-panes -a -F '    #{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command}' 2>/dev/null || echo "    (none)"
        else
            echo "  (tmux installed but no server running)"
        fi
    else
        echo "  (tmux not installed)"
    fi

    echo ""
    echo "===== /tmp/ultradebate_v2 health ====="
    if [ -d "$V2_ROOT" ]; then
        for peer in codex claude gemini; do
            ib="$V2_ROOT/$peer/inbox"
            if [ -d "$ib" ]; then
                printf "  %-7s inbox=%-3s msgs   path=%s\n" "$peer" "$(count_files "$ib")" "$ib"
            fi
        done
        printf "  sessions=%-3s dirs   path=%s/sessions\n" "$(ls -1d $V2_ROOT/sessions/*/ 2>/dev/null | wc -l | tr -d ' ')" "$V2_ROOT"
        if [ -d "$V2_ROOT/jobs" ]; then
            printf "  jobs=%-3s files   path=%s/jobs\n" "$(count_files "$V2_ROOT/jobs")" "$V2_ROOT"
        fi
    else
        echo "  (path $V2_ROOT does not exist)"
    fi

    echo ""
    echo "===== watcher daemons ====="
    printf "  codex_inbox_watcher  : %s\n" "$(lock_status /tmp/codex_inbox_watcher.lock)"
    printf "  claude_inbox_watcher : %s\n" "$(lock_status /tmp/claude_inbox_watcher.lock)"
    exit 0
fi

# -----------------------------------------------------------------------------
# JSON output (best-effort, single-shot)
# -----------------------------------------------------------------------------
python3 - "$V2_ROOT" <<'PY'
import json, os, subprocess, sys, re
v2 = sys.argv[1]

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""

def parse_ps(filter_re):
    out = run("ps -eo pid=,ppid=,tty=,etime=,cmd=")
    rows = []
    for line in out.splitlines():
        if not re.search(filter_re, line): continue
        parts = line.split(None, 4)
        if len(parts) < 5: continue
        pid, ppid, tty, etime, cmd = parts
        reasons = []
        if tty == "?": reasons.append("no-tty")
        if ppid == "1": reasons.append("orphan")
        rows.append({"pid": pid, "ppid": ppid, "tty": tty, "etime": etime,
                     "cmd": cmd[:120], "ghost": bool(reasons),
                     "ghost_reasons": reasons})
    return rows

def lock(p):
    if not os.path.exists(p): return {"state": "absent"}
    try: pid = open(p).read().strip()
    except Exception: return {"state": "unreadable"}
    if not pid: return {"state": "empty"}
    alive = os.path.isdir(f"/proc/{pid}")
    return {"state": "alive" if alive else "stale", "pid": pid}

tmux_sessions = run("tmux list-sessions 2>/dev/null").splitlines()
tmux_panes = run("tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index}|#{pane_pid}|#{pane_current_command}' 2>/dev/null").splitlines()

def count(p): return len(os.listdir(p)) if os.path.isdir(p) else 0

result = {
    "claude": parse_ps(r"claude --dangerously"),
    "codex":  parse_ps(r"(\s|/)codex(\s|$)"),
    "tmux":   {"sessions": tmux_sessions, "panes": tmux_panes},
    "v2": {
        "root_exists": os.path.isdir(v2),
        "inbox_codex":  count(f"{v2}/codex/inbox"),
        "inbox_claude": count(f"{v2}/claude/inbox"),
        "sessions":     count(f"{v2}/sessions"),
        "jobs":         count(f"{v2}/jobs"),
    },
    "watchers": {
        "codex":  lock("/tmp/codex_inbox_watcher.lock"),
        "claude": lock("/tmp/claude_inbox_watcher.lock"),
    },
}
print(json.dumps(result, indent=2, ensure_ascii=False))
PY
