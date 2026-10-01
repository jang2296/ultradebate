#!/bin/bash
# =============================================================================
# CRITICAL - TMUX MESSAGE INJECTION SAFE PATTERN (DO NOT REGRESS)
# =============================================================================
# This watcher uses load-buffer + paste-buffer -d + C-m submit (user mandate
# 2026-06-03: ALWAYS C-m, never Tab/Escape+Enter; see logic at the inject step).
# NEVER replace with a raw `tmux send-keys -t <pane> "$msg" C-m` (message inline):
#   tmux interprets literal \n inside the message as Enter, splitting the
#   payload mid-submit. This was the 2026-05-16 v2.5 "/n 처럼 된다" bug. The safe
#   pattern is flatten \n -> ONE paste-buffer atom -> C-m as a SEPARATE keystroke.
# Mirror helper on Python side: _v2_tmux_safe_send() in src/ultradebate/server.py.
# Memory: feedback_ultradebate_v25_cwd_binding.md
# =============================================================================
#
# Codex inbox watcher - auto-injects messages into Codex CLI via tmux send-keys.
#
# Whenever a v2 message arrives in /tmp/ultradebate_v2/codex/inbox/, this watcher
# constructs an appropriate Codex prompt + sends it via tmux paste-buffer with
# Codex-specific submit sequence (flattened payload + C-m).
#
# Result: Codex CLI auto-processes incoming messages without user intervention.
# Mirror of claude_inbox_watcher.sh on Claude side.
#
# Usage:
#   bash codex_inbox_watcher.sh <tmux_session>
#     default tmux_session = "3"
#
#   nohup bash codex_inbox_watcher.sh 3 > /tmp/codex_inbox_watcher.log 2>&1 &

set -uo pipefail

# v2.5+ DYNAMIC BINDING (2026-05-16):
# Watcher now reads /tmp/ultradebate_v2/sessions/*/session.json each loop and
# follows whichever tmux session is currently registered as Codex's pane via
# session_attach. The CLI arg is treated as a FALLBACK only — used when no
# dual_active session.json contains a codex.tmux_session field.
#
# This lets users open Codex in ANY tmux session (any name, any pane) and have
# the watcher auto-attach without restarting. Re-binding is checked every loop
# iteration (~2s), so when user closes one Codex and opens another the watcher
# follows within ~2 seconds.

TMUX_SESSION_FALLBACK=${1:-3}
TMUX_SESSION="$TMUX_SESSION_FALLBACK"  # may be overridden each loop by resolve_codex_pane()
INBOX=/tmp/ultradebate_v2/codex/inbox
PROCESSED_LOG=/tmp/ultradebate_v2/codex/.watcher_seen.log
LOCK=/tmp/codex_inbox_watcher.lock
V25_SESSIONS=/tmp/ultradebate_v2/sessions

# Resolve the currently bound Codex tmux session by scanning all v2.5 session.json.
# Prefers entries where codex is active (no unattached_at) AND tmux session exists.
# Echoes the resolved tmux session name (or fallback if none found).
resolve_codex_pane() {
    local found=""
    if [ -d "$V25_SESSIONS" ]; then
        for sj in "$V25_SESSIONS"/*/session.json; do
            [ -f "$sj" ] || continue
            local cand
            cand=$(python3 -c "
import json, sys
try:
    d = json.load(open('$sj'))
    ag = d.get('active_agents', {}).get('codex', {})
    if ag and not ag.get('unattached_at') and ag.get('tmux_session'):
        print(ag['tmux_session'])
except Exception:
    pass
" 2>/dev/null)
            if [ -n "$cand" ] && tmux has-session -t "$cand" 2>/dev/null; then
                found="$cand"
                break
            fi
        done
    fi
    if [ -n "$found" ]; then
        echo "$found"
    else
        echo "$TMUX_SESSION_FALLBACK"
    fi
}

mkdir -p $INBOX
touch $PROCESSED_LOG

ts() { date '+%H:%M:%S'; }

# Single instance lock
if [ -f $LOCK ]; then
    pid=$(cat $LOCK 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[$(ts)] V2_CODEX_WATCHER_DUP pid=$pid — exiting"
        exit 1
    fi
fi
echo $$ > $LOCK
trap "rm -f $LOCK" EXIT

echo "[$(ts)] V2_CODEX_WATCHER_START fallback_session=$TMUX_SESSION_FALLBACK inbox=$INBOX (v2.5 dynamic binding)"

ACK_TIMEOUT_SEC=${ACK_TIMEOUT_SEC:-300}  # event-based: max wait for Codex ack (5min fallback)
LAST_RESOLVED=""

while true; do
    # Re-resolve every loop — picks up new Codex sessions launched by user
    NEW_RESOLVED=$(resolve_codex_pane)
    if [ "$NEW_RESOLVED" != "$LAST_RESOLVED" ]; then
        echo "[$(ts)] V2_CODEX_WATCHER_REBIND from='$LAST_RESOLVED' to='$NEW_RESOLVED'"
        LAST_RESOLVED="$NEW_RESOLVED"
    fi
    TMUX_SESSION="$NEW_RESOLVED"
    # Sort by ts to process oldest first (FIFO)
    files=$(ls -1 "$INBOX"/*.json 2>/dev/null | sort)
    for msg in $files; do
        [ -f "$msg" ] || continue
        mid=$(basename "$msg" .json)
        if grep -q "^${mid}$" "$PROCESSED_LOG" 2>/dev/null; then
            continue
        fi

        # Read message
        sender=$(python3 -c "import json,sys; print(json.load(open('$msg')).get('sender','?'))" 2>/dev/null || echo "?")
        mtype=$(python3 -c "import json,sys; print(json.load(open('$msg')).get('type','?'))" 2>/dev/null || echo "?")

        # Build Codex prompt based on message type
        if [ "$mtype" = "command" ]; then
            cmd=$(python3 -c "
import json
d=json.load(open('$msg'))
print(d.get('payload',{}).get('command','?'))" 2>/dev/null || echo "?")
            prompt="Cross-tool command from $sender (id=$mid): execute this in Codex context: $cmd"
        elif [ "$mtype" = "ping" ]; then
            text=$(python3 -c "
import json
d=json.load(open('$msg'))
print(str(d.get('payload',{}).get('text',''))[:300])" 2>/dev/null || echo "")
            prompt="Ping from $sender (id=$mid): $text. Briefly acknowledge by calling ultradebate_send_to_peer back to claude with type='ack' payload={'received_id':'$mid'}."
        elif [ "$mtype" = "verdict" ] || [ "$mtype" = "meta_team_dispatch" ]; then
            payload=$(python3 -c "
import json
d=json.load(open('$msg'))
print(json.dumps(d.get('payload',{}), ensure_ascii=False)[:1000])" 2>/dev/null || echo "{}")
            prompt="$mtype from $sender (id=$mid): $payload. Analyze + respond via ultradebate_send_to_peer back to claude."
        else
            text=$(python3 -c "
import json
d=json.load(open('$msg'))
p=d.get('payload',{})
print(p.get('text', json.dumps(p, ensure_ascii=False))[:500])" 2>/dev/null || echo "")
            prompt="Message from $sender (id=$mid type=$mtype): $text. Process + ack via ultradebate_send_to_peer."
        fi

        # Inject into Codex CLI via paste-buffer + verified submit sequence.
        # 2026-05-10 06:27 KST: send-keys with literal "Enter" caused multi-line stack;
        #                       C-m raw fixed it for short prompts.
        # 2026-05-10 07:06 KST: large prompts via send-keys "$prompt" C-m got parsed as
        #                       "[Pasted Content N chars]" placeholders + residual text.
        #                       Switch to load-buffer + paste-buffer pattern: gives Codex
        #                       a single paste event boundary, then explicit C-m submit.
        # 2026-05-10 09:19 KST: User reported still needing manual Enter occasionally.
        #                       Root cause: paste-buffer preserves newlines → Codex sees
        #                       multi-line input → C-m may submit only last line OR fail
        #                       to fully submit. Two fixes:
        #                       1. FLATTEN newlines to space before paste so Codex sees
        #                          ONE logical line (atomic paste, no multi-line stack)
        #                       2. At that time, single C-m matched the user's
        #                          observed Enter behavior.
        # 2026-05-18 20:18 KST: current Codex TUI mode empirically required
        #                       Escape+Enter; C-m inserted a newline.
        if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
            # 2026-05-23 Tab-queue bidirectional behavior:
            #   Codex Working → paste + Tab (Queued follow-up inputs stack)
            #   Codex Idle    → paste + Escape+Enter (normal submit)
            # Previously this loop waited up to 30s (5 iterations × 6s) for Codex
            # to idle, but long jobs left the user staring at no progress. Short
            # transient idle wait now (≤6s), then route by busy state.
            # capture-pane -J joins wrapped lines (reduces multi-byte UTF-8
            # cuts at the wrap column). The iconv -c pipe drops invalid byte
            # sequences before grep — matches the 2026-05-22 text_safety fix
            # that strips lone surrogates from the Python-side capture path.
            for _w in 1 2 3; do
                pane_state=$(tmux capture-pane -J -t $TMUX_SESSION -p 2>/dev/null \
                    | tail -10 \
                    | iconv -f utf-8 -t utf-8 -c 2>/dev/null \
                    || tmux capture-pane -J -t $TMUX_SESSION -p 2>/dev/null \
                       | tail -10 \
                       | tr -d '\000-\010\013\014\016-\037')
                if echo "$pane_state" | grep -qE "esc to interrupt|Working \("; then
                    sleep 2
                else
                    break
                fi
            done
            # Re-capture for routing decision (busy vs idle).
            pane_state=$(tmux capture-pane -J -t $TMUX_SESSION -p 2>/dev/null \
                | tail -10 \
                | iconv -f utf-8 -t utf-8 -c 2>/dev/null \
                || tmux capture-pane -J -t $TMUX_SESSION -p 2>/dev/null \
                   | tail -10 \
                   | tr -d '\000-\010\013\014\016-\037')
            # Flatten ALL C0 control chars (newline/CR/Tab/ESC/etc.) + DEL to space
            # so Codex sees ONE inert logical line: no multi-line stack, no Tab
            # autocomplete, no ESC bracketed-paste breakout. USER MANDATE 2026-06-16
            # (mirrors server.py _flatten_payload_for_tmux). Only a separate C-m submits.
            flat_prompt=$(printf '%s' "$prompt" | LC_ALL=C tr '\001-\037\177' ' ')
            tmpf=$(mktemp /tmp/codex_inject_XXXXXX.txt)
            printf '%s' "$flat_prompt" > "$tmpf"
            tmux load-buffer -b codex_inject "$tmpf"
            # -p: bracketed-paste mode (required by Codex v0.130+, see
            # server.py:_v2_tmux_safe_send for the same fix on 2026-05-22).
            tmux paste-buffer -b codex_inject -p -d -t $TMUX_SESSION 2>/dev/null
            rm -f "$tmpf"
            sleep 2.0  # paste-accept window (codex CLI buffers paste atom)
            # USER MANDATE 2026-06-03: ALWAYS C-m, NEVER Tab / Escape+Enter.
            # When codex's prior job ran into the next, Tab-queued messages were
            # dropped; the injection must equal the user's own Enter (single C-m).
            # The "/n처럼 되는" bug is structurally prevented: flatten above +
            # ONE bracketed paste-buffer atom + C-m as a SEPARATE keystroke after
            # the paste-settle window (so C-m is the submit, never a payload '\r').
            # Emergency override: export CODEX_SUBMIT_KEY="Enter" (or "Escape Enter").
            CODEX_SUBMIT_KEY="${CODEX_SUBMIT_KEY:-C-m}"
            # Codex submit is C-m ONLY (mandate 2026-06-16). Never allow Tab — it
            # queues/autocompletes instead of submitting like the user's own Enter.
            # Even if the env override smuggles in a Tab, force back to C-m.
            case "$CODEX_SUBMIT_KEY" in *[Tt]ab*|"") CODEX_SUBMIT_KEY="C-m";; esac
            tmux send-keys -t $TMUX_SESSION $CODEX_SUBMIT_KEY 2>/dev/null
            sleep 2.0
            # Verify — same iconv -c scrub as the busy-wait loop above.
            _pane=$(tmux capture-pane -J -t $TMUX_SESSION -p 2>/dev/null \
                | tail -10 \
                | iconv -f utf-8 -t utf-8 -c 2>/dev/null \
                || tmux capture-pane -J -t $TMUX_SESSION -p 2>/dev/null \
                   | tail -10 \
                   | tr -d '\000-\010\013\014\016-\037')
            if echo "$_pane" | grep -qE "esc to interrupt|Working \("; then
                echo "[$(ts)] V2_CODEX_SUBMIT_OK id=$mid (C-m fired, Codex Working detected)"
            else
                echo "[$(ts)] V2_CODEX_SUBMIT_UNCERTAIN id=$mid (C-m fired, Codex not visibly Working — may be processing fast or stuck)"
            fi
            echo "[$(ts)] V2_CODEX_INJECTED id=$mid type=$mtype from=$sender → tmux:$TMUX_SESSION (paste-buffer flat + C-m forced)"
        else
            echo "[$(ts)] V2_CODEX_INJECT_FAIL id=$mid — tmux session '$TMUX_SESSION' missing"
            # Don't mark as seen — will retry on next cycle
            continue
        fi

        echo "$mid" >> "$PROCESSED_LOG"

        # EVENT-BASED SYNC: wait for Codex to ACK (write a message to Claude inbox
        # with payload.received_id == $mid). Fall back to TIMEOUT_SEC if no ack.
        echo "[$(ts)] V2_CODEX_AWAIT_ACK id=$mid (event-driven, max ${ACK_TIMEOUT_SEC}s)"
        wait_start=$(date +%s)
        ack_seen=0
        while true; do
            # Scan Claude inbox + processed for any message with received_id=$mid
            for cmsg in /tmp/ultradebate_v2/claude/inbox/*.json /tmp/ultradebate_v2/claude/processed/*.json; do
                [ -f "$cmsg" ] || continue
                rid=$(python3 -c "
import json,sys
try:
    d=json.load(open('$cmsg'))
    p=d.get('payload',{})
    print(p.get('received_id','') or p.get('ack_for','') or p.get('reply_to',''))
except Exception:
    print('')" 2>/dev/null)
                if [ "$rid" = "$mid" ]; then
                    ack_seen=1
                    break
                fi
            done
            [ $ack_seen -eq 1 ] && break

            # Timeout fallback
            elapsed=$(( $(date +%s) - wait_start ))
            if [ $elapsed -gt $ACK_TIMEOUT_SEC ]; then
                echo "[$(ts)] V2_CODEX_ACK_TIMEOUT id=$mid (${elapsed}s elapsed) — proceeding"
                break
            fi
            sleep 3
        done
        if [ $ack_seen -eq 1 ]; then
            ack_elapsed=$(( $(date +%s) - wait_start ))
            echo "[$(ts)] V2_CODEX_ACK_RECEIVED id=$mid latency=${ack_elapsed}s — next inject ready"
        fi
    done
    sleep 2
done
