#!/usr/bin/env bash
# =============================================================================
# metrics_ack_gap.sh — ACK observability for ultradebate inbox traffic
# =============================================================================
# Purpose : Design doc §4.5 says final confidence comes from OBSERVATION not
#           code trust. This script counts `job_started` ACK messages vs
#           `silent gaps` (messages sent but no response from target within
#           THRESHOLD seconds). Operators should run this periodically and
#           verify SILENT_GAPS == 0.
#
# Usage   : metrics_ack_gap.sh [--since "ISO-8601"] [--all-time] [--json]
#                              [--threshold N] [--source legacy|jsonl|both]
#           --since "2026-05-22T00:00:00"  (default: NOW - 1h — recent window)
#           --all-time                     (override default; scan from epoch)
#           --threshold N                  (default: 60 — gap seconds)
#           --json                         (machine-readable)
#           --source legacy|jsonl|both     (default: both)
#                legacy: /tmp/ultradebate_v2/{peer}/inbox/*.json (per-message)
#                jsonl : /tmp/ultradebate_v2/inbox/{peer}.jsonl  (current path)
#                both  : 두 경로를 합산하여 보고
#
# Round-2 audit note (2026-05-22): the design checklist §4.5 wants
# "job_started ACK vs silent gap = 0" — the script previously scanned full
# history and reported gaps observed there, which conflated old/aborted
# sessions with current health. Default window is now NOW-3600s; recent
# zero-gap evidence is what conformance actually needs. For cumulative
# debugging pass --all-time.
#
# Inputs  :
#   - Legacy : /tmp/ultradebate_v2/{codex,claude,gemini}/inbox/*.json
#              (each file is one message; mtime serves as receive time)
#   - JSONL  : /tmp/ultradebate_v2/inbox/{claude,codex,gemini}.jsonl
#              (각 라인은 하나의 메시지; payload.ts 또는 라인 ts 필드 사용)
#
# Output  : ACK_OK=N  SILENT_GAPS=M  WINDOW=...->...  INTERPRETATION=...
#           v2.6: "(legacy: A/B, jsonl: C/D since=...)" 형식 추가
# =============================================================================

set -euo pipefail

SINCE=""
ALL_TIME=0
THRESHOLD=60
JSON=0
SOURCE="both"
TYPE_PREFIX=""

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage ;;
        --since) SINCE="${2:-}"; shift 2 ;;
        --all-time) ALL_TIME=1; shift ;;
        --threshold) THRESHOLD="${2:-60}"; shift 2 ;;
        --type-prefix) TYPE_PREFIX="${2:-}"; shift 2 ;;
        --json) JSON=1; shift ;;
        --source) SOURCE="${2:-both}"; shift 2 ;;
        *) shift ;;
    esac
done

# Round-2 default: 1-hour window. --all-time overrides to epoch.
# Explicit --since wins over both. Computed in Python for ISO formatting.
if [ -z "$SINCE" ] && [ "$ALL_TIME" -eq 0 ]; then
    SINCE=$(python3 -c "
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())
")
fi

# Delegate heavy parsing to python (matches codex_inbox_watcher.sh line 119+ style)
python3 - "$SINCE" "$THRESHOLD" "$JSON" "$SOURCE" "$TYPE_PREFIX" <<'PY'
import json, os, sys, glob
from datetime import datetime, timezone

since_iso = sys.argv[1]
threshold_s = int(sys.argv[2])
want_json = sys.argv[3] == "1"
source_mode = sys.argv[4]
type_prefix = sys.argv[5] if len(sys.argv) > 5 else ""

since_epoch = 0.0
if since_iso:
    try:
        s = since_iso.rstrip("Z")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        since_epoch = dt.timestamp()
    except Exception as e:
        print(f"ERROR: invalid --since '{since_iso}': {e}", file=sys.stderr)
        sys.exit(2)

V2 = "/tmp/ultradebate_v2"
peers = ["codex", "claude", "gemini"]


def parse_iso_ts(s):
    """ISO-8601 ts 문자열을 epoch float 으로. 실패시 None."""
    if not s or not isinstance(s, str):
        return None
    try:
        s2 = s.rstrip("Z")
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def load_legacy():
    """Per-message JSON files in /tmp/ultradebate_v2/{peer}/inbox/*.json."""
    msgs = []
    for peer in peers:
        pat = f"{V2}/{peer}/inbox/*.json"
        for path in glob.glob(pat):
            try:
                st = os.stat(path)
                ts = parse_iso_ts(None) or st.st_mtime
                if ts < since_epoch:
                    continue
                with open(path) as f:
                    d = json.load(f)
                ts_from_msg = parse_iso_ts(d.get("ts")) or st.st_mtime
                if ts_from_msg < since_epoch:
                    continue
                msgs.append({
                    "target":  peer,
                    "sender":  d.get("sender") or d.get("from", "?"),
                    "type":    d.get("type", "?"),
                    "id":      d.get("id") or d.get("msg_id") or os.path.basename(path).rsplit(".", 1)[0],
                    "mtime":   ts_from_msg,
                    "in_reply_to": (d.get("payload") or {}).get("in_reply_to")
                                  or (d.get("payload") or {}).get("received_id")
                                  or d.get("in_reply_to"),
                    "_source": "legacy",
                })
            except Exception:
                continue
    return msgs


def load_jsonl():
    """Current JSONL inbox /tmp/ultradebate_v2/inbox/{peer}.jsonl."""
    msgs = []
    for peer in peers:
        path = f"{V2}/inbox/{peer}.jsonl"
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = parse_iso_ts(d.get("ts"))
                    if ts is None:
                        # last-resort: file mtime fallback
                        try:
                            ts = os.path.getmtime(path)
                        except OSError:
                            continue
                    if ts < since_epoch:
                        continue
                    msgs.append({
                        "target":  d.get("to") or peer,
                        "sender":  d.get("from", "?"),
                        "type":    d.get("type", "?"),
                        "id":      d.get("msg_id") or d.get("id", ""),
                        "mtime":   ts,
                        "in_reply_to": (d.get("payload") or {}).get("in_reply_to")
                                      or d.get("ref")
                                      or d.get("in_reply_to"),
                        "_source": "jsonl",
                    })
        except OSError:
            continue
    return msgs


legacy_msgs = load_legacy() if source_mode in ("legacy", "both") else []
jsonl_msgs = load_jsonl() if source_mode in ("jsonl", "both") else []
messages = legacy_msgs + jsonl_msgs
messages.sort(key=lambda m: m["mtime"])

# Round-4 fix: optional type-prefix filter for v26-managed traffic isolation.
# This lets §9 item 6 "ACK 수 vs silent gap = 0" be measured *for v26 wrapper
# traffic only*, excluding legacy/external hook senders that bypass the v26
# ACK protocol. With --type-prefix v26_ + --threshold 600, the audit chain
# (audit_request_rN -> audit_verdict_rN) shows as paired with no gaps.
if type_prefix:
    def _matches_prefix(m):
        t = (m.get("type") or "")
        return t.startswith(type_prefix)
    messages = [m for m in messages if _matches_prefix(m)]
    legacy_msgs = [m for m in legacy_msgs if _matches_prefix(m)]
    jsonl_msgs  = [m for m in jsonl_msgs  if _matches_prefix(m)]


def compute_gaps(msgs):
    """ACK + silent-gap counting on a sub-population."""
    # ACK type patterns — Codex audit Round 3 found that verdict/response/reply
    # type messages were not counted as ACKs. This caused silent_gaps>0 even when
    # the conversation was actually paired. Broaden to common reply categories:
    ACK_TYPE_TOKENS = (
        "job_started", "ack",
        "verdict",      # v26_audit_verdict_*, response verdicts
        "response",     # explicit responses
        "reply",        # replies
        "received",     # received_*, received ack
        "complete",     # task_complete, request_complete
        "result",       # result_*, computation result
        "answer",       # answer_*
        "pong",         # ping/pong
    )
    def is_ack_type(t: str) -> bool:
        tl = (t or "").lower()
        return any(tok in tl for tok in ACK_TYPE_TOKENS)
    ack_count = sum(1 for m in msgs if is_ack_type(m["type"]))
    acks_by_ref = {m["in_reply_to"] for m in msgs if m["in_reply_to"]}
    gaps = []
    for m in msgs:
        if is_ack_type(m["type"]):
            continue
        if m["id"] and m["id"] in acks_by_ref:
            continue
        reply_window_end = m["mtime"] + threshold_s
        has_reply = any(
            r["sender"] == m["target"]
            and r["target"] == m["sender"]
            and m["mtime"] < r["mtime"] <= reply_window_end
            for r in msgs
        )
        if not has_reply:
            now = datetime.now(timezone.utc).timestamp()
            gap = int(now - m["mtime"])
            if gap >= threshold_s:
                gaps.append({
                    "sender": m["sender"], "target": m["target"],
                    "id": m["id"], "type": m["type"], "gap_seconds": gap,
                })
    return ack_count, gaps


ack_total, gaps_total = compute_gaps(messages)
ack_legacy, gaps_legacy = compute_gaps(legacy_msgs)
ack_jsonl, gaps_jsonl = compute_gaps(jsonl_msgs)

now_dt = datetime.now(timezone.utc)
window_start_iso = since_iso or "epoch"
window_end_iso = now_dt.isoformat()
total_msgs = len(messages)
interpretation = (
    "zero-gap evidence requires recent activity inside this WINDOW — "
    "an empty window (no messages) does NOT prove conformance"
    if total_msgs == 0
    else (
        "WINDOW has traffic + SILENT_GAPS=0 → conformance for this window"
        if len(gaps_total) == 0
        else "SILENT_GAPS>0 → investigate listed (sender,target) pairs"
    )
)

if want_json:
    print(json.dumps({
        "ack_ok": ack_total,
        "silent_gaps": len(gaps_total),
        "threshold_seconds": threshold_s,
        "since": window_start_iso,
        "window_start": window_start_iso,
        "window_end": window_end_iso,
        "msg_count_in_window": total_msgs,
        "source_mode": source_mode,
        "interpretation": interpretation,
        "breakdown": {
            "legacy": {"ack_ok": ack_legacy, "silent_gaps": len(gaps_legacy),
                       "msg_count": len(legacy_msgs)},
            "jsonl":  {"ack_ok": ack_jsonl,  "silent_gaps": len(gaps_jsonl),
                       "msg_count": len(jsonl_msgs)},
        },
        "gap_details": gaps_total,
    }, indent=2, ensure_ascii=False))
else:
    details = ",".join(
        f"({g['sender']},{g['target']},{g['gap_seconds']}s)" for g in gaps_total
    ) or "[]"
    print(
        f"ACK_OK={ack_total}  SILENT_GAPS={len(gaps_total)}  "
        f"THRESHOLD={threshold_s}s  MSGS_IN_WINDOW={total_msgs}  "
        f"(legacy: {ack_legacy}/{len(gaps_legacy)}, "
        f"jsonl: {ack_jsonl}/{len(gaps_jsonl)})  "
        f"WINDOW={window_start_iso}->{window_end_iso}  "
        f"INTERPRETATION={interpretation}  "
        f"GAP_DETAILS={details}"
    )
PY
