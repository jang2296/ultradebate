"""ultradebate v2.3 — Shared Plan State

Canonical JSON state file shared between Claude and Codex via /tmp/ultradebate_v2/shared_plan.json.

Solves 5 idle-stall causes identified 2026-05-11:
  1. Shared plan state absent (each AI re-computed "where are we?")
  2. Codex pull pattern (Claude had to send explicit message)
  3. B6 1-cycle limit (reactive obligation, not proactive trigger)
  4. /loop prompt stale (30-60min lag forced "stale" detection each turn)
  5. Stage transition manual (W1 PASS → W2 START needed human prompt)

v2.3 architecture:
  - Both AIs read shared_plan.json at every turn → "current step" unambiguous
  - PASS criteria → auto-advance to next step (B9 rule)
  - Heartbeat freshness check (B10 rule) → detect stale peer in <10min
  - Codex auto-pull daemon polls plan every 60s → picks up its tasks without explicit ping
  - /loop pre-processor rewrites stale prompts using current plan state
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

PLAN_PATH = Path("/tmp/ultradebate_v2/shared_plan.json")
HISTORY_PATH = Path("/tmp/ultradebate_v2/shared_plan.history.jsonl")
SCHEMA_VERSION = "2.3.0"
HEARTBEAT_STALE_MIN = 10
CODEX_AUTO_PULL_INTERVAL_SEC = 60


def _now_iso() -> str:
    """ISO-8601 timestamp with local timezone offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _new_etag() -> str:
    return uuid.uuid4().hex[:12]


def read_plan() -> dict[str, Any]:
    """Read canonical plan state. Returns empty dict if not initialized."""
    if not PLAN_PATH.exists():
        return {}
    try:
        with PLAN_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_plan(
    plan: dict[str, Any],
    updated_by: str,
    expected_etag: Optional[str] = None,
) -> tuple[bool, str]:
    """Atomic write with optional etag check.

    Returns (success, new_etag_or_error_message).
    """
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Optimistic lock: if expected_etag provided, current must match
    if expected_etag is not None:
        current = read_plan()
        current_etag = current.get("etag")
        if current_etag and current_etag != expected_etag:
            return False, f"etag_conflict: expected={expected_etag} current={current_etag}"

    new_etag = _new_etag()
    plan["schema_version"] = SCHEMA_VERSION
    plan["etag"] = new_etag
    plan["updated_at"] = _now_iso()
    plan["updated_by"] = updated_by

    # Atomic write via tempfile + rename (POSIX rename is atomic on same fs)
    fd, tmp = tempfile.mkstemp(prefix="shared_plan.", suffix=".tmp", dir=str(PLAN_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        shutil.move(tmp, str(PLAN_PATH))
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False, f"write_error: {e}"

    # Append to history log (best-effort)
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": _now_iso(),
                "etag": new_etag,
                "updated_by": updated_by,
                "current_step": plan.get("current", {}).get("step_id"),
                "step_status": plan.get("current", {}).get("step_status"),
            }) + "\n")
    except OSError:
        pass

    return True, new_etag


def update_heartbeat(actor: str, action_summary: str = "") -> bool:
    """Update only the heartbeat field for `actor` (claude or codex)."""
    if actor not in ("claude", "codex"):
        return False
    plan = read_plan()
    if not plan:
        return False
    plan.setdefault("heartbeat", {})[actor] = {
        "timestamp": _now_iso(),
        "action": action_summary or plan.get("heartbeat", {}).get(actor, {}).get("action", ""),
    }
    ok, _ = write_plan(plan, updated_by=actor)
    return ok


def is_peer_stale(actor: str, threshold_min: int = HEARTBEAT_STALE_MIN) -> bool:
    """Return True if peer's last heartbeat is older than threshold_min."""
    peer = "codex" if actor == "claude" else "claude"
    plan = read_plan()
    peer_hb = plan.get("heartbeat", {}).get(peer, {})
    ts_str = peer_hb.get("timestamp")
    if not ts_str:
        return True
    try:
        ts = datetime.fromisoformat(ts_str)
    except ValueError:
        return True
    return datetime.now(ts.tzinfo) - ts > timedelta(minutes=threshold_min)


def transition_step(
    new_step_id: str,
    new_step_status: str,
    new_step_owner: str,
    updated_by: str,
    acceptance_gates: Optional[list[str]] = None,
) -> bool:
    """Promote a new step to current. Archives previous current to history."""
    plan = read_plan()
    if not plan:
        return False

    # Archive previous current
    prev = plan.get("current")
    if prev and prev.get("step_id"):
        plan.setdefault("history", []).append({**prev, "archived_at": _now_iso()})

    plan["current"] = {
        "step_id": new_step_id,
        "step_status": new_step_status,
        "step_owner": new_step_owner,
        "acceptance_gates": acceptance_gates or [],
        "gates_passed": [],
        "gates_failed": [],
        "started_at": _now_iso(),
    }
    ok, _ = write_plan(plan, updated_by=updated_by)
    return ok


def mark_step_status(step_status: str, updated_by: str, evidence: Optional[dict] = None) -> bool:
    """Set current step status (running/PASS/FAIL/BLOCKED). On PASS, B9 auto-advance fires."""
    plan = read_plan()
    if not plan:
        return False
    cur = plan.setdefault("current", {})
    cur["step_status"] = step_status
    if evidence:
        cur["evidence"] = evidence
    cur["status_updated_at"] = _now_iso()

    # B9 auto-advance trigger
    if step_status.upper() == "PASS" and plan.get("rules", {}).get("auto_advance_on_pass", False):
        nxt = plan.get("next")
        if nxt and nxt.get("step_id"):
            # Archive current
            plan.setdefault("history", []).append({**cur, "archived_at": _now_iso(), "verdict": "PASS"})
            plan["current"] = {
                "step_id": nxt["step_id"],
                "step_status": "pending",
                "step_owner": nxt.get("step_owner", "tbd"),
                "acceptance_gates": nxt.get("acceptance_gates", []),
                "gates_passed": [],
                "gates_failed": [],
                "started_at": _now_iso(),
                "advanced_via": "B9_auto_advance_on_PASS",
            }
            plan["next"] = nxt.get("then", {})
            plan.setdefault("notifications", []).append({
                "ts": _now_iso(),
                "type": "B9_auto_advance",
                "from": cur.get("step_id"),
                "to": plan["current"]["step_id"],
                "trigger": "PASS",
            })

    ok, _ = write_plan(plan, updated_by=updated_by)
    return ok


def current_step_summary() -> str:
    """Human-readable one-line summary of current step (for /loop rewriter)."""
    plan = read_plan()
    cur = plan.get("current", {})
    if not cur:
        return "(no current step)"
    return (
        f"{cur.get('step_id', '?')} "
        f"status={cur.get('step_status', '?')} "
        f"owner={cur.get('step_owner', '?')} "
        f"gates_passed={len(cur.get('gates_passed', []))}/{len(cur.get('acceptance_gates', []))}"
    )


def detect_codex_pending_action() -> Optional[dict]:
    """Return action descriptor if current step assigned to codex_* and is pending/running.

    Used by codex_auto_pull_daemon to know whether to ping Codex.
    """
    plan = read_plan()
    cur = plan.get("current", {})
    owner = (cur.get("step_owner") or "").lower()
    if owner.startswith("codex_") and cur.get("step_status") in ("pending", "running", "awaiting_b3"):
        return {
            "step_id": cur.get("step_id"),
            "owner": owner,
            "status": cur.get("step_status"),
            "acceptance_gates": cur.get("acceptance_gates", []),
        }
    return None


if __name__ == "__main__":
    # Self-check: print plan state when run directly
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "summary":
        print(current_step_summary())
    elif cmd == "read":
        print(json.dumps(read_plan(), indent=2, ensure_ascii=False))
    elif cmd == "heartbeat":
        actor = sys.argv[2] if len(sys.argv) > 2 else "claude"
        action = sys.argv[3] if len(sys.argv) > 3 else "heartbeat tick"
        print("OK" if update_heartbeat(actor, action) else "FAIL")
    elif cmd == "peer_stale":
        actor = sys.argv[2] if len(sys.argv) > 2 else "claude"
        print("STALE" if is_peer_stale(actor) else "FRESH")
    elif cmd == "codex_pending":
        a = detect_codex_pending_action()
        print(json.dumps(a, indent=2) if a else "NONE")
    else:
        print(f"unknown command: {cmd}")
        sys.exit(2)
