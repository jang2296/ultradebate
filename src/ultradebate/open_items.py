"""Multi-item verdict tracker — structural enforcement for 'numbered questions
must each receive an explicit verdict before the exchange is complete'.

This is distinct from `_V22_QUESTIONS_FILE` (single-question request/response).
That tracker handles 1:1 Q&A; this one handles 1:N (one exchange, N items).

Why this module exists (design doc §2.6, §4.4b):
A peer can answer some items with evidence, omit verdicts on others, and move
on — leaving the requester blocked but with no machine-readable signal. By
storing items as named slots and refusing `respond` when any slot is empty,
the 'evidence-only, verdict-deferred' pattern becomes structurally impossible.

What the module CAN enforce: structural completeness (every item slot filled).
What it CANNOT enforce: the *content* of each verdict — that is LLM judgment
and lives outside this module. The contract is "verdict exists, not empty".
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACK_DIR = Path("/tmp/ultradebate_v2/open_items")

ALLOWED_VERDICT_VALUES = {
    "agree", "disagree", "needs_more_info", "blocked", "ack", "reject",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_exchange_id(exchange_id: str) -> str:
    """Reject empty / traversal-prone exchange IDs and normalize separators."""
    if not isinstance(exchange_id, str) or not exchange_id.strip():
        raise ValueError("exchange_id must be a non-empty string")
    # Reject traversal segments outright; replace path separators with '_'.
    if ".." in exchange_id:
        raise ValueError("exchange_id must not contain '..'")
    safe = exchange_id.replace("/", "_").replace("\\", "_")
    if not safe or safe.startswith("."):
        raise ValueError("exchange_id resolves to an unsafe filename")
    return safe


def _track_path(exchange_id: str) -> Path:
    return TRACK_DIR / f"{_safe_exchange_id(exchange_id)}.json"


def _ensure_dir() -> None:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically: tmp file → os.replace (POSIX atomic swap)."""
    _ensure_dir()
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _normalize_verdict(raw: Any) -> dict[str, Any] | None:
    """Return a normalized verdict dict, or None if the value is empty/invalid.

    Accepts:
      - bare non-empty string  → {"value": "<str>"}
      - dict with non-empty 'value' field → kept as-is (other fields preserved)
    Rejects: None, "", whitespace-only string, dict missing/empty 'value'.
    """
    if isinstance(raw, str):
        v = raw.strip()
        if not v:
            return None
        return {"value": v}
    if isinstance(raw, dict):
        val = raw.get("value")
        if not isinstance(val, str) or not val.strip():
            return None
        out = dict(raw)
        out["value"] = val.strip()
        return out
    return None


def open_questions(
    exchange_id: str,
    items: list[str],
    *,
    sender: str | None = None,
    target_peer: str | None = None,
    context: dict | None = None,
) -> dict[str, Any]:
    """Register a set of numbered items as an open exchange.

    Idempotency: re-opening an existing exchange with the *same* items returns
    the existing record. Re-opening with *different* items is refused so that
    in-flight state cannot be silently overwritten.
    """
    try:
        path = _track_path(exchange_id)
    except ValueError as e:
        return {"ok": False, "error": "invalid_exchange_id", "message": str(e)}

    if not isinstance(items, list) or not items:
        return {"ok": False, "error": "items_required",
                "message": "items must be a non-empty list of identifiers"}
    norm_items: list[str] = []
    seen: set[str] = set()
    for it in items:
        if not isinstance(it, str) or not it.strip():
            return {"ok": False, "error": "invalid_item",
                    "message": "each item must be a non-empty string"}
        key = it.strip()
        if key in seen:
            return {"ok": False, "error": "duplicate_item", "item": key}
        seen.add(key)
        norm_items.append(key)

    existing = _load(path)
    if existing is not None:
        existing_items = list(existing.get("open", []))
        if existing_items == norm_items:
            return {"ok": True, "exchange_id": exchange_id, "open": existing_items,
                    "opened_at": existing.get("opened_at"), "reopened": True}
        return {"ok": False, "error": "exchange_already_exists",
                "existing_items": existing_items,
                "note": "Use a new exchange_id or close the existing exchange first"}

    record = {
        "exchange_id": exchange_id,
        "open": norm_items,
        "verdicts": {},
        "opened_at": _now_iso(),
        "sender": sender,
        "target_peer": target_peer,
        "context": context or {},
        "status": "open",
    }
    _atomic_write(path, record)
    return {"ok": True, "exchange_id": exchange_id, "open": norm_items,
            "opened_at": record["opened_at"]}


def respond_to_questions(
    exchange_id: str,
    verdicts: dict[str, Any],
    *,
    responder: str | None = None,
    require_known_verdict_values: bool = False,
) -> dict[str, Any]:
    """Fill verdicts for an exchange. STRUCTURAL REJECTION if any item missing.

    `verdicts` accepts both bare strings and dicts with a `value` key:
        {"W13": "agree", "W14": {"value": "disagree", "rationale": "..."}}
    Empty strings, whitespace-only strings, and dicts with empty `value` are
    treated as *missing* — that is the whole point of this module.
    """
    try:
        path = _track_path(exchange_id)
    except ValueError as e:
        return {"ok": False, "error": "invalid_exchange_id", "message": str(e)}

    record = _load(path)
    if record is None:
        return {"ok": False, "error": "unknown_exchange",
                "message": f"No open exchange named {exchange_id!r}"}
    if record.get("status") == "completed":
        return {"ok": False, "error": "already_completed",
                "completed_at": record.get("completed_at")}
    if record.get("status") == "closed":
        return {"ok": False, "error": "exchange_closed",
                "reason": record.get("close_reason")}

    if not isinstance(verdicts, dict):
        return {"ok": False, "error": "verdicts_must_be_dict"}

    open_items_list: list[str] = list(record.get("open", []))
    normalized: dict[str, dict[str, Any]] = {}
    bad_values: list[str] = []
    for item in open_items_list:
        nv = _normalize_verdict(verdicts.get(item))
        if nv is None:
            continue
        if require_known_verdict_values and nv["value"] not in ALLOWED_VERDICT_VALUES:
            bad_values.append(item)
            continue
        normalized[item] = nv

    missing = [i for i in open_items_list if i not in normalized]
    if missing or bad_values:
        return {
            "ok": False,
            "incomplete": True,
            "missing": missing,
            "bad_verdict_values": bad_values,
            "allowed_verdict_values": (
                sorted(ALLOWED_VERDICT_VALUES) if require_known_verdict_values else None
            ),
            "note": "All open items must have a verdict to complete this exchange",
        }

    record["verdicts"] = normalized
    record["completed_at"] = _now_iso()
    record["responder"] = responder
    record["status"] = "completed"
    _atomic_write(path, record)
    return {"ok": True, "exchange_id": exchange_id, "completed": open_items_list,
            "completed_at": record["completed_at"]}


def pending_verdicts(*, max_age_minutes: float | None = None) -> dict[str, Any]:
    """List all exchanges that are still incomplete.

    `max_age_minutes`: optional filter — only return exchanges older than this.
    Useful at turn boundaries to surface stalled exchanges.
    """
    if not TRACK_DIR.exists():
        return {"pending": []}

    now = datetime.now(timezone.utc)
    pending: list[dict[str, Any]] = []
    for entry in sorted(TRACK_DIR.glob("*.json")):
        if entry.name.startswith("."):
            continue
        record = _load(entry)
        if record is None or record.get("status") != "open":
            continue
        opened_at_raw = record.get("opened_at")
        age_seconds: float | None = None
        if isinstance(opened_at_raw, str):
            try:
                opened_dt = datetime.fromisoformat(opened_at_raw)
                age_seconds = (now - opened_dt).total_seconds()
            except ValueError:
                age_seconds = None
        if max_age_minutes is not None and age_seconds is not None:
            if age_seconds < max_age_minutes * 60:
                continue
        verdicts = record.get("verdicts", {}) or {}
        missing = [i for i in record.get("open", []) if i not in verdicts]
        pending.append({
            "exchange_id": record.get("exchange_id", entry.stem),
            "missing": missing,
            "opened_at": opened_at_raw,
            "age_seconds": age_seconds,
            "sender": record.get("sender"),
            "target_peer": record.get("target_peer"),
        })
    return {"pending": pending}


def get_exchange(exchange_id: str) -> dict[str, Any]:
    """Inspect a single exchange (debug/UI helper)."""
    try:
        path = _track_path(exchange_id)
    except ValueError as e:
        return {"ok": False, "error": "invalid_exchange_id", "message": str(e)}
    record = _load(path)
    if record is None:
        return {"ok": False, "error": "unknown_exchange"}
    return {"ok": True, "exchange": record}


def list_exchanges(*, include_completed: bool = False, limit: int = 100) -> dict[str, Any]:
    """List recent exchanges, newest first by file mtime."""
    if not TRACK_DIR.exists():
        return {"exchanges": []}
    try:
        limit = max(1, min(int(limit), 1000))
    except (TypeError, ValueError):
        limit = 100

    entries = [p for p in TRACK_DIR.glob("*.json") if not p.name.startswith(".")]
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, Any]] = []
    for entry in entries:
        record = _load(entry)
        if record is None:
            continue
        status = record.get("status", "open")
        if not include_completed and status != "open":
            continue
        verdicts = record.get("verdicts", {}) or {}
        out.append({
            "exchange_id": record.get("exchange_id", entry.stem),
            "status": status,
            "open": record.get("open", []),
            "filled": list(verdicts.keys()),
            "opened_at": record.get("opened_at"),
            "completed_at": record.get("completed_at"),
        })
        if len(out) >= limit:
            break
    return {"exchanges": out}


def close_exchange(exchange_id: str, *, reason: str = "manual_close") -> dict[str, Any]:
    """Force-close an exchange without verdicts. Records reason for audit.

    USE SPARINGLY — intended for cancellation by the requester, not as a bypass
    for the verdict-completeness contract.
    """
    try:
        path = _track_path(exchange_id)
    except ValueError as e:
        return {"ok": False, "error": "invalid_exchange_id", "message": str(e)}
    record = _load(path)
    if record is None:
        return {"ok": False, "error": "unknown_exchange"}
    if record.get("status") == "completed":
        return {"ok": False, "error": "already_completed"}
    if record.get("status") == "closed":
        return {"ok": True, "exchange_id": exchange_id, "already_closed": True,
                "closed_at": record.get("closed_at")}
    record["status"] = "closed"
    record["closed_at"] = _now_iso()
    record["close_reason"] = str(reason) if reason else "manual_close"
    _atomic_write(path, record)
    return {"ok": True, "exchange_id": exchange_id,
            "closed_at": record["closed_at"], "reason": record["close_reason"]}
