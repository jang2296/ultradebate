"""Blackboard — single source of truth for inter-LLM shared state.

Stored as `{session_dir}/blackboard.json` with flock-protected read/write
and a monotonic revision counter to detect lost updates.

Schema (all English internally):
{
  "session_id": str,
  "topic": str,
  "stage": "RESEARCH"|"DEBATE"|"PLAN"|"EXECUTE"|"REVIEW"|"DONE",
  "revision": int,
  "last_updated": float,
  "focus": str,
  "decided": [{"id": "D-001", "scope": "global|<agent>", "text": str}],
  "open_questions": [{"id": "Q-001", "text": str, "votes": {agent: stance}}],
  "assignments": [{"id": "T-...", "owner": str, "title": str, "status": str}],
  "risks": [{"id": "R-001", "text": str}],
  "team_roster": {"claude": 1, "codex": 1, "gemini": 1, "grok": 0}
}

Patch grammar (subset of JSON merge-patch + list ops):
  {"focus": "..."}                       → replace scalar
  {"decided": {"add": [...]}}            → append to list
  {"open_questions": {"resolve": "Q-1"}} → mark resolved + move to decided
  {"assignments": {"update": {"T-1": "claude"}}} → set owner
  {"team_roster": {"set": {"grok": 1}}}  → merge into roster dict
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import fcntl  # POSIX only
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False


class RevisionMismatch(Exception):
    """Raised when patch's expected_revision doesn't match the current blackboard."""

    def __init__(self, expected: int, actual: int):
        super().__init__(f"expected revision {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


@dataclass
class Blackboard:
    session_id: str
    topic: str
    stage: str = "DEBATE"
    revision: int = 0
    last_updated: float = 0.0
    focus: str = ""
    decided: list[dict] = field(default_factory=list)
    open_questions: list[dict] = field(default_factory=list)
    assignments: list[dict] = field(default_factory=list)
    risks: list[dict] = field(default_factory=list)
    team_roster: dict[str, int] = field(default_factory=dict)


def _path(session_dir: Path) -> Path:
    return session_dir / "blackboard.json"


@contextmanager
def _locked(path: Path, mode: str = "r+"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode.startswith("r") and not path.exists():
        path.write_text("{}", encoding="utf-8")
    elif mode == "w":
        path.touch(exist_ok=True)
    f = open(path, mode, encoding="utf-8")
    try:
        if _HAS_FCNTL:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield f
    finally:
        if _HAS_FCNTL:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        f.close()


def init(
    session_dir: Path,
    *,
    session_id: str,
    topic: str,
    team_roster: dict[str, int] | None = None,
    stage: str = "DEBATE",
) -> Blackboard:
    """Create a fresh blackboard. Idempotent: safe to call on existing session."""
    bb = Blackboard(
        session_id=session_id,
        topic=topic,
        stage=stage,
        revision=0,
        last_updated=time.time(),
        team_roster=team_roster or {},
    )
    path = _path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path, "w") as f:
        json.dump(asdict(bb), f, ensure_ascii=False, indent=2)
        f.flush()
    return bb


def load(session_dir: Path) -> Blackboard:
    path = _path(session_dir)
    if not path.exists():
        raise FileNotFoundError(f"blackboard not found: {path}")
    with _locked(path, "r+") as f:
        f.seek(0)
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    # Defensive: drop unknown keys, supply defaults for missing
    allowed = {fld.name for fld in Blackboard.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in allowed}
    return Blackboard(**{**_defaults(), **filtered})


def _defaults() -> dict:
    return {
        "session_id": "",
        "topic": "",
        "stage": "DEBATE",
        "revision": 0,
        "last_updated": 0.0,
        "focus": "",
        "decided": [],
        "open_questions": [],
        "assignments": [],
        "risks": [],
        "team_roster": {},
    }


def patch(
    session_dir: Path,
    patch_dict: dict,
    *,
    expected_revision: int | None = None,
) -> Blackboard:
    """Atomic patch with optional revision check.

    On expected_revision mismatch, raises RevisionMismatch (caller may retry
    after re-reading). Always increments revision and updates last_updated.
    """
    if not isinstance(patch_dict, dict):
        raise TypeError(f"patch_dict must be dict, got {type(patch_dict).__name__}")

    path = _path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}", encoding="utf-8")

    with _locked(path, "r+") as f:
        f.seek(0)
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}
        merged = {**_defaults(), **(data or {})}

        if expected_revision is not None and merged.get("revision", 0) != expected_revision:
            raise RevisionMismatch(expected_revision, merged.get("revision", 0))

        merged = _apply_patch(merged, patch_dict)
        merged["revision"] = int(merged.get("revision", 0)) + 1
        merged["last_updated"] = time.time()

        f.seek(0)
        f.truncate()
        json.dump(merged, f, ensure_ascii=False, indent=2)
        f.flush()

    allowed = {fld.name for fld in Blackboard.__dataclass_fields__.values()}
    filtered = {k: v for k, v in merged.items() if k in allowed}
    return Blackboard(**{**_defaults(), **filtered})


def _apply_patch(data: dict, patch_dict: dict) -> dict:
    """Apply our limited patch grammar in-place (returns the dict for chaining).

    Each top-level key in `patch_dict`:
      - If value is a scalar (str/int/float/bool) → replace data[key].
      - If value is a dict with op keys → apply ops (add/resolve/update/set).
      - If value is a list → replace data[key] (full overwrite).
    Unknown ops are silently ignored to keep the parser permissive.
    """
    for key, op in patch_dict.items():
        if isinstance(op, (str, int, float, bool)) or op is None:
            data[key] = op
            continue
        if isinstance(op, list):
            data[key] = op
            continue
        if not isinstance(op, dict):
            continue

        # List-shaped fields support add/resolve/remove/update
        if key in {"decided", "open_questions", "assignments", "risks"}:
            current = list(data.get(key, []) or [])

            add_items = op.get("add")
            if isinstance(add_items, list):
                for item in add_items:
                    if isinstance(item, str):
                        current.append({"id": _next_id(current, key), "text": item})
                    elif isinstance(item, dict):
                        if "id" not in item:
                            item = {**item, "id": _next_id(current, key)}
                        current.append(item)

            resolve_id = op.get("resolve")
            if isinstance(resolve_id, str) and key == "open_questions":
                # Move the resolved Q to "decided" with text + resolution note
                kept = []
                for q in current:
                    if isinstance(q, dict) and q.get("id") == resolve_id:
                        decided_list = list(data.get("decided", []) or [])
                        decided_list.append({
                            "id": _next_id(decided_list, "decided"),
                            "scope": "global",
                            "text": f"resolved {resolve_id}: {op.get('note', q.get('text', ''))}",
                        })
                        data["decided"] = decided_list
                    else:
                        kept.append(q)
                current = kept

            remove_id = op.get("remove")
            if isinstance(remove_id, str):
                current = [
                    item for item in current
                    if not (isinstance(item, dict) and item.get("id") == remove_id)
                ]

            update_map = op.get("update")
            if isinstance(update_map, dict) and key == "assignments":
                # {"T-1": "claude"} → set owner=claude on assignment with id=T-1
                for tid, owner in update_map.items():
                    found = False
                    for item in current:
                        if isinstance(item, dict) and item.get("id") == tid:
                            item["owner"] = owner
                            found = True
                            break
                    if not found:
                        current.append({"id": tid, "owner": owner, "status": "pending"})

            data[key] = current
            continue

        if key == "team_roster":
            current_dict = dict(data.get(key, {}) or {})
            set_map = op.get("set")
            if isinstance(set_map, dict):
                for agent, val in set_map.items():
                    try:
                        current_dict[agent] = int(bool(val))
                    except (TypeError, ValueError):
                        continue
            data[key] = current_dict
            continue

        # Fallback: treat unknown dict-valued keys as full replace
        data[key] = op
    return data


def _next_id(items: list[dict], category: str) -> str:
    prefix = {
        "decided": "D",
        "open_questions": "Q",
        "assignments": "T",
        "risks": "R",
    }.get(category, "X")
    used: set[int] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        i = it.get("id")
        if isinstance(i, str) and i.startswith(prefix + "-"):
            try:
                used.add(int(i.split("-", 1)[1]))
            except ValueError:
                continue
    n = 1
    while n in used:
        n += 1
    return f"{prefix}-{n:03d}"


def role_aware_view(bb: Blackboard, agent: str, *, max_chars: int = 2000) -> str:
    """Render a role-filtered Markdown slice for `agent` (≤max_chars).

    Includes: focus + global decisions + agent's own decisions + agent's
    assignments + open questions + active roster. Excludes other agents'
    private decisions (scope != "global" and scope != agent).
    """
    lines: list[str] = []
    lines.append(f"# Blackboard view for @{agent} (rev {bb.revision})")
    lines.append(f"topic: {bb.topic}")
    lines.append(f"stage: {bb.stage}")
    if bb.team_roster:
        roster_str = " ".join(f"{a}={v}" for a, v in bb.team_roster.items())
        lines.append(f"roster: {roster_str}")
    if bb.focus:
        lines.append(f"\n## Focus\n{bb.focus}")

    relevant_decisions = [
        d for d in bb.decided
        if isinstance(d, dict) and d.get("scope", "global") in {"global", agent}
    ]
    if relevant_decisions:
        lines.append("\n## Decided")
        for d in relevant_decisions:
            lines.append(f"- {d.get('id', '?')}: {d.get('text', '')}")

    if bb.open_questions:
        lines.append("\n## Open questions")
        for q in bb.open_questions:
            if isinstance(q, dict):
                lines.append(f"- {q.get('id', '?')}: {q.get('text', '')}")

    own_assigns = [
        a for a in bb.assignments
        if isinstance(a, dict) and a.get("owner") == agent
    ]
    other_assigns = [
        a for a in bb.assignments
        if isinstance(a, dict) and a.get("owner") != agent
    ]
    if own_assigns:
        lines.append("\n## Your assignments")
        for a in own_assigns:
            lines.append(
                f"- {a.get('id', '?')} [{a.get('status', 'pending')}]: {a.get('title', '')}"
            )
    if other_assigns:
        lines.append("\n## Teammates' assignments (read-only)")
        for a in other_assigns:
            lines.append(
                f"- {a.get('id', '?')} @{a.get('owner', '?')}: {a.get('title', '')}"
            )

    if bb.risks:
        lines.append("\n## Risks surfaced")
        for r in bb.risks:
            if isinstance(r, dict):
                lines.append(f"- {r.get('id', '?')}: {r.get('text', '')}")

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 30] + "\n[... truncated for token economy ...]"
    return out


def active_agents(bb: Blackboard) -> list[str]:
    """Return list of agents with roster value == 1, in insertion order."""
    return [a for a, v in bb.team_roster.items() if int(bool(v)) == 1]


def to_dict(bb: Blackboard) -> dict[str, Any]:
    return asdict(bb)
