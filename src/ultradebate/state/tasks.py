"""Shared task list with fcntl-based atomic claim."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import fcntl  # POSIX only
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False

VALID_STATUSES = {"pending", "in_progress", "done", "blocked"}


@dataclass
class Task:
    id: str
    title: str
    status: str = "pending"
    owner: str | None = None
    created_by: str = ""
    created_at: float = 0.0
    claimed_at: float | None = None
    completed_at: float | None = None


def _path(session_dir: Path) -> Path:
    return session_dir / "tasks.jsonl"


@contextmanager
def _locked(path: Path, mode: str = "r+"):
    """Open `path` and acquire an exclusive flock. Creates the file if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_all(f) -> list[dict]:
    f.seek(0)
    items: list[dict] = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def _write_all(f, items: list[dict]) -> None:
    f.seek(0)
    f.truncate()
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
    f.flush()


def _new_task_id() -> str:
    return "T-" + uuid.uuid4().hex[:6]


def add_task(session_dir: Path, title: str, created_by: str) -> str:
    """Add a new pending task; return its id."""
    if not title or not title.strip():
        raise ValueError("task title cannot be empty")
    tid = _new_task_id()
    with _locked(_path(session_dir)) as f:
        items = _read_all(f)
        items.append(
            asdict(
                Task(
                    id=tid,
                    title=title.strip(),
                    created_by=created_by,
                    created_at=time.time(),
                )
            )
        )
        _write_all(f, items)
    return tid


def claim_task(session_dir: Path, agent: str, task_id: str) -> bool:
    """Atomically transition task pending → in_progress, owned by `agent`.

    Returns True if claim succeeded. Returns False if the task was already
    claimed by someone else, completed, blocked, or non-existent.
    """
    path = _path(session_dir)
    if not path.exists():
        return False
    with _locked(path) as f:
        items = _read_all(f)
        for it in items:
            if it.get("id") == task_id and it.get("status") == "pending":
                it["status"] = "in_progress"
                it["owner"] = agent
                it["claimed_at"] = time.time()
                _write_all(f, items)
                return True
        return False


def complete_task(session_dir: Path, agent: str, task_id: str) -> bool:
    """Mark a task as done; only the current owner may complete it."""
    path = _path(session_dir)
    if not path.exists():
        return False
    with _locked(path) as f:
        items = _read_all(f)
        for it in items:
            if (
                it.get("id") == task_id
                and it.get("owner") == agent
                and it.get("status") == "in_progress"
            ):
                it["status"] = "done"
                it["completed_at"] = time.time()
                _write_all(f, items)
                return True
        return False


def block_task(session_dir: Path, agent: str, task_id: str) -> bool:
    """Mark a task as blocked; either owner or any agent may flag it."""
    path = _path(session_dir)
    if not path.exists():
        return False
    with _locked(path) as f:
        items = _read_all(f)
        for it in items:
            if it.get("id") == task_id and it.get("status") in {"pending", "in_progress"}:
                it["status"] = "blocked"
                _write_all(f, items)
                return True
        return False


def list_tasks(session_dir: Path) -> list[Task]:
    """Read all tasks (shared lock for consistency)."""
    path = _path(session_dir)
    if not path.exists():
        return []
    with _locked(path, mode="r+") as f:
        items = _read_all(f)
    out: list[Task] = []
    for it in items:
        try:
            out.append(Task(**it))
        except TypeError:
            continue
    return out
