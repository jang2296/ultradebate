"""Session lifecycle: create, persist meta, load."""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .security import safe_session_dir


@dataclass
class SessionMeta:
    session_id: str
    topic: str
    mode: str  # strict | lenient
    max_rounds: int
    token_budget: int
    created_at: float
    current_round: int = 0
    tokens_used: int = 0
    last_consensus: float | None = None
    completed: bool = False
    consensus_history: list[float] = field(default_factory=list)


def new_session_id() -> str:
    """16 hex chars from a CSPRNG."""
    return secrets.token_hex(8)


def init_session(
    base_dir: Path,
    topic: str,
    mode: str,
    max_rounds: int,
    token_budget: int,
) -> SessionMeta:
    sid = new_session_id()
    sdir = safe_session_dir(base_dir, sid)
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "mailbox" / "inbox").mkdir(parents=True, exist_ok=True)
    (sdir / "mailbox" / "processed").mkdir(parents=True, exist_ok=True)

    meta = SessionMeta(
        session_id=sid,
        topic=topic,
        mode=mode,
        max_rounds=max_rounds,
        token_budget=token_budget,
        created_at=time.time(),
    )
    save_meta(sdir, meta)
    return meta


def save_meta(session_dir: Path, meta: SessionMeta) -> None:
    path = session_dir / "meta.json"
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic on POSIX


def load_meta(session_dir: Path) -> SessionMeta:
    with open(session_dir / "meta.json", encoding="utf-8") as f:
        data = json.load(f)
    return SessionMeta(**data)
