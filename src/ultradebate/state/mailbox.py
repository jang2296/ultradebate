"""P2P message routing. Each agent has an inbox; processed messages move to archive.

The `deliver()` writer appends to `inbox/to_<recipient>.jsonl`. The `fetch_inbox()`
reader rotates the file to `processed/` so the same message is never delivered twice.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def _inbox(session_dir: Path, recipient: str) -> Path:
    p = session_dir / "mailbox" / "inbox" / f"to_{recipient}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _processed_dir(session_dir: Path) -> Path:
    p = session_dir / "mailbox" / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def deliver(
    session_dir: Path,
    sender: str,
    recipient: str,
    content: str,
    round_n: int,
) -> None:
    msg = {
        "from": sender,
        "to": recipient,
        "content": content,
        "round": round_n,
        "ts": time.time(),
    }
    path = _inbox(session_dir, recipient)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def fetch_inbox(
    session_dir: Path,
    recipient: str,
    *,
    mark_processed: bool = True,
) -> list[dict]:
    """Read & (optionally) rotate inbox. Returns list of message dicts."""
    path = _inbox(session_dir, recipient)
    if not path.exists():
        return []
    msgs: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if mark_processed and msgs:
        archive = _processed_dir(session_dir) / f"to_{recipient}_{int(time.time() * 1000)}.jsonl"
        try:
            path.rename(archive)
        except OSError:
            # Best effort: if rename fails, truncate to avoid duplicate delivery
            path.write_text("", encoding="utf-8")
    return msgs


def peek_inbox(session_dir: Path, recipient: str) -> int:
    """Return inbox message count without consuming."""
    path = _inbox(session_dir, recipient)
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
