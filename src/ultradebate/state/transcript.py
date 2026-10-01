"""Append-only JSONL transcript of every agent utterance."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TranscriptEntry:
    round: int
    speaker: str
    timestamp: float = field(default_factory=time.time)
    analysis: str = ""
    vote_position: str = "abstain"
    vote_confidence: float = 0.5
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    parse_warnings: list[str] = field(default_factory=list)


def _path(session_dir: Path) -> Path:
    return session_dir / "transcript.jsonl"


def append_entry(session_dir: Path, entry: TranscriptEntry) -> None:
    """Append a single entry. Append is atomic on POSIX local fs (single write())."""
    path = _path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def load_transcript(session_dir: Path) -> list[TranscriptEntry]:
    path = _path(session_dir)
    if not path.exists():
        return []
    out: list[TranscriptEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Backward-compat: silently drop unknown keys
            allowed = {f.name for f in TranscriptEntry.__dataclass_fields__.values()}
            d = {k: v for k, v in d.items() if k in allowed}
            try:
                out.append(TranscriptEntry(**d))
            except TypeError:
                continue
    return out


def transcript_excluding_round(entries: list[TranscriptEntry], round_n: int) -> list[TranscriptEntry]:
    """Return entries from rounds < round_n (used to prevent intra-round leakage)."""
    return [e for e in entries if e.round < round_n]
