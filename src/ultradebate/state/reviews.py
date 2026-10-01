"""R12 — Cross-review storage + 6-category matrix per design §5.9.

A ReviewResult records one reviewer's pass over one reviewee's artifact, using
the same 6 defect categories as round-level critique (UNIMPL/OMIT/UNCONN/
PARTIAL/APPROACH/CLARIFY). v1.0 stores reviews in `{session}/reviews/{ts}.json`;
the ultradebate_review server tool drives the workflow.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ReviewResult:
    id: str
    reviewer: str
    reviewee: str
    artifact_path: str
    omissions: list[str] = field(default_factory=list)        # OMIT
    wrong_links: list[str] = field(default_factory=list)      # UNCONN
    unimplemented: list[str] = field(default_factory=list)    # UNIMPL
    spec_drift: list[str] = field(default_factory=list)       # APPROACH/CLARIFY/PARTIAL
    overall_pass: bool = True
    severity: str = "minor"  # blocking | major | minor
    created_at: float = 0.0


def _new_review_id() -> str:
    return "RV-" + uuid.uuid4().hex[:6]


def reviews_dir(session_dir: Path) -> Path:
    p = session_dir / "reviews"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(session_dir: Path, review: ReviewResult) -> Path:
    """Persist a review to {session}/reviews/{review.id}.json."""
    review.created_at = time.time()
    path = reviews_dir(session_dir) / f"{review.id}.json"
    path.write_text(
        json.dumps(asdict(review), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_all(session_dir: Path) -> list[ReviewResult]:
    out: list[ReviewResult] = []
    rdir = reviews_dir(session_dir)
    for path in sorted(rdir.glob("RV-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out.append(ReviewResult(**data))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return out


def make_review(
    *,
    reviewer: str,
    reviewee: str,
    artifact_path: str,
) -> ReviewResult:
    return ReviewResult(
        id=_new_review_id(),
        reviewer=reviewer,
        reviewee=reviewee,
        artifact_path=artifact_path,
    )
