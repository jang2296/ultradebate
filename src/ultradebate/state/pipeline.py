"""R4 — 5-stage pipeline state machine.

Stages: RESEARCH → DEBATE → PLAN → EXECUTE → CROSS-REVIEW → DONE.

v1.0 scope: only DEBATE is fully wired (the existing round/round/finalize
loop). RESEARCH/PLAN/EXECUTE/CROSS-REVIEW are scaffolds that immediately
advance to the next stage so the state machine is observable end-to-end.
The full per-stage prompts + artifact handling land in v1.1 (design §10
Phase 7).

The blackboard's `stage` field is the single source of truth. All transitions
are atomic-patches with revision check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import blackboard as bb

VALID_STAGES = ("RESEARCH", "DEBATE", "PLAN", "EXECUTE", "CROSS-REVIEW", "DONE")
NEXT_STAGE = {
    "RESEARCH": "DEBATE",
    "DEBATE": "PLAN",
    "PLAN": "EXECUTE",
    "EXECUTE": "CROSS-REVIEW",
    "CROSS-REVIEW": "DONE",
    "DONE": "DONE",
}


@dataclass
class StageResult:
    completed_stage: str
    next_stage: str
    note: str = ""


def advance(session_dir: Path) -> StageResult:
    """Advance the blackboard to the next stage. Idempotent on DONE."""
    b = bb.load(session_dir)
    current = b.stage if b.stage in VALID_STAGES else "DEBATE"
    nxt = NEXT_STAGE.get(current, "DONE")
    bb.patch(session_dir, {"stage": nxt})
    return StageResult(completed_stage=current, next_stage=nxt)


def current_stage(session_dir: Path) -> str:
    b = bb.load(session_dir)
    return b.stage if b.stage in VALID_STAGES else "DEBATE"


def stage_description(stage: str) -> str:
    return {
        "RESEARCH": "Each LLM does light topic research (v1.1 expands).",
        "DEBATE": "Critical multi-AI debate with 6 defect categories (v1.0 wired).",
        "PLAN": "Decompose into tasks, assign owners (v1.1 expands).",
        "EXECUTE": "Each LLM produces artifacts in workspace/{agent}/ (v1.1).",
        "CROSS-REVIEW": "Cross-review artifacts via 6-category matrix (v1.1).",
        "DONE": "Pipeline complete; finalize emitted.",
    }.get(stage, "(unknown stage)")
