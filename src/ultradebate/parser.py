"""Robust parser for the v1.0 debate response format.

Expected tags from each AI:

    <analysis>...</analysis>
    <messages>[{
        "to": "claude|codex|gemini|grok",
        "category": "UNIMPL|OMIT|UNCONN|PARTIAL|APPROACH|CLARIFY|ACK|INFO",
        "cite_path": "path/to/file.ts" | "D-001",
        "cite_lines": "34-41" | null,
        "content": "...",
        "proposal": "one-line fix" | null
    }]</messages>
    <tasks>[{"action": "add|claim|complete|block", ...}]</tasks>
    <vote>{"position": "agree|disagree|abstain", "confidence": 0.0-1.0}</vote>
    <blackboard_patch>{"decided": {...}, "open_questions": {...}, ...}</blackboard_patch>

The parser is permissive: missing tags fall back to safe defaults and record warnings.
Vague messages (missing category, or INFO/no-proposal) are flagged so the consensus
engine can apply a small penalty.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

VALID_VOTES = {"agree", "disagree", "abstain"}
VALID_TASK_ACTIONS = {"add", "claim", "complete", "block"}
AGENT_NAMES = {"claude", "codex", "gemini", "grok"}

# 6 defect categories + 2 control categories
DEFECT_CATEGORIES = {"UNIMPL", "OMIT", "UNCONN", "PARTIAL", "APPROACH", "CLARIFY"}
CONTROL_CATEGORIES = {"ACK", "INFO"}
VALID_CATEGORIES = DEFECT_CATEGORIES | CONTROL_CATEGORIES


@dataclass
class Vote:
    position: str = "abstain"
    confidence: float = 0.5


@dataclass
class OutgoingMessage:
    to: str
    content: str
    category: str = "INFO"  # default if absent
    cite_path: str | None = None
    cite_lines: str | None = None
    proposal: str | None = None
    vague: bool = False  # set True when category was missing/invalid OR no cite for defect

    @property
    def is_ack(self) -> bool:
        return self.category == "ACK"

    @property
    def is_defect(self) -> bool:
        return self.category in DEFECT_CATEGORIES


@dataclass
class TaskAction:
    action: str
    title: str | None = None
    id: str | None = None


@dataclass
class CapabilityRequest:
    capability: str
    args: dict
    rationale: str = ""


@dataclass
class ParsedResponse:
    analysis: str = ""
    messages: list[OutgoingMessage] = field(default_factory=list)
    tasks: list[TaskAction] = field(default_factory=list)
    vote: Vote = field(default_factory=Vote)
    blackboard_patch: dict | None = None
    capability_request: CapabilityRequest | None = None
    parse_warnings: list[str] = field(default_factory=list)
    template_echo: bool = False  # True when LLM literally echoed prompt placeholders


def _extract_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(rf"<{tag}\b[^>]*>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _parse_json_loose(raw: str | None, default: Any) -> Any:
    """JSON-parse with fallbacks for common LLM output quirks."""
    if not raw:
        return default
    s = raw.strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    for opener, closer in (("[", "]"), ("{", "}")):
        i = s.find(opener)
        j = s.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                continue
    return default


def _normalize_to(value: Any) -> str:
    """Strip leading '@' and lowercase. Returns "" if not a valid agent name."""
    if not isinstance(value, str):
        return ""
    s = value.strip().lstrip("@").lower()
    return s if s in AGENT_NAMES else ""


def _normalize_category(value: Any) -> tuple[str, bool]:
    """Return (canonical_category, was_invalid_or_missing)."""
    if not isinstance(value, str):
        return "INFO", True
    cat = value.strip().upper()
    if cat in VALID_CATEGORIES:
        return cat, False
    return "INFO", True


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return None


def parse_response(text: str) -> ParsedResponse:
    """Parse a raw AI response. Always returns a valid ParsedResponse."""
    warnings: list[str] = []

    if not isinstance(text, str):
        return ParsedResponse(parse_warnings=["non-string response"])

    # --- analysis ---
    analysis = _extract_tag(text, "analysis")
    if analysis is None:
        analysis = text.strip()
        warnings.append("missing <analysis> tag")

    # --- messages ---
    messages_raw = _extract_tag(text, "messages")
    messages_data = _parse_json_loose(messages_raw, [])
    if messages_raw and not isinstance(messages_data, list):
        warnings.append("<messages> not a JSON list")
        messages_data = []
    messages: list[OutgoingMessage] = []
    for entry in messages_data or []:
        if not isinstance(entry, dict):
            continue
        to = _normalize_to(entry.get("to"))
        content = entry.get("content", "")
        if not to or not isinstance(content, str) or not content.strip():
            continue
        category, cat_missing = _normalize_category(entry.get("category"))
        cite_path = _str_or_none(entry.get("cite_path"))
        cite_lines = _str_or_none(entry.get("cite_lines"))
        proposal = _str_or_none(entry.get("proposal"))

        # Vague rule: defect category requires cite_path; INFO/ACK without proposal is fine
        # but counts as vague when category was missing entirely.
        is_defect_no_cite = category in DEFECT_CATEGORIES and not cite_path
        vague = cat_missing or is_defect_no_cite
        if cat_missing:
            warnings.append(f"message to {to!r} missing/invalid category — defaulted to INFO")
        if is_defect_no_cite:
            warnings.append(f"message to {to!r} category={category} missing cite_path")

        messages.append(
            OutgoingMessage(
                to=to,
                content=content.strip(),
                category=category,
                cite_path=cite_path,
                cite_lines=cite_lines,
                proposal=proposal,
                vague=vague,
            )
        )

    # --- tasks ---
    tasks_raw = _extract_tag(text, "tasks")
    tasks_data = _parse_json_loose(tasks_raw, [])
    if tasks_raw and not isinstance(tasks_data, list):
        warnings.append("<tasks> not a JSON list")
        tasks_data = []
    tasks: list[TaskAction] = []
    for entry in tasks_data or []:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action", "")).strip().lower()
        if action not in VALID_TASK_ACTIONS:
            continue
        title = entry.get("title")
        tid = entry.get("id")
        tasks.append(
            TaskAction(
                action=action,
                title=str(title).strip() if isinstance(title, str) and title.strip() else None,
                id=str(tid).strip() if isinstance(tid, str) and tid.strip() else None,
            )
        )

    # --- vote ---
    vote_raw = _extract_tag(text, "vote")
    vote_data = _parse_json_loose(vote_raw, None)
    if isinstance(vote_data, dict):
        position = str(vote_data.get("position", "abstain")).strip().lower()
        if position not in VALID_VOTES:
            warnings.append(f"invalid vote position {position!r}")
            position = "abstain"
        try:
            confidence = float(vote_data.get("confidence", 0.5))
        except (TypeError, ValueError):
            warnings.append("invalid vote confidence")
            confidence = 0.5
        # Defensive sanitization: NaN/Inf survives min/max comparisons in some
        # Python implementations and would corrupt the consensus engine.
        if math.isnan(confidence) or math.isinf(confidence):
            warnings.append("vote confidence was NaN/Inf — defaulted to 0.5")
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        vote = Vote(position=position, confidence=confidence)
    else:
        if vote_raw:
            warnings.append("<vote> not parseable JSON")
        vote = Vote()

    # --- blackboard_patch (optional) ---
    bb_raw = _extract_tag(text, "blackboard_patch")
    bb_patch: dict | None = None
    if bb_raw:
        parsed_bb = _parse_json_loose(bb_raw, None)
        if isinstance(parsed_bb, dict):
            bb_patch = parsed_bb
        elif parsed_bb is not None:
            warnings.append("<blackboard_patch> not a JSON object")

    # --- capability_request (R14, optional) ---
    cap_raw = _extract_tag(text, "capability_request")
    cap_req: CapabilityRequest | None = None
    if cap_raw:
        parsed_cap = _parse_json_loose(cap_raw, None)
        if isinstance(parsed_cap, dict):
            cap_id = parsed_cap.get("capability")
            cap_args = parsed_cap.get("args", {})
            cap_rationale = parsed_cap.get("rationale", "")
            if isinstance(cap_id, str) and cap_id.strip() and isinstance(cap_args, dict):
                cap_req = CapabilityRequest(
                    capability=cap_id.strip(),
                    args=cap_args,
                    rationale=str(cap_rationale or "").strip(),
                )
            else:
                warnings.append("<capability_request> missing required fields")
        elif parsed_cap is not None:
            warnings.append("<capability_request> not a JSON object")

    # Detect template-echo failures: some CLIs (notably Codex with --reasoning-effort
    # xhigh) consume massive reasoning tokens but emit prompt placeholders verbatim
    # in the final output. We scan for the REPLACE_WITH sentinel anywhere or the
    # literal pipe-list inside the raw <vote> tag (before normalization).
    raw_vote_str = vote_raw or ""
    template_echo = (
        "REPLACE_WITH" in (analysis or "")
        or "REPLACE_WITH" in raw_vote_str
        or "agree|disagree|abstain" in raw_vote_str.lower()
    )
    if template_echo:
        warnings.append("template-echo detected — LLM emitted placeholder text verbatim")
        # Unconditionally downgrade the vote to abstain so a placeholder echo can
        # never poison consensus — even if the upstream normalizer already
        # mapped an unknown position to "abstain", we ALSO zero the confidence.
        # (Earlier code gated this behind `not in VALID_VOTES`, which was
        # unreachable because normalization already enforced VALID_VOTES.)
        vote = Vote(position="abstain", confidence=0.0)

    return ParsedResponse(
        analysis=analysis,
        messages=messages,
        tasks=tasks,
        vote=vote,
        blackboard_patch=bb_patch,
        capability_request=cap_req,
        parse_warnings=warnings,
        template_echo=template_echo,
    )
