"""Capability catalog for R14 — Cross-MCP capability bridge.

Each entry registers a named capability that one LLM can request another LLM
to execute on its behalf, e.g. Codex asking Claude to apply SuperClaude
`/sc:implement` rigor, or Claude asking Gemini for a 1M-context repo scan.

The catalog is a hard-coded whitelist (security): unknown capability IDs are
rejected at the server boundary so a malicious or hallucinated capability
request cannot escalate privileges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Capability:
    id: str
    provider: str  # which agent fulfills it (claude/codex/gemini/grok)
    description: str
    input_keys: tuple[str, ...] = field(default_factory=tuple)
    output_summary: str = ""


# Whitelist of allowed capabilities. Adding a new entry is the only way to
# expand the bridge; LLM output cannot register new ones at runtime.
CAPABILITY_CATALOG: dict[str, Capability] = {
    "superclaude:implement": Capability(
        id="superclaude:implement",
        provider="claude",
        description="Apply SuperClaude `/sc:implement` methodology — disciplined "
                    "spec → impl → test → docs flow with persona coordination.",
        input_keys=("spec", "acceptance_criteria"),
        output_summary="diff + tests + doc",
    ),
    "codex:goal": Capability(
        id="codex:goal",
        provider="codex",
        description="Codex goal-oriented task tracker. Decomposes a goal into "
                    "milestones with explicit acceptance gates.",
        input_keys=("goal",),
        output_summary="structured plan with milestones",
    ),
    "gemini:long-context-scan": Capability(
        id="gemini:long-context-scan",
        provider="gemini",
        description="Gemini 1M-token context for repo-wide cross-file analysis. "
                    "Use when a question requires scanning >100KB of code.",
        input_keys=("repo_path", "question"),
        output_summary="cross-file analysis with citations",
    ),
    "claude:prompt-cache": Capability(
        id="claude:prompt-cache",
        provider="claude",
        description="Claude Anthropic prompt-cache reuse — cheaper iteration on "
                    "a stable prefix when only the suffix changes.",
        input_keys=("prefix", "suffix"),
        output_summary="response with cache hit",
    ),
}


def get(capability_id: str) -> Capability | None:
    return CAPABILITY_CATALOG.get(capability_id)


def list_capabilities() -> list[dict]:
    return [
        {
            "id": c.id,
            "provider": c.provider,
            "description": c.description,
            "input_keys": list(c.input_keys),
            "output_summary": c.output_summary,
        }
        for c in CAPABILITY_CATALOG.values()
    ]


def validate_args(capability: Capability, args: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, reason). Required keys must be present (non-empty); unknown
    keys are rejected to keep the contract tight (security + type safety).

    The reserved key `__rationale__` is allowed (used internally to pass the
    requester's rationale into the prompt) and never counted as unknown.
    """
    if not isinstance(args, dict):
        return False, "args must be a JSON object"
    missing = [k for k in capability.input_keys if not args.get(k)]
    if missing:
        return False, f"missing required keys: {missing}"
    allowed = set(capability.input_keys) | {"__rationale__"}
    unknown = [k for k in args if k not in allowed]
    if unknown:
        return False, f"unknown keys not in catalog: {unknown}"
    return True, ""


def render_catalog_for_prompt(max_chars: int = 1200) -> str:
    """Compact textual catalog for inclusion in the round prompt — under 1.2KB."""
    lines = ["# Capability catalog (R14 — request via <capability_request>):"]
    for c in CAPABILITY_CATALOG.values():
        keys = ",".join(c.input_keys) if c.input_keys else "(none)"
        lines.append(f"- {c.id:32s} → {c.provider:6s} | inputs: {keys}")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 30] + "\n[... catalog truncated ...]"
    return out
