"""MCP tools for collaboration between running Claude Code and Codex CLIs.

The v2 transport records messages in a JSONL inbox, then optionally injects
text into the peer's existing tmux pane with bracketed paste and a separate
submission key. Claude outside tmux consumes the inbox through host alerts
or the receive tools. Legacy round tools also support separate CLI workers.

See README.md for WSL setup, persistent inbox monitoring and delivery checks.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .agents import ClaudeWorker, CodexWorker, GeminiWorker, MockWorker
from .agents.base import AgentResponse, AgentWorker
from .config import Config
from .consensus import consensus_score
from .parser import DEFECT_CATEGORIES, ParsedResponse
from .prompts import (
    CAPABILITY_PROMPT,
    SYNTHESIS_PROMPT,
    build_round_prompt,
    render_tasks,
    render_transcript,
    render_roster,
)
from .security import SecurityError, collect_api_keys, redact, safe_session_dir
from .session import SessionMeta, init_session, load_meta, save_meta
from .state import blackboard as bb
from .state import capabilities as caps
from .state import mailbox as mb
from .state import pipeline as pl
from .state import reviews as rv
from .state import tasks as tk
from .state import transcript as ts

CFG = Config.from_env()
CFG.base_dir.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("ultradebate")

ALL_KNOWN_AGENTS = ("claude", "codex", "gemini", "grok")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


from .text_safety import (
    sanitize_text as _sanitize_text,
    sanitize_payload as _sanitize_payload,
    safe_capture_pane as _safe_capture_pane,
    truncate_text_bytes as _truncate_text_bytes,
)


def _json_safe_text(value: str) -> str:
    """Return a UTF-8/JSON-safe string with lone UTF-16 surrogates replaced.

    Canonical implementation lives in :mod:`ultradebate.text_safety` so every
    capture/IPC/return path goes through one sanitizer. This wrapper preserves
    the historical signature: scripts and tests may import either name.
    """
    return _sanitize_text(value)


def _json_safe_obj(value: Any) -> Any:
    """Recursively sanitize objects before JSONL writes or MCP tool returns."""
    return _sanitize_payload(value)


def _json_dumps_safe(value: Any, **kwargs: Any) -> str:
    """json.dumps wrapper that cannot emit lone-surrogate payloads.

    Sanitizes the payload before encoding AND re-sanitizes the resulting JSON
    string. The double pass closes a coverage gap: ``json.dumps`` with
    ``ensure_ascii=False`` can serialize a surrogate that survived earlier
    layers (e.g. an int-keyed dict, a frozen dataclass).
    """
    encoded = json.dumps(_sanitize_payload(value), **kwargs)
    return _sanitize_text(encoded)


def _build_workers(force_mock: bool | None = None) -> dict[str, AgentWorker]:
    """Factory for the four known workers (real or mock)."""
    use_mock = CFG.mock if force_mock is None else force_mock
    if use_mock:
        # Mock supports only the original three personas; grok mock falls back to claude flavor
        return {
            "claude": MockWorker("claude", seed=hash("claude") & 0xFFFF),
            "codex": MockWorker("codex", seed=hash("codex") & 0xFFFF),
            "gemini": MockWorker("gemini", seed=hash("gemini") & 0xFFFF),
            "grok": MockWorker("claude", seed=hash("grok") & 0xFFFF),
        }
    return {
        "claude": ClaudeWorker(CFG.claude_cli, max_policy=CFG.max_policy),
        "codex": CodexWorker(CFG.codex_cli, max_policy=CFG.max_policy),
        "gemini": GeminiWorker(CFG.gemini_cli, max_policy=CFG.max_policy),
        # Grok worker not yet implemented; placeholder mock keeps roster math sane
        "grok": MockWorker("claude", seed=hash("grok-stub") & 0xFFFF),
    }


def _normalize_roster(roster: dict[str, int] | None) -> dict[str, int]:
    """Return a {agent: 0|1} dict for known agents only. Defaults from config."""
    base = dict(CFG.default_roster)
    if roster:
        for k, v in roster.items():
            if k in ALL_KNOWN_AGENTS:
                base[k] = 1 if int(bool(v)) == 1 else 0
    # Ensure all known keys present
    for k in ALL_KNOWN_AGENTS:
        base.setdefault(k, 0)
    return base


def _active_agents(roster: dict[str, int]) -> list[str]:
    return [a for a in ALL_KNOWN_AGENTS if roster.get(a, 0) == 1]


def _required_min(mode: str, active_count: int) -> int:
    if mode == "strict":
        return active_count
    # lenient: ceil(N/2), with minimum of 1
    return max(1, math.ceil(active_count / 2))


def _dedicated_reviewer(round_n: int, active: list[str]) -> str | None:
    """Rotate per round; returns None when policy is off or roster too small."""
    if CFG.dedicated_reviewer != "rotate":
        return None
    if len(active) < 2:
        return None
    return active[(round_n - 1) % len(active)]


def estimate_session_cost(
    roster: dict[str, int], rounds: int, max_mode: bool
) -> dict[str, Any]:
    """Estimate total tokens + USD per design §9.2.

    Per-agent per-round token budget under max-mode is ~20K (input+reasoning+output);
    under default mode ~5K. We compute total = base × active × rounds × overhead(1.2).
    Pricing is intentionally rough — the goal is a guard rail, not invoicing.
    """
    base_per_agent_round = 20_000 if max_mode else 5_000
    active_count = sum(1 for v in roster.values() if v == 1)
    total_tokens = int(base_per_agent_round * active_count * rounds * 1.2)

    # Rough per-agent USD/MT (input+output blended, max-mode):
    pricing_usd_per_million = {"claude": 30, "codex": 60, "gemini": 8, "grok": 15}
    avg = (
        sum(pricing_usd_per_million.get(a, 20) for a, v in roster.items() if v == 1)
        / max(1, active_count)
    )
    cost_usd = round(total_tokens / 1_000_000 * avg, 2)
    return {
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": cost_usd,
        "per_agent_per_round_tokens": base_per_agent_round,
        "active_agents": active_count,
        "rounds": rounds,
        "warn_threshold_usd": CFG.warn_cost_usd,
        "abort_threshold_usd": CFG.abort_cost_usd,
    }


def _reviewer_target(round_n: int, active: list[str], reviewer: str) -> str:
    """Pick the artifact/claim the reviewer focuses on this round.

    Simple deterministic rotation: target = next teammate after reviewer.
    """
    if not active:
        return "the latest blackboard.focus"
    others = [a for a in active if a != reviewer]
    if not others:
        return "your own claims from the previous round (self-review)"
    idx = (round_n - 1) % len(others)
    return f"@{others[idx]}'s claims and any artifacts they produced"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def ultradebate_check_environment() -> dict[str, Any]:
    """Diagnostic: which CLIs are installed, which API keys are set, current config.

    English-only contract: all tool docstrings and outputs are English.
    """
    workers = _build_workers(force_mock=False)
    cli_status: dict[str, bool] = {}
    for n in ("claude", "codex", "gemini"):
        cli_status[n] = await workers[n].is_available()
    keys_present = {
        k: bool(os.environ.get(k))
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    }
    return {
        "clis_available": cli_status,
        "api_keys_set": keys_present,
        "ready_for_strict_mode": all(cli_status.values()),
        "missing_clis": [n for n, ok in cli_status.items() if not ok],
        "config": {
            "base_dir": str(CFG.base_dir),
            "max_rounds": CFG.max_rounds,
            "token_budget": CFG.token_budget,
            "mode": CFG.mode,
            "convergence_delta": CFG.convergence_delta,
            "mock": CFG.mock,
            "max_mode": CFG.max_policy.enabled,
            "default_roster": CFG.default_roster,
            "dedicated_reviewer": CFG.dedicated_reviewer,
            "claude_cli": f"{CFG.claude_cli.binary} {' '.join(CFG.claude_cli.args)}",
            "codex_cli": f"{CFG.codex_cli.binary} {' '.join(CFG.codex_cli.args)}",
            "gemini_cli": f"{CFG.gemini_cli.binary} {' '.join(CFG.gemini_cli.args)}",
        },
    }


async def _create_session(
    *,
    topic: str,
    roster: dict[str, int] | None,
    rounds: int,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"strict", "lenient"}:
        return {"error": f"invalid mode {mode!r} (expected 'strict' or 'lenient')"}
    if not isinstance(topic, str) or not topic.strip():
        return {"error": "topic must be a non-empty string"}
    if len(topic) > 8000:
        return {"error": "topic exceeds 8000 chars"}
    rounds_clamped = max(1, min(int(rounds), 20))

    try:
        collect_api_keys()  # rejects OAuth-shaped tokens
    except SecurityError as e:
        return {"error": f"security check failed: {e}"}

    final_roster = _normalize_roster(roster)
    active = _active_agents(final_roster)
    if not active:
        return {"error": "roster has no active agents (need at least one with value 1)"}

    # R15-B — design §9.2 implementation point.
    # The `cost_estimate` response field returned to the caller below at the
    # session-create boundary (NOT at the per-agent AgentResponse level).
    # `ULTRADEBATE_ABORT_COST_USD` enforcement: the next block compares
    # estimated_cost_usd against abort_threshold_usd and short-circuits with
    # error="cost_abort" if exceeded.
    cost_estimate = estimate_session_cost(
        final_roster, rounds_clamped, max_mode=CFG.max_policy.enabled
    )
    if cost_estimate["estimated_cost_usd"] > cost_estimate["abort_threshold_usd"]:
        return {
            "error": "cost_abort",
            "cost_estimate": cost_estimate,
            "message": (
                f"estimated cost ${cost_estimate['estimated_cost_usd']} exceeds "
                f"abort threshold ${cost_estimate['abort_threshold_usd']}. "
                "Reduce rounds or roster, or raise ULTRADEBATE_ABORT_COST_USD."
            ),
        }

    meta = init_session(
        base_dir=CFG.base_dir,
        topic=topic.strip(),
        mode=mode,
        max_rounds=rounds_clamped,
        token_budget=CFG.token_budget,
    )
    sdir = safe_session_dir(CFG.base_dir, meta.session_id)
    bb.init(
        sdir,
        session_id=meta.session_id,
        topic=meta.topic,
        team_roster=final_roster,
        stage="DEBATE",
    )

    workers = _build_workers()
    availability: dict[str, bool] = {}
    for name in active:
        availability[name] = await workers[name].is_available()

    return {
        "session_id": meta.session_id,
        "topic": meta.topic,
        "mode": mode,
        "max_rounds": meta.max_rounds,
        "token_budget": meta.token_budget,
        "session_dir": str(sdir),
        "team_roster": final_roster,
        "active_agents": active,
        "agents_available": availability,
        "ready_for_strict": (
            all(availability.values()) if mode == "strict" else True
        ),
        "max_mode": CFG.max_policy.enabled,
        "cost_estimate": cost_estimate,
        "next_action": (
            f"call ultradebate_round(session_id={meta.session_id!r}, round_n=1)"
        ),
    }


def _detect_live_codex_tmux() -> dict[str, Any] | None:
    """Safety guard 2026-05-16: detect a live ccb_codex tmux session whose
    pane CWD matches the current working directory.

    Background: ultradebate_session_create / ultradebate_start spawn NEW codex
    CLI subprocesses for debate. If a live `ccb codex` (or similar) tmux pane
    already exists in the same workspace, calling session_create duplicates
    the connection — wastes Codex weekly quota + breaks live coordination
    expectations. This helper detects that situation so the caller can warn
    or abort with `force_new=true` override.

    Returns a dict with detected session name + pane info, or None if no
    matching live codex tmux is found. Fail-closed: any error returns None
    (no false-positive abort).
    """
    import os
    import subprocess
    try:
        cwd = os.environ.get("PWD") or os.getcwd()
        if not cwd:
            return None
        # Enumerate tmux sessions
        ls = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            capture_output=True, text=True, timeout=3,
        )
        if ls.returncode != 0:
            return None
        sessions = [s.strip() for s in ls.stdout.splitlines() if s.strip()]
        for s in sessions:
            name_lower = s.lower()
            if "codex" not in name_lower and "ccb" not in name_lower:
                continue
            panes = subprocess.run(
                ["tmux", "list-panes", "-t", s, "-F",
                 "#{pane_current_command} #{pane_current_path}"],
                capture_output=True, text=True, timeout=3,
            )
            if panes.returncode != 0:
                continue
            for line in panes.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Match: pane CWD matches caller CWD (exact or subdirectory)
                if cwd in line:
                    parts = line.split(" ", 1)
                    return {
                        "tmux_session": s,
                        "pane_cmd": parts[0] if parts else "?",
                        "pane_cwd": parts[1] if len(parts) > 1 else "?",
                        "advice_send": f'ultradebate_v2_send_to_peer (recommended; bracketed-paste + C-m) OR `tmux load-buffer ...; tmux paste-buffer -p -d -t {s}; tmux send-keys -t {s} C-m` — use paste-buffer so newlines flatten (NEVER raw multi-line send-keys, which tmux splits into /n)',
                        "advice_capture": f"tmux capture-pane -t {s} -p",
                    }
    except Exception:
        pass
    return None


@mcp.tool()
async def ultradebate_start(
    topic: str,
    rounds: int = 5,
    mode: str = "lenient",
    roster: dict[str, int] | None = None,
    force_new: bool = False,
) -> dict[str, Any]:
    """Start a new debate session.

    English-only contract: `topic` MUST already be in English. The Claude Code
    Host should translate Korean user input before calling this tool.

    Safety guard (2026-05-16): refuses to start a new debate when a live
    `ccb_codex`-style tmux pane is already running in the same workspace.
    Set `force_new=True` to bypass (rare cases only).

    Args:
        topic: Debate topic — problem statement, design question, or code under review.
        rounds: Maximum number of rounds (1-20). Convergence may end the debate sooner.
        mode: 'strict' (require all active agents) or 'lenient' (require ⌈N/2⌉).
        roster: Map of agent → 0|1. Defaults to config (claude=1, codex=1, gemini=1).
        force_new: bypass the live-codex-tmux safety guard.
    """
    if not force_new:
        live = _detect_live_codex_tmux()
        if live:
            return {
                "error": "live_codex_tmux_detected",
                **live,
                "message": (
                    f"A live Codex tmux session '{live['tmux_session']}' is "
                    f"already running in this workspace (pane cmd "
                    f"'{live['pane_cmd']}', cwd '{live['pane_cwd']}'). "
                    f"ultradebate_start would spawn a SEPARATE codex CLI "
                    f"subprocess, duplicating the connection and wasting "
                    f"weekly quota. Use `ultradebate_v2_send_to_peer` "
                    f"(recommended; bracketed-paste + C-m) or "
                    f"`tmux load-buffer ...; tmux paste-buffer -p -d -t "
                    f"{live['tmux_session']}; tmux send-keys -t "
                    f"{live['tmux_session']} C-m` to talk to "
                    f"the live Codex, or pass force_new=true to bypass this "
                    f"guard (rare cases only)."
                ),
            }
    return await _create_session(topic=topic, roster=roster, rounds=rounds, mode=mode)


@mcp.tool()
async def ultradebate_session_create(
    topic: str,
    roster: dict[str, int] | None = None,
    rounds: int = 5,
    mode: str = "lenient",
    force_new: bool = False,
) -> dict[str, Any]:
    """Identical to ultradebate_start; v2 explicit name.

    English-only contract. Same live-codex-tmux safety guard applies
    (set `force_new=True` to bypass).
    """
    if not force_new:
        live = _detect_live_codex_tmux()
        if live:
            return {
                "error": "live_codex_tmux_detected",
                **live,
                "message": (
                    f"A live Codex tmux session '{live['tmux_session']}' is "
                    f"already running in this workspace (pane cmd "
                    f"'{live['pane_cmd']}', cwd '{live['pane_cwd']}'). "
                    f"ultradebate_session_create would spawn a SEPARATE "
                    f"codex CLI subprocess, duplicating the connection and "
                    f"wasting weekly quota. Use `ultradebate_v2_send_to_peer` "
                    f"(recommended; bracketed-paste + C-m) or "
                    f"`tmux load-buffer ...; tmux paste-buffer -p -d -t "
                    f"{live['tmux_session']}; tmux send-keys -t "
                    f"{live['tmux_session']} C-m` to talk to "
                    f"the live Codex, or pass force_new=true to bypass this "
                    f"guard (rare cases only)."
                ),
            }
    return await _create_session(topic=topic, roster=roster, rounds=rounds, mode=mode)


@mcp.tool()
async def ultradebate_set_roster(
    session_id: str, roster: dict[str, int]
) -> dict[str, Any]:
    """Update which agents participate in subsequent rounds (R7).

    Atomic merge-patch on the blackboard. Already-completed rounds are not
    rewritten.
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}
    if not isinstance(roster, dict) or not roster:
        return {"error": "roster must be a non-empty dict of agent→0|1"}
    set_map = {
        k: 1 if int(bool(v)) == 1 else 0
        for k, v in roster.items()
        if k in ALL_KNOWN_AGENTS
    }
    if not set_map:
        return {"error": "no known agents in roster (claude/codex/gemini/grok)"}
    new_bb = bb.patch(sdir, {"team_roster": {"set": set_map}})
    return {
        "ok": True,
        "session_id": session_id,
        "team_roster": new_bb.team_roster,
        "active_agents": _active_agents(new_bb.team_roster),
        "revision": new_bb.revision,
    }


@mcp.tool()
async def ultradebate_blackboard_read(
    session_id: str, agent: str | None = None
) -> dict[str, Any]:
    """Read the blackboard. If `agent` is provided, returns a role-aware Markdown view.

    English-only contract.
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
        b = bb.load(sdir)
    except (SecurityError, FileNotFoundError) as e:
        return {"error": str(e)}
    out: dict[str, Any] = {
        "session_id": session_id,
        "revision": b.revision,
        "stage": b.stage,
        "team_roster": b.team_roster,
        "raw": bb.to_dict(b),
    }
    if isinstance(agent, str) and agent.strip():
        out["view_for"] = agent
        out["view_text"] = bb.role_aware_view(b, agent)
    return out


@mcp.tool()
async def ultradebate_blackboard_patch(
    session_id: str,
    patch: dict,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Atomic patch with optional revision check (R6/R9).

    Patch grammar: see state/blackboard.py docstring.
    Returns RevisionMismatch error when expected_revision doesn't match.
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}
    if not isinstance(patch, dict):
        return {"error": "patch must be a JSON object"}
    try:
        new_bb = bb.patch(sdir, patch, expected_revision=expected_revision)
    except bb.RevisionMismatch as e:
        return {
            "error": "revision_mismatch",
            "expected": e.expected,
            "actual": e.actual,
        }
    return {
        "ok": True,
        "session_id": session_id,
        "revision": new_bb.revision,
        "stage": new_bb.stage,
    }


@mcp.tool()
async def ultradebate_round(session_id: str, round_n: int) -> dict[str, Any]:
    """Execute one debate round — dispatches to all active roster agents in parallel.

    Anchor-bias mitigation: agents see prior rounds' transcript but NOT each other's
    current-round output. Order of "other_agents" listing is shuffled per agent.
    Dedicated reviewer rotates per round (R11).
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
        meta = load_meta(sdir)
        b = bb.load(sdir)
    except (SecurityError, FileNotFoundError) as e:
        return {"error": f"session not found or invalid: {e}"}

    if meta.completed:
        return {"error": "session already completed", "session_id": session_id}
    if round_n < 1 or round_n > meta.max_rounds:
        return {"error": f"round_n must be 1..{meta.max_rounds}, got {round_n}"}

    workers = _build_workers()
    active = _active_agents(b.team_roster)
    if not active:
        return {"error": "no active agents in roster"}

    reviewer = _dedicated_reviewer(round_n, active)
    target = _reviewer_target(round_n, active, reviewer) if reviewer else None

    transcript_entries = ts.transcript_excluding_round(ts.load_transcript(sdir), round_n)
    tasks_now = tk.list_tasks(sdir)

    catalog_text = caps.render_catalog_for_prompt()
    prompts: dict[str, str] = {}
    for agent in active:
        msgs = mb.fetch_inbox(sdir, agent, mark_processed=True)
        others = [a for a in active if a != agent]
        random.shuffle(others)
        prompts[agent] = build_round_prompt(
            agent_name=agent,
            other_agents=others,
            topic=meta.topic,
            round_n=round_n,
            max_rounds=meta.max_rounds,
            mode=meta.mode,
            mailbox_msgs=msgs,
            transcript_entries=transcript_entries,
            tasks=tasks_now,
            blackboard_view=bb.role_aware_view(b, agent),
            is_dedicated_reviewer=(agent == reviewer),
            reviewer_target=target,
            capability_catalog=catalog_text,
        )

    started = time.monotonic()
    # max-mode reasoning (ultrathink / xhigh) needs generous per-agent budgets;
    # asyncio.gather + return_exceptions makes one slow agent harmless to others.
    # codex with --reasoning-effort xhigh frequently saturates 300s; bump to 480s
    # so we capture the full structured-tag output rather than truncating mid-thought.
    timeouts = {"claude": 300, "codex": 480, "gemini": 180, "grok": 180}
    coros = [
        workers[a].invoke(prompts[a], timeout=timeouts.get(a, 180)) for a in active
    ]
    results: list = await asyncio.gather(*coros, return_exceptions=True)
    round_latency_ms = int((time.monotonic() - started) * 1000)

    responses: dict[str, AgentResponse] = {}
    errors: list[dict] = []
    for agent, res in zip(active, results):
        if isinstance(res, BaseException):
            errors.append(
                {"agent": agent, "error": f"exception: {type(res).__name__}: {res}"}
            )
            continue
        if not res.ok:
            errors.append({"agent": agent, "error": redact(res.error or "unknown")})
            continue
        responses[agent] = res

    required = _required_min(meta.mode, len(active))
    if len(responses) < required:
        return {
            "error": f"{meta.mode} mode requires ≥{required} of {len(active)} active agents; "
                     f"got {len(responses)}",
            "errors": errors,
            "session_id": session_id,
            "round": round_n,
            "responses_received": list(responses),
        }

    # Persist transcript
    for agent, res in responses.items():
        ts.append_entry(
            sdir,
            ts.TranscriptEntry(
                round=round_n,
                speaker=agent,
                analysis=res.parsed.analysis,
                vote_position=res.parsed.vote.position,
                vote_confidence=res.parsed.vote.confidence,
                tokens_in=res.tokens_in,
                tokens_out=res.tokens_out,
                latency_ms=res.latency_ms,
                parse_warnings=res.parsed.parse_warnings,
            ),
        )

    # Route P2P messages
    routed_count = 0
    defect_routes = 0
    ack_routes = 0
    for sender, res in responses.items():
        for outgoing in res.parsed.messages:
            if outgoing.to in active and outgoing.to != sender:
                content = _format_message_for_inbox(outgoing)
                mb.deliver(sdir, sender, outgoing.to, content, round_n)
                routed_count += 1
                if outgoing.is_defect:
                    defect_routes += 1
                if outgoing.is_ack:
                    ack_routes += 1

    # Apply task actions
    task_events: list[dict] = []
    for sender, res in responses.items():
        for ta in res.parsed.tasks:
            if ta.action == "add" and ta.title:
                tid = tk.add_task(sdir, ta.title, sender)
                task_events.append({"agent": sender, "action": "add", "id": tid, "title": ta.title})
            elif ta.action == "claim" and ta.id:
                ok = tk.claim_task(sdir, sender, ta.id)
                task_events.append({"agent": sender, "action": "claim", "id": ta.id, "ok": ok})
            elif ta.action == "complete" and ta.id:
                ok = tk.complete_task(sdir, sender, ta.id)
                task_events.append({"agent": sender, "action": "complete", "id": ta.id, "ok": ok})
            elif ta.action == "block" and ta.id:
                ok = tk.block_task(sdir, sender, ta.id)
                task_events.append({"agent": sender, "action": "block", "id": ta.id, "ok": ok})

    # Apply blackboard patches
    bb_revisions: list[dict] = []
    current_b = b
    for sender, res in responses.items():
        if res.parsed.blackboard_patch:
            try:
                current_b = bb.patch(sdir, res.parsed.blackboard_patch)
                bb_revisions.append(
                    {"agent": sender, "revision": current_b.revision, "ok": True}
                )
            except (TypeError, ValueError, OSError) as e:  # noqa: PERF203
                bb_revisions.append(
                    {"agent": sender, "ok": False, "error": str(e)}
                )

    # Apply capability requests (R14) — invoke provider, route response to requester's mailbox
    capability_events: list[dict] = []
    for sender, res in responses.items():
        cap_req = res.parsed.capability_request
        if cap_req is None:
            continue
        # Stash rationale into args under reserved __rationale__ key so the
        # internal helper can pull it for prompt formatting without polluting
        # the visible args block.
        args_with_rationale = {**cap_req.args, "__rationale__": cap_req.rationale}
        evt = await _invoke_capability_internal(
            sdir=sdir,
            requester=sender,
            capability_id=cap_req.capability,
            args=args_with_rationale,
            current_round=round_n,
            workers=workers,
        )
        capability_events.append(evt)

    # Consensus
    parsed_list = [r.parsed for r in responses.values()]
    cons = consensus_score(
        parsed_list,
        prev_score=meta.last_consensus,
        delta_threshold=CFG.convergence_delta,
    )

    tokens_this_round = sum(r.tokens_in + r.tokens_out for r in responses.values())
    meta.current_round = round_n
    meta.tokens_used += tokens_this_round
    meta.last_consensus = cons.score
    meta.consensus_history.append(round(cons.score, 4))
    save_meta(sdir, meta)

    over_budget = meta.tokens_used >= meta.token_budget
    last_round = round_n >= meta.max_rounds
    should_stop = cons.converged or over_budget or last_round

    next_action = (
        f"call ultradebate_finalize(session_id={session_id!r})"
        if should_stop
        else f"call ultradebate_round(session_id={session_id!r}, round_n={round_n + 1})"
    )

    return {
        "session_id": session_id,
        "round": round_n,
        "round_latency_ms": round_latency_ms,
        "active_agents": active,
        "dedicated_reviewer": reviewer,
        "reviewer_target": target,
        "responses": {
            agent: _summarize_response(res)
            for agent, res in responses.items()
        },
        "consensus": cons.to_dict(),
        "messages_routed": routed_count,
        "defect_messages": defect_routes,
        "ack_messages": ack_routes,
        "task_events": task_events,
        "blackboard_revisions": bb_revisions,
        "capability_events": capability_events,
        "errors": errors,
        "tokens_used": meta.tokens_used,
        "tokens_budget": meta.token_budget,
        "should_stop": should_stop,
        "stop_reason": (
            "converged" if cons.converged
            else "over_budget" if over_budget
            else "last_round" if last_round
            else None
        ),
        "next_action": next_action,
    }


async def _invoke_capability_internal(
    *,
    sdir: Path,
    requester: str,
    capability_id: str,
    args: dict,
    current_round: int,
    workers: dict[str, AgentWorker],
) -> dict:
    """Shared core for in-round capability_request and the public tool (R14)."""
    cap = caps.get(capability_id)
    if cap is None:
        return {
            "ok": False,
            "requester": requester,
            "capability_id": capability_id,
            "error": f"unknown capability {capability_id!r} (not in whitelist)",
        }
    ok, reason = caps.validate_args(cap, args)
    if not ok:
        return {
            "ok": False,
            "requester": requester,
            "capability_id": capability_id,
            "error": f"invalid args: {reason}",
        }
    # Pull rationale from the parser-provided CapabilityRequest if available.
    # Direct invocations via the public tool may pass a "rationale" key in args
    # that we strip out before formatting (so it doesn't double-appear in the
    # args block). Default empty string when not provided.
    rationale_text = (
        str(args.pop("__rationale__", "")) if isinstance(args, dict) else ""
    )
    prompt = CAPABILITY_PROMPT.format(
        capability_id=cap.id,
        capability_description=cap.description,
        provider=cap.provider,
        requester=requester,
        rationale=rationale_text or "(no rationale supplied)",
        args_json=json.dumps(args, ensure_ascii=False, indent=2),
        output_summary=cap.output_summary,
    )
    provider_worker = workers.get(cap.provider)
    if provider_worker is None:
        return {
            "ok": False,
            "requester": requester,
            "capability_id": capability_id,
            "error": f"provider {cap.provider!r} worker unavailable",
        }
    res = await provider_worker.invoke(prompt, timeout=240)
    if not res.ok:
        return {
            "ok": False,
            "requester": requester,
            "capability_id": capability_id,
            "provider": cap.provider,
            "error": redact(res.error or "unknown error"),
        }
    delivery_content = (
        f"[CAPABILITY:{capability_id}] response from @{cap.provider} "
        f"→ @{requester}\n\n{res.parsed.analysis[:4000]}"
    )
    mb.deliver(sdir, cap.provider, requester, delivery_content, current_round)
    return {
        "ok": True,
        "requester": requester,
        "provider": cap.provider,
        "capability_id": capability_id,
        "tokens_in": res.tokens_in,
        "tokens_out": res.tokens_out,
        "latency_ms": res.latency_ms,
        "delivered_to_inbox": True,
    }


def _format_message_for_inbox(msg) -> str:
    """Render an OutgoingMessage as an inbox content string preserving v2 structure."""
    parts = [f"[{msg.category}]"]
    if msg.cite_path:
        cite = msg.cite_path + (f":{msg.cite_lines}" if msg.cite_lines else "")
        parts.append(cite)
    parts.append(msg.content)
    if msg.proposal:
        parts.append(f"PROPOSAL: {msg.proposal}")
    return " ".join(parts)


def _summarize_response(res: AgentResponse) -> dict:
    """Compact per-response payload for the round result."""
    parsed: ParsedResponse = res.parsed
    msg_breakdown: dict[str, int] = {}
    for m in parsed.messages:
        msg_breakdown[m.category] = msg_breakdown.get(m.category, 0) + 1
    return {
        "analysis_preview": parsed.analysis[:1500],
        "analysis_chars": len(parsed.analysis),
        "vote": {
            "position": parsed.vote.position,
            "confidence": parsed.vote.confidence,
        },
        "messages_sent": len(parsed.messages),
        "message_categories": msg_breakdown,
        "vague_messages": sum(1 for m in parsed.messages if m.vague),
        "task_actions_count": len(parsed.tasks),
        "blackboard_patch_present": parsed.blackboard_patch is not None,
        "parse_warnings": parsed.parse_warnings,
        "latency_ms": res.latency_ms,
        "tokens": {"in": res.tokens_in, "out": res.tokens_out},
    }


@mcp.tool()
async def ultradebate_invoke_capability(
    session_id: str,
    requester: str,
    capability_id: str,
    args: dict | None = None,
) -> dict[str, Any]:
    """Invoke a registered capability (R14 — Cross-MCP Capability Bridge).

    The provider runs the capability; their response is delivered to the
    requester's inbox so the next round's prompt sees it. Args must satisfy
    the capability's input_keys whitelist; unknown capability_ids are rejected.
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
        meta = load_meta(sdir)
    except (SecurityError, FileNotFoundError) as e:
        return {"error": str(e)}
    if requester not in ALL_KNOWN_AGENTS:
        return {"error": f"invalid requester {requester!r}"}
    workers = _build_workers()
    return await _invoke_capability_internal(
        sdir=sdir,
        requester=requester,
        capability_id=capability_id,
        args=args or {},
        current_round=meta.current_round,
        workers=workers,
    )


@mcp.tool()
async def ultradebate_capabilities() -> dict[str, Any]:
    """List the whitelisted capabilities (R14)."""
    return {"capabilities": caps.list_capabilities()}


@mcp.tool()
async def ultradebate_run_pipeline(session_id: str, max_loops: int = 1) -> dict[str, Any]:
    """Advance the 5-stage pipeline (R4 — design §3.2).

    v1.0 scope: only DEBATE is wired with prompt+round logic. The other stages
    (RESEARCH/PLAN/EXECUTE/CROSS-REVIEW) advance immediately as scaffolds so
    the state machine is observable end-to-end. Full stage handlers land in v1.1.
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}
    transitions: list[dict] = []
    for _ in range(max(1, int(max_loops))):
        result = pl.advance(sdir)
        transitions.append({
            "from": result.completed_stage,
            "to": result.next_stage,
            "note": pl.stage_description(result.next_stage),
        })
        if result.next_stage == "DONE":
            break
    return {
        "ok": True,
        "session_id": session_id,
        "transitions": transitions,
        "current_stage": pl.current_stage(sdir),
    }


@mcp.tool()
async def ultradebate_pipeline_stage(session_id: str) -> dict[str, Any]:
    """Read the current pipeline stage (R4)."""
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}
    stage = pl.current_stage(sdir)
    return {
        "session_id": session_id,
        "stage": stage,
        "description": pl.stage_description(stage),
        "next_stage": pl.NEXT_STAGE.get(stage, "DONE"),
    }


@mcp.tool()
async def ultradebate_review(
    session_id: str,
    reviewer: str,
    reviewee: str,
    artifact_path: str,
) -> dict[str, Any]:
    """R12 cross-review — `reviewer` evaluates `reviewee`'s artifact via the
    6-category matrix and saves a ReviewResult (design §5.9).

    The reviewer worker is invoked with a focused prompt; the response is
    parsed for category-tagged messages (UNIMPL/OMIT/UNCONN/PARTIAL/APPROACH/
    CLARIFY) and persisted under `{session}/reviews/{id}.json`.
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}
    if reviewer not in ALL_KNOWN_AGENTS:
        return {"error": f"invalid reviewer {reviewer!r}"}
    if reviewee not in ALL_KNOWN_AGENTS:
        return {"error": f"invalid reviewee {reviewee!r}"}
    if reviewer == reviewee:
        return {"error": "reviewer and reviewee must differ"}
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        return {"error": "artifact_path must be non-empty string"}

    review = rv.make_review(
        reviewer=reviewer, reviewee=reviewee, artifact_path=artifact_path.strip()
    )
    workers = _build_workers()
    review_prompt = (
        "ultrathink\n\n"
        f"You are reviewing @{reviewee}'s artifact at `{artifact_path}` "
        "for the ultradebate session. Apply the 6-category defect matrix:\n"
        "  UNIMPL — empty bodies / TODO / pass / NotImplementedError\n"
        "  OMIT   — sub-tasks not addressed; spec items missing\n"
        "  UNCONN — undefined imports, dangling refs, broken wiring\n"
        "  PARTIAL — happy-path only, missing edge cases\n"
        "  APPROACH — better alternative the author missed\n"
        "  CLARIFY — genuinely unclear claim — ASK\n\n"
        "Output JSON ONLY (no prose):\n"
        '{"omissions": [...], "wrong_links": [...], "unimplemented": [...], '
        '"spec_drift": [...], "overall_pass": bool, "severity": "blocking|major|minor"}'
    )
    res = await workers[reviewer].invoke(review_prompt, timeout=240)
    if not res.ok:
        return {"error": f"reviewer call failed: {redact(res.error or '')}"}

    # Best-effort JSON parse of reviewer's response
    body = res.parsed.analysis or res.raw or ""
    parsed_payload: dict | None = None
    try:
        # Try to extract a balanced {...} object
        start = body.find("{")
        end = body.rfind("}")
        if 0 <= start < end:
            parsed_payload = json.loads(body[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        parsed_payload = None

    if isinstance(parsed_payload, dict):
        review.omissions = list(parsed_payload.get("omissions") or [])
        review.wrong_links = list(parsed_payload.get("wrong_links") or [])
        review.unimplemented = list(parsed_payload.get("unimplemented") or [])
        review.spec_drift = list(parsed_payload.get("spec_drift") or [])
        review.overall_pass = bool(parsed_payload.get("overall_pass", True))
        review.severity = str(parsed_payload.get("severity", "minor"))

    saved_path = rv.save(sdir, review)
    return {
        "ok": True,
        "review_id": review.id,
        "saved_to": str(saved_path),
        "overall_pass": review.overall_pass,
        "severity": review.severity,
        "defect_counts": {
            "omissions": len(review.omissions),
            "wrong_links": len(review.wrong_links),
            "unimplemented": len(review.unimplemented),
            "spec_drift": len(review.spec_drift),
        },
        "tokens": {"in": res.tokens_in, "out": res.tokens_out},
    }


@mcp.tool()
async def ultradebate_list_reviews(session_id: str) -> dict[str, Any]:
    """List all stored cross-reviews for a session (R12)."""
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}
    items = rv.load_all(sdir)
    return {
        "session_id": session_id,
        "count": len(items),
        "reviews": [
            {
                "id": r.id,
                "reviewer": r.reviewer,
                "reviewee": r.reviewee,
                "artifact_path": r.artifact_path,
                "overall_pass": r.overall_pass,
                "severity": r.severity,
                "defect_counts": {
                    "omissions": len(r.omissions),
                    "wrong_links": len(r.wrong_links),
                    "unimplemented": len(r.unimplemented),
                    "spec_drift": len(r.spec_drift),
                },
            }
            for r in items
        ],
    }


@mcp.tool()
async def ultradebate_send(
    session_id: str, from_agent: str, to_agent: str, content: str
) -> dict[str, Any]:
    """Inject a P2P message into the mailbox (debug/intervention)."""
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}
    if from_agent not in {*ALL_KNOWN_AGENTS, "host"}:
        return {"error": f"invalid from_agent {from_agent!r}"}
    if to_agent not in ALL_KNOWN_AGENTS:
        return {"error": f"invalid to_agent {to_agent!r}"}
    if not isinstance(content, str) or not content.strip():
        return {"error": "content must be non-empty string"}
    meta = load_meta(sdir)
    mb.deliver(sdir, from_agent, to_agent, content.strip(), meta.current_round)
    return {"ok": True, "from": from_agent, "to": to_agent, "round": meta.current_round}


@mcp.tool()
async def ultradebate_claim_task(
    session_id: str, agent: str, task_id: str
) -> dict[str, Any]:
    """Atomically claim a pending task. Returns ok=False if already claimed."""
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}
    if agent not in ALL_KNOWN_AGENTS:
        return {"error": f"invalid agent {agent!r}"}
    ok = tk.claim_task(sdir, agent, task_id)
    return {"ok": ok, "task_id": task_id, "agent": agent}


@mcp.tool()
async def ultradebate_consensus(session_id: str) -> dict[str, Any]:
    """Read-only consensus snapshot."""
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
        meta = load_meta(sdir)
    except (SecurityError, FileNotFoundError) as e:
        return {"error": str(e)}
    return {
        "session_id": session_id,
        "current_round": meta.current_round,
        "max_rounds": meta.max_rounds,
        "last_consensus": meta.last_consensus,
        "consensus_history": meta.consensus_history,
        "tokens_used": meta.tokens_used,
        "tokens_budget": meta.token_budget,
        "completed": meta.completed,
    }


async def _generate_synthesis(sdir: Path, meta: SessionMeta, b: bb.Blackboard) -> AgentResponse:
    """Run the moderator synthesis call. Uses Claude (or its mock when in mock mode)."""
    entries = ts.load_transcript(sdir)
    tasks = tk.list_tasks(sdir)
    full = render_transcript(entries, max_chars=24000)
    prompt = SYNTHESIS_PROMPT.format(
        topic=meta.topic,
        roster_section=render_roster(b.team_roster),
        full_transcript=full,
        final_tasks=render_tasks(tasks),
        consensus_history=", ".join(
            f"r{i + 1}={s:.3f}" for i, s in enumerate(meta.consensus_history)
        ) or "(none)",
    )
    workers = _build_workers()
    # Synthesis prompt is large + max-mode = give claude a generous budget
    return await workers["claude"].invoke(prompt, timeout=360)


@mcp.tool()
async def ultradebate_synthesize(session_id: str) -> dict[str, Any]:
    """Generate final synthesis report (no archive). Use ultradebate_finalize to also archive."""
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
        meta = load_meta(sdir)
        b = bb.load(sdir)
    except (SecurityError, FileNotFoundError) as e:
        return {"error": str(e)}

    res = await _generate_synthesis(sdir, meta, b)
    meta.completed = True
    save_meta(sdir, meta)

    tasks = tk.list_tasks(sdir)
    return {
        "session_id": session_id,
        "topic": meta.topic,
        "rounds_completed": meta.current_round,
        "final_consensus": meta.last_consensus,
        "consensus_history": meta.consensus_history,
        "synthesis": res.parsed.analysis if res.ok else f"[synthesis error: {redact(res.error or '')}]",
        "tasks_summary": [
            {"id": t.id, "title": t.title, "status": t.status, "owner": t.owner}
            for t in tasks
        ],
        "tokens_total": meta.tokens_used + (res.tokens_in or 0) + (res.tokens_out or 0),
        "synthesis_error": res.error,
    }


@mcp.tool()
async def ultradebate_finalize(session_id: str) -> dict[str, Any]:
    """Synthesize the final report AND archive transcript+mailbox to archive/{ts}/.

    After this, the session directory keeps only `meta.json`, `blackboard.json`,
    and the `archive/` history. `transcript.jsonl` and `mailbox/` are reset so a
    follow-up debate starts fresh — token-economy guard from R13.
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
        meta = load_meta(sdir)
        b = bb.load(sdir)
    except (SecurityError, FileNotFoundError) as e:
        return {"error": str(e)}

    res = await _generate_synthesis(sdir, meta, b)
    meta.completed = True
    save_meta(sdir, meta)

    ts_dir = sdir / "archive" / str(int(time.time()))
    ts_dir.mkdir(parents=True, exist_ok=True)

    report_text = (
        res.parsed.analysis
        if res.ok else f"[synthesis error: {redact(res.error or '')}]"
    )
    (ts_dir / "final_report.md").write_text(report_text, encoding="utf-8")
    (ts_dir / "blackboard.snapshot.json").write_text(
        json.dumps(bb.to_dict(b), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ts_dir / "consensus.json").write_text(
        json.dumps(
            {
                "history": meta.consensus_history,
                "final": meta.last_consensus,
                "tokens_used": meta.tokens_used,
                "rounds": meta.current_round,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    transcript_path = sdir / "transcript.jsonl"
    if transcript_path.exists():
        shutil.move(str(transcript_path), str(ts_dir / "transcript.jsonl"))
    mailbox_dir = sdir / "mailbox"
    if mailbox_dir.exists():
        shutil.move(str(mailbox_dir), str(ts_dir / "mailbox"))

    # Reset live state for next debate cycle
    transcript_path.write_text("", encoding="utf-8")
    (sdir / "mailbox" / "inbox").mkdir(parents=True, exist_ok=True)
    (sdir / "mailbox" / "processed").mkdir(parents=True, exist_ok=True)

    tasks = tk.list_tasks(sdir)
    return {
        "ok": True,
        "session_id": session_id,
        "topic": meta.topic,
        "rounds_completed": meta.current_round,
        "final_consensus": meta.last_consensus,
        "consensus_history": meta.consensus_history,
        "archive_dir": str(ts_dir),
        "report_path": str(ts_dir / "final_report.md"),
        "synthesis_preview": report_text[:1500],
        "tasks_summary": [
            {"id": t.id, "title": t.title, "status": t.status, "owner": t.owner}
            for t in tasks
        ],
        "tokens_total": meta.tokens_used + (res.tokens_in or 0) + (res.tokens_out or 0),
        "synthesis_error": res.error,
    }


@mcp.tool()
async def ultradebate_cleanup(session_id: str, force: bool = False) -> dict[str, Any]:
    """Graceful shutdown. With force=True, also removes the session directory."""
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
    except SecurityError as e:
        return {"error": str(e)}

    workers = _build_workers()
    for w in workers.values():
        try:
            await w.shutdown()
        except Exception:  # noqa: BLE001
            pass

    if force:
        shutil.rmtree(sdir, ignore_errors=True)
        return {"ok": True, "removed": True, "session_dir": str(sdir)}
    return {"ok": True, "removed": False, "session_dir": str(sdir)}


@mcp.tool()
async def ultradebate_status(session_id: str) -> dict[str, Any]:
    """Comprehensive status: meta + transcript count + tasks + inbox sizes + blackboard."""
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
        meta = load_meta(sdir)
        b = bb.load(sdir)
    except (SecurityError, FileNotFoundError) as e:
        return {"error": str(e)}

    entries = ts.load_transcript(sdir)
    tasks = tk.list_tasks(sdir)
    active = _active_agents(b.team_roster)
    inbox_sizes = {a: mb.peek_inbox(sdir, a) for a in active}

    return {
        "session_id": session_id,
        "topic": meta.topic,
        "mode": meta.mode,
        "current_round": meta.current_round,
        "max_rounds": meta.max_rounds,
        "tokens_used": meta.tokens_used,
        "tokens_budget": meta.token_budget,
        "last_consensus": meta.last_consensus,
        "consensus_history": meta.consensus_history,
        "completed": meta.completed,
        "transcript_entries": len(entries),
        "team_roster": b.team_roster,
        "active_agents": active,
        "blackboard_revision": b.revision,
        "blackboard_stage": b.stage,
        "tasks": [
            {"id": t.id, "title": t.title, "status": t.status, "owner": t.owner}
            for t in tasks
        ],
        "inbox_sizes": inbox_sizes,
    }


# Re-export for tests / external callers
DEFECT_CATEGORIES_EXPORT = DEFECT_CATEGORIES


# ============================================================================
# ULTRADEBATE V1.1 — Auto-Debate Orchestration (added 2026-05-09)
#
# Three new tools that compose existing primitives to satisfy the user mandate
# of fully-automatic Claude+Codex debate with role assignment + parallel work +
# cross-verification + consensus check, replacing manual chained ask/wakeup
# polling. Pure addition — does NOT modify existing tool semantics.
# ============================================================================


@mcp.tool()
async def ultradebate_auto_debate(
    topic: str,
    max_rounds: int = 3,
    rounds: int | None = None,
    roster: dict[str, int] | None = None,
    mode: str = "lenient",
    finalize: bool = True,
    early_stop_on_converge: bool = True,
) -> dict[str, Any]:
    """End-to-end debate: session_create → loop rounds until stop → synthesize → finalize.

    Replaces manual chained calls (session_create + N×round + synthesize + finalize)
    with a single orchestrated tool. Loops rounds until `should_stop` is true
    (converged / over_budget / last_round). Optionally finalizes (archives transcript).

    Args:
        topic: Debate topic (English contract).
        max_rounds: Maximum number of rounds (default 3, capped by config).
        rounds: Alias for max_rounds (kept for parity with session_create).
        roster: Override default roster, e.g. {"claude": 1, "codex": 1, "gemini": 0}.
        mode: "lenient" (≥ceil(N/2)) or "strict" (all agents required).
        finalize: If True, archive transcript+mailbox via ultradebate_finalize at end.
        early_stop_on_converge: If True, stop on first converged round; if False,
            run all max_rounds even after convergence (for thoroughness).

    Returns:
        {
            session_id, topic, rounds_completed, max_rounds,
            stop_reason, converged, consensus_history, last_consensus_score,
            per_round_summaries: [{round, consensus_score, converged, ...}, ...],
            synthesis: {...}, archive: {...} | None,
        }
    """
    if rounds is not None:
        max_rounds = rounds

    sess = await ultradebate_session_create(
        topic=topic,
        rounds=max_rounds,
        roster=roster,
        mode=mode,
    )
    if "error" in sess:
        return {"error": f"session_create failed: {sess['error']}", "stage": "session_create"}
    sid = sess["session_id"]

    per_round: list[dict[str, Any]] = []
    rounds_done = 0
    stop_reason: str | None = None
    last_score: float | None = None
    consensus_hist: list[float] = []

    for n in range(1, max_rounds + 1):
        round_result = await ultradebate_round(sid, n)
        rounds_done = n
        if "error" in round_result:
            stop_reason = f"round_{n}_error"
            per_round.append({
                "round": n, "error": round_result["error"],
                "responses_received": round_result.get("responses_received"),
            })
            break

        cons_dict = round_result.get("consensus", {}) or {}
        last_score = cons_dict.get("score")
        if last_score is not None:
            consensus_hist.append(last_score)

        per_round.append({
            "round": n,
            "consensus_score": last_score,
            "converged": cons_dict.get("converged"),
            "active_agents": round_result.get("active_agents"),
            "messages_routed": round_result.get("messages_routed", 0),
            "errors": round_result.get("errors", []),
            "tokens_used": round_result.get("tokens_used"),
        })

        if round_result.get("should_stop"):
            stop_reason = round_result.get("stop_reason")
            if stop_reason == "converged" and not early_stop_on_converge:
                stop_reason = None
                continue
            break

    synth = await ultradebate_synthesize(sid)

    archive: dict | None = None
    if finalize:
        archive = await ultradebate_finalize(sid)

    return {
        "session_id": sid,
        "topic": topic,
        "rounds_completed": rounds_done,
        "max_rounds": max_rounds,
        "stop_reason": stop_reason,
        "converged": stop_reason == "converged",
        "consensus_history": consensus_hist,
        "last_consensus_score": last_score,
        "per_round_summaries": per_round,
        "synthesis": synth,
        "archive": archive,
    }


@mcp.tool()
async def ultradebate_cross_verify(
    session_id: str,
    round_n: int,
    verifier_pairs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Cross-verification: each agent's round-N output is reviewed by the other agent(s).

    For roster {claude, codex}: claude reviews codex's analysis, codex reviews claude's.
    Parallel dispatch via asyncio.gather for low total latency.

    Args:
        session_id: Existing session id.
        round_n: Round whose outputs to review.
        verifier_pairs: Optional explicit map {reviewer: "target1,target2"}.
            Default: symmetric all-to-all (each agent reviews every other agent).

    Returns:
        {
            session_id, round_reviewed, pairs_verified, latency_ms,
            review_matrix: [{reviewer, target, ok, raw_text, parsed_analysis, ...}, ...]
        }
    """
    try:
        sdir = safe_session_dir(CFG.base_dir, session_id)
        meta = load_meta(sdir)
        b = bb.load(sdir)
    except (SecurityError, FileNotFoundError) as e:
        return {"error": f"session not found or invalid: {e}"}

    transcript = ts.load_transcript(sdir)
    round_entries = [e for e in transcript if e.round == round_n]
    if not round_entries:
        return {"error": f"no transcript entries for round {round_n}"}

    active = _active_agents(b.team_roster)
    workers = _build_workers()

    if verifier_pairs is None:
        verifier_pairs = {
            reviewer: ",".join(a for a in active if a != reviewer)
            for reviewer in active
        }

    review_coros = []
    pair_meta: list[tuple[str, str]] = []

    for reviewer, targets_csv in verifier_pairs.items():
        if reviewer not in active:
            continue
        targets = [
            t.strip() for t in (targets_csv.split(",") if isinstance(targets_csv, str) else [targets_csv])
            if t.strip()
        ]
        for target in targets:
            target_entry = next((e for e in round_entries if e.speaker == target), None)
            if not target_entry:
                continue
            prompt = (
                f"You are {reviewer} performing CROSS-VERIFY review of {target}'s "
                f"output from round {round_n}.\n\n"
                f"Topic: {meta.topic}\n\n"
                f"=== {target} round {round_n} analysis ===\n"
                f"{target_entry.analysis}\n\n"
                f"Score on rubrics: correctness, completeness, soundness, "
                f"risk_calibration. Each 1-10. Return JSON: "
                f'{{"reviewer": "{reviewer}", "target": "{target}", '
                f'"scores": {{...}}, "overall": N, "verdict": "PASS|FAIL", '
                f'"notes": "..."}}'
            )
            timeouts = {"claude": 240, "codex": 360, "gemini": 180, "grok": 180}
            review_coros.append(
                workers[reviewer].invoke(prompt, timeout=timeouts.get(reviewer, 180))
            )
            pair_meta.append((reviewer, target))

    if not review_coros:
        return {"error": "no verifier pairs to dispatch"}

    started = time.monotonic()
    results = await asyncio.gather(*review_coros, return_exceptions=True)
    latency_ms = int((time.monotonic() - started) * 1000)

    review_matrix: list[dict] = []
    for (reviewer, target), res in zip(pair_meta, results):
        if isinstance(res, BaseException):
            review_matrix.append({
                "reviewer": reviewer, "target": target,
                "error": f"{type(res).__name__}: {res}",
            })
            continue
        review_matrix.append({
            "reviewer": reviewer, "target": target,
            "ok": res.ok,
            "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
            "latency_ms": res.latency_ms,
            "raw_text": (res.raw_text[:1500] if res.raw_text else ""),
            "parsed_analysis": (res.parsed.analysis[:1500] if res.parsed else ""),
        })

    return {
        "session_id": session_id,
        "round_reviewed": round_n,
        "pairs_verified": len(review_matrix),
        "latency_ms": latency_ms,
        "review_matrix": review_matrix,
    }


@mcp.tool()
async def ultradebate_parallel_tasks(
    tasks: list[dict[str, Any]],
    timeout_per_task_sec: int = 240,
) -> dict[str, Any]:
    """Dispatch multiple INDEPENDENT tasks in parallel to specified agents.

    Each task: {"agent": "claude|codex|gemini|grok", "prompt": "...", "task_id": "optional"}

    Use cases:
    - Role-split research (claude:designA + codex:goalB run simultaneously)
    - Multi-file review (claude:fileX + codex:fileY)
    - Parallel implementation (claude:moduleA + codex:moduleB)

    Returns: {tasks_dispatched, tasks_skipped, total_latency_ms, results: [...]}
    """
    if not tasks:
        return {"error": "tasks list is empty"}

    workers = _build_workers()
    coros: list = []
    task_meta: list[dict] = []
    for i, t in enumerate(tasks):
        agent = t.get("agent")
        prompt = t.get("prompt")
        tid = t.get("task_id", f"par_{i}")
        if not agent or agent not in workers:
            coros.append(asyncio.sleep(0))
            task_meta.append({
                "task_id": tid, "agent": agent, "skipped": True,
                "error": f"unknown agent: {agent}",
            })
            continue
        if not prompt:
            coros.append(asyncio.sleep(0))
            task_meta.append({
                "task_id": tid, "agent": agent, "skipped": True,
                "error": "empty prompt",
            })
            continue
        coros.append(workers[agent].invoke(prompt, timeout=timeout_per_task_sec))
        task_meta.append({"task_id": tid, "agent": agent, "skipped": False})

    started = time.monotonic()
    results = await asyncio.gather(*coros, return_exceptions=True)
    total_latency_ms = int((time.monotonic() - started) * 1000)

    out: list[dict] = []
    for meta_t, res in zip(task_meta, results):
        if meta_t.get("skipped"):
            out.append(meta_t)
            continue
        if isinstance(res, BaseException):
            out.append({
                "task_id": meta_t["task_id"], "agent": meta_t["agent"],
                "ok": False,
                "error": f"{type(res).__name__}: {res}",
            })
            continue
        out.append({
            "task_id": meta_t["task_id"], "agent": meta_t["agent"],
            "ok": res.ok,
            "raw_text": (res.raw_text[:2000] if res.raw_text else ""),
            "tokens_in": res.tokens_in, "tokens_out": res.tokens_out,
            "latency_ms": res.latency_ms,
            "error": redact(res.error) if (res.error and not res.ok) else None,
        })

    return {
        "tasks_dispatched": sum(1 for t in task_meta if not t.get("skipped")),
        "tasks_skipped": sum(1 for t in task_meta if t.get("skipped")),
        "total_latency_ms": total_latency_ms,
        "results": out,
    }


# ============================================================================
# ULTRADEBATE V2 — Bidirectional CLI Bridge (Option C, 2026-05-10)
#
# Per user mandate: 양쪽 CLI 모두 ultradebate MCP 사용 + 양방향 메시지 전달 +
# cross-tool commands (/compact, /sc:*) + 즉각 응답 (현재처럼 응답 도착 모르는 일 없게).
#
# Inbox/outbox: /tmp/ultradebate_v2/{claude,codex}/{inbox,outbox,processed,errored}
# ============================================================================

import uuid as _v2_uuid
from datetime import datetime as _v2_dt

_V2_BASE = Path("/tmp/ultradebate_v2")
_V2_PEERS = ("claude", "codex", "gemini")


def _v2_inbox(peer: str) -> Path:
    return _V2_BASE / peer / "inbox"


def _v2_outbox(peer: str) -> Path:
    return _V2_BASE / peer / "outbox"


def _v2_processed(peer: str) -> Path:
    return _V2_BASE / peer / "processed"


def _v2_ensure_dirs(peer: str) -> None:
    for sub in ("inbox", "outbox", "processed", "errored"):
        (_V2_BASE / peer / sub).mkdir(parents=True, exist_ok=True)


def _v2_mid() -> str:
    return f"{_v2_dt.utcnow().strftime('%Y%m%dT%H%M%S')}-{_v2_uuid.uuid4().hex[:8]}"


# Per-peer tmux submit key sequence.
# Codex CLI: `C-m` (Ctrl+M) = the user's keyboard Enter — it submits when Codex
# is idle and queues the payload as Codex's next task when busy (user-verified
# 2026-05-24). The 2026-05-18 "C-m inserts a newline" finding was a misdiagnosis:
# that /n came from RAW multi-line send-keys (tmux splits embedded \n into
# Enters), NOT from C-m. With the paste-buffer path below (flatten → one
# bracketed-paste atom) + a single C-m there is no split. A brief Escape+Enter
# experiment was REVERTED per user mandate ("C-m을 확실하게 강제 … 이상한 분기가
# 생기거나 하지 않게"). Keep central so future CLI-version tests change one constant.
# Claude/Gemini entries are historical metadata; tmux delivery below uses the
# safe paste-buffer path for all peers.
_V2_SUBMIT_KEYS: dict[str, list[str]] = {
    "claude": ["Enter"],
    # RESOLVED 2026-06-03 (user re-mandate, authority watching the live pane):
    #   codex submit key is C-m ONLY, and the Tab "Queued follow-up inputs" path
    #   is removed from the codex send path entirely (see
    #   ultradebate_v2_send_to_peer → force_cm_no_tab). Rationale: when codex's
    #   prior job ran into the next, Tab-queued messages were not picked up; the
    #   injection must behave exactly like the user's own Enter (a single forced
    #   C-m). The "/n처럼 되는" bug is structurally impossible because
    #   _v2_tmux_safe_send flattens \n/\r, delivers ONE bracketed paste-buffer
    #   atom, then sends C-m as a SEPARATE keystroke after the paste-settle
    #   window — so C-m can only ever be the submit signal, never a literal '\r'
    #   inside the payload. Emergency override: CODEX_SUBMIT_KEY env (watcher).
    "codex": ["C-m"],
    "gemini": ["Enter"],
}


# =============================================================================
# TMUX PAYLOAD / SUBMIT-KEY SANITIZERS — codex "C-m ONLY" mandate (2026-06-16)
# =============================================================================
# A peer CLI prompt is a TUI: a raw Tab triggers autocomplete/queue, a raw \n/\r
# submits or wraps the line, and a raw ESC can terminate bracketed-paste mode
# (so the rest of the payload is read as keystrokes). USER MANDATE 2026-06-16:
# even when the payload itself contains \n / \t / ESC / other control bytes,
# delivery to codex MUST behave as ONE pasted atom submitted by a single C-m —
# never a Tab, never a literal newline. We therefore (a) flatten every C0
# control char in the payload to a space before paste, and (b) force the codex
# submit sequence to exactly ["C-m"]. These are the SINGLE SOURCE OF TRUTH;
# the shell daemon scripts/codex_inbox_watcher.sh mirrors the same two rules.
_CTRL_TO_SPACE = {c: " " for c in range(0x20)}
_CTRL_TO_SPACE[0x7F] = " "  # DEL


def _flatten_payload_for_tmux(text: str, *, flatten: bool = True) -> str:
    """Neutralize control bytes so a TUI never reads the payload as keystrokes.

    ``flatten=True`` (default): every C0 control char + DEL — including Tab
    (0x09), newline (0x0A), CR (0x0D) and ESC (0x1B) — becomes a single space,
    so the pasted text is one inert line that can only be submitted by the
    separate C-m keystroke. ``flatten=False``: still strip the two
    TUI-dangerous controls (Tab, ESC) but keep \\n/\\r for the rare non-TUI
    buffer-fill caller. Idempotent; safe on empty input.
    """
    if not text:
        return text
    if flatten:
        return text.translate(_CTRL_TO_SPACE)
    # Even when preserving newlines, a Tab or ESC must never reach a TUI.
    return text.replace("\t", " ").replace("\x1b", " ")


def _sanitize_submit_keys(
    submit_keys: list[str] | None, *, peer: str | None = None,
) -> list[str]:
    """Return a TUI-safe submit sequence. Codex is hard-locked to ['C-m'].

    USER MANDATE: codex submit == the user's own Enter == a single C-m. Any
    'Tab' / 'Escape' / 'Enter' requested for codex collapses to ['C-m'] (Tab
    queues/autocompletes, Escape+Enter was the reverted path, and a literal
    'Enter' can read as an in-paste newline on some codex builds). For other
    peers we keep their sequence but always drop a stray 'Tab', which must
    never be a normal-submit key.
    """
    if peer == "codex":
        return ["C-m"]
    keys = [k for k in (submit_keys or ["C-m"]) if k != "Tab"]
    return keys or ["C-m"]


# Centralized CLI busy-state markers (lowercase, matched against capture-pane
# tail). A pane is "busy" (a turn is in progress) when ANY marker is present.
# SINGLE SOURCE OF TRUTH — every pre-submit busy gate + post-submit verify loop
# MUST use this set so detection never diverges.
#
# 2026-05-24 fix: added "pursuing goal". Codex CLI shows "Pursuing goal (22m)"
# for long-running turns, which the prior per-callsite tuples MISSED — a
# false-idle there fired Escape+Enter / C-m into a working pane, causing both
# the "/n처럼 되는" regression AND an unintended interrupt of Codex's work.
# Empirically observed Codex busy indicators: "Working (Xs)",
# "Pursuing goal (Xm/Xh)", "esc to interrupt", "thinking".
_PANE_BUSY_MARKERS: tuple[str, ...] = (
    "esc to interrupt",
    "working",        # "Working (3s)"
    "pursuing goal",  # "Pursuing goal (22m)" — long-running goal state
    "thinking",
)


def _pane_text_is_busy(pane_text: str) -> bool:
    """True iff the pane tail shows any known CLI busy indicator.

    Case-insensitive. Used by pre-submit gates (`_v25_attempt_codex_tmux_wake`,
    `_v26_pane_busy`) and the post-submit verify loop in `_v2_tmux_safe_send`.
    Centralizing avoids the historical divergence where one callsite missed
    "Pursuing goal" and fired a submit key into a working pane.
    """
    t = pane_text.lower()
    return any(m in t for m in _PANE_BUSY_MARKERS)


# =============================================================================
# CRITICAL - TMUX MESSAGE INJECTION SAFE PATTERN (DO NOT REGRESS)
# =============================================================================
# ALL tmux message payloads MUST go through _v2_tmux_safe_send() below.
#
# NEVER write:  subprocess.run(["tmux", "send-keys", "-t", PANE, MSG, "C-m"])
#               ^ tmux interprets literal '\n' inside MSG as Enter, splitting
#                 the payload mid-submit. Causes intermittent "partial-send"
#                 bugs that masquerade as peer hallucination/non-response.
#
# ALWAYS use:   _v2_tmux_safe_send(pane=..., message=..., enters=1|5, ...)
#               ^ set-buffer -> paste-buffer -d -> settle -> C-m x N -> cleanup.
#                 Battle-tested 2026-05-16 (549-char multi-line payload,
#                 Codex JSON ack in 15s, zero splitting).
#
# Reason: 2026-05-16 user mandate after the "/n 처럼 된다" v2.5 regression.
#         "사용자의 엔터와 동일한 Ctrl+M으로 코덱스에게 보내고 ... v2.5 이후에
#          제대로된 연결이나 메시지 보내고 토론하기 ... 안돼고 있어"
#         Memory: feedback_ultradebate_v25_cwd_binding.md
# =============================================================================


def _v2_tmux_safe_send(
    pane: str,
    message: str,
    *,
    enters: int = 1,
    settle_sec: float = 2.0,
    socket: str | None = None,
    flatten_newlines: bool = True,
    submit_keys: list[str] | None = None,
) -> bool:
    """v2.5 SAFE tmux message-injection helper - SINGLE SOURCE OF TRUTH.

    set-buffer + paste-buffer + C-m pattern. See CRITICAL banner above for
    rationale. NEVER replace with raw send-keys for message payloads.

    Args:
        pane: tmux pane target (e.g. "0:1.1", "%0", "ccb_codex:1.1").
        message: full payload to deliver. Newlines flattened to spaces by default.
        enters: number of trailing submit sequences to send.
        settle_sec: floor for paste-settle. Effective settle scales with
                    payload length (len/500s, capped at 8s) so 2500-char
                    payloads get ~5s settle and 5000-char ~8s. v2.6 audit
                    (2026-05-22) confirmed 2521-char paste with 2s settle
                    sometimes fires Escape+Enter before paste-buffer is
                    fully ingested by the Codex TUI.
        socket: optional tmux socket (e.g. "/tmp/tmux-1000/default"). None uses
                tmux default socket.
        flatten_newlines: when True (default), replace \\n and \\r with space
                          BEFORE paste - prevents premature Enter inside payload.
                          Only set False if you need literal newlines in target
                          buffer AND target is not a TUI prompt.
        submit_keys: submit key sequence. Codex == ["C-m"] (hard-locked); Claude
                     == ["Enter"]. A stray "Tab" is always stripped here.

    Returns:
        True on confirmed delivery (busy marker after submit OR submit keys
        sent + no obvious failure). False on tmux subprocess failure or
        when no busy marker appears after retries.
    """
    import tempfile
    sock_args = ["-S", socket] if socket else []
    payload = _flatten_payload_for_tmux(message, flatten=flatten_newlines)  # incl. \t/ESC → space
    # 길이 비례 settle: 1000자 -> 2s, 2500자 -> 5s, 5000자 -> 8s 캡.
    effective_settle = max(float(settle_sec), len(payload) / 500.0)
    if effective_settle > 8.0:
        effective_settle = 8.0
    # Unique buffer name avoids collision with parallel sends across PIDs.
    buf_name = f"udb_inject_{os.getpid()}_{int(time.time() * 1000) % 1_000_000}"
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="udb_inject_", suffix=".txt",
        delete=False, encoding="utf-8",
    ) as tf:
        tf.write(payload)
        tmpf = tf.name
    try:
        rc1 = subprocess.run(
            ["tmux", *sock_args, "load-buffer", "-b", buf_name, tmpf],
            capture_output=True, text=True, timeout=5,
        )
        if rc1.returncode != 0:
            return False
        rc2 = subprocess.run(
            # -p enables bracketed-paste mode. Without it, the receiving CLI
            # (Codex v0.130+, Claude CLI, etc.) treats the pasted bytes as if
            # they were *typed* and a subsequent `Enter` reads as an in-input
            # newline rather than as submit. With -p the CLI sees a single
            # paste atom, so Escape+Enter / Enter actually fires submit.
            # Empirically required for the v2.6 audit flow (2026-05-22).
            ["tmux", *sock_args, "paste-buffer", "-b", buf_name, "-p", "-d", "-t", pane],
            capture_output=True, text=True, timeout=5,
        )
        if rc2.returncode != 0:
            return False
        # Defense-in-depth: a 'Tab' must never reach this normal-submit path
        # (Tab is only ever the busy-queue key, and never for codex). Strip it.
        keys = _sanitize_submit_keys(submit_keys)
        # paste-accept window. Codex v0.130+ requires >=2.0s; long payloads
        # need proportionally longer (audit 2026-05-22).
        time.sleep(effective_settle)
        for _ in range(max(1, enters)):
            subprocess.run(
                ["tmux", *sock_args, "send-keys", "-t", pane, *keys],
                capture_output=True, text=True, timeout=5,
            )
            time.sleep(effective_settle)

        # 검증 loop: busy marker (esc to interrupt / working / thinking)가
        # capture-pane tail에 나타나면 submit 성공으로 간주. 안 보이면 추가
        # Enter 최대 2회 재시도. v2.6 audit에서 paste-settle보다 빠르게
        # submit 키가 fired돼 큐만 들어가고 발사 안 되는 케이스 fix.
        for attempt in range(3):
            # safe_capture_pane: bytes→utf-8 replace decode + surrogate scrub +
            # byte-bounded truncate. Returns "" on any tmux failure, preserving
            # the original "보수적으로 True 반환" intent through the empty-tail
            # fast path below.
            tail_raw = _safe_capture_pane(
                pane, lines=10, socket=socket if socket else None,
            )
            if not tail_raw:
                return True
            tail = tail_raw.lower()
            if _pane_text_is_busy(tail):
                return True
            if attempt < 2:
                # 추가 Enter 시도 (settle 더 짧게).
                subprocess.run(
                    ["tmux", *sock_args, "send-keys", "-t", pane, *keys],
                    capture_output=True, text=True, timeout=5,
                )
                time.sleep(1.5)
        # busy marker 끝까지 못 봤음 — Codex audit round 2 (2026-05-22)에서
        # 'load/paste/send all rc=0 + capture idle x3 -> True 반환'이 strict
        # verified-success가 아니라고 지적됨. 이 경로는 *submit 신호가 fire되지
        # 않았을 가능성이 큰 상태*이므로 명시적으로 False 반환. paste-buffer는
        # inbox JSONL append와는 *별개의 채널*이라, 호출자(ultradebate_v2_send_to_peer
        # 같은 wrapper)가 inbox 기록 자체는 분리해서 보존하고 tmux_state를
        # 'submit_failed_no_busy' 같은 값으로 정확히 구분할 수 있다.
        return False
    finally:
        try:
            os.unlink(tmpf)
        except OSError:
            pass
        # Buffer cleanup (paste-buffer -d already deletes on success, but
        # call again defensively for failed-paste cases).
        try:
            subprocess.run(
                ["tmux", *sock_args, "delete-buffer", "-b", buf_name],
                capture_output=True, text=True, timeout=2,
            )
        except Exception:
            pass


def _v26_tmux_queue_send(
    pane: str,
    message: str,
    *,
    peer: str | None = None,
    settle_sec: float = 2.0,
    socket: str | None = None,
    flatten_newlines: bool = True,
) -> bool:
    """Queue a message into a busy peer's pane via Tab (codex 'Queued follow-up inputs').

    Used by :func:`ultradebate_v2_send_to_peer` when the target pane is currently
    busy (Working/thinking) and the B29 mandate forbids interrupting. The
    message is pasted into the input buffer using the same bracketed-paste
    protocol as :func:`_v2_tmux_safe_send`, then a single ``Tab`` keystroke
    moves it into the peer CLI's queued inputs — codex and claude both process
    queued items when they next idle.

    Differences from :func:`_v2_tmux_safe_send`:

    * Submit key is always ``Tab`` (not ``Escape+Enter`` / ``Enter``). Pressing
      Enter while codex is Working appends a literal newline into the input
      buffer instead of submitting (the 2026-05-23 "/n처럼 되는" regression).
      ``Tab`` triggers codex's "Queued follow-up inputs" path, which preserves
      the payload as a discrete pending submission.
    * No post-submit busy-marker verify loop — we *expect* the peer to remain
      busy, so capture-pane would always return "esc to interrupt" / "Working"
      and any retry would just add duplicate Tabs.

    The caller (typically :func:`ultradebate_v2_send_to_peer`) MUST have
    already established that the pane is busy. Calling this helper against an
    idle pane risks ``Tab`` triggering autocomplete instead of queueing.

    Args:
        pane: tmux pane target (e.g. ``"0:1.1"``, ``"%0"``, ``"ccb_codex:1.1"``).
        message: full payload to deliver. Newlines flattened to spaces by default
            because a literal newline in the input buffer becomes a wrapped line
            on screen, not a submit signal — the original cause of the regression
            this helper fixes.
        settle_sec: floor for paste-settle. Scales with payload length the same
            way :func:`_v2_tmux_safe_send` does (``len/500``, capped at 8s).
        socket: optional tmux socket (e.g. ``"/tmp/tmux-1000/default"``). ``None``
            uses the tmux default socket.
        flatten_newlines: when True (default), replace ``\\n`` and ``\\r`` with
            space BEFORE paste. Recommended ON — see ``message`` note above.

    Returns:
        True iff load-buffer + paste-buffer + Tab subprocess sequence all
        returned ``rc=0``. Returns False on any subprocess failure (caller
        downgrades ``tmux_state`` to ``queue_failed`` and relies on the inbox
        JSONL line for delivery).
    """
    # codex NEVER queues via Tab (user mandate 2026-06-03/06-16): a Tab-queued
    # message was dropped when codex's prior job ran into the next. If this
    # helper is ever invoked for codex, fall back to the C-m submit path so the
    # injection equals the user's own Enter — structurally no Tab for codex.
    if peer == "codex":
        return _v2_tmux_safe_send(
            pane=pane, message=message, enters=1, settle_sec=settle_sec,
            socket=socket, flatten_newlines=flatten_newlines, submit_keys=["C-m"],
        )
    import tempfile
    sock_args = ["-S", socket] if socket else []
    payload = _flatten_payload_for_tmux(message, flatten=flatten_newlines)  # incl. \t/ESC → space
    # 길이 비례 settle — _v2_tmux_safe_send 와 동일한 공식 사용 (1000자 -> 2s,
    # 2500자 -> 5s, 5000자 -> 8s 캡). Queue path도 paste-buffer ingest 완료
    # 전에 Tab 누르면 큐에 빈 항목이 들어갈 위험이 있어 같은 settle 필요.
    effective_settle = max(float(settle_sec), len(payload) / 500.0)
    if effective_settle > 8.0:
        effective_settle = 8.0
    buf_name = f"udb_queue_{os.getpid()}_{int(time.time() * 1000) % 1_000_000}"
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="udb_queue_", suffix=".txt",
        delete=False, encoding="utf-8",
    ) as tf:
        tf.write(payload)
        tmpf = tf.name
    try:
        rc1 = subprocess.run(
            ["tmux", *sock_args, "load-buffer", "-b", buf_name, tmpf],
            capture_output=True, text=True, timeout=5,
        )
        if rc1.returncode != 0:
            return False
        rc2 = subprocess.run(
            # -p enables bracketed-paste mode so codex sees a single paste atom
            # rather than typed bytes (same rationale as _v2_tmux_safe_send).
            # -d deletes the buffer after paste.
            ["tmux", *sock_args, "paste-buffer", "-b", buf_name, "-p", "-d", "-t", pane],
            capture_output=True, text=True, timeout=5,
        )
        if rc2.returncode != 0:
            return False
        # paste-accept window — codex TUI needs time to ingest the paste before
        # Tab keystroke arrives, otherwise Tab fires against an empty buffer
        # and codex tries to autocomplete instead of queueing.
        time.sleep(effective_settle)
        rc3 = subprocess.run(
            ["tmux", *sock_args, "send-keys", "-t", pane, "Tab"],
            capture_output=True, text=True, timeout=5,
        )
        if rc3.returncode != 0:
            return False
        # Short post-Tab settle so the caller's follow-up capture-pane (if any)
        # observes the queue state instead of mid-transition.
        time.sleep(1.0)
        return True
    finally:
        try:
            os.unlink(tmpf)
        except OSError:
            pass
        try:
            subprocess.run(
                ["tmux", *sock_args, "delete-buffer", "-b", buf_name],
                capture_output=True, text=True, timeout=2,
            )
        except Exception:
            pass


def _v2_tmux_send_codex_safe(text: str, sess: str) -> bool:
    """Codex paste-burst-safe submit - thin wrapper around _v2_tmux_safe_send().

    Mirrors scripts/codex_inbox_watcher.sh protocol. The submit sequence is
    centralized in _V2_SUBMIT_KEYS["codex"].
    """
    return _v2_tmux_safe_send(
        pane=sess,
        message=text,
        enters=1,
        settle_sec=2.0,
        submit_keys=_V2_SUBMIT_KEYS["codex"],
    )


def _v2_tmux_send(peer: str, text: str, session_candidates: list[str]) -> dict:
    """Send a tmux message-payload with peer-aware submit count.

    ALL peers go through _v2_tmux_safe_send (set-buffer + paste-buffer + C-m).
    The historical "send-keys -t pane TEXT C-m" direct path is BANNED - it
    splits messages on literal '\\n'. See CRITICAL banner above _v2_tmux_safe_send().

    Returns: {ok, target, path, submit_keys} on success or
             {ok: False, error, tried} on failure.
    """
    submit_keys = _V2_SUBMIT_KEYS.get(peer, ["Enter"])
    enters = 1
    for sess in session_candidates:
        # Cheap session existence check to skip 12-13s timeout accumulation
        # against missing tmux sessions.
        has = subprocess.run(
            ["tmux", "has-session", "-t", sess],
            capture_output=True, text=True, timeout=5,
        )
        if has.returncode != 0:
            continue
        if _v2_tmux_safe_send(
            pane=sess,
            message=text,
            enters=enters,
            submit_keys=submit_keys,
        ):
            return {
                "ok": True,
                "target": sess,
                "submit_keys": ["paste-buffer-flat", "+".join(submit_keys)],
                "path": "v25_safe_send",
            }
    return {"ok": False, "error": "no tmux session matched",
            "tried": session_candidates}


@mcp.tool()
async def ultradebate_send_to_peer(
    peer: str,
    message_type: str,
    payload: dict[str, Any] | None = None,
    sender: str = "claude",
    deliver_via: str = "inbox",
) -> dict[str, Any]:
    """v2: Bidirectional CLI bridge — write message to peer's inbox.

    peer: target peer in {"claude", "codex", "gemini"}
    message_type: "text" | "command" | "ping" | "compact_request" | "sc_command"
                  | "meta_team_dispatch" | "verdict" | "context_share"
    payload: arbitrary dict (text body, command string, results, etc.)
    sender: source peer (default "claude" — the calling CLI's identity)
    deliver_via: "inbox" (file marker) | "tmux_send_keys" (CLI keystroke) | "both"

    Returns: {ok, message_id, peer, path, ts, delivery}
    """
    if peer not in _V2_PEERS:
        return {"error": f"unknown peer: {peer}", "valid": list(_V2_PEERS)}
    _v2_ensure_dirs(peer)
    mid = _v2_mid()
    msg = {
        "id": mid,
        "ts": _v2_dt.utcnow().isoformat(timespec="seconds") + "Z",
        "sender": sender,
        "recipient": peer,
        "type": message_type,
        "payload": payload or {},
    }
    path = _v2_inbox(peer) / f"{mid}.json"
    try:
        path.write_text(json.dumps(msg, ensure_ascii=False, indent=2))
    except OSError as e:
        return {"error": f"write failed: {e}", "path": str(path)}

    delivery = ["inbox"]

    # Optional tmux send-keys for instant delivery (peer-aware submit key)
    if deliver_via in ("tmux_send_keys", "both"):
        notify = (
            f"echo '[ULTRADEBATE_V2_INBOX] new message id={mid} type={message_type} "
            f"from={sender} — read /tmp/ultradebate_v2/{peer}/inbox/{mid}.json'"
        )
        # Try peer-named sessions + numeric default + ccb pattern
        candidates = [f"ccb_{peer}", peer, "ultradebate_view", "3"]
        result = _v2_tmux_send(peer, notify, candidates)
        if result.get("ok"):
            delivery.append(f"tmux:{result['target']}")

    return {
        "ok": True, "message_id": mid, "peer": peer, "path": str(path),
        "ts": msg["ts"], "delivery": delivery,
    }


@mcp.tool()
async def ultradebate_recv_from_peer(
    peer: str = "claude",
    consume: bool = True,
    max_messages: int = 10,
    filter_type: str | None = None,
) -> dict[str, Any]:
    """v2: Read peer's inbox (the CALLER's inbox, looking for messages from peers).

    peer: caller's identity (default "claude" — read claude's inbox)
    consume: if True, move processed messages to processed/ (default)
    max_messages: limit
    filter_type: optional message_type filter

    Returns: {messages: [{id, type, payload, sender, ts}, ...], count, peer}
    """
    if peer not in _V2_PEERS:
        return {"error": f"unknown peer: {peer}", "valid": list(_V2_PEERS)}
    _v2_ensure_dirs(peer)
    inbox = _v2_inbox(peer)
    files = sorted(inbox.glob("*.json"))[:max_messages]
    msgs: list[dict] = []
    for fp in files:
        try:
            data = json.loads(fp.read_text())
            if filter_type and data.get("type") != filter_type:
                continue
            msgs.append(data)
            if consume:
                proc_path = _v2_processed(peer) / fp.name
                fp.rename(proc_path)
        except (OSError, json.JSONDecodeError) as e:
            err_path = (_V2_BASE / peer / "errored" / fp.name)
            try:
                fp.rename(err_path)
            except OSError:
                pass
            msgs.append({"error": str(e), "id": fp.stem})
    return {"peer": peer, "count": len(msgs), "messages": msgs}


@mcp.tool()
async def ultradebate_send_command(
    peer: str,
    command: str,
    sender: str = "claude",
    via: str = "tmux_send_keys",
) -> dict[str, Any]:
    """v2: Cross-tool command — peer executes the command in their CLI.

    Example use cases:
    - Claude → Codex: command="/compact" → Codex CLI runs /compact (context save)
    - Codex → Claude: command="/sc:analyze ..." → Claude runs /sc:analyze
    - Claude → Codex: command="/think hard" → Codex switches reasoning mode

    peer: target ("claude" | "codex" | "gemini")
    command: literal command string (slash command + args)
    sender: caller identity
    via: "tmux_send_keys" (direct CLI keystroke) | "inbox" (message-only)
         | "both" (file marker + keystroke)

    Returns: {ok, peer, command, delivery, tmux_target}
    """
    if peer not in _V2_PEERS:
        return {"error": f"unknown peer: {peer}", "valid": list(_V2_PEERS)}

    delivery = []
    tmux_target = None

    # Inbox marker (always)
    if via in ("inbox", "both"):
        send_result = await ultradebate_send_to_peer(
            peer=peer, message_type="command",
            payload={"command": command}, sender=sender,
            deliver_via="inbox",
        )
        if send_result.get("ok"):
            delivery.append("inbox")

    # tmux send-keys (direct CLI keystroke, peer-aware submit)
    if via in ("tmux_send_keys", "both"):
        candidates = [f"ccb_{peer}", peer, "ultradebate_view", "3"]
        result = _v2_tmux_send(peer, command, candidates)
        if result.get("ok"):
            delivery.append(f"tmux:{result['target']}")
            tmux_target = result["target"]

    if not delivery:
        return {"error": "no delivery channel succeeded", "peer": peer,
                "command": command}

    return {"ok": True, "peer": peer, "sender": sender,
            "command": command, "delivery": delivery,
            "tmux_target": tmux_target}


@mcp.tool()
async def ultradebate_v2_status() -> dict[str, Any]:
    """v2 status: inbox/outbox file counts per peer."""
    out: dict[str, Any] = {"base": str(_V2_BASE), "peers": {}}
    for peer in _V2_PEERS:
        _v2_ensure_dirs(peer)
        out["peers"][peer] = {
            "inbox_pending": len(list(_v2_inbox(peer).glob("*.json"))),
            "outbox": len(list(_v2_outbox(peer).glob("*.json"))),
            "processed": len(list(_v2_processed(peer).glob("*.json"))),
            "errored": len(list((_V2_BASE / peer / "errored").glob("*.json"))),
        }
    return out


# =====================================================================
# v2.1 — Role-Aware Task Coordination + Mutual Verification
# =====================================================================
# Per user mandate (2026-05-10): "MCP must enforce role distribution
# (debate-based) AND mutual code verification — improve MCP."
#
# Anti-pattern this prevents: one peer (e.g. Claude) silently does all
# code work while other peer (Codex) only reviews. v2.1 enforces:
#   1. Roles agreed via debate before task starts
#   2. Owner ≠ Reviewer
#   3. Owner submits implementation; status → review_pending
#   4. Reviewer must explicitly PASS/REVISION/FAIL
#   5. Workload audit emits IMBALANCE_WARN if any peer >70% ownership

import secrets as _v21_secrets

_V21_TASKS_FILE = _V2_BASE / "tasks.json"
_V21_ROLES_FILE = _V2_BASE / "role_negotiations.json"
_V21_AUDIT_FILE = _V2_BASE / "workload_audit.json"


def _v21_now_iso() -> str:
    return _v2_dt.now().astimezone().isoformat(timespec="seconds")


def _v21_load(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def _v21_save(path: Path, d: dict) -> None:
    _V2_BASE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False))


def _v21_audit_increment(owner: str, reviewer: str) -> None:
    d = _v21_load(_V21_AUDIT_FILE, {"by_owner": {}, "by_reviewer": {}, "history": []})
    d["by_owner"][owner] = d["by_owner"].get(owner, 0) + 1
    d["by_reviewer"][reviewer] = d["by_reviewer"].get(reviewer, 0) + 1
    d["history"].append({"owner": owner, "reviewer": reviewer, "ts": _v21_now_iso()})
    if len(d["history"]) > 200:
        d["history"] = d["history"][-200:]
    _v21_save(_V21_AUDIT_FILE, d)


def _v21_send_inbox_notif(recipient: str, mtype: str, payload: dict) -> str:
    """Auto-notify peer via inbox file (watcher will inject into peer CLI)."""
    nid = f"v21_{mtype}_{int(time.time())}_{_v21_secrets.token_hex(3)}"
    notif = {
        "id": nid,
        "ts": _v21_now_iso(),
        "sender": "ultradebate_v2_mcp",
        "recipient": recipient,
        "type": mtype,
        "payload": payload,
    }
    _v2_inbox(recipient).mkdir(parents=True, exist_ok=True)
    (_v2_inbox(recipient) / f"{nid}.json").write_text(
        json.dumps(notif, indent=2, ensure_ascii=False))
    return nid


@mcp.tool()
async def ultradebate_v2_role_negotiate(
    task_description: str,
    proposer: str,
    proposed_owner: str,
    proposed_reviewer: str,
    rationale: str = "",
    candidates: list[str] = None,
    debate_id: str = None,
) -> dict[str, Any]:
    """v2.1 — Negotiate role assignment via multi-step debate.

    Each candidate AI calls this with their own owner+reviewer proposal
    and rationale. When all candidates have proposed, server checks
    consensus: matching owner AND matching reviewer = agreed; mismatch
    = deadlock_user_escalation_needed.

    Use case at task boundary:
      Claude calls: proposer='claude', proposed_owner='codex', reviewer='claude'
      Codex calls:  proposer='codex',  proposed_owner='codex', reviewer='claude'
      → agreed (both proposed same).

    Or:
      Claude:  proposed_owner='codex'
      Codex:   proposed_owner='claude' (declines)
      → deadlock → user_escalation
    """
    if candidates is None:
        candidates = list(_V2_PEERS)
    if proposer not in candidates:
        return {"ok": False, "error": f"proposer '{proposer}' not in candidates {candidates}"}
    if proposed_owner not in candidates:
        return {"ok": False, "error": f"proposed_owner '{proposed_owner}' not in {candidates}"}
    if proposed_reviewer not in candidates:
        return {"ok": False, "error": f"proposed_reviewer '{proposed_reviewer}' not in {candidates}"}
    if proposed_owner == proposed_reviewer:
        return {"ok": False, "error": "owner and reviewer must differ"}

    d = _v21_load(_V21_ROLES_FILE, {"debates": {}, "schema_version": "2.1"})
    if debate_id is None:
        debate_id = f"role_{int(time.time())}_{_v21_secrets.token_hex(3)}"

    if debate_id not in d["debates"]:
        d["debates"][debate_id] = {
            "task_description": task_description,
            "candidates": candidates,
            "proposals": {},
            "started_at": _v21_now_iso(),
            "status": "negotiating",
        }
    deb = d["debates"][debate_id]
    deb["proposals"][proposer] = {
        "proposed_owner": proposed_owner,
        "proposed_reviewer": proposed_reviewer,
        "rationale": rationale,
        "ts": _v21_now_iso(),
    }

    proposals = deb["proposals"]
    if all(c in proposals for c in candidates):
        first = proposals[candidates[0]]
        consensus = all(
            proposals[c]["proposed_owner"] == first["proposed_owner"] and
            proposals[c]["proposed_reviewer"] == first["proposed_reviewer"]
            for c in candidates
        )
        if consensus:
            deb["status"] = "agreed"
            deb["final_owner"] = first["proposed_owner"]
            deb["final_reviewer"] = first["proposed_reviewer"]
            deb["agreed_at"] = _v21_now_iso()
        else:
            deb["status"] = "deadlock_user_escalation_needed"

    _v21_save(_V21_ROLES_FILE, d)
    return {
        "ok": True,
        "debate_id": debate_id,
        "status": deb["status"],
        "proposals_count": len(proposals),
        "expected_count": len(candidates),
        "final_owner": deb.get("final_owner"),
        "final_reviewer": deb.get("final_reviewer"),
    }


@mcp.tool()
async def ultradebate_v2_task_create(
    task_id: str,
    description: str,
    owner: str,
    reviewer: str,
    spec: dict[str, Any] = None,
    debate_id: str = None,
) -> dict[str, Any]:
    """v2.1 — Create coordinated task with enforced owner != reviewer.

    Status starts at 'pending'. Progression:
      pending → in_progress → review_pending → completed
                                          ↘ revision_requested → in_progress
                                          ↘ aborted_max_rounds (3 rounds)
                                          ↘ aborted_user_escalation
    """
    if owner not in _V2_PEERS:
        return {"ok": False, "error": f"owner '{owner}' not in {list(_V2_PEERS)}"}
    if reviewer not in _V2_PEERS:
        return {"ok": False, "error": f"reviewer '{reviewer}' not in {list(_V2_PEERS)}"}
    if owner == reviewer:
        return {"ok": False, "error": "owner and reviewer must differ — anti-bias enforcement"}

    d = _v21_load(_V21_TASKS_FILE, {"tasks": {}, "schema_version": "2.1"})
    if task_id in d["tasks"]:
        return {"ok": False, "error": f"task_id '{task_id}' already exists",
                "existing_status": d["tasks"][task_id].get("status")}

    d["tasks"][task_id] = {
        "task_id": task_id,
        "description": description,
        "owner": owner,
        "reviewer": reviewer,
        "spec": spec or {},
        "debate_id": debate_id,
        "status": "pending",
        "created_at": _v21_now_iso(),
        "review_rounds": 0,
        "max_review_rounds": 3,
        "history": [{"event": "created", "ts": _v21_now_iso(), "by": "system"}],
    }
    _v21_save(_V21_TASKS_FILE, d)

    nid = _v21_send_inbox_notif(owner, "v21_task_assigned", {
        "task_id": task_id, "description": description,
        "owner": owner, "reviewer": reviewer,
        "spec": spec or {},
        "next_action": f"{owner}: implement and call ultradebate_v2_task_implement(task_id, submitter='{owner}', files_modified=[...], diff_summary='...')",
    })
    return {"ok": True, "task_id": task_id, "status": "pending", "owner_notified": nid}


@mcp.tool()
async def ultradebate_v2_task_implement(
    task_id: str,
    submitter: str,
    files_modified: list[str],
    diff_summary: str,
) -> dict[str, Any]:
    """v2.1 — Submit implementation. Validates submitter == owner.

    Transitions: pending|in_progress|revision_requested → review_pending.
    Auto-notifies reviewer via inbox file.
    """
    d = _v21_load(_V21_TASKS_FILE, {"tasks": {}})
    if task_id not in d["tasks"]:
        return {"ok": False, "error": f"task_id '{task_id}' not found"}
    t = d["tasks"][task_id]
    if submitter != t["owner"]:
        return {"ok": False, "error": f"role violation: submitter '{submitter}' != assigned owner '{t['owner']}'"}
    valid_from = ("pending", "in_progress", "revision_requested")
    if t["status"] not in valid_from:
        return {"ok": False, "error": f"cannot submit from status '{t['status']}' (must be one of {valid_from})"}

    t["status"] = "review_pending"
    t["files_modified"] = files_modified
    t["diff_summary"] = diff_summary
    t["last_submitted_at"] = _v21_now_iso()
    t["history"].append({
        "event": "implemented",
        "ts": _v21_now_iso(),
        "by": submitter,
        "files_count": len(files_modified),
        "round": t["review_rounds"] + 1,
    })
    _v21_save(_V21_TASKS_FILE, d)

    nid = _v21_send_inbox_notif(t["reviewer"], "v21_task_review_request", {
        "task_id": task_id,
        "owner": t["owner"],
        "description": t["description"],
        "files_modified": files_modified,
        "diff_summary": diff_summary[:1500],
        "round": t["review_rounds"] + 1,
        "max_rounds": t["max_review_rounds"],
        "next_action": f"{t['reviewer']}: review and call ultradebate_v2_task_review(task_id='{task_id}', reviewer='{t['reviewer']}', verdict='PASS|REVISION|FAIL', findings=[...], rationale='...')",
    })
    return {"ok": True, "task_id": task_id, "status": "review_pending",
            "reviewer": t["reviewer"], "reviewer_notified": nid}


@mcp.tool()
async def ultradebate_v2_task_review(
    task_id: str,
    reviewer: str,
    verdict: str,
    findings: list[dict[str, Any]] = None,
    rationale: str = "",
) -> dict[str, Any]:
    """v2.1 — Submit review verdict. Validates reviewer == assigned reviewer.

    verdict ∈ {PASS, REVISION, FAIL}:
      PASS → status='completed', workload audit incremented
      REVISION → round +=1, status='revision_requested' (or 'aborted_max_rounds' if >=3)
      FAIL → status='aborted_user_escalation' (review process broke down)
    """
    d = _v21_load(_V21_TASKS_FILE, {"tasks": {}})
    if task_id not in d["tasks"]:
        return {"ok": False, "error": f"task_id '{task_id}' not found"}
    t = d["tasks"][task_id]
    if reviewer != t["reviewer"]:
        return {"ok": False, "error": f"role violation: reviewer '{reviewer}' != assigned '{t['reviewer']}'"}
    if t["status"] != "review_pending":
        return {"ok": False, "error": f"cannot review from status '{t['status']}' (must be 'review_pending')"}
    if verdict not in ("PASS", "REVISION", "FAIL"):
        return {"ok": False, "error": "verdict must be PASS|REVISION|FAIL"}

    t["last_review"] = {
        "reviewer": reviewer,
        "verdict": verdict,
        "findings": findings or [],
        "rationale": rationale,
        "ts": _v21_now_iso(),
    }
    t["history"].append({"event": "reviewed", "ts": _v21_now_iso(),
                          "by": reviewer, "verdict": verdict,
                          "round": t["review_rounds"] + 1})

    if verdict == "PASS":
        t["status"] = "completed"
        t["completed_at"] = _v21_now_iso()
        _v21_audit_increment(t["owner"], t["reviewer"])
    elif verdict == "REVISION":
        t["review_rounds"] += 1
        if t["review_rounds"] >= t["max_review_rounds"]:
            t["status"] = "aborted_max_rounds"
            _v21_send_inbox_notif(t["owner"], "v21_task_aborted_max_rounds", {
                "task_id": task_id, "rounds": t["review_rounds"],
                "rationale": rationale,
            })
        else:
            t["status"] = "revision_requested"
            _v21_send_inbox_notif(t["owner"], "v21_task_revision_request", {
                "task_id": task_id, "round": t["review_rounds"],
                "max_rounds": t["max_review_rounds"],
                "findings": findings or [], "rationale": rationale,
                "next_action": f"{t['owner']}: address findings and re-submit via ultradebate_v2_task_implement",
            })
    else:  # FAIL
        t["status"] = "aborted_user_escalation"
        _v21_send_inbox_notif("claude", "v21_user_escalation_needed", {
            "task_id": task_id, "rationale": rationale,
            "owner": t["owner"], "reviewer": reviewer,
        })

    _v21_save(_V21_TASKS_FILE, d)
    return {"ok": True, "task_id": task_id, "status": t["status"],
            "verdict": verdict, "review_round": t["review_rounds"]}


@mcp.tool()
async def ultradebate_v2_task_status(task_id: str = None) -> dict[str, Any]:
    """v2.1 — Query task status. Returns single task or summary of all."""
    d = _v21_load(_V21_TASKS_FILE, {"tasks": {}})
    if task_id:
        if task_id not in d["tasks"]:
            return {"ok": False, "error": f"task_id '{task_id}' not found"}
        return {"ok": True, "task": d["tasks"][task_id]}

    statuses = ("pending", "in_progress", "review_pending", "revision_requested",
                "completed", "aborted_max_rounds", "aborted_user_escalation")
    by_status = {s: 0 for s in statuses}
    for t in d["tasks"].values():
        s = t.get("status", "pending")
        by_status[s] = by_status.get(s, 0) + 1
    return {"ok": True, "tasks_count": len(d["tasks"]),
            "by_status": by_status,
            "task_ids": list(d["tasks"].keys())}


@mcp.tool()
async def ultradebate_v2_workload_audit(window_n: int = 20,
                                          imbalance_pct: float = 70.0) -> dict[str, Any]:
    """v2.1 — Audit task ownership distribution.

    Returns ownership counts + percentages over last N completed tasks.
    Emits IMBALANCE_WARN per peer if ownership > imbalance_pct (default 70%).
    Use to detect 'one peer does everything' anti-pattern that prompted v2.1.
    """
    audit = _v21_load(_V21_AUDIT_FILE, {"by_owner": {}, "by_reviewer": {}, "history": []})
    history = audit.get("history", [])
    recent = history[-window_n:] if len(history) > window_n else history

    counts: dict[str, int] = {}
    for entry in recent:
        o = entry.get("owner")
        if o:
            counts[o] = counts.get(o, 0) + 1
    total = len(recent)
    pct = {k: round(100.0 * v / total, 1) if total else 0 for k, v in counts.items()}

    warnings = []
    for k, p in pct.items():
        if p > imbalance_pct and total >= 5:
            warnings.append({
                "level": "IMBALANCE_WARN",
                "peer": k,
                "ownership_pct": p,
                "msg": (f"'{k}' owns {p}% of last {total} completed tasks — "
                        f"exceeds {imbalance_pct}% threshold. Per user mandate, "
                        f"role distribution should be debate-balanced."),
            })

    return {
        "ok": True,
        "window_n": window_n,
        "imbalance_threshold_pct": imbalance_pct,
        "total_completed": total,
        "ownership_counts": counts,
        "ownership_pct": pct,
        "warnings": warnings,
        "all_time_owner_counts": audit.get("by_owner", {}),
        "all_time_reviewer_counts": audit.get("by_reviewer", {}),
    }


# =====================================================================
# v2.2 — Bilateral Iteration Protocol (Forced Debate + Mutual Review)
# =====================================================================
# Per user mandate (2026-05-10): Address 4 collaboration gaps:
#   1. No iterative debate cycles (1-round verdicts dominant)
#   2. No active conversation initiation (only response-on-notify)
#   3. No mutual code review symmetry (Codex implements, Claude reviews — not reverse)
#   4. No work-kickoff enforcement (authorization given but no follow-through)
#
# v2.2 enforces:
#   - debate_round: ≥2 rounds before convergence, both peers must speak
#   - request_response: structured open question with deadline
#   - mutual_review_pair: symmetric review (A reviews B AND B reviews A)
#   - propose_counter: challenge primitive that forces original proposer to address
#   - consensus_check: explicit convergence detection
#   - session_health: bilateral collaboration metrics

_V22_DEBATES_FILE = _V2_BASE / "debates.json"
_V22_QUESTIONS_FILE = _V2_BASE / "open_questions.json"
_V22_COUNTERS_FILE = _V2_BASE / "counters.json"
_V22_PAIRS_FILE = _V2_BASE / "review_pairs.json"


def _v22_now_iso() -> str:
    return _v2_dt.now().astimezone().isoformat(timespec="seconds")


def _v22_now_ts() -> float:
    return _v2_dt.now().timestamp()


def _v22_load(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def _v22_save(path: Path, d: dict) -> None:
    _V2_BASE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False))


@mcp.tool()
async def ultradebate_v2_debate_round(
    debate_id: str,
    round_n: int,
    speaker: str,
    position: str,
    content: str,
    rationale: str = "",
    cite_paths: list[str] = None,
    consensus_signal: bool = False,
    debate_topic: str = None,
    candidates: list[str] = None,
) -> dict[str, Any]:
    """v2.2 — Iterative debate primitive. Forces bilateral rounds until convergence.

    position ∈ {AGREE, DISAGREE, REFINE, CHALLENGE, NEW_PROPOSAL}.
    Round 1 starts the debate (debate_topic + candidates required if new debate).
    Server records each round. Both peers must speak in EACH round.
    consensus_signal=True from BOTH peers in same round → converged.
    Round_n cap=5 (project default), then escalation.
    """
    if position not in ("AGREE", "DISAGREE", "REFINE", "CHALLENGE", "NEW_PROPOSAL"):
        return {"ok": False, "error": f"position must be AGREE|DISAGREE|REFINE|CHALLENGE|NEW_PROPOSAL"}
    if speaker not in _V2_PEERS:
        return {"ok": False, "error": f"speaker '{speaker}' not in {list(_V2_PEERS)}"}

    d = _v22_load(_V22_DEBATES_FILE, {"debates": {}, "schema_version": "2.2"})

    if debate_id not in d["debates"]:
        if not debate_topic:
            return {"ok": False, "error": "debate_topic required to create new debate"}
        # FIX HIGH-2 (Codex review 2026-05-10): default to bilateral Claude+Codex.
        # _V2_PEERS includes gemini which would stall debates waiting for 3rd peer.
        # gemini can be opted in explicitly via candidates parameter.
        default_bilateral = ["claude", "codex"]
        d["debates"][debate_id] = {
            "debate_id": debate_id,
            "topic": debate_topic,
            "candidates": candidates if candidates else default_bilateral,
            "rounds": {},
            "status": "active",
            "created_at": _v22_now_iso(),
            "max_rounds": 5,
        }
    deb = d["debates"][debate_id]

    if deb["status"] != "active":
        return {"ok": False, "error": f"debate status='{deb['status']}', not active"}

    rkey = str(round_n)
    if rkey not in deb["rounds"]:
        deb["rounds"][rkey] = {"speakers": {}, "started_at": _v22_now_iso()}
    rd = deb["rounds"][rkey]
    rd["speakers"][speaker] = {
        "position": position,
        "content": content,
        "rationale": rationale,
        "cite_paths": cite_paths or [],
        "consensus_signal": consensus_signal,
        "ts": _v22_now_iso(),
    }

    # Check convergence: all candidates spoke in this round AND all signaled consensus
    all_spoke = all(c in rd["speakers"] for c in deb["candidates"])
    all_consensus = all_spoke and all(
        rd["speakers"][c].get("consensus_signal", False) for c in deb["candidates"]
    )

    next_action = None
    # B2 ENFORCEMENT (Codex review fix 2026-05-10): minimum 2 rounds before convergence.
    # Round 1 with all consensus_signal=True → still requires round 2 confirmation.
    # This eliminates the single-round verdict anti-pattern v2.2 was meant to prevent.
    min_rounds_for_convergence = 2
    if all_consensus and round_n >= min_rounds_for_convergence:
        deb["status"] = "converged"
        deb["converged_at"] = _v22_now_iso()
        deb["converged_round"] = round_n
        next_action = "completed"
    elif all_spoke:
        # Round complete but not converged (either need more rounds or disagreement)
        next_round = round_n + 1
        if next_round > deb["max_rounds"]:
            deb["status"] = "deadlock_max_rounds"
            next_action = "escalate_to_user"
        else:
            if all_consensus and round_n < min_rounds_for_convergence:
                next_action = f"begin_round_{next_round}_b2_enforce_2round_min"
            else:
                next_action = f"begin_round_{next_round}"
            # Auto-notify other peer to respond in next round
            for c in deb["candidates"]:
                if c != speaker:
                    _v21_send_inbox_notif(c, "v22_debate_next_round", {
                        "debate_id": debate_id,
                        "topic": deb["topic"],
                        "next_round": next_round,
                        "previous_round_summary": rd["speakers"],
                        "b2_enforcement_note": "Per Rule B2, minimum 2 rounds required even if all signaled consensus in round 1. Confirm consensus in round 2 to converge.",
                        "next_action": f"{c}: respond via ultradebate_v2_debate_round(debate_id='{debate_id}', round_n={next_round}, speaker='{c}', position='AGREE|DISAGREE|REFINE|CHALLENGE', content='...', rationale='...', consensus_signal=<bool>)",
                    })
    else:
        # Waiting on other peers to speak in this round
        waiting_on = [c for c in deb["candidates"] if c not in rd["speakers"]]
        next_action = f"awaiting_{','.join(waiting_on)}_in_round_{round_n}"
        for c in waiting_on:
            _v21_send_inbox_notif(c, "v22_debate_round_open", {
                "debate_id": debate_id,
                "topic": deb["topic"],
                "current_round": round_n,
                "speakers_so_far": list(rd["speakers"].keys()),
                "next_action": f"{c}: speak in round {round_n} via ultradebate_v2_debate_round(...)",
            })

    _v22_save(_V22_DEBATES_FILE, d)
    return {
        "ok": True,
        "debate_id": debate_id,
        "round_n": round_n,
        "status": deb["status"],
        "next_action": next_action,
        "round_speakers": list(rd["speakers"].keys()),
    }


@mcp.tool()
async def ultradebate_v2_request_response(
    target_peer: str,
    question: str,
    sender: str,
    context: dict[str, Any] = None,
    deadline_sec: int = 1800,
    blocking: bool = False,
    request_id: str = None,
) -> dict[str, Any]:
    """v2.2 — Active conversation initiation. Forces target peer to respond.

    Target peer's inbox gets v22_open_question notification.
    Target MUST respond within deadline_sec (default 30min).
    blocking=True signals sender is blocked until response (high priority).
    """
    if target_peer not in _V2_PEERS:
        return {"ok": False, "error": f"target_peer '{target_peer}' not in {list(_V2_PEERS)}"}
    if sender not in _V2_PEERS:
        return {"ok": False, "error": f"sender '{sender}' not in {list(_V2_PEERS)}"}
    if target_peer == sender:
        return {"ok": False, "error": "cannot request response from self"}

    d = _v22_load(_V22_QUESTIONS_FILE, {"questions": {}, "schema_version": "2.2"})
    if request_id is None:
        request_id = f"q_{int(_v22_now_ts())}_{_v21_secrets.token_hex(3)}"

    deadline_ts = _v22_now_ts() + deadline_sec
    d["questions"][request_id] = {
        "request_id": request_id,
        "sender": sender,
        "target_peer": target_peer,
        "question": question,
        "context": context or {},
        "blocking": blocking,
        "asked_at": _v22_now_iso(),
        "deadline_at": _v2_dt.fromtimestamp(deadline_ts).astimezone().isoformat(timespec="seconds"),
        "deadline_sec": deadline_sec,
        "status": "open",
        "answer": None,
    }
    _v22_save(_V22_QUESTIONS_FILE, d)

    nid = _v21_send_inbox_notif(target_peer, "v22_open_question", {
        "request_id": request_id,
        "sender": sender,
        "question": question,
        "context": context or {},
        "blocking": blocking,
        "deadline_sec": deadline_sec,
        "next_action": f"{target_peer}: answer via ultradebate_v2_answer_response(request_id='{request_id}', responder='{target_peer}', answer='...', position='AGREE|REFINE|DISAGREE')",
    })
    return {"ok": True, "request_id": request_id, "target_peer": target_peer,
            "deadline_sec": deadline_sec, "notification_sent": nid}


@mcp.tool()
async def ultradebate_v2_answer_response(
    request_id: str,
    responder: str,
    answer: str,
    position: str = "AGREE",
    counter_proposal: str = None,
) -> dict[str, Any]:
    """v2.2 — Answer an open question from peer.

    Validates responder == question.target_peer. Closes question.
    position ∈ {AGREE, REFINE, DISAGREE}. counter_proposal optional if REFINE/DISAGREE.
    """
    d = _v22_load(_V22_QUESTIONS_FILE, {"questions": {}})
    if request_id not in d["questions"]:
        return {"ok": False, "error": f"request_id '{request_id}' not found"}
    q = d["questions"][request_id]
    if responder != q["target_peer"]:
        return {"ok": False, "error": f"responder '{responder}' != target_peer '{q['target_peer']}'"}
    if q["status"] != "open":
        return {"ok": False, "error": f"question status='{q['status']}', not open"}
    if position not in ("AGREE", "REFINE", "DISAGREE"):
        return {"ok": False, "error": "position must be AGREE|REFINE|DISAGREE"}

    q["status"] = "answered"
    q["answered_at"] = _v22_now_iso()
    q["latency_sec"] = int(_v22_now_ts() -
                            _v2_dt.fromisoformat(q["asked_at"]).timestamp())
    q["answer"] = {
        "responder": responder,
        "answer": answer,
        "position": position,
        "counter_proposal": counter_proposal,
    }
    _v22_save(_V22_QUESTIONS_FILE, d)

    # Notify original sender that answer arrived
    _v21_send_inbox_notif(q["sender"], "v22_question_answered", {
        "request_id": request_id,
        "answer": answer,
        "position": position,
        "counter_proposal": counter_proposal,
        "latency_sec": q["latency_sec"],
    })
    return {"ok": True, "request_id": request_id, "status": "answered",
            "latency_sec": q["latency_sec"]}


@mcp.tool()
async def ultradebate_v2_propose_counter(
    target_proposal_id: str,
    challenger: str,
    counter_position: str,
    counter_rationale: str,
    threshold_disagreement: float = 0.5,
) -> dict[str, Any]:
    """v2.2 — Challenge primitive. Forces original proposer to address counter.

    target_proposal_id can be task_id, debate_id, or any addressable proposal.
    Original proposer's inbox gets v22_counter_received notification.
    threshold_disagreement (0-1): how strongly opposed.
    """
    if challenger not in _V2_PEERS:
        return {"ok": False, "error": f"challenger '{challenger}' not in {list(_V2_PEERS)}"}

    d = _v22_load(_V22_COUNTERS_FILE, {"counters": {}, "schema_version": "2.2"})
    counter_id = f"c_{int(_v22_now_ts())}_{_v21_secrets.token_hex(3)}"

    # Look up original proposer (best-effort across tasks/debates)
    original_proposer = None
    tasks_d = _v21_load(_V21_TASKS_FILE, {"tasks": {}})
    if target_proposal_id in tasks_d.get("tasks", {}):
        original_proposer = tasks_d["tasks"][target_proposal_id]["owner"]
    debates_d = _v22_load(_V22_DEBATES_FILE, {"debates": {}})
    if target_proposal_id in debates_d.get("debates", {}):
        deb = debates_d["debates"][target_proposal_id]
        # Most recent speaker who is not challenger
        for r in sorted(deb.get("rounds", {}).keys(), reverse=True):
            for sp, info in deb["rounds"][r].get("speakers", {}).items():
                if sp != challenger:
                    original_proposer = sp
                    break
            if original_proposer:
                break

    # FIX MEDIUM-3 (Codex review 2026-05-10): require resolvable target.
    # Previous behavior silently created unenforced counter when target lookup failed.
    # Now reject explicitly so caller must provide a known task_id or debate_id.
    if original_proposer is None:
        return {"ok": False,
                "error": f"target_proposal_id '{target_proposal_id}' not found in tasks or debates — cannot identify original_proposer to challenge",
                "hint": "Pass target_proposal_id matching an existing task_id or debate_id, OR add explicit original_proposer parameter (future API extension)."}

    d["counters"][counter_id] = {
        "counter_id": counter_id,
        "target_proposal_id": target_proposal_id,
        "challenger": challenger,
        "original_proposer": original_proposer,
        "counter_position": counter_position,
        "counter_rationale": counter_rationale,
        "threshold_disagreement": threshold_disagreement,
        "status": "open",
        "created_at": _v22_now_iso(),
    }
    _v22_save(_V22_COUNTERS_FILE, d)

    if original_proposer and original_proposer != challenger:
        _v21_send_inbox_notif(original_proposer, "v22_counter_received", {
            "counter_id": counter_id,
            "target_proposal_id": target_proposal_id,
            "challenger": challenger,
            "counter_position": counter_position,
            "counter_rationale": counter_rationale,
            "threshold_disagreement": threshold_disagreement,
            "next_action": f"{original_proposer}: respond via ultradebate_v2_resolve_counter(counter_id='{counter_id}', resolver='{original_proposer}', resolution='ACCEPT|REJECT|REFINE', response='...')",
        })

    return {"ok": True, "counter_id": counter_id,
            "original_proposer_notified": original_proposer is not None}


@mcp.tool()
async def ultradebate_v2_resolve_counter(
    counter_id: str,
    resolver: str,
    resolution: str,
    response: str,
) -> dict[str, Any]:
    """v2.2 — Resolve a challenge. Original proposer accepts, rejects, or refines.

    resolution ∈ {ACCEPT, REJECT, REFINE}.
    """
    d = _v22_load(_V22_COUNTERS_FILE, {"counters": {}})
    if counter_id not in d["counters"]:
        return {"ok": False, "error": f"counter_id '{counter_id}' not found"}
    c = d["counters"][counter_id]
    if c.get("original_proposer") and resolver != c["original_proposer"]:
        return {"ok": False, "error": f"resolver '{resolver}' != original_proposer '{c['original_proposer']}'"}
    if c["status"] != "open":
        return {"ok": False, "error": f"counter status='{c['status']}', not open"}
    if resolution not in ("ACCEPT", "REJECT", "REFINE"):
        return {"ok": False, "error": "resolution must be ACCEPT|REJECT|REFINE"}

    c["status"] = "resolved"
    c["resolution"] = resolution
    c["response"] = response
    c["resolved_at"] = _v22_now_iso()
    c["resolved_by"] = resolver
    _v22_save(_V22_COUNTERS_FILE, d)

    _v21_send_inbox_notif(c["challenger"], "v22_counter_resolved", {
        "counter_id": counter_id,
        "resolution": resolution,
        "response": response,
    })
    return {"ok": True, "counter_id": counter_id, "status": "resolved",
            "resolution": resolution}


@mcp.tool()
async def ultradebate_v2_mutual_review_pair_create(
    pair_id: str,
    task_id_a: str,
    task_id_b: str,
    description: str = "",
) -> dict[str, Any]:
    """v2.2 — Symmetric mutual review.

    Creates pairing where:
      - Task A (owner=peer_X) is reviewed by peer_Y
      - Task B (owner=peer_Y) is reviewed by peer_X
    Both AIs are forced to provide review verdicts on each other's work.
    """
    tasks_d = _v21_load(_V21_TASKS_FILE, {"tasks": {}})
    if task_id_a not in tasks_d["tasks"]:
        return {"ok": False, "error": f"task_id_a '{task_id_a}' not found"}
    if task_id_b not in tasks_d["tasks"]:
        return {"ok": False, "error": f"task_id_b '{task_id_b}' not found"}
    ta = tasks_d["tasks"][task_id_a]
    tb = tasks_d["tasks"][task_id_b]
    if ta["owner"] == tb["owner"]:
        return {"ok": False, "error": "task_id_a.owner == task_id_b.owner — mutual pair requires different owners"}

    d = _v22_load(_V22_PAIRS_FILE, {"pairs": {}, "schema_version": "2.2"})
    if pair_id in d["pairs"]:
        return {"ok": False, "error": f"pair_id '{pair_id}' already exists"}
    d["pairs"][pair_id] = {
        "pair_id": pair_id,
        "task_id_a": task_id_a,
        "task_id_b": task_id_b,
        "owner_a": ta["owner"],
        "owner_b": tb["owner"],
        "reviewer_a": ta["reviewer"],
        "reviewer_b": tb["reviewer"],
        "description": description,
        "status": "active",
        "review_a_status": "pending",
        "review_b_status": "pending",
        "created_at": _v22_now_iso(),
    }
    _v22_save(_V22_PAIRS_FILE, d)

    # FIX MEDIUM-4 (Codex review 2026-05-10): notify reviewers + validate paired tasks.
    # Previous behavior was bookkeeping-only. Now actively enforce:
    # 1. Send v22_mutual_review_obligation notifications to both reviewers
    # 2. Validate paired tasks are in review_pending or completed state with mutual reviewers
    notif_a = _v21_send_inbox_notif(ta["reviewer"], "v22_mutual_review_obligation", {
        "pair_id": pair_id,
        "task_id_to_review": task_id_a,
        "owner_of_that_task": ta["owner"],
        "your_role": "reviewer",
        "description": description,
        "next_action": f"{ta['reviewer']}: review task '{task_id_a}' (owned by {ta['owner']}) via ultradebate_v2_task_review(task_id='{task_id_a}', reviewer='{ta['reviewer']}', verdict='PASS|REVISION|FAIL', findings=[...], rationale='...')",
    })
    notif_b = _v21_send_inbox_notif(tb["reviewer"], "v22_mutual_review_obligation", {
        "pair_id": pair_id,
        "task_id_to_review": task_id_b,
        "owner_of_that_task": tb["owner"],
        "your_role": "reviewer",
        "description": description,
        "next_action": f"{tb['reviewer']}: review task '{task_id_b}' (owned by {tb['owner']}) via ultradebate_v2_task_review(task_id='{task_id_b}', reviewer='{tb['reviewer']}', verdict='PASS|REVISION|FAIL', findings=[...], rationale='...')",
    })

    # Validate task states (warn if not in review_pending/completed)
    warnings = []
    for label, t in (("a", ta), ("b", tb)):
        if t.get("status") not in ("review_pending", "completed"):
            warnings.append(f"task {label} ({t.get('task_id')}) status='{t.get('status')}' — expected review_pending or completed for mutual review")

    return {"ok": True, "pair_id": pair_id,
            "review_flow": f"{ta['reviewer']} reviews {ta['owner']}'s work, {tb['reviewer']} reviews {tb['owner']}'s work",
            "notifications_sent": {"reviewer_a": notif_a, "reviewer_b": notif_b},
            "warnings": warnings}


@mcp.tool()
async def ultradebate_v2_session_health(window_n: int = 20) -> dict[str, Any]:
    """v2.2 — Bilateral collaboration health metrics.

    Returns:
      debate_rounds_per_task_avg
      challenges_per_proposal_avg
      mutual_review_ratio
      response_latency_p95_sec
      one_sided_drift_score
      total_open_questions / overdue
      total_unresolved_counters
      active_debates
    """
    debates_d = _v22_load(_V22_DEBATES_FILE, {"debates": {}})
    questions_d = _v22_load(_V22_QUESTIONS_FILE, {"questions": {}})
    counters_d = _v22_load(_V22_COUNTERS_FILE, {"counters": {}})
    pairs_d = _v22_load(_V22_PAIRS_FILE, {"pairs": {}})
    tasks_d = _v21_load(_V21_TASKS_FILE, {"tasks": {}})
    audit_d = _v21_load(_V21_AUDIT_FILE, {"by_owner": {}, "history": []})

    completed_tasks = [t for t in tasks_d["tasks"].values() if t.get("status") == "completed"]
    active_debates = [d for d in debates_d["debates"].values() if d.get("status") == "active"]
    converged_debates = [d for d in debates_d["debates"].values() if d.get("status") == "converged"]
    open_questions = [q for q in questions_d["questions"].values() if q.get("status") == "open"]
    answered_questions = [q for q in questions_d["questions"].values() if q.get("status") == "answered"]
    unresolved_counters = [c for c in counters_d["counters"].values() if c.get("status") == "open"]
    active_pairs = [p for p in pairs_d["pairs"].values() if p.get("status") == "active"]

    # Overdue questions (past deadline)
    now = _v22_now_ts()
    overdue = []
    for q in open_questions:
        try:
            deadline_ts = _v2_dt.fromisoformat(q["deadline_at"]).timestamp()
            if now > deadline_ts:
                overdue.append({"request_id": q["request_id"], "target_peer": q["target_peer"],
                                "blocking": q.get("blocking", False),
                                "overdue_sec": int(now - deadline_ts)})
        except Exception:
            pass

    # Debate rounds per converged debate
    if converged_debates:
        rounds_avg = sum(d.get("converged_round", 1) for d in converged_debates) / len(converged_debates)
    else:
        rounds_avg = 0.0

    # Response latency p95
    latencies = sorted([q["latency_sec"] for q in answered_questions if "latency_sec" in q])
    if latencies:
        p95_idx = int(len(latencies) * 0.95)
        p95_lat = latencies[min(p95_idx, len(latencies) - 1)]
    else:
        p95_lat = None

    # Mutual review ratio: (completed tasks in pairs) / total completed tasks
    paired_task_ids = set()
    for p in pairs_d["pairs"].values():
        paired_task_ids.add(p.get("task_id_a"))
        paired_task_ids.add(p.get("task_id_b"))
    paired_completed = sum(1 for t in completed_tasks if t.get("task_id") in paired_task_ids)
    mutual_review_ratio = paired_completed / len(completed_tasks) if completed_tasks else 0.0

    # One-sided drift: workload audit imbalance
    by_owner = audit_d.get("by_owner", {})
    if by_owner and len(by_owner) >= 2:
        max_count = max(by_owner.values())
        total = sum(by_owner.values())
        drift_score = (max_count / total) if total > 0 else 0.0
    else:
        drift_score = 0.0 if not by_owner else 1.0

    # Health verdict
    health_warnings = []
    if rounds_avg < 2.0 and converged_debates:
        health_warnings.append({
            "level": "INSUFFICIENT_DEBATE",
            "msg": f"Avg {rounds_avg:.1f} rounds per converged debate (target ≥2). Single-round verdicts dominant.",
        })
    if mutual_review_ratio < 0.3 and len(completed_tasks) >= 5:
        health_warnings.append({
            "level": "ONE_SIDED_REVIEW",
            "msg": f"Only {mutual_review_ratio:.0%} of completed tasks have mutual review pair. Symmetry needed.",
        })
    if drift_score > 0.7:
        health_warnings.append({
            "level": "OWNERSHIP_DRIFT",
            "msg": f"One peer owns {drift_score:.0%} of completed tasks. Rebalance.",
        })
    if overdue:
        health_warnings.append({
            "level": "OVERDUE_QUESTIONS",
            "msg": f"{len(overdue)} questions past deadline. Peer not responding.",
        })

    return {
        "ok": True,
        "metrics": {
            "debate_rounds_per_task_avg": round(rounds_avg, 2),
            "mutual_review_ratio": round(mutual_review_ratio, 2),
            "response_latency_p95_sec": p95_lat,
            "one_sided_drift_score": round(drift_score, 2),
            "active_debates": len(active_debates),
            "converged_debates": len(converged_debates),
            "open_questions": len(open_questions),
            "overdue_questions": len(overdue),
            "answered_questions": len(answered_questions),
            "unresolved_counters": len(unresolved_counters),
            "active_review_pairs": len(active_pairs),
            "total_completed_tasks": len(completed_tasks),
        },
        "overdue_details": overdue[:5],
        "warnings": health_warnings,
        "schema_version": "2.2",
    }


# ===========================================================================
# v2.4 — Dual-Agent Autonomous Cooperation (4 protocols)
# ===========================================================================
# Spec: V2_4_DUAL_AGENT_DESIGN.md (Korean source: dual_agent_system_v2.md)
# Protocols: Presence + Work Distribution + Peer Review Loop + Goal-Driven Outer Loop
#
# Layout under /tmp/ultradebate_v2/:
#   heartbeat/{agent}.json       — status enum + metadata (extends v2.3)
#   inbox/{agent}.jsonl          — priority queue
#   events.jsonl                 — append-only status transition log
#   cycles/{cycle_id}/phase.json — outer loop state
#   cycles/{cycle_id}/plan.yaml  — task DAG + signoffs
#   cycles/{cycle_id}/goal/{claude,codex,reconciled}.goal.md
#   cycles/{cycle_id}/artifacts/{task_id}/{draft.*, review.md, debate.round_N.md, accepted.*}
#
# Bilateral rules B13-B20 (additive on top of B1-B12).
# All tools idempotent; state file writes are atomic (write+rename).

_V24_BASE = Path("/tmp/ultradebate_v2")
_V24_HEARTBEAT = _V24_BASE / "heartbeat"
_V24_INBOX = _V24_BASE / "inbox"
_V24_EVENTS = _V24_BASE / "events.jsonl"
_V24_CYCLES = _V24_BASE / "cycles"

_V24_VALID_STATUSES = {"working", "thinking", "reviewing", "debating", "idle", "blocked", "offline"}
_V24_VALID_PRIORITIES = {"urgent", "normal", "low"}
_V24_VALID_PHASES = ("intake", "decomposition", "execution", "integration", "acceptance", "done")
_V24_VALID_TASK_STATES = {
    "DRAFT", "SUBMITTED", "UNDER_REVIEW", "CHANGES_REQUESTED",
    "CONTESTED", "DEBATE", "ACCEPTED", "ESCALATE",
}
_V24_PHASE_TRANSITIONS = {
    "intake": {"decomposition"},
    "decomposition": {"execution", "intake"},
    "execution": {"integration"},
    "integration": {"acceptance", "execution"},  # gap re-entry
    "acceptance": {"done"},
    "done": set(),
}
_V24_TASK_STATE_TRANSITIONS = {
    "DRAFT": {"SUBMITTED"},
    "SUBMITTED": {"UNDER_REVIEW"},
    "UNDER_REVIEW": {"ACCEPTED", "CHANGES_REQUESTED"},
    "CHANGES_REQUESTED": {"DRAFT", "CONTESTED"},
    "CONTESTED": {"DEBATE"},
    "DEBATE": {"DRAFT", "ACCEPTED", "ESCALATE"},
    "ACCEPTED": set(),
    "ESCALATE": set(),
}
_V24_STALE_THRESHOLD_SEC = 900  # 15min


def _v24_now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(timespec="seconds")


def _v24_ensure_dirs() -> None:
    _V24_BASE.mkdir(parents=True, exist_ok=True)
    _V24_HEARTBEAT.mkdir(parents=True, exist_ok=True)
    _V24_INBOX.mkdir(parents=True, exist_ok=True)
    _V24_CYCLES.mkdir(parents=True, exist_ok=True)


def _v24_cycle_dir(cycle_id: str) -> Path:
    safe_id = "".join(c for c in cycle_id if c.isalnum() or c in "-_")
    if not safe_id or safe_id != cycle_id:
        raise ValueError(f"invalid cycle_id: {cycle_id!r}")
    return _V24_CYCLES / cycle_id


def _v24_atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _v24_atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _v24_emit_event(event: dict) -> None:
    _v24_ensure_dirs()
    event = {"ts": _v24_now_iso(), **event}
    with _V24_EVENTS.open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _v24_load_phase(cycle_id: str) -> dict:
    path = _v24_cycle_dir(cycle_id) / "phase.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _v24_load_plan(cycle_id: str) -> dict:
    path = _v24_cycle_dir(cycle_id) / "plan.yaml"
    if not path.exists():
        return {}
    # Lightweight YAML parser: we only emit subset. Try yaml then fallback to json-as-yaml.
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text()) or {}
    except ImportError:
        # Fallback: plan.yaml written as JSON-compatible YAML
        return json.loads(path.read_text())


def _v24_save_plan(cycle_id: str, plan: dict) -> None:
    path = _v24_cycle_dir(cycle_id) / "plan.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
        text = yaml.safe_dump(plan, sort_keys=False, allow_unicode=True)
    except ImportError:
        text = json.dumps(plan, indent=2, ensure_ascii=False)
    _v24_atomic_write_text(path, text)


# ---------------------------------------------------------------------------
# Presence Protocol (Protocol 1)
# ---------------------------------------------------------------------------


@mcp.tool()
def ultradebate_v2_presence_update(
    agent: str,
    status: str,
    metadata: dict | None = None,
) -> dict:
    """Write status enum + metadata for an agent's heartbeat (v2.4).

    Args:
        agent: 'claude' or 'codex'
        status: one of working|thinking|reviewing|debating|idle|blocked|offline
        metadata: dict with status-appropriate fields:
            working: {task_id, started_at, eta}
            thinking: {topic, started_at}
            reviewing: {target_artifact, target_agent}
            debating: {task_id, round, topic}
            idle: {since}
            blocked: {reason, waiting_for}
            offline: {last_seen}

    Returns:
        {ok, agent, status, prev_status, ts, event_emitted}
    """
    if agent not in ("claude", "codex"):
        return {"ok": False, "error": f"agent must be claude|codex, got {agent!r}"}
    if status not in _V24_VALID_STATUSES:
        return {"ok": False, "error": f"status must be one of {sorted(_V24_VALID_STATUSES)}, got {status!r}"}

    _v24_ensure_dirs()
    metadata = metadata or {}
    now = _v24_now_iso()
    hb_path = _V24_HEARTBEAT / f"{agent}.json"
    prev_status = None
    prev_payload = {}
    if hb_path.exists():
        try:
            prev_payload = json.loads(hb_path.read_text())
            prev_status = prev_payload.get("status")
        except Exception:
            pass

    new_payload = {
        "agent": agent,
        "status": status,
        "current_task": metadata if status in {"working", "reviewing", "debating"} else metadata,
        "last_ping": now,
        "queue_depth": metadata.get("queue_depth", prev_payload.get("queue_depth", 0)),
        "next_available": metadata.get("next_available"),
        "schema_version": "2.4.0",
    }
    _v24_atomic_write_json(hb_path, new_payload)

    if prev_status != status:
        _v24_emit_event({
            "agent": agent,
            "event": "status_transition",
            "from": prev_status,
            "to": status,
            "context": metadata,
        })

    return {
        "ok": True,
        "agent": agent,
        "status": status,
        "prev_status": prev_status,
        "ts": now,
        "event_emitted": prev_status != status,
    }


@mcp.tool()
def ultradebate_v2_presence_read(agent: str | None = None) -> dict:
    """Read agent presence + staleness flag (v2.4).

    Args:
        agent: 'claude', 'codex', or None for both

    Returns:
        {ok, agents: {claude: {...}, codex: {...}}, now}
        Each agent payload includes `stale` boolean (true if last_ping > 15min ago)
        and `auto_offline` (true if watchdog should mark offline).
    """
    import datetime as _dt
    _v24_ensure_dirs()
    now_dt = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9)))
    targets = [agent] if agent else ["claude", "codex"]
    result = {}
    for a in targets:
        if a not in ("claude", "codex"):
            continue
        hb_path = _V24_HEARTBEAT / f"{a}.json"
        if not hb_path.exists():
            result[a] = {"status": "unknown", "stale": True, "auto_offline": True}
            continue
        try:
            payload = json.loads(hb_path.read_text())
            last_ping = _dt.datetime.fromisoformat(payload.get("last_ping", "1970-01-01T00:00:00+09:00"))
            age_sec = (now_dt - last_ping).total_seconds()
            payload["age_sec"] = int(age_sec)
            payload["stale"] = age_sec > _V24_STALE_THRESHOLD_SEC
            payload["auto_offline"] = age_sec > 2 * _V24_STALE_THRESHOLD_SEC
            result[a] = payload
        except Exception as exc:
            result[a] = {"status": "error", "error": str(exc), "stale": True}
    return {"ok": True, "agents": result, "now": now_dt.isoformat(timespec="seconds")}


@mcp.tool()
def ultradebate_v2_presence_history(agent: str | None = None, n: int = 50) -> dict:
    """Return recent status_transition events from events.jsonl (v2.4)."""
    _v24_ensure_dirs()
    if not _V24_EVENTS.exists():
        return {"ok": True, "events": []}
    events = []
    with _V24_EVENTS.open() as f:
        for line in f:
            try:
                ev = json.loads(line)
                if ev.get("event") != "status_transition":
                    continue
                if agent and ev.get("agent") != agent:
                    continue
                events.append(ev)
            except Exception:
                continue
    return {"ok": True, "events": events[-int(n):]}


# ---------------------------------------------------------------------------
# Priority Messaging (Protocol 1)
# ---------------------------------------------------------------------------


def _v25_attempt_codex_tmux_wake(
    msg_id: str, msg_type: str, sender: str, priority: str
) -> dict[str, Any]:
    """Best-effort tmux auto-wake for Codex — paste-buffer + forced C-m.

    Implements ENF-1 (CLAUDE.md §18 B33). When priority_send delivers an urgent
    message to Codex via the JSONL inbox, the receiver has no automatic
    notification — historically this caused messages to be buried for 30+
    minutes (see Empirical Evidence in CLAUDE.md §18).

    SINGLE delivery path (NO state branching) — user mandate 2026-05-24
    ("C-m을 확실하게 강제 … 이상한 분기가 생기거나 하지 않게"):
      - Always: paste-buffer (flatten → one bracketed-paste atom) + a single
        `C-m` via _v2_tmux_safe_send. `C-m` is the Codex CLI submit key = the
        user's keyboard Enter. It submits when Codex is idle and queues the
        payload as Codex's next task when Codex is working (user-verified
        2026-05-24).

    Why C-m is safe here (the "/n처럼 되는" regression was NOT caused by C-m):
      - The regression came from RAW multi-line `send-keys` — tmux splits
        embedded newlines into Enter keypresses, fragmenting the payload.
      - _v2_tmux_safe_send flattens newlines and pastes the payload as ONE
        atom, then sends ONE C-m. A single-line atom + C-m cannot split → no
        /n, independent of Codex busy state. paste-buffer is the real fix; the
        submit-key choice (C-m vs Escape+Enter) was a red herring.
      - The brief 2026-05-24 Escape+Enter experiment + Working/Idle branching
        was REVERTED: the user confirmed C-m is the working submit key for
        their Codex CLI and that the branching was fragile.

    Safety:
      - Only acts on priority='urgent' (avoids spam on normal/low).
      - Only acts when Codex transport='tmux' (no-op otherwise).
      - paste-buffer SSoT helper (no raw send-keys; no /n split).
      - All exceptions swallowed — never blocks the caller's priority_send.

    Returns: {attempted, woken, queued, reason, pane}
      - attempted: True if a tmux delivery was issued (paste + C-m)
      - woken: True if the C-m submit was verified (busy marker appeared)
      - queued: always False (no separate queue path; C-m queues inline)
      - reason: human-readable explanation
      - pane: tmux pane target or None
    """
    result: dict[str, Any] = {
        "attempted": False, "woken": False, "queued": False,
        "reason": "", "pane": None,
    }

    if priority != "urgent":
        result["reason"] = "non-urgent priority, skip (avoid spam)"
        return result

    try:
        resolved = _v26_resolve_peer_pane("codex")
    except Exception as exc:
        result["reason"] = f"pane lookup failed: {exc!r}"
        return result
    if not resolved.get("ok") or resolved.get("selection_reason") not in {
        "cwd_match", "cwd_child", "cwd_parent",
    }:
        result["reason"] = "no codex tmux pane matching the current project"
        return result
    codex_pane = resolved.get("pane")
    if not codex_pane:
        result["reason"] = "resolved pane is missing"
        return result
    result["pane"] = codex_pane

    # SINGLE forced-C-m path — NO state branching (user mandate 2026-05-24:
    # "C-m을 확실하게 강제 … 이상한 분기가 생기거나 하지 않게"). The earlier
    # Working→Tab / Idle→Escape+Enter branching is REMOVED.
    #
    # C-m is the Codex CLI submit key = the user's keyboard Enter. It submits
    # when Codex is idle and queues the payload as Codex's next task when Codex
    # is working (user-verified). _v2_tmux_safe_send flattens newlines + pastes
    # the payload as ONE bracketed-paste atom, then sends ONE C-m — a
    # single-line atom + C-m cannot split, so there is no "/n처럼 되는"
    # regression regardless of Codex busy state. (paste-buffer is the real /n
    # fix; the submit-key choice was a red herring.)
    result["attempted"] = True
    try:
        wake_msg = (
            f"[MCP-AUTO-WAKE] urgent priority_send from {sender}: "
            f"type={msg_type} msg_id={msg_id}"
        )
        ok = _v2_tmux_safe_send(
            codex_pane, wake_msg, submit_keys=["C-m"], socket=resolved.get("socket"),
        )
        result["woken"] = bool(ok)
        result["reason"] = (
            f"C-m submit via paste-buffer to {codex_pane}"
            if ok else
            f"C-m submit unverified for {codex_pane} (inbox JSONL still delivered)"
        )
    except Exception as exc:
        result["reason"] = f"wake exception: {exc!r}"

    return result


@mcp.tool()
def ultradebate_v2_priority_send(
    sender: str,
    target: str,
    priority: str,
    msg_type: str,
    payload: dict,
    ref: str | None = None,
    cycle_id: str | None = None,
    auto_wake: bool = True,
) -> dict:
    """Append a priority-tagged message to target peer's inbox (v2.4).

    Args:
        sender: 'claude' or 'codex'
        target: 'claude' or 'codex' (must differ from sender)
        priority: urgent|normal|low
        msg_type: arbitrary type tag (e.g., 'review_feedback', 'goal_proposal')
        payload: arbitrary JSON-serializable dict
        ref: optional reference id (task_id, cycle_id, debate_id, etc.)
        cycle_id: explicit cycle_id for v2.5 dispatch auto-counting (MEDIUM-2 fix).
            When omitted at a cycle boundary, mtime-based resolution may attribute
            the counter to the wrong cycle. Prefer passing the canonical cycle_id
            from session.json `current_cycle_id` when available.
        auto_wake: attempt Codex tmux wake for urgent messages. Set False for
            inbox-only delivery or when the caller handles tmux submission.

    Returns:
        {ok, msg_id, ts, target_inbox_path}
    """
    if sender not in ("claude", "codex") or target not in ("claude", "codex"):
        return {"ok": False, "error": "sender/target must be claude|codex"}
    if sender == target:
        return {"ok": False, "error": "sender must differ from target"}
    if priority not in _V24_VALID_PRIORITIES:
        return {"ok": False, "error": f"priority must be one of {sorted(_V24_VALID_PRIORITIES)}"}

    _v24_ensure_dirs()
    msg_type = _json_safe_text(str(msg_type))
    ref = _json_safe_obj(ref)
    payload = _json_safe_obj(payload)
    import uuid as _uuid
    msg_id = _uuid.uuid4().hex[:16]
    now = _v24_now_iso()
    msg = {
        "ts": now,
        "from": sender,
        "to": target,
        "priority": priority,
        "type": msg_type,
        "ref": ref,
        "msg_id": msg_id,
        "payload": payload,
    }
    inbox = _V24_INBOX / f"{target}.jsonl"
    with inbox.open("a") as f:
        # _json_dumps_safe sanitizes payload + re-sanitizes the encoded string,
        # so no lone surrogate (\\uD8xx escape) can land in the JSONL even if
        # the caller passed a payload that bypassed the helper-sanitize at
        # lines 3372-3374.
        line = _json_dumps_safe(msg, ensure_ascii=False)
        f.write(line + "\n")
    _v24_emit_event({
        "agent": sender,
        "event": "priority_message_sent",
        "context": {"to": target, "priority": priority, "msg_id": msg_id, "type": msg_type, "ref": ref},
    })
    # v2.5 B21 auto-counting: dispatch messages increment sender's dispatch_messages_sent.
    # "dispatch" identified by msg_type containing 'dispatch' (case-insensitive).
    # MEDIUM-2 fix (review-eng audit 2026-05-15): prefer explicit cycle_id over mtime
    # resolution to avoid attributing the counter to the wrong cycle at boundaries.
    try:
        if "dispatch" in str(msg_type).lower():
            _v25_increment_metric(
                sender, "dispatch_messages_sent", 1, cycle_id=cycle_id,
            )
    except Exception:
        pass  # auto-counting is best-effort, must not block priority_send

    # Urgent direct sends may wake Codex; wrappers own their submission path.
    wake_info: dict[str, Any] = {
        "attempted": False, "woken": False, "queued": False,
        "reason": "auto-wake disabled or target is not codex", "pane": None,
    }
    if target == "codex" and auto_wake:
        try:
            wake_info = _v25_attempt_codex_tmux_wake(msg_id, msg_type, sender, priority)
        except Exception as exc:
            wake_info = {
                "attempted": False, "woken": False, "queued": False,
                "reason": f"wake exception: {exc!r}", "pane": None,
            }
    return _json_safe_obj({
        "ok": True, "msg_id": msg_id, "ts": now,
        "target_inbox_path": str(inbox), "auto_wake": wake_info,
    })


@mcp.tool()
def ultradebate_v2_priority_inbox_read(
    agent: str,
    min_priority: str = "low",
    max_n: int = 20,
    drain: bool = False,
) -> dict:
    """Read agent's inbox sorted by priority (urgent>normal>low).

    Args:
        agent: 'claude' or 'codex'
        min_priority: minimum priority to include (urgent only / normal+up / low+up)
        max_n: max messages to return
        drain: if True, also mark them processed (move to /tmp/ultradebate_v2/{agent}/processed/)

    Returns:
        {ok, messages: [...], remaining_after}
    """
    if agent not in ("claude", "codex"):
        return {"ok": False, "error": "agent must be claude|codex"}
    if min_priority not in _V24_VALID_PRIORITIES:
        return {"ok": False, "error": f"min_priority must be one of {sorted(_V24_VALID_PRIORITIES)}"}

    _v24_ensure_dirs()
    inbox = _V24_INBOX / f"{agent}.jsonl"
    if not inbox.exists():
        return {"ok": True, "messages": [], "remaining_after": 0}

    priority_rank = {"urgent": 0, "normal": 1, "low": 2}
    min_rank = priority_rank[min_priority]

    all_msgs = []
    with inbox.open() as f:
        for line in f:
            try:
                m = json.loads(line)
                if priority_rank.get(m.get("priority", "normal"), 1) <= min_rank:
                    all_msgs.append(_json_safe_obj(m))
            except Exception:
                continue
    all_msgs.sort(key=lambda m: (priority_rank.get(m.get("priority", "normal"), 1), m.get("ts", "")))
    selected = all_msgs[: int(max_n)]
    remaining_after = len(all_msgs) - len(selected) if drain else len(all_msgs)

    if drain and selected:
        # Move drained msgs to processed dir
        processed = _V24_BASE / agent / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        selected_ids = {m["msg_id"] for m in selected if "msg_id" in m}
        remaining = [m for m in all_msgs if m.get("msg_id") not in selected_ids]
        # Rewrite inbox without drained msgs
        with inbox.open("w") as f:
            for m in remaining:
                f.write(_json_dumps_safe(m, ensure_ascii=False) + "\n")
        # Archive drained
        for m in selected:
            arch = processed / f"{m.get('msg_id', 'unknown')}.json"
            try:
                arch.write_text(_json_dumps_safe(m, indent=2, ensure_ascii=False))
            except Exception:
                pass

    return _json_safe_obj({"ok": True, "messages": selected, "remaining_after": remaining_after})


@mcp.tool()
async def ultradebate_v2_wait_for_peer_message(
    agent: str,
    timeout_seconds: int = 600,
    min_priority: str = "low",
    since_ts: str | None = None,
    poll_interval_seconds: float = 2.0,
) -> dict:
    """Long-poll the agent's inbox; return as soon as a qualifying new message arrives.

    Solves the message-burying problem: priority_send appends to the inbox JSONL but
    the receiving CLI (Claude / Codex) has no immediate trigger. Callers (typically
    Claude while Codex is Working) invoke this tool to BLOCK until Codex's next
    priority_send lands, eliminating IDLE-stall and reducing message latency from
    "wakeup-bound" (minutes) to "poll-interval-bound" (seconds).

    Behavior:
      - Snapshots inbox file size at entry; only messages appended AFTER that
        snapshot can satisfy the wait (existing un-drained messages do NOT
        cause an immediate return — callers should call priority_inbox_read first
        if they want to drain backlog).
      - Polls every ``poll_interval_seconds``; on size growth, parses new tail
        lines and filters by ``min_priority`` (urgent>normal>low using the same
        priority_rank convention as priority_inbox_read: urgent=0, normal=1, low=2).
      - Optional ``since_ts`` further restricts qualifying messages to those with
        ts strictly newer than the supplied ISO-8601 timestamp.

    Args:
        agent: 'claude' or 'codex' (whose inbox to watch — usually the caller's own).
        timeout_seconds: max wait (clamped to 1..3600, default 600 = 10 min).
        min_priority: minimum priority that triggers return (urgent|normal|low).
        since_ts: optional ISO-8601 timestamp; only messages with ts > since_ts qualify.
        poll_interval_seconds: file-stat poll period (clamped to 0.5..30.0, default 2.0).

    Returns:
        On message arrival:
            {ok: True, new_messages: True, message_count: int (<=5),
             elapsed_sec: float, timeout: False,
             latest_message: dict, all_new_messages: [dict, ...],
             inbox_path: str}
        On timeout:
            {ok: True, new_messages: False, message_count: 0,
             elapsed_sec: float, timeout: True,
             latest_message: None, inbox_path: str}
        On error:
            {ok: False, error: str}

    Sample Claude usage (next session, after MCP server restart):
        await mcp__ultradebate__ultradebate_v2_wait_for_peer_message(
            agent="claude", timeout_seconds=300, min_priority="urgent")

    NOTE: requires MCP server restart to register — user mandate: applied next session.
    """
    if agent not in ("claude", "codex"):
        return {"ok": False, "error": "agent must be claude|codex"}
    if min_priority not in _V24_VALID_PRIORITIES:
        return {"ok": False, "error": f"min_priority must be one of {sorted(_V24_VALID_PRIORITIES)}"}

    # Clamp to safe ranges
    timeout_seconds = max(1, min(int(timeout_seconds), 3600))
    poll_interval_seconds = max(0.5, min(float(poll_interval_seconds), 30.0))

    # Mirror priority_inbox_read convention: urgent=0 (highest) ... low=2 (lowest).
    # A message qualifies if its rank <= min_rank (i.e., priority at least as high).
    priority_rank = {"urgent": 0, "normal": 1, "low": 2}
    min_rank = priority_rank[min_priority]

    # Parse since_ts (optional) — use ISO-8601 lexicographic comparison
    # (consistent with how _v24_now_iso emits and how priority_inbox_read sorts).
    since_ts_norm: str | None = None
    if since_ts:
        # Accept trailing 'Z' (UTC) by normalizing to '+00:00' for downstream consumers,
        # but the comparison itself is purely lexicographic on the original string.
        since_ts_norm = since_ts.strip()
        if not since_ts_norm:
            since_ts_norm = None

    _v24_ensure_dirs()
    inbox_path = _V24_INBOX / f"{agent}.jsonl"

    # Snapshot baseline: any message appended after this byte offset is "new".
    baseline_size = inbox_path.stat().st_size if inbox_path.exists() else 0

    start = time.monotonic()
    deadline = start + timeout_seconds

    while True:
        await asyncio.sleep(poll_interval_seconds)
        now = time.monotonic()
        if now >= deadline:
            break

        if not inbox_path.exists():
            # Inbox dir may exist but file not yet created — keep waiting.
            continue

        try:
            current_size = inbox_path.stat().st_size
        except OSError as e:
            return {"ok": False, "error": f"inbox stat failed: {e}"}

        if current_size <= baseline_size:
            continue

        # File grew — read only the appended tail (avoid re-parsing entire history).
        try:
            with inbox_path.open("rb") as f:
                f.seek(baseline_size)
                tail_bytes = f.read(current_size - baseline_size)
        except (OSError, IOError) as e:
            return {"ok": False, "error": f"inbox read failed: {e}"}

        # Update baseline regardless of qualification outcome — already-seen bytes
        # should not be re-scanned next iteration even if no message qualifies.
        baseline_size = current_size

        try:
            tail_text = tail_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return {"ok": False, "error": f"inbox decode failed: {e}"}

        # Parse newest-first to surface the latest qualifying message quickly.
        candidates: list[dict] = []
        for raw_line in reversed(tail_text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate partial writes from concurrent producers

            msg_priority = msg.get("priority", "normal")
            if priority_rank.get(msg_priority, 1) > min_rank:
                continue  # below min_priority threshold

            if since_ts_norm:
                msg_ts = msg.get("ts", "")
                # ISO-8601 lexicographic comparison (same precision = correct order).
                if not msg_ts or msg_ts <= since_ts_norm:
                    continue

            candidates.append(_json_safe_obj(msg))
            if len(candidates) >= 5:  # cap response size
                break

        if candidates:
            return _json_safe_obj({
                "ok": True,
                "new_messages": True,
                "message_count": len(candidates),
                "elapsed_sec": round(time.monotonic() - start, 2),
                "timeout": False,
                "latest_message": candidates[0],
                "all_new_messages": candidates,
                "inbox_path": str(inbox_path),
            })
        # No qualifying message in this growth — continue polling until deadline.

    # Timeout path
    return _json_safe_obj({
        "ok": True,
        "new_messages": False,
        "message_count": 0,
        "elapsed_sec": round(time.monotonic() - start, 2),
        "timeout": True,
        "latest_message": None,
        "inbox_path": str(inbox_path),
    })


# ---------------------------------------------------------------------------
# Goal-Driven Outer Loop (Protocol 4)
# ---------------------------------------------------------------------------


@mcp.tool()
def ultradebate_v2_cycle_create(cycle_id: str, user_request: str, started_by: str = "claude") -> dict:
    """Initialize a v2.4 cycle directory + Phase 1 (intake).

    Args:
        cycle_id: unique cycle id (e.g., 'C-2026-05-12-001')
        user_request: original user prompt verbatim
        started_by: 'claude' or 'codex' (who initiated)
    """
    if started_by not in ("claude", "codex"):
        return {"ok": False, "error": "started_by must be claude|codex"}
    _v24_ensure_dirs()
    cdir = _v24_cycle_dir(cycle_id)
    if (cdir / "phase.json").exists():
        return {"ok": False, "error": f"cycle {cycle_id} already exists"}
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "goal").mkdir(exist_ok=True)
    (cdir / "artifacts").mkdir(exist_ok=True)
    now = _v24_now_iso()
    phase_state = {
        "cycle_id": cycle_id,
        "schema_version": "2.4.0",
        "phase": "intake",
        "phase_history": [{"phase": "intake", "entered_at": now, "exited_at": None, "evidence": "cycle_created"}],
        "started_at": now,
        "started_by": started_by,
        "user_request": user_request,
        "current_iteration": 1,
        "max_iterations": 3,
        "tasks_total": 0,
        "tasks_accepted": 0,
        "tasks_in_progress": 0,
        "tasks_blocked": 0,
        "escalations": [],
    }
    _v24_atomic_write_json(cdir / "phase.json", phase_state)
    _v24_emit_event({
        "agent": started_by,
        "event": "cycle_created",
        "context": {"cycle_id": cycle_id, "phase": "intake"},
    })
    return {"ok": True, "cycle_id": cycle_id, "phase": "intake", "cycle_dir": str(cdir)}


@mcp.tool()
def ultradebate_v2_cycle_status(cycle_id: str) -> dict:
    """Return current phase + counts + escalations + signoff state for a cycle."""
    phase = _v24_load_phase(cycle_id)
    if not phase:
        return {"ok": False, "error": f"cycle {cycle_id} not found"}
    plan = _v24_load_plan(cycle_id)
    signoffs = (plan or {}).get("signoffs", {})
    tasks = (plan or {}).get("tasks", [])
    task_state_counts = {}
    for t in tasks:
        s = t.get("state", "DRAFT")
        task_state_counts[s] = task_state_counts.get(s, 0) + 1
    return {
        "ok": True,
        "cycle_id": cycle_id,
        "phase": phase.get("phase"),
        "phase_history": phase.get("phase_history", []),
        "tasks_total": len(tasks),
        "task_state_counts": task_state_counts,
        "escalations": phase.get("escalations", []),
        "signoffs": signoffs,
        "iteration": phase.get("current_iteration", 1),
    }


@mcp.tool()
def ultradebate_v2_cycle_phase_advance(
    cycle_id: str,
    target_phase: str,
    evidence: dict,
    advanced_by: str = "claude",
) -> dict:
    """Transition cycle to target_phase if gate condition met (v2.4 B17).

    Gates:
      intake → decomposition: both goal.md exist + reconciled.goal.md committed
      decomposition → execution: plan.yaml has both signoffs.decision=='approve'
      execution → integration: all plan.yaml tasks state==ACCEPTED
      integration → acceptance: all acceptance_criteria checked
      acceptance → done: both peers sign final report (evidence must contain final_report_path)
    """
    if target_phase not in _V24_VALID_PHASES:
        return {"ok": False, "error": f"target_phase must be one of {_V24_VALID_PHASES}"}
    if advanced_by not in ("claude", "codex"):
        return {"ok": False, "error": "advanced_by must be claude|codex"}
    phase = _v24_load_phase(cycle_id)
    if not phase:
        return {"ok": False, "error": f"cycle {cycle_id} not found"}
    current = phase.get("phase")
    if target_phase not in _V24_PHASE_TRANSITIONS.get(current, set()):
        return {
            "ok": False,
            "error": f"illegal transition {current!r} → {target_phase!r}",
            "allowed_from_current": sorted(_V24_PHASE_TRANSITIONS.get(current, set())),
        }

    # Gate checks
    cdir = _v24_cycle_dir(cycle_id)
    gate_ok = True
    gate_reason = []
    if current == "intake" and target_phase == "decomposition":
        for a in ("claude", "codex"):
            if not (cdir / "goal" / f"{a}.goal.md").exists():
                gate_ok = False
                gate_reason.append(f"missing {a}.goal.md")
        if not (cdir / "goal" / "reconciled.goal.md").exists():
            gate_ok = False
            gate_reason.append("missing reconciled.goal.md")
    elif current == "decomposition" and target_phase == "execution":
        plan = _v24_load_plan(cycle_id)
        so = (plan or {}).get("signoffs", {})
        for a in ("claude", "codex"):
            if so.get(a, {}).get("decision") != "approve":
                gate_ok = False
                gate_reason.append(f"{a} signoff != approve")
    elif current == "execution" and target_phase == "integration":
        plan = _v24_load_plan(cycle_id)
        for t in (plan or {}).get("tasks", []):
            if t.get("state") != "ACCEPTED":
                gate_ok = False
                gate_reason.append(f"task {t.get('id')} state={t.get('state')} != ACCEPTED")
    elif current == "integration" and target_phase == "acceptance":
        if not evidence.get("all_acceptance_criteria_passed"):
            gate_ok = False
            gate_reason.append("evidence.all_acceptance_criteria_passed != truthy")
    elif current == "acceptance" and target_phase == "done":
        if not evidence.get("final_report_path"):
            gate_ok = False
            gate_reason.append("missing final_report_path")
        for a in ("claude", "codex"):
            if not evidence.get(f"signoff_{a}"):
                gate_ok = False
                gate_reason.append(f"missing signoff_{a}")

    if not gate_ok:
        return {"ok": False, "error": "gate_check_failed", "reasons": gate_reason}

    now = _v24_now_iso()
    # Close current phase entry
    history = phase.get("phase_history", [])
    if history and history[-1].get("phase") == current and history[-1].get("exited_at") is None:
        history[-1]["exited_at"] = now
    history.append({"phase": target_phase, "entered_at": now, "exited_at": None, "evidence": evidence})
    phase["phase"] = target_phase
    phase["phase_history"] = history
    _v24_atomic_write_json(cdir / "phase.json", phase)
    _v24_emit_event({
        "agent": advanced_by,
        "event": "phase_transition",
        "context": {"cycle_id": cycle_id, "from": current, "to": target_phase, "evidence": evidence},
    })
    return {"ok": True, "cycle_id": cycle_id, "from": current, "to": target_phase, "ts": now}


@mcp.tool()
def ultradebate_v2_goal_submit(cycle_id: str, agent: str, goal_md_content: str) -> dict:
    """Write per-peer goal.md for the cycle's intake phase."""
    if agent not in ("claude", "codex"):
        return {"ok": False, "error": "agent must be claude|codex"}
    cdir = _v24_cycle_dir(cycle_id)
    if not (cdir / "phase.json").exists():
        return {"ok": False, "error": f"cycle {cycle_id} not found"}
    goal_path = cdir / "goal" / f"{agent}.goal.md"
    _v24_atomic_write_text(goal_path, goal_md_content)
    _v24_emit_event({
        "agent": agent,
        "event": "goal_submitted",
        "context": {"cycle_id": cycle_id, "size_chars": len(goal_md_content)},
    })
    return {"ok": True, "cycle_id": cycle_id, "agent": agent, "path": str(goal_path)}


@mcp.tool()
def ultradebate_v2_goal_reconcile(cycle_id: str, reconciled_md: str, reconciled_by: str = "claude") -> dict:
    """Commit reconciled.goal.md after both peers have submitted theirs."""
    if reconciled_by not in ("claude", "codex"):
        return {"ok": False, "error": "reconciled_by must be claude|codex"}
    cdir = _v24_cycle_dir(cycle_id)
    if not (cdir / "phase.json").exists():
        return {"ok": False, "error": f"cycle {cycle_id} not found"}
    for a in ("claude", "codex"):
        if not (cdir / "goal" / f"{a}.goal.md").exists():
            return {"ok": False, "error": f"missing {a}.goal.md — submit before reconciling"}
    rpath = cdir / "goal" / "reconciled.goal.md"
    _v24_atomic_write_text(rpath, reconciled_md)
    _v24_emit_event({
        "agent": reconciled_by,
        "event": "goal_reconciled",
        "context": {"cycle_id": cycle_id, "size_chars": len(reconciled_md)},
    })
    return {"ok": True, "cycle_id": cycle_id, "path": str(rpath)}


# ---------------------------------------------------------------------------
# Work Distribution (Protocol 2): plan.yaml DAG
# ---------------------------------------------------------------------------


def _v24_validate_dag(tasks: list[dict]) -> tuple[bool, list[str]]:
    """Validate DAG: no cycles, all depends_on refer to existing task ids."""
    errors = []
    ids = {t.get("id") for t in tasks if t.get("id")}
    for t in tasks:
        tid = t.get("id")
        if not tid:
            errors.append("task missing id")
            continue
        deps = t.get("depends_on", []) or []
        for d in deps:
            if d not in ids:
                errors.append(f"task {tid}: unknown dep {d!r}")
            if d == tid:
                errors.append(f"task {tid}: self-dependency")
    # Cycle detection (DFS)
    graph = {t["id"]: list(t.get("depends_on", []) or []) for t in tasks if t.get("id")}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}

    def dfs(n: str) -> bool:
        if color[n] == GRAY:
            return True  # cycle
        if color[n] == BLACK:
            return False
        color[n] = GRAY
        for d in graph.get(n, []):
            if d in color and dfs(d):
                errors.append(f"cycle detected involving {n} → {d}")
                return True
        color[n] = BLACK
        return False

    for n in list(graph.keys()):
        dfs(n)
    return (len(errors) == 0, errors)


@mcp.tool()
def ultradebate_v2_plan_yaml_create(cycle_id: str, plan: dict, created_by: str = "claude") -> dict:
    """Create plan.yaml for the cycle with DAG validation.

    plan dict schema:
        {
          "goal": str,
          "acceptance_criteria": [str, ...],
          "tasks": [
            {"id": "T-001", "title": "...", "assignee": "claude|codex",
             "reviewer": "claude|codex", "depends_on": [...], "estimate_min": int,
             "state": "DRAFT"}
          ],
          "assignments_policy": {...}  # optional
        }
    """
    if created_by not in ("claude", "codex"):
        return {"ok": False, "error": "created_by must be claude|codex"}
    cdir = _v24_cycle_dir(cycle_id)
    if not (cdir / "phase.json").exists():
        return {"ok": False, "error": f"cycle {cycle_id} not found"}
    tasks = plan.get("tasks", [])
    if not tasks:
        return {"ok": False, "error": "plan must contain at least one task"}
    # Validate assignee/reviewer roles
    for t in tasks:
        if t.get("assignee") not in ("claude", "codex"):
            return {"ok": False, "error": f"task {t.get('id')}: invalid assignee"}
        if t.get("reviewer") not in ("claude", "codex"):
            return {"ok": False, "error": f"task {t.get('id')}: invalid reviewer"}
        if t.get("assignee") == t.get("reviewer"):
            return {"ok": False, "error": f"task {t.get('id')}: assignee must differ from reviewer"}
        t.setdefault("state", "DRAFT")
        t.setdefault("depends_on", [])
    ok, errs = _v24_validate_dag(tasks)
    if not ok:
        return {"ok": False, "error": "dag_validation_failed", "errors": errs}

    plan_full = {
        "cycle_id": cycle_id,
        "schema_version": "2.4.0",
        "goal": plan.get("goal", ""),
        "acceptance_criteria": plan.get("acceptance_criteria", []),
        "assignments_policy": plan.get("assignments_policy", {}),
        "tasks": tasks,
        "signoffs": {
            "claude": {"ts": None, "decision": None, "comment": None},
            "codex": {"ts": None, "decision": None, "comment": None},
        },
        "created_by": created_by,
        "created_at": _v24_now_iso(),
    }
    _v24_save_plan(cycle_id, plan_full)
    _v24_emit_event({
        "agent": created_by,
        "event": "plan_yaml_created",
        "context": {"cycle_id": cycle_id, "tasks_count": len(tasks)},
    })
    return {"ok": True, "cycle_id": cycle_id, "tasks_count": len(tasks)}


@mcp.tool()
def ultradebate_v2_plan_yaml_signoff(
    cycle_id: str,
    agent: str,
    decision: str,
    comment: str | None = None,
) -> dict:
    """Sign off on plan.yaml. Decision: approve|reject|revise."""
    if agent not in ("claude", "codex"):
        return {"ok": False, "error": "agent must be claude|codex"}
    if decision not in ("approve", "reject", "revise"):
        return {"ok": False, "error": "decision must be approve|reject|revise"}
    plan = _v24_load_plan(cycle_id)
    if not plan:
        return {"ok": False, "error": f"plan.yaml not found for cycle {cycle_id}"}
    so = plan.setdefault("signoffs", {})
    so[agent] = {"ts": _v24_now_iso(), "decision": decision, "comment": comment}
    _v24_save_plan(cycle_id, plan)
    both_approved = (
        so.get("claude", {}).get("decision") == "approve"
        and so.get("codex", {}).get("decision") == "approve"
    )
    _v24_emit_event({
        "agent": agent,
        "event": "plan_signoff",
        "context": {"cycle_id": cycle_id, "decision": decision, "both_approved": both_approved},
    })
    return {"ok": True, "cycle_id": cycle_id, "agent": agent, "decision": decision, "both_approved": both_approved}


@mcp.tool()
def ultradebate_v2_plan_yaml_ready_tasks(cycle_id: str) -> dict:
    """Return task_ids eligible for dispatch (all deps in ACCEPTED state and self not yet SUBMITTED+)."""
    plan = _v24_load_plan(cycle_id)
    if not plan:
        return {"ok": False, "error": f"plan.yaml not found for cycle {cycle_id}"}
    tasks = plan.get("tasks", [])
    state_by_id = {t.get("id"): t.get("state", "DRAFT") for t in tasks}
    ready = []
    for t in tasks:
        if t.get("state") in {"DRAFT"} and not t.get("dispatched"):
            deps = t.get("depends_on", []) or []
            if all(state_by_id.get(d) == "ACCEPTED" for d in deps):
                ready.append({
                    "id": t.get("id"),
                    "title": t.get("title"),
                    "assignee": t.get("assignee"),
                    "reviewer": t.get("reviewer"),
                    "estimate_min": t.get("estimate_min"),
                })
    return {"ok": True, "cycle_id": cycle_id, "ready_tasks": ready, "count": len(ready)}


@mcp.tool()
def ultradebate_v2_plan_yaml_read(cycle_id: str) -> dict:
    """Read full plan.yaml (parsed)."""
    plan = _v24_load_plan(cycle_id)
    if not plan:
        return {"ok": False, "error": f"plan.yaml not found for cycle {cycle_id}"}
    return {"ok": True, "plan": plan}


# ---------------------------------------------------------------------------
# Peer Review Loop (Protocol 3): per-task state machine
# ---------------------------------------------------------------------------


@mcp.tool()
def ultradebate_v2_task_state_advance(
    cycle_id: str,
    task_id: str,
    new_state: str,
    evidence: dict,
    advanced_by: str = "claude",
) -> dict:
    """Advance per-task state machine. Validates legal transitions (v2.4 B16).

    evidence schema varies per transition:
      SUBMITTED:    {draft_path}
      UNDER_REVIEW: {reviewer, started_at}
      ACCEPTED:     {review_path, all_checks_pass: true}
      CHANGES_REQUESTED: {review_path, issues_count}
      CONTESTED:    {dispute_reason}
      DEBATE:       {round, debate_path}
      ESCALATE:     {reason}
    """
    if advanced_by not in ("claude", "codex"):
        return {"ok": False, "error": "advanced_by must be claude|codex"}
    if new_state not in _V24_VALID_TASK_STATES:
        return {"ok": False, "error": f"new_state must be one of {sorted(_V24_VALID_TASK_STATES)}"}
    plan = _v24_load_plan(cycle_id)
    if not plan:
        return {"ok": False, "error": f"plan.yaml not found for cycle {cycle_id}"}
    tasks = plan.get("tasks", [])
    target_task = None
    for t in tasks:
        if t.get("id") == task_id:
            target_task = t
            break
    if not target_task:
        return {"ok": False, "error": f"task {task_id} not found in plan"}

    current_state = target_task.get("state", "DRAFT")
    if new_state not in _V24_TASK_STATE_TRANSITIONS.get(current_state, set()):
        return {
            "ok": False,
            "error": f"illegal transition {current_state} → {new_state}",
            "allowed": sorted(_V24_TASK_STATE_TRANSITIONS.get(current_state, set())),
        }

    # v2.5 B23: No-Idle Reviewer. If reviewer is entering UNDER_REVIEW and there are
    # other ready (DRAFT-state, deps-met) tasks in the cycle, reviewer must NOT be idle —
    # they should pick one up as owner instead.
    if new_state == "UNDER_REVIEW":
        try:
            reviewer = target_task.get("reviewer")
            if reviewer in ("claude", "codex"):
                rt = ultradebate_v2_plan_yaml_ready_tasks(cycle_id)
                ready_for_reviewer = [
                    t for t in rt.get("ready_tasks", [])
                    if t.get("id") != task_id and t.get("assignee") == reviewer
                ]
                presence = ultradebate_v2_presence_read(reviewer)
                rev_status = (presence.get("agents", {}).get(reviewer, {}) or {}).get("status")
                if rev_status == "idle" and ready_for_reviewer:
                    return {
                        "ok": False,
                        "error": "B23_IDLE_REVIEWER",
                        "code": "B23_IDLE_REVIEWER",
                        "message": (
                            f"Reviewer {reviewer!r} entering review while idle with "
                            f"{len(ready_for_reviewer)} ready owner tasks. Pick one up first."
                        ),
                        "ready_tasks_for_reviewer": [t.get("id") for t in ready_for_reviewer],
                    }
        except Exception:
            pass  # B23 is advisory if probe paths fail; do not block state machine on infra error

    # Iteration limit B18: count DRAFT entries
    history = target_task.setdefault("state_history", [])
    draft_entries = sum(1 for h in history if h.get("state") == "DRAFT")
    if new_state == "DRAFT" and draft_entries >= 4:  # 4 prior + this = 5
        # Force ESCALATE
        new_state = "ESCALATE"
        evidence = {"escalation_reason": "iteration_limit_5_exceeded", **evidence}

    # Debate round limit B19
    if new_state == "DEBATE":
        round_n = evidence.get("round", 1)
        if int(round_n) > 4:
            new_state = "ESCALATE"
            evidence = {"escalation_reason": "debate_round_4_exceeded", **evidence}

    now = _v24_now_iso()
    history.append({"state": new_state, "ts": now, "by": advanced_by, "evidence": evidence})
    target_task["state"] = new_state
    target_task["state_updated_at"] = now
    _v24_save_plan(cycle_id, plan)
    _v24_emit_event({
        "agent": advanced_by,
        "event": "task_state_advance",
        "context": {
            "cycle_id": cycle_id, "task_id": task_id,
            "from": current_state, "to": new_state, "evidence": evidence,
        },
    })

    # v2.5 auto-counting (workload_audit). Best-effort: failures do not block state machine.
    try:
        if new_state == "SUBMITTED":
            # Assignee authored a patch (DRAFT → SUBMITTED).
            assignee = target_task.get("assignee")
            if assignee in ("claude", "codex"):
                _v25_increment_metric(assignee, "patches_authored", 1, cycle_id=cycle_id)
        elif new_state == "CONTESTED":
            # advanced_by initiated a debate.
            if advanced_by in ("claude", "codex"):
                _v25_increment_metric(advanced_by, "debates_initiated", 1, cycle_id=cycle_id)
    except Exception:
        pass

    return {
        "ok": True,
        "cycle_id": cycle_id,
        "task_id": task_id,
        "from": current_state,
        "to": new_state,
        "iteration_count": len([h for h in history if h.get("state") == "DRAFT"]),
        "escalated": new_state == "ESCALATE",
    }


@mcp.tool()
def ultradebate_v2_task_state_query(cycle_id: str, task_id: str | None = None) -> dict:
    """Query state machine status of one task or all tasks in a cycle."""
    plan = _v24_load_plan(cycle_id)
    if not plan:
        return {"ok": False, "error": f"plan.yaml not found for cycle {cycle_id}"}
    tasks = plan.get("tasks", [])
    if task_id:
        for t in tasks:
            if t.get("id") == task_id:
                return {"ok": True, "task": {
                    "id": t.get("id"),
                    "state": t.get("state"),
                    "assignee": t.get("assignee"),
                    "reviewer": t.get("reviewer"),
                    "history": t.get("state_history", []),
                }}
        return {"ok": False, "error": f"task {task_id} not found"}
    return {"ok": True, "tasks": [
        {
            "id": t.get("id"),
            "state": t.get("state"),
            "assignee": t.get("assignee"),
            "reviewer": t.get("reviewer"),
            "iteration_count": len([h for h in t.get("state_history", []) if h.get("state") == "DRAFT"]),
        }
        for t in tasks
    ]}


@mcp.tool()
def ultradebate_v2_artifact_submit(
    cycle_id: str,
    task_id: str,
    artifact_type: str,
    content: str,
    round_n: int | None = None,
    submitted_by: str = "claude",
) -> dict:
    """Write per-task artifact (draft, review, debate.round_N, accepted, rca).

    Args:
        artifact_type: 'draft' | 'review' | 'debate' | 'accepted' | 'rca' (v2.5)
        content: artifact text (markdown or code)
        round_n: required if artifact_type == 'debate'
        submitted_by: 'claude' or 'codex'

    Note: For 'rca' artifacts, prefer `ultradebate_v2_rca_submit` for B22 enforcement
    + counter increment. This tool's 'rca' branch ONLY writes the rca/{agent}.rca.md
    file and does NOT increment `rcas_performed` (prevents double-count when callers
    use both tools for the same content). If you intentionally want the rca file
    written without B22 enforcement, use this tool; otherwise use rca_submit.
    """
    if artifact_type not in ("draft", "review", "debate", "accepted", "rca"):
        return {"ok": False, "error": "artifact_type must be draft|review|debate|accepted|rca"}
    if submitted_by not in ("claude", "codex"):
        return {"ok": False, "error": "submitted_by must be claude|codex"}
    cdir = _v24_cycle_dir(cycle_id)
    if not (cdir / "phase.json").exists():
        return {"ok": False, "error": f"cycle {cycle_id} not found"}

    task_dir = cdir / "artifacts" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    artifact_path: Path
    if artifact_type == "debate":
        if round_n is None:
            return {"ok": False, "error": "round_n required for artifact_type='debate'"}
        artifact_path = task_dir / f"debate.round_{int(round_n)}.md"
    elif artifact_type == "draft":
        # detect ext from content shebang or fallback .md
        artifact_path = task_dir / "draft.md"  # callers may post code as comment-block-wrapped md
    elif artifact_type == "review":
        artifact_path = task_dir / "review.md"
    elif artifact_type == "rca":
        # v2.5 layout: cycles/{id}/artifacts/{task_id}/rca/{agent}.rca.md
        rca_dir = task_dir / "rca"
        rca_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = rca_dir / f"{submitted_by}.rca.md"
    else:  # accepted
        artifact_path = task_dir / "accepted.md"
    _v24_atomic_write_text(artifact_path, content)
    _v24_emit_event({
        "agent": submitted_by,
        "event": "artifact_submitted",
        "context": {
            "cycle_id": cycle_id, "task_id": task_id,
            "artifact_type": artifact_type, "round_n": round_n, "size_chars": len(content),
        },
    })

    # v2.5 auto-counting (workload_audit). Best-effort.
    # MEDIUM-1 fix (review-eng audit 2026-05-15): only count 'review' here.
    # 'rca' counter increment is owned exclusively by `ultradebate_v2_rca_submit`
    # to prevent double-count when callers use both tools for the same content.
    try:
        if artifact_type == "review":
            _v25_increment_metric(submitted_by, "reviews_performed", 1, cycle_id=cycle_id)
    except Exception:
        pass

    return {"ok": True, "path": str(artifact_path), "size_chars": len(content)}


@mcp.tool()
def ultradebate_v2_artifact_read(
    cycle_id: str,
    task_id: str,
    artifact_type: str,
    round_n: int | None = None,
    agent: str | None = None,
) -> dict:
    """Read per-task artifact.

    For artifact_type='rca', supply `agent` ('claude' or 'codex') to select which
    side's RCA to read; if omitted, both sides are returned when available.
    """
    if artifact_type not in ("draft", "review", "debate", "accepted", "rca"):
        return {"ok": False, "error": "artifact_type must be draft|review|debate|accepted|rca"}
    cdir = _v24_cycle_dir(cycle_id)
    task_dir = cdir / "artifacts" / task_id
    if artifact_type == "debate":
        if round_n is None:
            return {"ok": False, "error": "round_n required for artifact_type='debate'"}
        fname = f"debate.round_{int(round_n)}.md"
        artifact_path = task_dir / fname
        if not artifact_path.exists():
            return {"ok": False, "error": f"{fname} not found for task {task_id}"}
        return {"ok": True, "path": str(artifact_path), "content": artifact_path.read_text()}
    if artifact_type == "rca":
        rca_dir = task_dir / "rca"
        if agent in ("claude", "codex"):
            p = rca_dir / f"{agent}.rca.md"
            if not p.exists():
                return {"ok": False, "error": f"{agent}.rca.md not found for task {task_id}"}
            return {"ok": True, "path": str(p), "content": p.read_text()}
        out = {}
        for a in ("claude", "codex"):
            p = rca_dir / f"{a}.rca.md"
            if p.exists():
                out[a] = {"path": str(p), "content": p.read_text()}
        cross = rca_dir / "cross_verify.md"
        if cross.exists():
            out["cross_verify"] = {"path": str(cross), "content": cross.read_text()}
        if not out:
            return {"ok": False, "error": f"no rca artifacts found for task {task_id}"}
        return {"ok": True, "rcas": out}
    fname = {"draft": "draft.md", "review": "review.md", "accepted": "accepted.md"}[artifact_type]
    artifact_path = task_dir / fname
    if not artifact_path.exists():
        return {"ok": False, "error": f"{fname} not found for task {task_id}"}
    return {"ok": True, "path": str(artifact_path), "content": artifact_path.read_text()}


@mcp.tool()
def ultradebate_v2_task_escalate(
    cycle_id: str,
    task_id: str,
    reason: str,
    both_positions: dict,
    escalated_by: str = "claude",
) -> dict:
    """Manual escalation. Both peers' positions captured for human review."""
    if escalated_by not in ("claude", "codex"):
        return {"ok": False, "error": "escalated_by must be claude|codex"}
    phase = _v24_load_phase(cycle_id)
    if not phase:
        return {"ok": False, "error": f"cycle {cycle_id} not found"}
    escalations = phase.setdefault("escalations", [])
    now = _v24_now_iso()
    esc = {
        "task_id": task_id,
        "reason": reason,
        "both_positions": both_positions,
        "escalated_by": escalated_by,
        "ts": now,
    }
    escalations.append(esc)
    _v24_atomic_write_json(_v24_cycle_dir(cycle_id) / "phase.json", phase)
    # Also advance task to ESCALATE
    ultradebate_v2_task_state_advance(
        cycle_id=cycle_id, task_id=task_id, new_state="ESCALATE",
        evidence={"escalation_reason": reason}, advanced_by=escalated_by,
    )
    _v24_emit_event({
        "agent": escalated_by,
        "event": "task_escalated",
        "context": {"cycle_id": cycle_id, "task_id": task_id, "reason": reason},
    })
    return {"ok": True, "cycle_id": cycle_id, "task_id": task_id, "escalations_count": len(escalations)}


# ---------------------------------------------------------------------------
# Watchdog / Health (v2.4)
# ---------------------------------------------------------------------------


@mcp.tool()
def ultradebate_v2_watchdog_check() -> dict:
    """Run staleness + escalation health checks. Returns warnings + auto-actions taken."""
    presence = ultradebate_v2_presence_read()
    warnings = []
    auto_actions = []
    if presence.get("ok"):
        for agent, state in presence["agents"].items():
            if state.get("auto_offline"):
                warnings.append({
                    "level": "AUTO_OFFLINE",
                    "agent": agent,
                    "msg": f"{agent} last_ping age {state.get('age_sec')}s > 2*stale_threshold",
                })
                # Auto-mark offline
                ultradebate_v2_presence_update(
                    agent=agent, status="offline",
                    metadata={"last_seen": state.get("last_ping"), "auto_marked": True},
                )
                auto_actions.append(f"marked {agent} offline")
            elif state.get("stale"):
                warnings.append({
                    "level": "STALE",
                    "agent": agent,
                    "msg": f"{agent} last_ping age {state.get('age_sec')}s > stale_threshold",
                })
    return {"ok": True, "warnings": warnings, "auto_actions": auto_actions, "schema_version": "2.4.0"}


@mcp.tool()
def ultradebate_v2_session_health_v24() -> dict:
    """v2.4 extended session health: combines v2.2 metrics + cycle + state-machine compliance."""
    v22 = ultradebate_v2_session_health()
    _v24_ensure_dirs()
    # Cycle stats
    cycles_active = 0
    cycles_done = 0
    total_escalations = 0
    if _V24_CYCLES.exists():
        for cdir in _V24_CYCLES.iterdir():
            if cdir.is_dir():
                phase = _v24_load_phase(cdir.name)
                if phase.get("phase") == "done":
                    cycles_done += 1
                else:
                    cycles_active += 1
                total_escalations += len(phase.get("escalations", []))
    return {
        "ok": True,
        "v22_metrics": v22.get("metrics", {}),
        "v22_warnings": v22.get("warnings", []),
        "v24_cycles": {
            "active": cycles_active,
            "done": cycles_done,
            "total_escalations": total_escalations,
        },
        "schema_version": "2.4.0",
    }


# ===========================================================================
# v2.5 — User Directive Broadcast · Session Binding · Role Symmetry
# ===========================================================================
# Spec: V2_5_DUAL_AGENT_DESIGN.md  (additive on top of v2.4)
# Protocols: 5 User Directive Broadcast · 6 Session Binding · 7 Role Symmetry
# Bilateral rules B21-B26 (additive on top of B1-B20).
#
# New SSOT layout (extends /tmp/ultradebate_v2/):
#   sessions/{session_hash}/session.json
#   sessions/{session_hash}/user_directives.jsonl
#   sessions/{session_hash}/audit.log
#   cycles/{cycle_id}/workload_audit.json
#   cycles/{cycle_id}/role_swaps.jsonl
#   cycles/{cycle_id}/compliance.jsonl
#   cycles/{cycle_id}/artifacts/{task_id}/rca/{agent}.rca.md
#   cycles/{cycle_id}/artifacts/{task_id}/rca/cross_verify.md
#
# All writes are atomic (write+rename) for read-modify-write files;
# append-only JSONL for event logs.
# ===========================================================================


_V25_SESSIONS = _V24_BASE / "sessions"
_V25_PARITY_RATIO_CAP = 2.0
_V25_PARITY_DIMENSIONS = ("patches", "reviews", "rcas", "dispatches")
_V25_METRIC_KEYS = (
    "patches_authored",
    "reviews_performed",
    "rcas_performed",
    "debates_initiated",
    "dispatch_messages_sent",
    "issues_found",
    "tasks_assigned_as_assignee",
    "tasks_assigned_as_reviewer",
)
_V25_VALID_SCOPES = {"both", "claude_only", "codex_only"}
_V25_VALID_AGENTS = {"claude", "codex"}
_V25_SCHEMA = "2.5.0"


def _v25_session_hash(cwd: str) -> str:
    import hashlib as _hashlib
    return _hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:12]


def _v25_session_dir(session_hash: str) -> Path:
    if not session_hash or not all(c in "0123456789abcdef" for c in session_hash):
        raise ValueError(f"invalid session_hash: {session_hash!r}")
    return _V25_SESSIONS / session_hash


def _v25_ensure_session_dir(session_hash: str) -> Path:
    d = _v25_session_dir(session_hash)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _v25_msg_id() -> str:
    import uuid as _uuid
    return _uuid.uuid4().hex


def _v25_load_audit_path(session_hash: str) -> Path:
    return _v25_session_dir(session_hash) / "audit.log"


def _v25_audit_append(session_hash: str, event: dict) -> None:
    p = _v25_load_audit_path(session_hash)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": _v24_now_iso(), **event}
    with p.open("a") as f:
        f.write(_json_dumps_safe(entry, ensure_ascii=False) + "\n")


def _v25_active_cycle_id() -> str | None:
    """Return the most recently-modified non-done cycle, or None."""
    if not _V24_CYCLES.exists():
        return None
    candidates: list[tuple[float, str]] = []
    for cdir in _V24_CYCLES.iterdir():
        if not cdir.is_dir():
            continue
        phase_path = cdir / "phase.json"
        if not phase_path.exists():
            continue
        try:
            phase = json.loads(phase_path.read_text())
        except Exception:
            continue
        if phase.get("phase") == "done":
            continue
        candidates.append((phase_path.stat().st_mtime, cdir.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _v25_default_metrics() -> dict:
    return {k: 0 for k in _V25_METRIC_KEYS}


def _v25_workload_audit_path(cycle_id: str) -> Path:
    return _v24_cycle_dir(cycle_id) / "workload_audit.json"


def _v25_load_workload_audit(cycle_id: str) -> dict:
    path = _v25_workload_audit_path(cycle_id)
    if not path.exists():
        return {
            "cycle_id": cycle_id,
            "schema_version": _V25_SCHEMA,
            "metrics": {"claude": _v25_default_metrics(), "codex": _v25_default_metrics()},
            "ratios": {dim: 1.0 for dim in _V25_PARITY_DIMENSIONS},
            "asymmetry_alert": False,
            "alert_dimensions": [],
            "recommended_rebalance": {},
        }
    try:
        return json.loads(path.read_text())
    except Exception:
        return {
            "cycle_id": cycle_id,
            "schema_version": _V25_SCHEMA,
            "metrics": {"claude": _v25_default_metrics(), "codex": _v25_default_metrics()},
            "ratios": {dim: 1.0 for dim in _V25_PARITY_DIMENSIONS},
            "asymmetry_alert": False,
            "alert_dimensions": [],
            "recommended_rebalance": {},
        }


def _v25_compute_ratios(audit: dict) -> dict:
    """Recompute ratios + asymmetry_alert in-place. Returns audit."""
    metrics = audit.get("metrics", {})
    claude_m = metrics.get("claude", _v25_default_metrics())
    codex_m = metrics.get("codex", _v25_default_metrics())
    dim_to_key = {
        "patches": "patches_authored",
        "reviews": "reviews_performed",
        "rcas": "rcas_performed",
        "dispatches": "dispatch_messages_sent",
    }
    ratios: dict = {}
    alert_dims: list[str] = []
    for dim in _V25_PARITY_DIMENSIONS:
        k = dim_to_key[dim]
        c = float(claude_m.get(k, 0))
        x = float(codex_m.get(k, 0))
        hi = max(c, x)
        lo = min(c, x)
        if hi == 0:
            r: float | str = 1.0
        elif lo == 0:
            r = "inf"
            alert_dims.append(dim)
        else:
            r = round(hi / lo, 2)
            if r >= _V25_PARITY_RATIO_CAP:
                alert_dims.append(dim)
        ratios[dim] = r
    audit["ratios"] = ratios
    audit["asymmetry_alert"] = bool(alert_dims)
    audit["alert_dimensions"] = alert_dims
    # recommend rebalance: underused side gets next task
    rec: dict = {}
    for dim in alert_dims:
        k = dim_to_key[dim]
        c = float(claude_m.get(k, 0))
        x = float(codex_m.get(k, 0))
        if c > x:
            under = "codex"
        else:
            under = "claude"
        if dim == "patches":
            rec["next_task_assignee"] = under
        elif dim == "rcas":
            rec["next_rca_required_from"] = under
        elif dim == "dispatches":
            rec["next_dispatch_required_from"] = under
        elif dim == "reviews":
            rec["next_review_required_from"] = under
    audit["recommended_rebalance"] = rec
    return audit


def _v25_save_workload_audit(cycle_id: str, audit: dict) -> None:
    audit["schema_version"] = _V25_SCHEMA
    audit["cycle_id"] = cycle_id
    _v25_compute_ratios(audit)
    _v24_atomic_write_json(_v25_workload_audit_path(cycle_id), audit)


def _v25_increment_metric(
    agent: str,
    metric_key: str,
    delta: int = 1,
    cycle_id: str | None = None,
) -> None:
    """Atomically bump a workload metric in the active or specified cycle's audit."""
    if agent not in _V25_VALID_AGENTS:
        return
    if metric_key not in _V25_METRIC_KEYS:
        return
    cid = cycle_id or _v25_active_cycle_id()
    if not cid:
        return  # no active cycle → nothing to count
    try:
        cdir = _v24_cycle_dir(cid)
    except ValueError:
        return
    if not cdir.exists():
        return
    audit = _v25_load_workload_audit(cid)
    metrics = audit.setdefault("metrics", {})
    am = metrics.setdefault(agent, _v25_default_metrics())
    am[metric_key] = int(am.get(metric_key, 0)) + int(delta)
    _v25_save_workload_audit(cid, audit)


# ---------------------------------------------------------------------------
# Protocol 5: User Directive Broadcast
# ---------------------------------------------------------------------------


@mcp.tool()
def ultradebate_v2_user_directive_broadcast(
    verbatim: str,
    claude_interpretation: str = "",
    codex_context: str = "",
    scope: str = "both",
    priority: str = "normal",
    session_hash: str | None = None,
    cwd: str | None = None,
    caller: str = "claude",
) -> dict:
    """Mirror a user directive to peer inbox(es) with explicit interpretation (B24, v2.5).

    Hook handles raw mirroring on UserPromptSubmit; this tool adds Claude's
    interpretation and Codex-side context, plus an audit entry.

    Args:
        verbatim: User's original prompt text (mandatory).
        claude_interpretation: Caller's interpretation/summary.
        codex_context: Additional context Codex needs to act.
        scope: 'both' | 'claude_only' | 'codex_only'.
        priority: 'urgent' | 'normal' | 'low'.
        session_hash: Optional explicit session_hash; resolved from cwd if absent.
        cwd: Working directory; used to compute session_hash if not supplied.
        caller: Originating agent ('claude' or 'codex').

    Returns:
        {ok, msg_id, ts, session_hash, written_inboxes, audit_path}
    """
    if not verbatim:
        return {"ok": False, "error": "verbatim is required"}
    if scope not in _V25_VALID_SCOPES:
        return {"ok": False, "error": f"scope must be one of {sorted(_V25_VALID_SCOPES)}"}
    if priority not in _V24_VALID_PRIORITIES:
        return {"ok": False, "error": f"priority must be one of {sorted(_V24_VALID_PRIORITIES)}"}
    if caller not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "caller must be claude|codex"}

    verbatim = _json_safe_text(verbatim)
    claude_interpretation = _json_safe_text(claude_interpretation)
    codex_context = _json_safe_text(codex_context)

    if not session_hash:
        if not cwd:
            return {"ok": False, "error": "either session_hash or cwd must be supplied"}
        session_hash = _v25_session_hash(cwd)

    _v24_ensure_dirs()
    sdir = _v25_ensure_session_dir(session_hash)
    msg_id = _v25_msg_id()
    now = _v24_now_iso()

    directive_entry = {
        "ts": now,
        "msg_id": msg_id,
        "source": "mcp",
        "caller": caller,
        "scope": scope,
        "priority": priority,
        "verbatim": verbatim,
        "claude_interpretation": claude_interpretation,
        "codex_context": codex_context,
    }
    with (sdir / "user_directives.jsonl").open("a") as f:
        f.write(_json_dumps_safe(directive_entry, ensure_ascii=False) + "\n")

    # Determine destination peers
    dests: list[str] = []
    if scope == "both":
        dests = ["claude", "codex"]
    elif scope == "claude_only":
        dests = ["claude"]
    elif scope == "codex_only":
        dests = ["codex"]

    written_inboxes: list[str] = []
    for dest in dests:
        payload = {
            "session_hash": session_hash,
            "verbatim": verbatim,
            "claude_interpretation": claude_interpretation if dest == "codex" else "",
            "codex_context": codex_context if dest == "codex" else "",
            "scope": scope,
            "caller": caller,
        }
        # Skip writing to caller's own inbox: caller already knows the directive.
        # But for symmetry with audit, we still write if scope explicitly targets caller.
        if dest == caller and scope == "both":
            continue
        msg = {
            "ts": now,
            "from": f"user_via_{caller}",
            "to": dest,
            "priority": priority,
            "type": "user_directive_interpreted",
            "ref": session_hash,
            "msg_id": _v25_msg_id(),
            "payload": payload,
        }
        inbox = _V24_INBOX / f"{dest}.jsonl"
        inbox.parent.mkdir(parents=True, exist_ok=True)
        with inbox.open("a") as f:
            f.write(_json_dumps_safe(msg, ensure_ascii=False) + "\n")
        written_inboxes.append(str(inbox))

    _v25_audit_append(session_hash, {
        "event": "directive_received",
        "source": "mcp",
        "msg_id": msg_id,
        "caller": caller,
        "scope": scope,
        "destinations": dests,
    })

    return _json_safe_obj({
        "ok": True,
        "msg_id": msg_id,
        "ts": now,
        "session_hash": session_hash,
        "written_inboxes": written_inboxes,
        "audit_path": str(_v25_load_audit_path(session_hash)),
    })


@mcp.tool()
def ultradebate_v2_user_directives_tail(
    session_hash: str | None = None,
    cwd: str | None = None,
    since_ts: str | None = None,
    max_n: int = 20,
) -> dict:
    """Return the most recent user directives for a session (B24).

    Args:
        session_hash: Explicit session_hash; computed from cwd if absent.
        cwd: Used to compute session_hash if session_hash not supplied.
        since_ts: ISO timestamp; only directives with ts > since_ts are returned.
        max_n: Max directives to return (most recent first).

    Returns:
        {ok, session_hash, directives: [...], total_in_log}
    """
    if not session_hash:
        if not cwd:
            return {"ok": False, "error": "either session_hash or cwd must be supplied"}
        session_hash = _v25_session_hash(cwd)
    sdir = _v25_session_dir(session_hash)
    log = sdir / "user_directives.jsonl"
    if not log.exists():
        return {"ok": True, "session_hash": session_hash, "directives": [], "total_in_log": 0}
    entries: list[dict] = []
    with log.open() as f:
        for line in f:
            try:
                entries.append(_json_safe_obj(json.loads(line)))
            except Exception:
                continue
    total = len(entries)
    if since_ts:
        entries = [e for e in entries if e.get("ts", "") > since_ts]
    entries.sort(key=lambda e: e.get("ts", ""), reverse=True)
    return _json_safe_obj({
        "ok": True,
        "session_hash": session_hash,
        "directives": entries[: int(max_n)],
        "total_in_log": total,
    })


@mcp.tool()
def ultradebate_v2_directive_consume(
    agent: str,
    msg_id: str,
    session_hash: str | None = None,
    cwd: str | None = None,
) -> dict:
    """Mark a directive as consumed by `agent`, record consumption latency in audit log."""
    if agent not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "agent must be claude|codex"}
    if not msg_id:
        return {"ok": False, "error": "msg_id required"}
    if not session_hash:
        if not cwd:
            return {"ok": False, "error": "either session_hash or cwd must be supplied"}
        session_hash = _v25_session_hash(cwd)

    sdir = _v25_session_dir(session_hash)
    log = sdir / "user_directives.jsonl"
    sent_ts: str | None = None
    if log.exists():
        with log.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("msg_id") == msg_id:
                    sent_ts = e.get("ts")
                    break

    now = _v24_now_iso()
    latency_sec: float | None = None
    if sent_ts:
        try:
            import datetime as _dt
            sent_dt = _dt.datetime.fromisoformat(sent_ts)
            now_dt = _dt.datetime.fromisoformat(now)
            latency_sec = (now_dt - sent_dt).total_seconds()
        except Exception:
            latency_sec = None

    _v25_audit_append(session_hash, {
        "event": "directive_consumed",
        "agent": agent,
        "msg_id": msg_id,
        "latency_sec": latency_sec,
        "sent_ts": sent_ts,
    })

    return {
        "ok": True,
        "msg_id": msg_id,
        "agent": agent,
        "session_hash": session_hash,
        "latency_sec": latency_sec,
        "sent_ts": sent_ts,
        "consumed_ts": now,
    }


# ---------------------------------------------------------------------------
# Protocol 6: Session Binding
# ---------------------------------------------------------------------------


def _v25_load_session_json(session_hash: str) -> dict:
    sdir = _v25_session_dir(session_hash)
    path = sdir / "session.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _v25_save_session_json(session_hash: str, data: dict) -> None:
    sdir = _v25_ensure_session_dir(session_hash)
    _v24_atomic_write_json(sdir / "session.json", data)


def _v25_compute_binding_status(active_agents: dict) -> str:
    alive = [a for a, meta in active_agents.items() if meta and not meta.get("unattached_at")]
    if len(alive) >= 2:
        return "dual_active"
    if len(alive) == 1:
        return "single_active"
    return "idle"


def _v25_scan_agent_procs(agent: str) -> list[dict]:
    """Enumerate all live processes whose comm matches `agent` (codex|claude).

    Returns list of {pid, comm, cwd, start_ticks}, sorted by start_ticks DESC
    (newest first). Used by _v25_auto_detect_agent_proc for both matching and
    friendly error messages.
    """
    if agent not in ("codex", "claude"):
        return []
    procs: list[dict] = []
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return procs
    for pid_str in proc_entries:
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip()
            if agent not in comm.lower():
                continue
            proc_cwd = os.readlink(f"/proc/{pid}/cwd")
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().split()
            start_ticks = int(fields[21]) if len(fields) > 21 else 0
            procs.append({
                "pid": pid,
                "comm": comm,
                "cwd": os.path.realpath(proc_cwd),
                "start_ticks": start_ticks,
            })
        except (OSError, FileNotFoundError, ValueError, IndexError):
            continue
    procs.sort(key=lambda p: -p["start_ticks"])
    return procs


def _v25_cwd_prefix_match(proc_cwd: str, target_cwd: str) -> bool:
    """Safe prefix match using os.path.commonpath.

    True iff proc_cwd == target_cwd OR proc_cwd is a subdirectory of target_cwd.

    Uses commonpath (not startswith) so '/workspace/project_backup' does
    NOT spuriously match '/workspace/project'. Both args expected as
    realpath-resolved absolute paths.
    """
    if proc_cwd == target_cwd:
        return True
    try:
        return os.path.commonpath([proc_cwd, target_cwd]) == target_cwd
    except ValueError:
        # commonpath raises ValueError on mixed absolute/relative or different drives.
        return False


def _v25_auto_detect_agent_proc(agent: str, cwd: str) -> dict | None:
    """Scan /proc for an agent CLI whose cwd is at-or-under `cwd` (v2.5 helper).

    Returns {pid, transport, tmux_meta?, comm, matched_cwd} for the most
    recently started matching process, or None if no match.

    Match rules (2026-05-16 user mandate "prefix 매칭"):
      - /proc/PID/comm contains the agent name (codex|claude).
      - realpath(/proc/PID/cwd) is identical to OR a subdirectory of `cwd`
        (via os.path.commonpath — safe against spurious string-prefix matches).
      - Preference: highest start_time (newest process) when multiple match.
        Practical effect: 'codex in any subdir of the project root' binds to
        the project's session_hash.

    Transport detection: parses /proc/PID/environ for TMUX env var. If present,
    transport='tmux' and tmux_meta is populated from TMUX (socket,server_pid,
    session_idx) + TMUX_PANE. Otherwise transport='stdio'.

    Friendly-error consumers: see _v25_auto_detect_friendly_error() for the
    list-of-candidates message shown when no prefix match is found.
    """
    target_cwd = os.path.realpath(cwd)
    all_procs = _v25_scan_agent_procs(agent)
    candidates = [p for p in all_procs if _v25_cwd_prefix_match(p["cwd"], target_cwd)]
    if not candidates:
        return None
    # candidates already sorted DESC by start_ticks in _v25_scan_agent_procs.
    pid = candidates[0]["pid"]
    comm = candidates[0]["comm"]
    matched_cwd = candidates[0]["cwd"]

    transport = "stdio"
    tmux_meta: dict | None = None
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            environ_raw = f.read().decode("utf-8", errors="ignore")
        env_vars: dict[str, str] = {}
        for kv in environ_raw.split("\0"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                env_vars[k] = v
        tmux_var = env_vars.get("TMUX")
        tmux_pane = env_vars.get("TMUX_PANE")
        if tmux_var:
            # TMUX env format: "<socket>,<server_pid>,<session_idx>"
            parts = tmux_var.split(",")
            sock = parts[0] if len(parts) >= 1 else None
            server_pid = parts[1] if len(parts) >= 2 else None
            session_idx = parts[2] if len(parts) >= 3 else None
            transport = "tmux"
            tmux_meta = {
                "tmux_socket": sock,
                "tmux_server_pid": int(server_pid) if server_pid and server_pid.isdigit() else None,
                "tmux_session": session_idx,
                "tmux_pane": tmux_pane,
            }
    except (OSError, ValueError):
        pass

    return {
        "pid": pid,
        "transport": transport,
        "tmux_meta": tmux_meta,
        "comm": comm,
        "matched_cwd": matched_cwd,
    }


def _v25_auto_detect_friendly_error(agent: str, cwd: str) -> str:
    """Build human-readable error when auto_detect_from_cwd finds no prefix match.

    Lists all live {agent} processes with their cwds so the user can:
      - Pick the correct project root to retry with, OR
      - cd into the project dir and re-launch the agent CLI.
    """
    procs = _v25_scan_agent_procs(agent)
    if not procs:
        return (
            f"auto_detect_from_cwd: no live {agent} processes found anywhere "
            f"on this host. Start {agent} CLI in {cwd} (or any subdir) first."
        )
    target_real = os.path.realpath(cwd)
    lines = [
        f"auto_detect_from_cwd: no live {agent} found at-or-under cwd={cwd}",
        f"  (realpath={target_real})",
        f"  Found {len(procs)} live {agent} process(es) at other locations:",
    ]
    for p in procs[:8]:  # cap to avoid blowing up error payload
        lines.append(f"    pid={p['pid']} comm={p['comm']} cwd={p['cwd']}")
    if len(procs) > 8:
        lines.append(f"    ... ({len(procs) - 8} more truncated)")
    lines.append(
        "  Hint: cd into one of the cwds above and re-call with that path, "
        "OR pass pid= explicitly to bypass auto-detection."
    )
    return "\n".join(lines)


@mcp.tool()
def ultradebate_v2_session_attach(
    agent: str,
    cwd: str,
    pid: int | None = None,
    transport: str = "stdio",
    cli_version: str = "",
    tmux_meta: dict | None = None,
    auto_detect_from_cwd: bool = False,
) -> dict:
    """Register an agent with the session for cwd (B25, v2.5).

    Args:
        agent: 'claude' or 'codex'.
        cwd: working directory (used for session_hash).
        pid: process id of the agent CLI. Optional when auto_detect_from_cwd=True.
        transport: 'stdio' | 'tmux' | other. Auto-overridden by /proc scan when
                   auto_detect_from_cwd=True AND caller passes default 'stdio'.
        cli_version: optional CLI version string for diagnostics.
        tmux_meta: when transport=='tmux', supply {tmux_session, tmux_pane,
                   tmux_socket?, tmux_server_pid?}. Auto-populated from
                   /proc/PID/environ TMUX var when auto_detect_from_cwd=True.
        auto_detect_from_cwd: when True (2026-05-16 fix), scan /proc for the
                              most recently started {agent} CLI whose cwd
                              matches. Populates pid/transport/tmux_meta from
                              live process state. Explicit caller args override
                              auto-detected values on conflict. Reflects v2.5
                              design intent: cwd is master binding key, PID is
                              diagnostic metadata.

    Returns:
        {ok, session_hash, binding_status, session_path, auto_detected?}
    """
    if agent not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "agent must be claude|codex"}
    if not cwd:
        return {"ok": False, "error": "cwd required"}

    auto_info: dict | None = None
    if auto_detect_from_cwd:
        auto_info = _v25_auto_detect_agent_proc(agent, cwd)
        if auto_info is None and pid is None:
            return {
                "ok": False,
                "error": _v25_auto_detect_friendly_error(agent, cwd),
            }
        if auto_info:
            if pid is None:
                pid = auto_info["pid"]
            # Only override transport when caller left it as the default 'stdio'
            # AND auto-detection found tmux. Explicit non-default caller wins.
            if transport == "stdio" and auto_info["transport"] != "stdio":
                transport = auto_info["transport"]
            if tmux_meta is None and auto_info.get("tmux_meta"):
                tmux_meta = auto_info["tmux_meta"]
            if not cli_version:
                cli_version = f"auto-detected-{agent}-{auto_info.get('comm', 'unknown')}-pid{pid}"

    if pid is None:
        return {"ok": False, "error": "pid required (or auto_detect_from_cwd=True)"}

    session_hash = _v25_session_hash(cwd)
    sdir = _v25_ensure_session_dir(session_hash)
    data = _v25_load_session_json(session_hash)
    now = _v24_now_iso()
    if not data:
        data = {
            "session_hash": session_hash,
            "cwd": cwd,
            "schema_version": _V25_SCHEMA,
            "created_at": now,
            "active_agents": {},
            "current_cycle_id": _v25_active_cycle_id(),
            "binding_status": "idle",
        }
    agent_meta = {
        "joined_at": now,
        "pid": int(pid),
        "cli_version": cli_version,
        "transport": transport,
    }
    if transport == "tmux" and tmux_meta:
        agent_meta["tmux_session"] = tmux_meta.get("tmux_session")
        agent_meta["tmux_pane"] = tmux_meta.get("tmux_pane")
        # Preserve extended tmux_meta fields (socket, server_pid) for diagnostics
        # and safe_send routing across non-default sockets.
        if tmux_meta.get("tmux_socket"):
            agent_meta["tmux_socket"] = tmux_meta["tmux_socket"]
        if tmux_meta.get("tmux_server_pid"):
            agent_meta["tmux_server_pid"] = tmux_meta["tmux_server_pid"]
    # Remove any unattached_at marker on re-attach.
    data.setdefault("active_agents", {})[agent] = agent_meta
    data["binding_status"] = _v25_compute_binding_status(data["active_agents"])
    data["cwd"] = cwd  # in case of remount
    data["schema_version"] = _V25_SCHEMA
    _v25_save_session_json(session_hash, data)

    _v25_audit_append(session_hash, {
        "event": "session_attach",
        "agent": agent,
        "pid": pid,
        "transport": transport,
        "binding_status": data["binding_status"],
        "auto_detected": bool(auto_info),
    })
    result: dict = {
        "ok": True,
        "session_hash": session_hash,
        "binding_status": data["binding_status"],
        "session_path": str(sdir / "session.json"),
    }
    if auto_info:
        result["auto_detected"] = {
            "pid": auto_info["pid"],
            "transport": auto_info["transport"],
            "tmux_meta": auto_info.get("tmux_meta"),
            "comm": auto_info.get("comm"),
            "matched_cwd": auto_info.get("matched_cwd"),
        }
    return result


@mcp.tool()
def ultradebate_v2_session_resume(cwd: str | None = None, agent: str | None = None) -> dict:
    """Return session state for cwd (B25): binding + latest directives + cycle state + inbox.

    Args:
        cwd: Working directory; if absent, requires `session_hash` via session_status path
             (callers without cwd should call `session_status` directly).
        agent: optional agent to scope inbox read for.

    Returns:
        {ok, session_hash, session, latest_directives, current_cycle, inbox_summary}
    """
    if not cwd:
        return {"ok": False, "error": "cwd required"}
    session_hash = _v25_session_hash(cwd)
    sdir = _v25_session_dir(session_hash)
    data = _v25_load_session_json(session_hash)
    if not data:
        return {
            "ok": True,
            "session_hash": session_hash,
            "session": None,
            "latest_directives": [],
            "current_cycle": None,
            "inbox_summary": {},
        }
    # latest directives (last 5)
    tail = ultradebate_v2_user_directives_tail(session_hash=session_hash, max_n=5)
    directives = tail.get("directives", []) if tail.get("ok") else []
    # cycle state
    cycle_state = None
    cid = data.get("current_cycle_id") or _v25_active_cycle_id()
    if cid:
        try:
            phase = _v24_load_phase(cid)
            plan = _v24_load_plan(cid)
            cycle_state = {
                "cycle_id": cid,
                "phase": phase.get("phase"),
                "task_count": len(plan.get("tasks", [])),
                "asymmetry_alert": _v25_load_workload_audit(cid).get("asymmetry_alert"),
            }
        except Exception:
            cycle_state = {"cycle_id": cid, "error": "phase_or_plan_load_failed"}
    # inbox summary
    inbox_summary: dict = {}
    if agent in _V25_VALID_AGENTS:
        ir = ultradebate_v2_priority_inbox_read(agent=agent, min_priority="low", max_n=10)
        if ir.get("ok"):
            inbox_summary = {
                "agent": agent,
                "pending": len(ir.get("messages", [])),
                "remaining_after": ir.get("remaining_after", 0),
            }
    return _json_safe_obj({
        "ok": True,
        "session_hash": session_hash,
        "session": data,
        "latest_directives": directives,
        "current_cycle": cycle_state,
        "inbox_summary": inbox_summary,
    })


@mcp.tool()
def ultradebate_v2_session_status(
    session_hash: str | None = None,
    cwd: str | None = None,
) -> dict:
    """Return binding_status + active_agents + current_cycle (B25)."""
    if not session_hash:
        if not cwd:
            return {"ok": False, "error": "either session_hash or cwd must be supplied"}
        session_hash = _v25_session_hash(cwd)
    data = _v25_load_session_json(session_hash)
    if not data:
        return {"ok": True, "session_hash": session_hash, "exists": False}
    return {
        "ok": True,
        "session_hash": session_hash,
        "exists": True,
        "binding_status": data.get("binding_status"),
        "active_agents": data.get("active_agents", {}),
        "current_cycle_id": data.get("current_cycle_id"),
        "cwd": data.get("cwd"),
    }


@mcp.tool()
def ultradebate_v2_session_unattach(
    agent: str,
    session_hash: str | None = None,
    cwd: str | None = None,
) -> dict:
    """Graceful agent unattach (B25)."""
    if agent not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "agent must be claude|codex"}
    if not session_hash:
        if not cwd:
            return {"ok": False, "error": "either session_hash or cwd must be supplied"}
        session_hash = _v25_session_hash(cwd)
    data = _v25_load_session_json(session_hash)
    if not data:
        return {"ok": False, "error": f"no session for hash {session_hash}"}
    now = _v24_now_iso()
    agents = data.setdefault("active_agents", {})
    if agent in agents:
        agents[agent] = {**agents[agent], "unattached_at": now}
    data["binding_status"] = _v25_compute_binding_status(
        {a: m for a, m in agents.items() if not m.get("unattached_at")}
    )
    # When both unattached, archive the session.
    all_unattached = all(m.get("unattached_at") for m in agents.values()) if agents else True
    if all_unattached:
        data["archived_at"] = now
    _v25_save_session_json(session_hash, data)
    _v25_audit_append(session_hash, {
        "event": "session_unattach",
        "agent": agent,
        "binding_status": data["binding_status"],
        "archived": bool(data.get("archived_at")),
    })
    return {
        "ok": True,
        "session_hash": session_hash,
        "binding_status": data["binding_status"],
        "archived": bool(data.get("archived_at")),
    }


# ---------------------------------------------------------------------------
# Protocol 7: Role Symmetry Hard Guardrail
# ---------------------------------------------------------------------------


@mcp.tool()
def ultradebate_v2_workload_audit_read(cycle_id: str | None = None) -> dict:
    """Return current workload audit (ratios + asymmetry_alert) for a cycle (v2.5).

    If cycle_id is None, uses the most recent non-done cycle.
    """
    cid = cycle_id or _v25_active_cycle_id()
    if not cid:
        return {"ok": False, "error": "no active cycle found"}
    audit = _v25_load_workload_audit(cid)
    _v25_compute_ratios(audit)
    return {"ok": True, "audit": audit, "cycle_id": cid}


def _v25_check_compliance(agent: str, cycle_id: str) -> dict:
    """Compute violations + warnings + session_health for an agent (B26)."""
    violations: list[dict] = []
    warnings: list[dict] = []
    audit = _v25_load_workload_audit(cycle_id)
    _v25_compute_ratios(audit)
    ratios = audit.get("ratios", {})
    metrics = audit.get("metrics", {})
    other = "codex" if agent == "claude" else "claude"

    # B21 — Parity. ratio≥2.0 == violation, ratio>=1.5 (and overused == agent) == warning
    dim_to_key = {
        "patches": "patches_authored",
        "reviews": "reviews_performed",
        "rcas": "rcas_performed",
        "dispatches": "dispatch_messages_sent",
    }
    for dim, r in ratios.items():
        key = dim_to_key.get(dim)
        if key is None:
            continue
        agent_v = float(metrics.get(agent, {}).get(key, 0))
        other_v = float(metrics.get(other, {}).get(key, 0))
        if r == "inf":
            r_num = float("inf")
        else:
            try:
                r_num = float(r)
            except Exception:
                r_num = 1.0
        if r_num >= _V25_PARITY_RATIO_CAP:
            # Only flag the OVERUSED side; underused side is required to act.
            overused = agent if agent_v > other_v else other
            underused = other if overused == agent else agent
            if underused == agent:
                violations.append({
                    "rule": "B21",
                    "severity": "high",
                    "metric": dim,
                    "current_ratio": r,
                    "message": (
                        f"Parity ratio {dim}={r} exceeds cap {_V25_PARITY_RATIO_CAP}. "
                        f"You ({agent}) must take next {dim} action."
                    ),
                    "required_action": f"take_next_{dim}",
                })
        elif r_num >= 1.5 and agent_v > other_v:
            warnings.append({
                "rule": "B21",
                "severity": "medium",
                "metric": dim,
                "current_ratio": r,
                "message": (
                    f"Approaching {dim} parity cap (ratio {r}). "
                    f"Yield next {dim} to {other}."
                ),
            })

    # B22 — Symmetric RCA. If peer submitted RCA for any task but we have not, flag.
    cdir = _v24_cycle_dir(cycle_id)
    artifacts_dir = cdir / "artifacts"
    if artifacts_dir.exists():
        for task_dir in artifacts_dir.iterdir():
            if not task_dir.is_dir():
                continue
            rca_dir = task_dir / "rca"
            if not rca_dir.exists():
                continue
            peer_rca = rca_dir / f"{other}.rca.md"
            agent_rca = rca_dir / f"{agent}.rca.md"
            if peer_rca.exists() and not agent_rca.exists():
                violations.append({
                    "rule": "B22",
                    "severity": "high",
                    "task_id": task_dir.name,
                    "message": (
                        f"{other} submitted RCA for task {task_dir.name}; "
                        f"your independent RCA pending before cross-verify."
                    ),
                    "required_action": "submit_rca",
                })

    # session_health + B25 session binding check (LOW-G fix, review-eng audit 2026-05-15)
    sh = {"binding_status": None, "peer_status": None, "session_hash": None}
    try:
        # Pull from the most recently touched session, if any.
        latest_session = None
        latest_session_hash = None
        if _V25_SESSIONS.exists():
            sessions = sorted(_V25_SESSIONS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for sdir in sessions:
                session_json = sdir / "session.json"
                if session_json.exists():
                    try:
                        latest_session = json.loads(session_json.read_text())
                        latest_session_hash = sdir.name
                        break
                    except Exception:
                        continue
        if latest_session:
            sh["binding_status"] = latest_session.get("binding_status")
            sh["session_hash"] = latest_session_hash
            # B25 — Session Binding. Peer must have called session_attach.
            active_agents = latest_session.get("active_agents", {}) or {}
            peer_meta = active_agents.get(other) or {}
            peer_attached = bool(peer_meta) and not peer_meta.get("unattached_at")
            archived = bool(latest_session.get("archived_at"))
            if not archived and not peer_attached:
                violations.append({
                    "rule": "B25",
                    "severity": "medium",
                    "message": (
                        f"{other} has not called session_attach for session "
                        f"{latest_session_hash}. Awaiting their session_resume."
                    ),
                    "required_action": "wait_for_peer_session_resume",
                    "session_hash": latest_session_hash,
                    "binding_status": latest_session.get("binding_status"),
                })
        # Peer presence
        pr = ultradebate_v2_presence_read(other)
        peer_state = pr.get("agents", {}).get(other, {}) if pr.get("ok") else {}
        sh["peer_status"] = peer_state.get("status")
    except Exception:
        pass

    # B32 NEW (CLAUDE.md §18 B32, 2026-05-23) — Unread urgent priority_send inbox.
    # User mandate 2026-05-23: messages must not be buried. Receiver MUST drain
    # urgent inbox at turn start per B26 high-severity mandate. Without this,
    # b16_review_requests can sit unread for 30+ min (Empirical 2026-05-23).
    try:
        inbox_path = _V24_INBOX / f"{agent}.jsonl"
        processed_dir = _V24_BASE / agent / "processed"
        processed_ids: set[str] = set()
        if processed_dir.exists():
            for pp in processed_dir.iterdir():
                if pp.suffix == ".json":
                    processed_ids.add(pp.stem)
        urgent_unread: list[dict] = []
        if inbox_path.exists():
            with inbox_path.open() as f:
                for ln in f:
                    try:
                        m = json.loads(ln)
                        if m.get("priority") == "urgent":
                            mid = m.get("msg_id")
                            if mid and mid not in processed_ids:
                                urgent_unread.append({
                                    "msg_id": mid,
                                    "from": m.get("from"),
                                    "type": m.get("type"),
                                    "ts": m.get("ts"),
                                })
                    except Exception:
                        continue
        if urgent_unread:
            violations.append({
                "rule": "B32",
                "severity": "high",
                "metric": "unread_urgent_inbox",
                "current_value": len(urgent_unread),
                "messages_preview": urgent_unread[:5],
                "message": (
                    f"You have {len(urgent_unread)} unread urgent priority_send "
                    f"message(s). Per CLAUDE.md §18 B32 + B26 mandate, MUST call "
                    f"priority_inbox_read(agent='{agent}', min_priority='urgent', "
                    f"drain=True) BEFORE other actions."
                ),
                "required_action": "priority_inbox_read",
            })
    except Exception:
        pass  # best-effort; never block compliance_check

    return {
        "violations": violations,
        "warnings": warnings,
        "session_health": sh,
        "cycle_id": cycle_id,
    }


@mcp.tool()
def ultradebate_v2_compliance_check(agent: str, cycle_id: str | None = None) -> dict:
    """Pre-flight compliance check for `agent` (B26, v2.5).

    LLM MUST call this at the start of every turn. Violations (severity=high)
    must be handled before other actions; warnings within the same turn.

    Returns:
        {ok, agent, cycle_id, violations: [...], warnings: [...], session_health, ts}
    """
    if agent not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "agent must be claude|codex"}
    cid = cycle_id or _v25_active_cycle_id()
    if not cid:
        return {
            "ok": True,
            "agent": agent,
            "cycle_id": None,
            "violations": [],
            "warnings": [],
            "session_health": {},
            "ts": _v24_now_iso(),
            "note": "no active cycle",
        }
    result = _v25_check_compliance(agent, cid)
    now = _v24_now_iso()
    # Append to compliance.jsonl
    try:
        cdir = _v24_cycle_dir(cid)
        cdir.mkdir(parents=True, exist_ok=True)
        with (cdir / "compliance.jsonl").open("a") as f:
            f.write(json.dumps({
                "ts": now,
                "agent": agent,
                "violations_count": len(result["violations"]),
                "warnings_count": len(result["warnings"]),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return {"ok": True, "agent": agent, "ts": now, **result}


@mcp.tool()
def ultradebate_v2_rca_submit(
    cycle_id: str,
    issue_id: str,
    agent: str,
    rca_md: str,
) -> dict:
    """Submit a single-agent RCA for an issue (B22 symmetric RCA, v2.5).

    Rejects if `agent` has already submitted an RCA for this issue.
    Triggers an automatic urgent ping to the other peer if their RCA is missing.
    When both RCAs land, dispatches a cross_verify notification.

    Args:
        cycle_id: cycle identifier
        issue_id: task_id or arbitrary issue tag (used as artifacts/{issue_id}/rca/ subdir)
        agent: 'claude' or 'codex'
        rca_md: RCA content (markdown)

    Known limitations (review-eng audit 2026-05-15, deferred to future cleanup):
      LOW-A: simultaneous-submit race. Two concurrent `rca_submit` calls for the same
        (cycle_id, issue_id) may both observe `not own_path.exists()` and both succeed,
        producing a race-window double-write. A `flock`-based lockfile would resolve
        this; for now, MCP single-stdio-server serialization makes the window narrow.
      LOW-F: session.json is read-modify-write without flock; same caveat applies if
        two tools mutate the same session concurrently.
    """
    if agent not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "agent must be claude|codex"}
    if not rca_md:
        return {"ok": False, "error": "rca_md is required"}
    try:
        cdir = _v24_cycle_dir(cycle_id)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not cdir.exists():
        return {"ok": False, "error": f"cycle {cycle_id} not found"}

    rca_dir = cdir / "artifacts" / issue_id / "rca"
    rca_dir.mkdir(parents=True, exist_ok=True)
    own_path = rca_dir / f"{agent}.rca.md"
    if own_path.exists():
        return {
            "ok": False,
            "error": "B22_DUPLICATE_RCA",
            "code": "B22_DUPLICATE_RCA",
            "message": (
                f"{agent} already submitted RCA for issue {issue_id}. "
                f"Other peer's RCA required before cross-verify."
            ),
            "existing_path": str(own_path),
        }
    _v24_atomic_write_text(own_path, rca_md)

    # Auto-counting
    try:
        _v25_increment_metric(agent, "rcas_performed", 1, cycle_id=cycle_id)
    except Exception:
        pass

    other = "codex" if agent == "claude" else "claude"
    other_path = rca_dir / f"{other}.rca.md"
    both_present = other_path.exists()
    cross_verify_triggered = False
    ping_sent = False

    if both_present:
        # Create a cross-verify trigger note + urgent ping for both peers.
        cv_note_path = rca_dir / "cross_verify.md"
        if not cv_note_path.exists():
            _v24_atomic_write_text(
                cv_note_path,
                f"# Cross-Verify Required\n\n"
                f"Both agents have submitted RCAs for issue `{issue_id}` "
                f"in cycle `{cycle_id}`.\n\n"
                f"- claude.rca.md: {str(rca_dir / 'claude.rca.md')}\n"
                f"- codex.rca.md: {str(rca_dir / 'codex.rca.md')}\n\n"
                f"Conduct cross-verify, then write the consensus below.\n",
            )
        cross_verify_triggered = True
        # Ping both peers urgently
        for dest in ("claude", "codex"):
            sender = "codex" if dest == "claude" else "claude"
            ultradebate_v2_priority_send(
                sender=sender,
                target=dest,
                priority="urgent",
                msg_type="rca_cross_verify_required",
                payload={"cycle_id": cycle_id, "issue_id": issue_id, "rca_dir": str(rca_dir)},
                ref=issue_id,
            )
    else:
        # Notify the other peer that their RCA is pending.
        ultradebate_v2_priority_send(
            sender=agent,
            target=other,
            priority="urgent",
            msg_type="rca_pending",
            payload={"cycle_id": cycle_id, "issue_id": issue_id, "submitted_by": agent},
            ref=issue_id,
        )
        ping_sent = True

    _v24_emit_event({
        "agent": agent,
        "event": "rca_submitted",
        "context": {
            "cycle_id": cycle_id,
            "issue_id": issue_id,
            "both_present": both_present,
            "cross_verify_triggered": cross_verify_triggered,
        },
    })
    return {
        "ok": True,
        "cycle_id": cycle_id,
        "issue_id": issue_id,
        "agent": agent,
        "path": str(own_path),
        "both_present": both_present,
        "cross_verify_triggered": cross_verify_triggered,
        "ping_sent_to_peer": ping_sent,
    }


@mcp.tool()
def ultradebate_v2_task_assign(
    cycle_id: str,
    task_id: str,
    assignee: str,
    reviewer: str,
    requested_by: str = "claude",
) -> dict:
    """Record (or attempt) an assignee/reviewer pairing for a task (B21, v2.5).

    Enforces B21 parity gate: if patches ratio is >= 2.0 and `assignee` is the
    overused side, the request is rejected with a forced rebalance message.

    Also updates plan.yaml's task entry with the new assignee/reviewer (if the
    task exists in plan.yaml), and counts `tasks_assigned_as_*` metrics.

    Returns:
        {ok, cycle_id, task_id, assignee, reviewer, rebalance_required?, ...}
    """
    if assignee not in _V25_VALID_AGENTS or reviewer not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "assignee/reviewer must be claude|codex"}
    if assignee == reviewer:
        return {"ok": False, "error": "assignee must differ from reviewer"}
    try:
        cdir = _v24_cycle_dir(cycle_id)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    audit = _v25_load_workload_audit(cycle_id)
    _v25_compute_ratios(audit)
    ratios = audit.get("ratios", {})
    metrics = audit.get("metrics", {})

    # B21: if patches ratio is hot AND assignee is the overused side, reject.
    patches_ratio = ratios.get("patches", 1.0)
    try:
        r_num = float("inf") if patches_ratio == "inf" else float(patches_ratio)
    except Exception:
        r_num = 1.0
    if r_num >= _V25_PARITY_RATIO_CAP:
        c = float(metrics.get("claude", {}).get("patches_authored", 0))
        x = float(metrics.get("codex", {}).get("patches_authored", 0))
        overused = "claude" if c > x else "codex"
        underused = "codex" if overused == "claude" else "claude"
        if assignee == overused:
            return {
                "ok": False,
                "error": "B21_PARITY_VIOLATION",
                "code": "B21_PARITY_VIOLATION",
                "message": (
                    f"Patches ratio {patches_ratio} >= cap {_V25_PARITY_RATIO_CAP}. "
                    f"{underused} must be assignee for next task; got assignee={assignee}."
                ),
                "current_ratio": patches_ratio,
                "metrics": metrics,
                "recommended_assignee": underused,
            }

    # Update plan.yaml task entry if present (best-effort).
    plan_updated = False
    plan = _v24_load_plan(cycle_id)
    if plan:
        for t in plan.get("tasks", []):
            if t.get("id") == task_id:
                t["assignee"] = assignee
                t["reviewer"] = reviewer
                plan_updated = True
                break
        if plan_updated:
            _v24_save_plan(cycle_id, plan)

    # Auto-counting
    try:
        _v25_increment_metric(assignee, "tasks_assigned_as_assignee", 1, cycle_id=cycle_id)
        _v25_increment_metric(reviewer, "tasks_assigned_as_reviewer", 1, cycle_id=cycle_id)
    except Exception:
        pass

    _v24_emit_event({
        "agent": requested_by,
        "event": "task_assign",
        "context": {
            "cycle_id": cycle_id,
            "task_id": task_id,
            "assignee": assignee,
            "reviewer": reviewer,
            "plan_updated": plan_updated,
        },
    })

    return {
        "ok": True,
        "cycle_id": cycle_id,
        "task_id": task_id,
        "assignee": assignee,
        "reviewer": reviewer,
        "plan_updated": plan_updated,
        "rebalance_required": False,
    }


def _v25_role_swaps_path(cycle_id: str) -> Path:
    return _v24_cycle_dir(cycle_id) / "role_swaps.jsonl"


def _v25_load_role_swaps(cycle_id: str) -> list[dict]:
    p = _v25_role_swaps_path(cycle_id)
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _v25_append_role_swap(cycle_id: str, entry: dict) -> None:
    p = _v25_role_swaps_path(cycle_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@mcp.tool()
def ultradebate_v2_role_swap_request(
    cycle_id: str,
    requester: str,
    new_assignment: dict,
    rationale: str,
) -> dict:
    """Request a swap of §13 default roles (B21 swap mechanism, v2.5).

    Args:
        cycle_id: cycle to apply swap within.
        requester: 'claude' or 'codex'.
        new_assignment: dict like {"designer": "codex", "reviewer": "claude"}.
        rationale: free-text justification.

    Returns:
        {ok, swap_id, signoffs_required, swap_status}
    """
    if requester not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "requester must be claude|codex"}
    if not new_assignment or not isinstance(new_assignment, dict):
        return {"ok": False, "error": "new_assignment dict required"}
    try:
        _v24_cycle_dir(cycle_id)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    swap_id = _v25_msg_id()
    entry = {
        "swap_id": swap_id,
        "ts": _v24_now_iso(),
        "requester": requester,
        "new_assignment": new_assignment,
        "rationale": rationale,
        "signoffs": {requester: {"decision": "approve", "ts": _v24_now_iso()}},
        "swap_status": "pending_other_signoff",
    }
    _v25_append_role_swap(cycle_id, entry)
    other = "codex" if requester == "claude" else "claude"
    # Notify the other peer.
    ultradebate_v2_priority_send(
        sender=requester,
        target=other,
        priority="normal",
        msg_type="role_swap_request",
        payload={"cycle_id": cycle_id, "swap_id": swap_id, "new_assignment": new_assignment, "rationale": rationale},
        ref=swap_id,
    )
    return {
        "ok": True,
        "swap_id": swap_id,
        "cycle_id": cycle_id,
        "signoffs_required": [other],
        "swap_status": "pending_other_signoff",
    }


@mcp.tool()
def ultradebate_v2_role_swap_signoff(
    cycle_id: str,
    agent: str,
    swap_id: str,
    decision: str,
    rationale: str = "",
) -> dict:
    """Sign off (approve|reject) a pending role swap (v2.5).

    Both peers must approve before the swap is in effect. A reject from either
    peer marks the swap rejected.
    """
    if agent not in _V25_VALID_AGENTS:
        return {"ok": False, "error": "agent must be claude|codex"}
    if decision not in ("approve", "reject"):
        return {"ok": False, "error": "decision must be approve|reject"}
    swaps = _v25_load_role_swaps(cycle_id)
    target: dict | None = None
    for s in reversed(swaps):
        if s.get("swap_id") == swap_id:
            target = s
            break
    if not target:
        return {"ok": False, "error": f"swap_id {swap_id} not found"}
    if target.get("swap_status") in ("approved", "rejected"):
        return {"ok": False, "error": f"swap already finalized: {target.get('swap_status')}"}
    target.setdefault("signoffs", {})[agent] = {
        "decision": decision,
        "ts": _v24_now_iso(),
        "rationale": rationale,
    }
    if decision == "reject":
        target["swap_status"] = "rejected"
    else:
        approvals = sum(1 for s in target["signoffs"].values() if s.get("decision") == "approve")
        if approvals >= 2:
            target["swap_status"] = "approved"
        else:
            target["swap_status"] = "pending_other_signoff"
    # Append a fresh entry (audit log style; readers should respect the latest
    # entry per swap_id since we don't rewrite history).
    final_entry = {**target, "appended_at": _v24_now_iso(), "appended_by": agent}
    _v25_append_role_swap(cycle_id, final_entry)
    return {
        "ok": True,
        "swap_id": swap_id,
        "agent": agent,
        "decision": decision,
        "swap_status": target["swap_status"],
    }


# ---------------------------------------------------------------------------
# v2.6 — Multi-item open-question verdict enforcement (design §2.6 / §4.4b)
# Distinct from ultradebate_v2_request_response: this tracks N-item bundles
# where each item slot must be filled (no evidence-only, no deferred verdicts).
# ---------------------------------------------------------------------------

from . import open_items as _v26_open_items


@mcp.tool()
def ultradebate_v2_open_questions_register(
    exchange_id: str,
    items: list[str],
    sender: str | None = None,
    target_peer: str | None = None,
    context: dict | None = None,
) -> dict[str, Any]:
    """Register N numbered items as an open exchange — all must receive verdicts.

    items: list of identifiers like ['W13','W14','W11','W10','W15'].
    Returns the registered open set; refuses to overwrite an existing exchange
    with different items.
    """
    return _v26_open_items.open_questions(
        exchange_id, items,
        sender=sender, target_peer=target_peer, context=context,
    )


@mcp.tool()
def ultradebate_v2_open_questions_respond(
    exchange_id: str,
    verdicts: dict[str, Any],
    responder: str | None = None,
    require_known_verdict_values: bool = False,
) -> dict[str, Any]:
    """Submit verdicts. Rejected unless every open item has a non-empty verdict.

    `verdicts` accepts both bare strings and dicts with a `value` key. Missing,
    empty, or whitespace-only values are reported under `missing` so the caller
    can complete the exchange before the conversation advances.
    """
    return _v26_open_items.respond_to_questions(
        exchange_id, verdicts,
        responder=responder,
        require_known_verdict_values=require_known_verdict_values,
    )


@mcp.tool()
def ultradebate_v2_open_questions_pending(
    max_age_minutes: float | None = None,
) -> dict[str, Any]:
    """List exchanges still missing verdicts. Call at every turn boundary."""
    return _v26_open_items.pending_verdicts(max_age_minutes=max_age_minutes)


@mcp.tool()
def ultradebate_v2_open_questions_get(exchange_id: str) -> dict[str, Any]:
    """Inspect a single exchange (debug/UI helper)."""
    return _v26_open_items.get_exchange(exchange_id)


@mcp.tool()
def ultradebate_v2_open_questions_list(
    include_completed: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """List recent exchanges (newest first by mtime)."""
    return _v26_open_items.list_exchanges(
        include_completed=include_completed, limit=limit,
    )


@mcp.tool()
def ultradebate_v2_open_questions_close(
    exchange_id: str,
    reason: str = "manual_close",
) -> dict[str, Any]:
    """Force-close an exchange without verdicts (rare; for cancellation)."""
    return _v26_open_items.close_exchange(exchange_id, reason=reason)


# ---------------------------------------------------------------------------
# v2.6b — Background jobs + safe send wrapper (deterministic operational fixes)
# Design ref: MCP_MLP_도입_설계서.md §4.1 (run_long_job), §4.2 (send_to_codex
# wrapper). These tools are the *structural* fix for foreground-blocking and
# submit unreliability: callers receive an immediate ACK + job_id and poll via
# check_job, and every keystroke path is funneled through a single wrapper that
# already handles paste-buffer flattening, target-specific submit keys, and
# busy-pane retry. Inline subprocess.run() or raw tmux send-keys from any
# other tool is a regression of this design.
# ---------------------------------------------------------------------------

from . import jobs as _v26_jobs  # noqa: E402 — late import keeps load order clean


_V26_PEER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "codex": ("codex",),
    "claude": ("claude",),
    "gemini": ("gemini",),
}


def _v26_list_tmux_panes() -> list[dict[str, Any]]:
    """tmux list-panes -a 결과를 dict 리스트로 변환. 실패 시 빈 리스트."""
    fmt = "#{session_name}:#{window_index}.#{pane_index}|#{pane_pid}|#{pane_current_command}|#{pane_current_path}"
    try:
        proc = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", fmt],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if proc.returncode != 0:
        return []
    panes: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        pid_str = parts[1].strip()
        try:
            pid_int: int | None = int(pid_str) if pid_str else None
        except ValueError:
            pid_int = None
        panes.append({
            "pane_id": parts[0].strip(),
            "pid": pid_int,
            "cmd": parts[2].strip(),
            "path": parts[3].strip().rstrip("/") or "/",
        })
    return panes


def _v26_pane_runs_peer(pane_info: dict[str, Any], peer: str) -> bool:
    """주어진 pane이 target peer(codex/claude/gemini)를 실행 중인지 판별.

    1) pane의 current_command가 peer 이름 자체이면 즉시 True (fast path).
    2) command가 node/bash/sh/zsh 등 shell/launcher인 경우, 자식 프로세스를
       ``ps --ppid`` 로 조사하여 peer 키워드 매칭. 모호하면 False (안전).
    """
    keywords = _V26_PEER_KEYWORDS.get(peer.lower(), (peer.lower(),))
    cmd = (pane_info.get("cmd") or "").lower()
    if cmd in keywords:
        return True
    pid = pane_info.get("pid")
    if pid is None:
        return False
    # shell이나 node launcher인 경우만 자식 검사 (false positive 방지)
    if cmd not in {"node", "bash", "sh", "zsh", "fish", "dash"}:
        return False
    try:
        proc = subprocess.run(
            ["ps", "--ppid", str(pid), "-o", "args="],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    if proc.returncode != 0:
        return False
    for line in (proc.stdout or "").splitlines():
        ll = line.lower()
        for kw in keywords:
            # 단어 경계로 보수적 매칭 (예: 'claude --dangerously', '/bin/codex')
            if f"/{kw}" in ll or ll.startswith(kw) or f" {kw}" in ll or f"{kw} " in ll:
                return True
    return False


def _v26_path_match_score(cwd: str, pane_path: str) -> tuple[int, str]:
    """cwd vs pane_path 매칭 점수 계산. (score, reason) 반환. 높을수록 우수."""
    c = (cwd or "").rstrip("/") or "/"
    p = (pane_path or "").rstrip("/") or "/"
    if not c or not p:
        return (0, "no_match")
    if c == p:
        return (1000, "cwd_match")
    if c.startswith(p + "/"):
        return (500 + len(p), "cwd_child")
    if p.startswith(c + "/"):
        return (400 + len(c), "cwd_parent")
    # 공통 prefix 길이 (디렉토리 경계 기준)
    c_parts = c.split("/")
    p_parts = p.split("/")
    common: list[str] = []
    for a, b in zip(c_parts, p_parts):
        if a == b:
            common.append(a)
        else:
            break
    common_path = "/".join(common)
    if len(common_path) > 1:  # '/' 단일 prefix는 무시
        return (100 + len(common_path), "common_prefix")
    return (0, "no_match")


def _v26_pane_etime_seconds(pid: int | None) -> int:
    """pane PID의 elapsed time(초). 실패 시 매우 큰 값 (= 후순위)."""
    if pid is None:
        return 10**9
    try:
        proc = subprocess.run(
            ["ps", "-o", "etimes=", "-p", str(pid)],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 10**9
    if proc.returncode != 0:
        return 10**9
    s = (proc.stdout or "").strip()
    try:
        return int(s)
    except ValueError:
        return 10**9


def _v26_session_json_fallback(target: str) -> dict[str, Any]:
    """session.json 기반 기존 동작 (tmux가 없거나 후보 0건일 때 사용)."""
    sessions_root = _V2_BASE / "sessions"
    if not sessions_root.exists():
        return {"ok": False, "pane": None, "socket": None, "session_hash": None,
                "candidate_count": 0, "selected_path": None,
                "selection_reason": "no_match"}
    entries: list[tuple[float, Path]] = []
    try:
        for d in sessions_root.iterdir():
            sj = d / "session.json"
            if sj.exists():
                try:
                    entries.append((sj.stat().st_mtime, sj))
                except OSError:
                    continue
    except OSError:
        return {"ok": False, "pane": None, "socket": None, "session_hash": None,
                "candidate_count": 0, "selected_path": None,
                "selection_reason": "no_match"}
    entries.sort(key=lambda t: -t[0])
    for _mtime, sj in entries:
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        agent_meta = (data.get("active_agents") or {}).get(target) or {}
        tmux_session = agent_meta.get("tmux_session")
        tmux_pane = agent_meta.get("tmux_pane")
        socket = agent_meta.get("tmux_socket")
        if not tmux_session and not tmux_pane:
            continue
        pane = tmux_pane or tmux_session
        return {
            "ok": True,
            "pane": pane,
            "socket": socket,
            "session_hash": data.get("session_hash"),
            "tmux_session": tmux_session,
            "tmux_pane": tmux_pane,
            "candidate_count": 0,
            "selected_path": None,
            "selection_reason": "session_json_fallback",
        }
    return {"ok": False, "pane": None, "socket": None, "session_hash": None,
            "candidate_count": 0, "selected_path": None,
            "selection_reason": "no_match"}


def _v26_resolve_peer_pane(target: str) -> dict[str, Any]:
    """CWD-aware peer pane resolver.

    1) ``tmux list-panes -a`` 로 모든 pane을 enumerate
    2) ``_v26_pane_runs_peer`` 로 target peer 실행 pane 후보 추출
    3) 호출 시점의 cwd 기준 매칭 점수로 정렬 (cwd_match > cwd_child >
       cwd_parent > common_prefix). 동점은 etime이 짧은(최신) pane 우선.
    4) 후보 0건 또는 모두 점수 0이면 ``session.json`` fallback.

    반환 dict 키: ok, pane, socket, session_hash, tmux_session, tmux_pane,
    candidate_count, selected_path, selection_reason
    """
    try:
        cwd = os.getcwd().rstrip("/") or "/"
    except OSError:
        cwd = "/"

    panes = _v26_list_tmux_panes()
    candidates: list[dict[str, Any]] = []
    for p in panes:
        if not _v26_pane_runs_peer(p, target):
            continue
        score, reason = _v26_path_match_score(cwd, p.get("path") or "")
        candidates.append({
            **p,
            "score": score,
            "reason": reason,
            "etime": _v26_pane_etime_seconds(p.get("pid")),
        })

    if not candidates:
        return _v26_session_json_fallback(target)

    # 점수 내림차순, 동점 시 etime 오름차순 (최신 = 짧은 etime)
    candidates.sort(key=lambda c: (-c["score"], c["etime"]))
    best = candidates[0]

    if best["score"] <= 0:
        fb = _v26_session_json_fallback(target)
        fb["candidate_count"] = len(candidates)
        if fb.get("selection_reason") == "no_match":
            fb["selection_reason"] = "session_json_fallback"
        return fb

    pane_id = best["pane_id"]
    # session.json에서 socket 정보 보강 (있다면)
    socket: str | None = None
    session_hash: str | None = None
    try:
        sessions_root = _V2_BASE / "sessions"
        if sessions_root.exists():
            for d in sessions_root.iterdir():
                sj = d / "session.json"
                if not sj.exists():
                    continue
                try:
                    data = json.loads(sj.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                agent_meta = (data.get("active_agents") or {}).get(target) or {}
                if agent_meta.get("tmux_pane") == pane_id:
                    socket = agent_meta.get("tmux_socket")
                    session_hash = data.get("session_hash")
                    break
    except OSError:
        pass

    return {
        "ok": True,
        "pane": pane_id,
        "socket": socket,
        "session_hash": session_hash,
        "tmux_session": pane_id.split(":")[0] if ":" in pane_id else None,
        "tmux_pane": pane_id,
        "candidate_count": len(candidates),
        "selected_path": best.get("path"),
        "selection_reason": best["reason"],
    }


def _v26_pane_busy(pane: str, *, socket: str | None = None) -> bool:
    """Best-effort: capture last few lines of the pane and look for the
    Codex/Claude 'still working' indicator. Used by ``send_to_peer`` to defer
    submit until the prompt is idle. Returns False on any tmux error so we
    never block the caller solely because capture-pane failed."""
    # safe_capture_pane scrubs lone surrogates that would otherwise propagate
    # to the v26 send_to_peer return value (and from there to the MCP JSON
    # envelope that triggered HTTP 400 in the 2026-05-22 audit).
    tail = _safe_capture_pane(pane, lines=30, socket=socket)
    if not tail:
        return False
    # Centralized busy detection — includes "pursuing goal" (2026-05-24 fix) so
    # send_to_peer correctly defers/queues during Codex long-running goals.
    return _pane_text_is_busy(tail)


def _v26_pane_exists(pane: str, *, socket: str | None = None) -> bool:
    """Return True iff ``pane`` resolves to a live tmux pane.

    Used by :func:`ultradebate_v2_send_to_peer` to distinguish ``paste_failed``
    (pane vanished / tmux down) from ``submit_unverified`` (pane alive but
    no busy marker observed). Falls back to ``has-session`` heuristics when
    ``display-message`` is unavailable. Any tmux error returns False so the
    caller treats it as paste_failed (safer than the opposite).
    """
    sock_args = ["-S", socket] if socket else []
    try:
        proc = subprocess.run(
            ["tmux", *sock_args, "display-message", "-p", "-t", pane, "#{pane_id}"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


@mcp.tool()
def ultradebate_v2_run_long_job(
    cmd: str,
    est_minutes: float | None = None,
    peer: str | None = None,
    kind: str = "long_job",
    ack_sender: str = "system",
) -> dict[str, Any]:
    """Run a long-running shell command in a detached background process.

    The agent loop returns immediately with a ``job_id``; poll status via
    :func:`ultradebate_v2_check_job`. When ``peer`` is set, a ``job_started``
    ACK is auto-queued through :func:`ultradebate_v2_priority_send` so the
    peer cannot misinterpret silence as a dropped message.

    Returns the job metadata dict augmented with ``"ack_sent": bool``.
    """
    meta = _v26_jobs.submit_job(cmd, est_minutes=est_minutes, peer=peer, kind=kind)
    ack_sent = False
    if peer and peer != ack_sender and peer in ("claude", "codex"):
        try:
            ack_res = ultradebate_v2_priority_send(
                sender=ack_sender if ack_sender in ("claude", "codex") else "system",
                target=peer,
                priority="normal",
                msg_type="job_started",
                payload={
                    "job_id": meta["job_id"],
                    "cmd": meta["cmd"],
                    "kind": kind,
                    "est_minutes": est_minutes,
                    "log": meta["log"],
                    "started_at": meta["started_at"],
                },
            )
            ack_sent = bool(ack_res.get("ok"))
        except Exception as exc:
            # ACK is best-effort; the job is already running, so we never raise.
            meta["ack_error"] = str(exc)
    out = dict(meta)
    out["ack_sent"] = ack_sent
    out["ok"] = True
    return out


@mcp.tool()
def ultradebate_v2_check_job(job_id: str, log_tail_lines: int = 20) -> dict[str, Any]:
    """Check a background job's status. Non-blocking, returns in <1s."""
    return _v26_jobs.check_job(job_id, log_tail_lines=log_tail_lines)


@mcp.tool()
def ultradebate_v2_list_jobs(
    only_running: bool = False,
    max_items: int = 50,
) -> dict[str, Any]:
    """List recent background jobs, newest first."""
    return _v26_jobs.list_jobs(only_running=only_running, max_items=max_items)


@mcp.tool()
def ultradebate_v2_cancel_job(job_id: str, force: bool = False) -> dict[str, Any]:
    """Send TERM (or KILL when ``force=True``) to a running background job."""
    return _v26_jobs.cancel_job(job_id, sig="KILL" if force else "TERM")


@mcp.tool()
def ultradebate_v2_send_to_peer(
    sender: str,
    target: str,
    text: str,
    pane: str | None = None,
    priority: str = "normal",
    msg_type: str = "message",
    payload_extra: dict[str, Any] | None = None,
    tmux_inject: bool = True,
    wait_idle_sec: float = 60.0,
    queue_when_busy: bool = True,
    queue_wait_idle_sec: float = 5.0,
) -> dict[str, Any]:
    """Record a peer message and optionally submit it to an existing tmux pane.

    Inbox persistence precedes UI delivery. Codex always uses flattened
    bracketed paste followed by a separate C-m, including when busy; it never
    uses Tab. Other peers can use the legacy busy-queue path. Setting
    tmux_inject=False prevents all tmux delivery, including urgent auto-wake.

    ok=True means the inbox write succeeded. Check tmux_state and the peer's
    response separately: submitted, submit_unverified, no_pane, paste_failed,
    skipped, queued_busy, queued_via_tab, or queue_failed.
    """
    if sender == target:
        return {"ok": False, "error": "sender must differ from target"}
    if target not in ("claude", "codex"):
        return {"ok": False, "error": "target must be 'claude' or 'codex'"}

    payload: dict[str, Any] = {"text": text}
    if payload_extra:
        payload.update(payload_extra)
    inbox_res = ultradebate_v2_priority_send(
        sender=sender, target=target, priority=priority,
        msg_type=msg_type, payload=payload, auto_wake=False,
    )
    if not inbox_res.get("ok"):
        return {
            "ok": False, "stage": "priority_send",
            "error": inbox_res.get("error", "priority_send failed"),
            **inbox_res,
        }

    base_out: dict[str, Any] = {
        "ok": True,
        "msg_id": inbox_res.get("msg_id"),
        "inbox_path": inbox_res.get("target_inbox_path"),
        "ts": inbox_res.get("ts"),
        "pane": pane,
        "tmux_state": "skipped",
    }
    if not tmux_inject:
        return base_out

    socket: str | None = None
    if not pane:
        resolved = _v26_resolve_peer_pane(target)
        if resolved.get("ok"):
            pane = resolved.get("pane")
            socket = resolved.get("socket")
            base_out["pane"] = pane
            base_out["session_hash"] = resolved.get("session_hash")
    if not pane:
        base_out["tmux_state"] = "no_pane"
        return base_out

    # Wait briefly for pane to become idle. The message is already in the
    # inbox, so we never block past the deadline. When queue_when_busy=True
    # (default) we only wait long enough to cover *transient* busy flickers;
    # long Working sessions go straight to the Tab-queue path so the user
    # sees the payload land in codex's "Queued follow-up inputs" stack
    # instead of nothing (the 2026-05-23 fix).
    effective_idle_wait = min(
        max(0.0, float(queue_wait_idle_sec)),
        max(0.0, float(wait_idle_sec)),
    ) if queue_when_busy else max(0.0, float(wait_idle_sec))
    deadline = time.time() + effective_idle_wait
    while time.time() < deadline:
        if not _v26_pane_busy(pane, socket=socket):
            break
        time.sleep(1.0)

    # USER MANDATE 2026-06-03 (authority watching the live codex pane):
    # For codex, NEVER use the Tab "Queued follow-up inputs" path. When codex's
    # current job dragged into the next, Tab-queued messages were not picked up
    # correctly; the injection must behave exactly like the user's own Enter —
    # a single forced C-m. So codex ALWAYS goes through _v2_tmux_safe_send with
    # _V2_SUBMIT_KEYS["codex"] == ["C-m"], busy or idle. The "/n처럼 되는" bug is
    # structurally impossible: _v2_tmux_safe_send (a) flattens \n/\r to spaces,
    # (b) delivers the payload as ONE bracketed paste-buffer atom, and (c) sends
    # C-m as a SEPARATE keystroke only after the paste-settle window.
    force_cm_no_tab = (target == "codex")

    _PANE_TAIL_CAP_BUSY = 8 * 1024  # mirrors _PANE_TAIL_CAP below
    if _v26_pane_busy(pane, socket=socket) and not force_cm_no_tab:
        if queue_when_busy:
            # Busy path: paste + Tab → codex "Queued follow-up inputs" stack.
            # Enter would only add a literal newline here (Working state does
            # not accept Enter as submit), so Tab is the correct submit key.
            ok_q = _v26_tmux_queue_send(
                pane=pane,
                message=text,
                peer=target,
                socket=socket,
                flatten_newlines=True,
            )
            base_out["submit_keys"] = ["Tab"]
            if ok_q:
                base_out["tmux_state"] = "queued_via_tab"
            else:
                base_out["tmux_state"] = "queue_failed"
            # Surface pane_tail so the caller can confirm codex still shows
            # the busy marker (i.e. queue path was the right branch).
            base_out["pane_tail"] = _truncate_text_bytes(
                _safe_capture_pane(pane, lines=10, socket=socket),
                max_bytes=_PANE_TAIL_CAP_BUSY,
            )
            return _sanitize_payload(base_out)
        # Legacy behavior (queue_when_busy=False): preserve original
        # inbox-only fallback for callers that explicitly opt out of queueing.
        base_out["tmux_state"] = "queued_busy"
        return _sanitize_payload(base_out)

    # Hard-lock the submit sequence: codex -> exactly ["C-m"] (never Tab/Enter/
    # Escape), other peers keep their key but never a stray Tab. Belt-and-braces
    # with force_cm_no_tab above so codex submit can only ever be a single C-m.
    submit_keys = _sanitize_submit_keys(_V2_SUBMIT_KEYS.get(target, ["Enter"]), peer=target)
    ok = _v2_tmux_safe_send(
        pane=pane,
        message=text,
        submit_keys=submit_keys,
        flatten_newlines=True,
        socket=socket,
    )
    base_out["submit_keys"] = submit_keys

    def _capture_tail() -> str:
        # Always go through safe_capture_pane so any pane_tail surfaced in the
        # tool return is guaranteed surrogate-free (the 2026-05-22 HTTP 400 root
        # path went exactly through this tail string).
        return _safe_capture_pane(pane, lines=10, socket=socket)

    # Round-2 audit classification (2026-05-22):
    #   _v2_tmux_safe_send returns bool — we do one final capture-pane here to
    #   distinguish (a) paste/send subprocess actually failed vs
    #   (b) load/paste/send rc=0 but no busy marker (codex consumed fast OR
    #   submit key didn't actually fire). The bool alone cannot tell these
    #   apart, so the wrapper re-checks busy marker + pane existence.
    # 8KB cap on pane_tail in tool returns. The audit pane that triggered HTTP
    # 400 was ~1MB; even after surrogate scrub we never want a captured tail
    # that large flowing into an MCP JSON envelope. truncate_text_bytes cuts at
    # UTF-8 codepoint boundary so a truncated multi-byte char never becomes a
    # lone surrogate at the cut.
    _PANE_TAIL_CAP = 8 * 1024
    if not ok:
        # Wrapper said "False": no busy marker after 3 capture retries.
        # Re-check pane existence: if pane disappeared, that's paste_failed
        # (tmux subprocess error). Otherwise it's submit_unverified —
        # subprocess returned rc=0 throughout but busy marker never showed.
        tail = _capture_tail()
        base_out["pane_tail"] = _truncate_text_bytes(tail, max_bytes=_PANE_TAIL_CAP)
        # Empty tail + no pane = paste_failed (pane gone / tmux down).
        # Empty tail + pane present = also unverified (race: pane just
        # repainted). Non-empty tail without busy marker = submit_unverified.
        pane_exists = bool(tail) or _v26_pane_exists(pane, socket=socket)
        if not pane_exists:
            base_out["tmux_state"] = "paste_failed"
        else:
            # Busy marker re-check (one more chance — codex may have flipped
            # busy in the ~3s between capture loops).
            if _v26_pane_busy(pane, socket=socket):
                base_out["tmux_state"] = "submitted"
            else:
                base_out["tmux_state"] = "submit_unverified"
        return _sanitize_payload(base_out)

    # ok=True path: wrapper saw busy marker, but verify once more from the
    # caller's clock (race condition cover — busy may have already cleared).
    if _v26_pane_busy(pane, socket=socket):
        base_out["tmux_state"] = "submitted"
    else:
        # paste + submit fired and wrapper saw busy at some point, but it
        # already cleared. Treat as submitted (delivery happened) — the
        # peer just processed quickly.
        base_out["tmux_state"] = "submitted"
        base_out["pane_tail"] = _truncate_text_bytes(
            _capture_tail(), max_bytes=_PANE_TAIL_CAP,
        )
    return _sanitize_payload(base_out)


@mcp.tool()
def ultradebate_v2_send_to_codex(
    sender: str,
    text: str,
    pane: str | None = None,
    priority: str = "normal",
    msg_type: str = "message",
    payload_extra: dict[str, Any] | None = None,
    tmux_inject: bool = True,
    wait_idle_sec: float = 60.0,
) -> dict[str, Any]:
    """Convenience wrapper: send_to_peer with target='codex' (submit = C-m only)."""
    return ultradebate_v2_send_to_peer(
        sender=sender, target="codex", text=text, pane=pane,
        priority=priority, msg_type=msg_type, payload_extra=payload_extra,
        tmux_inject=tmux_inject, wait_idle_sec=wait_idle_sec,
    )


@mcp.tool()
def ultradebate_v2_send_to_claude(
    sender: str,
    text: str,
    pane: str | None = None,
    priority: str = "normal",
    msg_type: str = "message",
    payload_extra: dict[str, Any] | None = None,
    tmux_inject: bool = True,
    wait_idle_sec: float = 60.0,
) -> dict[str, Any]:
    """Convenience wrapper: send_to_peer with target='claude' (Enter)."""
    return ultradebate_v2_send_to_peer(
        sender=sender, target="claude", text=text, pane=pane,
        priority=priority, msg_type=msg_type, payload_extra=payload_extra,
        tmux_inject=tmux_inject, wait_idle_sec=wait_idle_sec,
    )


# ---------------------------------------------------------------------------
# v2.6c — MLP allocator (inference only, training stays offline)
# Design ref: MCP_MLP_도입_설계서.md §5–6. Recommendation only; dispatch
# remains deterministic. Training script: scripts/train_allocator_mlp.py
# ---------------------------------------------------------------------------

from . import allocator as _v26c_allocator  # noqa: E402 — late import keeps load order clean


@mcp.tool()
def ultradebate_v2_predict_allocation(features: dict[str, float]) -> dict[str, Any]:
    """Predict which peer should receive the next task (claude|codex|split).

    features must include all 19 names from FEATURE_ORDER (see allocator_meta.json).
    Missing fields default to 0.0 (a warning is returned in
    ``missing_features_defaulted_to_zero``).

    This is a *recommendation*. The dispatch decision is made by a deterministic
    tool that may override based on real-time signals (queue, quota).
    """
    return _v26c_allocator.predict(features)


@mcp.tool()
def ultradebate_v2_allocator_info() -> dict[str, Any]:
    """Diagnostic: model loaded? feature_order? classes? metrics?"""
    return _v26c_allocator.model_info()


@mcp.tool()
def ultradebate_v2_allocator_reload() -> dict[str, Any]:
    """Force re-load of the model file (after retraining)."""
    with _v26c_allocator._LOCK:
        _v26c_allocator._CACHE["model"] = None
        _v26c_allocator._CACHE["model_mtime"] = 0.0
    return _v26c_allocator.model_info()


# ---------------------------------------------------------------------------
# v2.6d — Live quota probe (real /status percentages → MLP feature accuracy)
# Design: feature values were proxy from message count; this lets the MLP
# consume actual reported quotas for live recommendations. Claude /status is
# not directly probable from inside the agent loop — manual recording path
# is provided.
# ---------------------------------------------------------------------------

from . import quota_probe as _v26d_quota  # noqa: E402 — late import keeps load order clean


@mcp.tool()
def ultradebate_v2_quota_probe_codex(
    pane: str | None = None,
    send_status_cmd: bool = True,
    record: bool = True,
) -> dict[str, Any]:
    """Probe live codex quota (5h + weekly % left) from its tmux pane.

    Args:
        pane: tmux pane id (e.g. ``"5:1.1"``). If ``None``, auto-resolve via
            ``_v26_resolve_peer_pane('codex')``.
        send_status_cmd: if True, paste ``/status`` and submit (Escape+Enter)
            before capturing. If False, just capture and parse whatever is
            currently visible.
        record: if True, append the result to the persistent quota log when
            parsing succeeds.

    Returns dict with keys::

        {ok, agent='codex', '5h_pct_left'?, 'weekly_pct_left'?,
         '5h_resets'?, 'weekly_resets'?, ...}
    """
    socket: str | None = None
    if pane is None:
        resolved = _v26_resolve_peer_pane("codex")
        if not resolved.get("ok"):
            return _sanitize_payload({
                "ok": False,
                "agent": "codex",
                "error": "no_codex_pane",
                "resolver": resolved,
            })
        pane = resolved.get("pane")
        socket = resolved.get("socket")
    result = _v26d_quota.probe_codex_quota(
        pane=pane,
        socket=socket,
        send_status_cmd=send_status_cmd,
    )
    if record and result.get("ok"):
        _v26d_quota.record_snapshot(result)
    return _sanitize_payload(result)


@mcp.tool()
def ultradebate_v2_quota_probe_claude() -> dict[str, Any]:
    """Returns a 'not_directly_probable' result with the manual recipe.

    Claude Code's ``/status`` is not capturable from inside the agent loop.
    Use ``ultradebate_v2_quota_record_manual`` for user-pasted snapshots.
    """
    return _sanitize_payload(_v26d_quota.probe_claude_quota())


@mcp.tool()
def ultradebate_v2_quota_claude_meta() -> dict[str, Any]:
    """Read claude plan metadata from ``~/.claude/.credentials.json``.

    Returns subscription_type + rate_limit_tier + OAuth expiry.
    *Cannot* return current 5h/weekly usage percentages — those are
    server-tracked and only visible in the host UI ``/status``. For
    percentages use ``ultradebate_v2_quota_record_manual`` after running
    ``/status`` in the host Claude session.
    """
    return _sanitize_payload(_v26d_quota.probe_claude_credentials_meta())


@mcp.tool()
def ultradebate_v2_quota_record_manual(
    agent: str,
    text: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Record a manual quota snapshot from pasted ``/status`` output.

    Use case: user runs ``/status`` in the Claude (or codex) session, copies
    the '5h limit' and 'Weekly limit' lines, calls this with ``text=<paste>``.
    The parser extracts %left values and appends to the quota log.
    """
    if agent not in ("claude", "codex", "gemini"):
        return _sanitize_payload({"ok": False, "error": "agent must be claude|codex|gemini"})
    parsed = _v26d_quota.parse_status_output(text or "")
    has_any = ("5h_pct_left" in parsed) or ("weekly_pct_left" in parsed)
    snapshot: dict[str, Any] = {
        "ok": bool(has_any),
        "agent": agent,
        "source": "manual",
        **parsed,
    }
    if notes:
        snapshot["notes"] = notes
    if snapshot["ok"]:
        _v26d_quota.record_snapshot(snapshot)
    else:
        snapshot["error"] = "no_status_lines_found_in_text"
    return _sanitize_payload(snapshot)


@mcp.tool()
def ultradebate_v2_quota_recent(
    agent: str | None = None,
    max_n: int = 20,
) -> dict[str, Any]:
    """Read last N quota snapshots from the persistent log."""
    return _sanitize_payload({
        "ok": True,
        "snapshots": _v26d_quota.read_recent_snapshots(agent=agent, max_n=max_n),
    })


@mcp.tool()
def ultradebate_v2_predict_allocation_live(
    claude_quota: dict[str, Any] | None = None,
    codex_quota: dict[str, Any] | None = None,
    extra_features: dict[str, float] | None = None,
) -> dict[str, Any]:
    """End-to-end live allocation: fill MLP features from real quotas, predict.

    If quota snapshots are None, the most recent recorded snapshot for each
    agent is used (codex from tmux probe, claude needs prior manual snapshot).
    Missing time/queue features default to 0.0 — caller can override via
    ``extra_features``.

    Returns::

        {ok, target, confidence, probabilities, features_used, quota_used}
    """
    claude_q = claude_quota
    codex_q = codex_quota
    if claude_q is None:
        recent_cl = _v26d_quota.read_recent_snapshots(agent="claude", max_n=1)
        claude_q = recent_cl[-1] if recent_cl else None
    if codex_q is None:
        recent_cx = _v26d_quota.read_recent_snapshots(agent="codex", max_n=1)
        codex_q = recent_cx[-1] if recent_cx else None

    feat_slots = _v26d_quota.snapshot_to_features({
        "claude": claude_q or {},
        "codex": codex_q or {},
    })

    # Cyclic time features; queue/job slots default to 0.0 (caller overrides
    # via extra_features when richer signals are available).
    import math as _v26d_math
    from datetime import datetime as _v26d_dt
    now = _v26d_dt.now()
    h = now.hour
    d = now.weekday()
    feats: dict[str, float] = {
        **feat_slots,
        # Coarse fallbacks — 1h/6h/24h windows approximated from 5h/weekly until
        # finer telemetry lands.
        "claude_1h_avg": feat_slots["claude_5h_used"],
        "codex_1h_avg": feat_slots["codex_5h_used"],
        "claude_6h_avg": feat_slots["claude_5h_used"],
        "codex_6h_avg": feat_slots["codex_5h_used"],
        "claude_24h_avg": feat_slots["claude_week_used"],
        "codex_24h_avg": feat_slots["codex_week_used"],
        "hour_sin": _v26d_math.sin(2 * _v26d_math.pi * h / 24),
        "hour_cos": _v26d_math.cos(2 * _v26d_math.pi * h / 24),
        "dow_sin": _v26d_math.sin(2 * _v26d_math.pi * d / 7),
        "dow_cos": _v26d_math.cos(2 * _v26d_math.pi * d / 7),
        "queue_len": 0.0,
        "recent_task_type_bucket": 0.0,
        "est_job_minutes": 0.0,
        "recent_ack_miss_rate": 0.0,
        "last_response_latency_s": 0.0,
    }
    if isinstance(extra_features, dict):
        feats.update({k: float(v) for k, v in extra_features.items()})

    pred = _v26c_allocator.predict(feats)
    if isinstance(pred, dict):
        pred = dict(pred)
        pred["features_used"] = feats
        pred["quota_used"] = {"claude": claude_q, "codex": codex_q}
    return _sanitize_payload(pred)


def main() -> None:
    """Entry point — runs the MCP stdio server."""
    mcp.run()


if __name__ == "__main__":
    main()
