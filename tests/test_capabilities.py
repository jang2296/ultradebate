"""R14 Capability Bridge tests — catalog whitelist, parser tag, server tool."""
from __future__ import annotations

import os

import pytest

os.environ["ULTRADEBATE_MOCK"] = "1"

from ultradebate.parser import parse_response  # noqa: E402
from ultradebate.state import capabilities as caps  # noqa: E402
from ultradebate import server as srv  # noqa: E402


def test_catalog_has_four_known_capabilities():
    ids = {c["id"] for c in caps.list_capabilities()}
    assert ids == {
        "superclaude:implement",
        "codex:goal",
        "gemini:long-context-scan",
        "claude:prompt-cache",
    }


def test_get_unknown_returns_none():
    assert caps.get("evil:exec") is None


def test_validate_args_missing_required():
    cap = caps.get("superclaude:implement")
    ok, reason = caps.validate_args(cap, {"spec": "x"})  # missing acceptance_criteria
    assert ok is False
    assert "acceptance_criteria" in reason


def test_validate_args_all_present():
    cap = caps.get("superclaude:implement")
    ok, _ = caps.validate_args(
        cap, {"spec": "build X", "acceptance_criteria": ["a", "b"]}
    )
    assert ok is True


def test_validate_args_rejects_unknown_keys():
    """LLM-found defect: unknown keys must be rejected (not silently ignored)."""
    cap = caps.get("superclaude:implement")
    ok, reason = caps.validate_args(
        cap,
        {"spec": "x", "acceptance_criteria": ["a"], "evil_extra": "drop_table"},
    )
    assert ok is False
    assert "unknown keys" in reason
    assert "evil_extra" in reason


def test_validate_args_allows_internal_rationale_marker():
    """Reserved __rationale__ key is allowed for internal use."""
    cap = caps.get("superclaude:implement")
    ok, _ = caps.validate_args(
        cap,
        {"spec": "x", "acceptance_criteria": ["a"], "__rationale__": "why"},
    )
    assert ok is True


def test_render_catalog_for_prompt_under_cap():
    text = caps.render_catalog_for_prompt(max_chars=2000)
    assert "superclaude:implement" in text
    assert "claude" in text
    assert len(text) <= 2030


def test_parser_extracts_capability_request():
    text = (
        "<analysis>x</analysis>"
        "<messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"agree","confidence":0.5}</vote>'
        '<capability_request>{"capability":"superclaude:implement",'
        '"args":{"spec":"X","acceptance_criteria":["a"]},'
        '"rationale":"need rigor"}</capability_request>'
    )
    r = parse_response(text)
    assert r.capability_request is not None
    assert r.capability_request.capability == "superclaude:implement"
    assert r.capability_request.args == {"spec": "X", "acceptance_criteria": ["a"]}
    assert r.capability_request.rationale == "need rigor"


def test_parser_capability_request_null_or_missing():
    text_null = (
        "<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"agree","confidence":0.5}</vote>'
        '<capability_request>null</capability_request>'
    )
    assert parse_response(text_null).capability_request is None

    text_missing = (
        "<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"agree","confidence":0.5}</vote>'
    )
    assert parse_response(text_missing).capability_request is None


@pytest.mark.asyncio
async def test_capabilities_tool_lists_whitelist():
    res = await srv.ultradebate_capabilities()
    ids = {c["id"] for c in res["capabilities"]}
    assert "superclaude:implement" in ids


@pytest.mark.asyncio
async def test_invoke_capability_unknown_rejected():
    sess = await srv.ultradebate_session_create(
        topic="cap test",
        roster={"claude": 1, "codex": 1, "gemini": 1},
        rounds=1,
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_invoke_capability(
        session_id=sid,
        requester="claude",
        capability_id="evil:exec",
        args={},
    )
    assert res["ok"] is False
    assert "unknown capability" in res["error"]


@pytest.mark.asyncio
async def test_invoke_capability_missing_args_rejected():
    sess = await srv.ultradebate_session_create(
        topic="cap missing args",
        roster={"claude": 1, "codex": 1, "gemini": 1},
        rounds=1,
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_invoke_capability(
        session_id=sid,
        requester="claude",
        capability_id="superclaude:implement",
        args={"spec": "x"},  # missing acceptance_criteria
    )
    assert res["ok"] is False
    assert "missing required keys" in res["error"]


@pytest.mark.asyncio
async def test_invoke_capability_happy_path_in_mock():
    sess = await srv.ultradebate_session_create(
        topic="cap happy path",
        roster={"claude": 1, "codex": 1, "gemini": 1},
        rounds=1,
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_invoke_capability(
        session_id=sid,
        requester="codex",
        capability_id="superclaude:implement",
        args={"spec": "add X", "acceptance_criteria": ["fast", "safe"]},
    )
    assert res["ok"] is True
    assert res["provider"] == "claude"
    assert res["delivered_to_inbox"] is True


def test_round_prompt_includes_capability_section():
    """Smoke check that build_round_prompt accepts capability_catalog."""
    from ultradebate.prompts import build_round_prompt

    prompt = build_round_prompt(
        agent_name="claude",
        other_agents=["codex"],
        topic="x",
        round_n=1,
        max_rounds=2,
        mode="lenient",
        mailbox_msgs=[],
        transcript_entries=[],
        tasks=[],
        capability_catalog="# Capability catalog\n- superclaude:implement",
    )
    assert "CAPABILITY BRIDGE" in prompt
    assert "superclaude:implement" in prompt
    assert "<capability_request>" in prompt
    # Vote placeholder must NOT be a pipe-list-as-value; v1.0 uses
    # explicit REPLACE_WITH markers. Either format is acceptable as long as
    # codex isn't tempted to echo "agree|disagree|abstain" verbatim.
    assert '"agree|disagree|abstain"' not in prompt
    assert "REPLACE_WITH_ONE_OF_agree_disagree_abstain" in prompt
