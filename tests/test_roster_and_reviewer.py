"""v1.0 server tests: roster toggle, lenient threshold, dedicated-reviewer rotation."""
from __future__ import annotations

import os

import pytest

# Force mock workers BEFORE importing server (module-level Config.from_env())
os.environ["ULTRADEBATE_MOCK"] = "1"

from ultradebate import server as srv  # noqa: E402


@pytest.mark.asyncio
async def test_session_create_with_partial_roster():
    res = await srv.ultradebate_session_create(
        topic="test partial roster",
        roster={"claude": 1, "codex": 0, "gemini": 1},
        rounds=2,
        mode="lenient",
    )
    assert "session_id" in res
    assert res["team_roster"]["claude"] == 1
    assert res["team_roster"]["codex"] == 0
    assert res["team_roster"]["gemini"] == 1
    assert res["active_agents"] == ["claude", "gemini"]


@pytest.mark.asyncio
async def test_set_roster_runtime_toggle():
    sess = await srv.ultradebate_start(
        topic="test runtime toggle",
        rounds=2,
        mode="lenient",
        roster={"claude": 1, "codex": 1, "gemini": 1},
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_set_roster(
        session_id=sid, roster={"gemini": 0}
    )
    assert res["ok"] is True
    assert res["team_roster"]["gemini"] == 0
    assert res["active_agents"] == ["claude", "codex"]
    assert res["revision"] >= 1


@pytest.mark.asyncio
async def test_lenient_threshold_scales_with_active_count():
    """lenient = ⌈N/2⌉ — for active=2, need 1; for active=3, need 2."""
    from ultradebate.server import _required_min

    assert _required_min("lenient", 1) == 1
    assert _required_min("lenient", 2) == 1
    assert _required_min("lenient", 3) == 2
    assert _required_min("lenient", 4) == 2
    assert _required_min("strict", 3) == 3
    assert _required_min("strict", 2) == 2


@pytest.mark.asyncio
async def test_dedicated_reviewer_rotates_per_round():
    """Reviewer index = (round_n - 1) % len(active)."""
    from ultradebate.server import _dedicated_reviewer

    active = ["claude", "codex", "gemini"]
    # rotate policy is on by default in mock-test config
    assert _dedicated_reviewer(1, active) == "claude"
    assert _dedicated_reviewer(2, active) == "codex"
    assert _dedicated_reviewer(3, active) == "gemini"
    assert _dedicated_reviewer(4, active) == "claude"  # wraps


@pytest.mark.asyncio
async def test_dedicated_reviewer_disabled_when_roster_too_small():
    from ultradebate.server import _dedicated_reviewer

    assert _dedicated_reviewer(1, []) is None
    assert _dedicated_reviewer(1, ["claude"]) is None


@pytest.mark.asyncio
async def test_blackboard_read_after_session_create():
    sess = await srv.ultradebate_session_create(
        topic="test blackboard read",
        roster={"claude": 1, "codex": 1, "gemini": 1},
        rounds=2,
        mode="lenient",
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_blackboard_read(session_id=sid, agent="claude")
    assert res["session_id"] == sid
    assert res["revision"] == 0
    assert "view_text" in res
    assert "@claude" in res["view_text"]


@pytest.mark.asyncio
async def test_blackboard_patch_revision_check():
    sess = await srv.ultradebate_start(
        topic="test patch revision",
        rounds=2,
        mode="lenient",
        roster={"claude": 1, "codex": 1, "gemini": 1},
    )
    sid = sess["session_id"]
    # Wrong expected revision should fail
    res = await srv.ultradebate_blackboard_patch(
        session_id=sid, patch={"focus": "x"}, expected_revision=99
    )
    assert res.get("error") == "revision_mismatch"
    # Correct revision should succeed
    res2 = await srv.ultradebate_blackboard_patch(
        session_id=sid, patch={"focus": "x"}, expected_revision=0
    )
    assert res2["ok"] is True
    assert res2["revision"] == 1


@pytest.mark.asyncio
async def test_round_with_two_agent_roster_lenient_passes():
    sess = await srv.ultradebate_start(
        topic="test two-agent debate",
        rounds=2,
        mode="lenient",
        roster={"claude": 1, "codex": 0, "gemini": 1},
    )
    sid = sess["session_id"]
    r1 = await srv.ultradebate_round(session_id=sid, round_n=1)
    assert "responses" in r1
    assert set(r1["active_agents"]) == {"claude", "gemini"}
    assert r1["dedicated_reviewer"] in {"claude", "gemini"}


@pytest.mark.asyncio
async def test_finalize_creates_archive_and_resets_transcript():
    sess = await srv.ultradebate_start(
        topic="test finalize archive",
        rounds=2,
        mode="lenient",
        roster={"claude": 1, "codex": 1, "gemini": 1},
    )
    sid = sess["session_id"]
    await srv.ultradebate_round(session_id=sid, round_n=1)
    res = await srv.ultradebate_finalize(session_id=sid)
    assert res["ok"] is True
    archive_dir = res["archive_dir"]
    from pathlib import Path

    assert Path(archive_dir, "final_report.md").exists()
    assert Path(archive_dir, "blackboard.snapshot.json").exists()
    assert Path(archive_dir, "consensus.json").exists()
    # transcript reset
    sdir = Path(res["report_path"]).parent.parent.parent
    assert (sdir / "transcript.jsonl").exists()
    assert (sdir / "transcript.jsonl").read_text() == ""
