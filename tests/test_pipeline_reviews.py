"""R4 pipeline + R12 cross-review tests."""
from __future__ import annotations

import os

import pytest

os.environ["ULTRADEBATE_MOCK"] = "1"

from ultradebate.state import blackboard as bb  # noqa: E402
from ultradebate.state import pipeline as pl  # noqa: E402
from ultradebate.state import reviews as rv  # noqa: E402
from ultradebate import server as srv  # noqa: E402


def _sdir(tmp_path):
    d = tmp_path / "sess"
    d.mkdir()
    return d


def test_pipeline_advances_through_all_stages(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="t", stage="RESEARCH")
    seen = []
    for _ in range(6):
        r = pl.advance(sdir)
        seen.append(r.next_stage)
        if r.next_stage == "DONE":
            break
    assert seen == ["DEBATE", "PLAN", "EXECUTE", "CROSS-REVIEW", "DONE"]


def test_pipeline_idempotent_on_done(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="t", stage="DONE")
    r = pl.advance(sdir)
    assert r.next_stage == "DONE"


def test_pipeline_stage_descriptions_present():
    for s in pl.VALID_STAGES:
        d = pl.stage_description(s)
        assert d  # non-empty


def test_review_dataclass_creation():
    r = rv.make_review(
        reviewer="claude", reviewee="codex", artifact_path="src/auth.ts"
    )
    assert r.reviewer == "claude"
    assert r.reviewee == "codex"
    assert r.artifact_path == "src/auth.ts"
    assert r.id.startswith("RV-")
    assert r.overall_pass is True
    assert r.severity == "minor"


def test_review_save_load_roundtrip(tmp_path):
    sdir = _sdir(tmp_path)
    r = rv.make_review(reviewer="claude", reviewee="codex", artifact_path="x.py")
    r.unimplemented = ["TODO at line 3"]
    r.overall_pass = False
    r.severity = "blocking"
    rv.save(sdir, r)
    loaded = rv.load_all(sdir)
    assert len(loaded) == 1
    assert loaded[0].overall_pass is False
    assert loaded[0].severity == "blocking"
    assert loaded[0].unimplemented == ["TODO at line 3"]


@pytest.mark.asyncio
async def test_run_pipeline_tool_advances_stage():
    sess = await srv.ultradebate_session_create(
        topic="pipeline test",
        roster={"claude": 1, "codex": 1, "gemini": 1, "grok": 0},
        rounds=1,
        mode="lenient",
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_run_pipeline(session_id=sid, max_loops=2)
    assert res["ok"] is True
    assert len(res["transitions"]) >= 1
    assert res["current_stage"] in pl.VALID_STAGES


@pytest.mark.asyncio
async def test_pipeline_stage_tool_reads_current():
    sess = await srv.ultradebate_session_create(
        topic="stage read test",
        roster={"claude": 1, "codex": 1, "gemini": 1, "grok": 0},
        rounds=1,
        mode="lenient",
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_pipeline_stage(session_id=sid)
    # Default stage from session_create is DEBATE
    assert res["stage"] == "DEBATE"
    assert res["next_stage"] == "PLAN"


@pytest.mark.asyncio
async def test_review_tool_persists_result():
    sess = await srv.ultradebate_session_create(
        topic="review test",
        roster={"claude": 1, "codex": 1, "gemini": 1, "grok": 0},
        rounds=1,
        mode="lenient",
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_review(
        session_id=sid,
        reviewer="claude",
        reviewee="codex",
        artifact_path="src/parser.py",
    )
    # Mock worker output won't be valid JSON — defaults preserved
    assert res["ok"] is True
    assert res["review_id"].startswith("RV-")
    listing = await srv.ultradebate_list_reviews(session_id=sid)
    assert listing["count"] == 1


@pytest.mark.asyncio
async def test_review_self_review_rejected():
    sess = await srv.ultradebate_session_create(
        topic="x",
        roster={"claude": 1, "codex": 1, "gemini": 1},
        rounds=1,
    )
    sid = sess["session_id"]
    res = await srv.ultradebate_review(
        session_id=sid, reviewer="claude", reviewee="claude", artifact_path="x.py"
    )
    assert "error" in res
    assert "differ" in res["error"]


def test_rolling_summary_directive_appears_in_long_transcript():
    """R10 — when entry count > threshold, instruction is prepended."""
    from ultradebate.prompts import render_transcript

    class E:
        def __init__(self, r, s):
            self.round = r
            self.speaker = s
            self.vote_position = "agree"
            self.analysis = "x" * 200

    entries_short = [E(1, "claude"), E(1, "codex")]
    out_short = render_transcript(entries_short)
    assert "ROLLING SUMMARY" not in out_short

    entries_long = [E(r, s) for r in range(1, 4) for s in ("claude", "codex", "gemini")]
    # 9 entries > threshold 6
    out_long = render_transcript(entries_long)
    assert "ROLLING SUMMARY" in out_long
    assert "≤5 bullet" in out_long
