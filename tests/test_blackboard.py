"""Blackboard tests: load/patch/revision-conflict/role-aware view."""
from __future__ import annotations

import pytest

from ultradebate.state import blackboard as bb


def _sdir(tmp_path):
    d = tmp_path / "sess"
    d.mkdir()
    return d


def test_init_creates_file_with_revision_zero(tmp_path):
    sdir = _sdir(tmp_path)
    b = bb.init(sdir, session_id="sid", topic="T", team_roster={"claude": 1, "codex": 1})
    assert b.revision == 0
    assert (sdir / "blackboard.json").exists()
    loaded = bb.load(sdir)
    assert loaded.session_id == "sid"
    assert loaded.team_roster == {"claude": 1, "codex": 1}


def test_patch_increments_revision(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T")
    b1 = bb.patch(sdir, {"focus": "first"})
    assert b1.revision == 1
    assert b1.focus == "first"
    b2 = bb.patch(sdir, {"focus": "second"})
    assert b2.revision == 2
    assert b2.focus == "second"


def test_patch_with_correct_expected_revision_succeeds(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T")
    b = bb.patch(sdir, {"focus": "v1"}, expected_revision=0)
    assert b.revision == 1


def test_patch_with_wrong_expected_revision_raises(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T")
    bb.patch(sdir, {"focus": "v1"})  # rev=1
    with pytest.raises(bb.RevisionMismatch) as exc:
        bb.patch(sdir, {"focus": "v2"}, expected_revision=99)
    assert exc.value.expected == 99
    assert exc.value.actual == 1


def test_decided_add_string_assigns_id(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T")
    b = bb.patch(sdir, {"decided": {"add": ["use RS256"]}})
    assert len(b.decided) == 1
    assert b.decided[0]["text"] == "use RS256"
    assert b.decided[0]["id"].startswith("D-")


def test_decided_add_dict_preserves_explicit_id(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T")
    b = bb.patch(
        sdir,
        {"decided": {"add": [{"id": "D-042", "text": "explicit", "scope": "global"}]}},
    )
    assert any(d["id"] == "D-042" for d in b.decided)


def test_resolve_open_question_moves_to_decided(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T")
    bb.patch(sdir, {"open_questions": {"add": [{"id": "Q-001", "text": "x?"}]}})
    b = bb.patch(sdir, {"open_questions": {"resolve": "Q-001", "note": "go with X"}})
    assert all(q["id"] != "Q-001" for q in b.open_questions)
    assert any("Q-001" in d.get("text", "") for d in b.decided)


def test_team_roster_set_merges(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T", team_roster={"claude": 1, "codex": 1})
    b = bb.patch(sdir, {"team_roster": {"set": {"gemini": 1, "claude": 0}}})
    assert b.team_roster["gemini"] == 1
    assert b.team_roster["claude"] == 0
    assert b.team_roster["codex"] == 1


def test_assignments_update_sets_owner(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T")
    bb.patch(sdir, {"assignments": {"add": [{"id": "T-1", "title": "x"}]}})
    b = bb.patch(sdir, {"assignments": {"update": {"T-1": "claude"}}})
    found = next(a for a in b.assignments if a["id"] == "T-1")
    assert found["owner"] == "claude"


def test_role_aware_view_filters_other_agent_decisions(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="JWT auth")
    bb.patch(
        sdir,
        {"decided": {"add": [{"id": "D-1", "scope": "claude", "text": "claude-only"}]}},
    )
    bb.patch(
        sdir,
        {"decided": {"add": [{"id": "D-2", "scope": "global", "text": "global"}]}},
    )
    bb.patch(
        sdir,
        {"decided": {"add": [{"id": "D-3", "scope": "codex", "text": "codex-only"}]}},
    )
    b = bb.load(sdir)
    view = bb.role_aware_view(b, "claude")
    assert "claude-only" in view
    assert "global" in view
    assert "codex-only" not in view


def test_role_aware_view_caps_size(tmp_path):
    sdir = _sdir(tmp_path)
    bb.init(sdir, session_id="sid", topic="T")
    huge = "x" * 5000
    bb.patch(sdir, {"focus": huge})
    b = bb.load(sdir)
    view = bb.role_aware_view(b, "claude", max_chars=500)
    assert len(view) <= 530


def test_active_agents_filter(tmp_path):
    sdir = _sdir(tmp_path)
    b = bb.init(
        sdir,
        session_id="sid",
        topic="T",
        team_roster={"claude": 1, "codex": 0, "gemini": 1, "grok": 0},
    )
    assert bb.active_agents(b) == ["claude", "gemini"]
