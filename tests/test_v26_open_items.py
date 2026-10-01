"""Tests for v2.6 open_items — structural verdict enforcement.

If respond_to_questions ever accepts an incomplete verdict set, the design
constraint 'every numbered item must have a verdict before exchange completes'
is gone. These tests pin that contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ultradebate import open_items as oi


@pytest.fixture(autouse=True)
def isolated_track_dir(tmp_path, monkeypatch):
    """Each test gets a clean TRACK_DIR under tmp_path."""
    test_dir = tmp_path / "open_items"
    monkeypatch.setattr(oi, "TRACK_DIR", test_dir)
    yield test_dir


def test_register_then_full_respond_succeeds():
    """Happy path: open with 3 items, respond to all 3 → completed."""
    r = oi.open_questions("ex1", ["W13", "W14", "W15"])
    assert r["ok"] is True
    full = oi.respond_to_questions(
        "ex1", {"W13": "agree", "W14": "disagree", "W15": "ack"}
    )
    assert full["ok"] is True
    assert sorted(full["completed"]) == ["W13", "W14", "W15"]


def test_partial_respond_rejected():
    """CRITICAL: missing verdicts must be rejected at module level."""
    oi.open_questions("ex2", ["W1", "W2", "W3"])
    r = oi.respond_to_questions("ex2", {"W1": "agree"})  # missing W2, W3
    assert r["ok"] is False
    assert r["incomplete"] is True
    assert set(r["missing"]) == {"W2", "W3"}


def test_empty_string_verdict_rejected_as_missing():
    """A bare '' or whitespace verdict must be treated as missing."""
    oi.open_questions("ex3", ["A", "B"])
    r = oi.respond_to_questions("ex3", {"A": "agree", "B": ""})
    assert r["ok"] is False
    assert "B" in r["missing"]
    oi.open_questions("ex3b", ["A", "B"])
    r2 = oi.respond_to_questions("ex3b", {"A": "agree", "B": "   "})
    assert r2["ok"] is False
    assert "B" in r2["missing"]


def test_dict_verdict_with_empty_value_rejected():
    """A dict verdict with empty 'value' field is treated as missing."""
    oi.open_questions("ex4", ["X"])
    r = oi.respond_to_questions("ex4", {"X": {"value": "", "rationale": "x"}})
    assert r["ok"] is False
    assert "X" in r["missing"]


def test_dict_verdict_with_value_accepted():
    """A dict verdict with non-empty 'value' field is accepted."""
    oi.open_questions("ex5", ["X"])
    r = oi.respond_to_questions(
        "ex5", {"X": {"value": "disagree", "rationale": "y"}}
    )
    assert r["ok"] is True


def test_pending_verdicts_shows_open():
    """pending_verdicts lists open exchanges, omits completed ones."""
    oi.open_questions("pending-1", ["W1", "W2"])
    oi.open_questions("pending-2", ["A"])
    oi.respond_to_questions("pending-2", {"A": "ack"})
    p = oi.pending_verdicts()
    ids = {x["exchange_id"] for x in p["pending"]}
    assert "pending-1" in ids
    assert "pending-2" not in ids  # already completed


def test_path_traversal_blocked(isolated_track_dir):
    """exchange_id with '..' must be rejected. Files must stay inside TRACK_DIR."""
    r = oi.open_questions("../../../etc/passwd", ["X"])
    assert r["ok"] is False
    assert r.get("error") == "invalid_exchange_id"
    # Slash-containing IDs get sanitized to '_' (allowed), still inside TRACK_DIR.
    r2 = oi.open_questions("safe/sub/id", ["Y"])
    assert r2["ok"] is True
    # All written files must live under TRACK_DIR
    if isolated_track_dir.exists():
        for f in isolated_track_dir.glob("**/*"):
            assert str(f).startswith(str(isolated_track_dir))


def test_unknown_verdict_value_optional_strict():
    """By default any non-empty string is accepted; strict mode enforces enum."""
    oi.open_questions("strict-1", ["X"])
    r = oi.respond_to_questions("strict-1", {"X": "weird_value"})
    assert r["ok"] is True  # default: any non-empty string OK

    oi.open_questions("strict-2", ["X"])
    r2 = oi.respond_to_questions(
        "strict-2", {"X": "weird_value"}, require_known_verdict_values=True
    )
    assert r2["ok"] is False
    # Module reports bad_verdict_values list
    assert "X" in (r2.get("bad_verdict_values") or [])


def test_unknown_exchange_id():
    """respond_to_questions on a non-existent exchange returns unknown_exchange."""
    r = oi.respond_to_questions("does-not-exist", {"X": "agree"})
    assert r["ok"] is False
    assert r.get("error") == "unknown_exchange"


def test_reopen_same_items_idempotent():
    """Re-opening with identical items returns the existing record (reopened=True)."""
    oi.open_questions("dup-1", ["A", "B"])
    r = oi.open_questions("dup-1", ["A", "B"])
    assert r["ok"] is True
    assert r.get("reopened") is True


def test_reopen_different_items_rejected():
    """Re-opening with different items is refused — no silent state overwrite."""
    oi.open_questions("dup-2", ["A", "B"])
    r = oi.open_questions("dup-2", ["A", "C"])
    assert r["ok"] is False
    assert r.get("error") == "exchange_already_exists"


def test_empty_items_list_rejected():
    """open_questions requires non-empty items list."""
    r = oi.open_questions("empty-1", [])
    assert r["ok"] is False
    assert r.get("error") == "items_required"


def test_double_complete_rejected():
    """Once completed, respond_to_questions rejects further updates."""
    oi.open_questions("done-1", ["X"])
    oi.respond_to_questions("done-1", {"X": "agree"})
    r = oi.respond_to_questions("done-1", {"X": "disagree"})
    assert r["ok"] is False
    assert r.get("error") == "already_completed"
