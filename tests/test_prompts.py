"""Prompt rendering tests."""
from __future__ import annotations

from ultradebate.prompts import (
    build_round_prompt,
    render_mailbox,
    render_tasks,
    render_transcript,
)
from ultradebate.state.tasks import Task
from ultradebate.state.transcript import TranscriptEntry


def test_render_transcript_empty():
    """v1.0: prompts are English internal contract (§5.4)."""
    assert "first round" in render_transcript([])


def test_render_transcript_truncates():
    long_analysis = "가나다" * 5000
    entry = TranscriptEntry(round=1, speaker="claude", analysis=long_analysis)
    out = render_transcript([entry], max_chars=2000)
    assert len(out) <= 2200  # max_chars + small buffer
    # Truncation marker may or may not appear depending on snippet trim — what
    # matters is the output respects the cap.


def test_render_mailbox_empty():
    assert "no incoming messages" in render_mailbox([])


def test_render_mailbox_with_msgs():
    out = render_mailbox([{"from": "codex", "to": "claude", "content": "hi", "round": 2}])
    assert "codex" in out
    assert "hi" in out


def test_render_tasks_empty():
    assert "empty task list" in render_tasks([])


def test_render_tasks_with_owner():
    t = Task(id="T-abc123", title="do X", status="in_progress", owner="codex")
    out = render_tasks([t])
    assert "T-abc123" in out
    assert "@codex" in out


def test_build_round_prompt_includes_required_sections():
    prompt = build_round_prompt(
        agent_name="claude",
        other_agents=["codex", "gemini"],
        topic="Should we adopt PEP 703?",
        round_n=1,
        max_rounds=5,
        mode="strict",
        mailbox_msgs=[],
        transcript_entries=[],
        tasks=[],
    )
    assert "claude" in prompt
    assert "codex" in prompt and "gemini" in prompt
    assert "PEP 703" in prompt
    # All four output tag mentions
    assert "<analysis>" in prompt
    assert "<messages>" in prompt
    assert "<tasks>" in prompt
    assert "<vote>" in prompt
