"""Mailbox routing tests."""
from __future__ import annotations

from ultradebate.state import mailbox as mb


def test_deliver_then_fetch(session_dir):
    mb.deliver(session_dir, "claude", "codex", "hello", round_n=1)
    msgs = mb.fetch_inbox(session_dir, "codex")
    assert len(msgs) == 1
    assert msgs[0]["from"] == "claude"
    assert msgs[0]["content"] == "hello"
    # Second fetch should be empty (rotated to processed/).
    assert mb.fetch_inbox(session_dir, "codex") == []


def test_peek_does_not_consume(session_dir):
    mb.deliver(session_dir, "gemini", "claude", "ping", round_n=2)
    assert mb.peek_inbox(session_dir, "claude") == 1
    assert mb.peek_inbox(session_dir, "claude") == 1  # idempotent
    msgs = mb.fetch_inbox(session_dir, "claude")
    assert len(msgs) == 1


def test_isolation_per_recipient(session_dir):
    mb.deliver(session_dir, "claude", "codex", "for codex", round_n=1)
    mb.deliver(session_dir, "claude", "gemini", "for gemini", round_n=1)
    codex_msgs = mb.fetch_inbox(session_dir, "codex")
    gemini_msgs = mb.fetch_inbox(session_dir, "gemini")
    assert len(codex_msgs) == 1 and codex_msgs[0]["content"] == "for codex"
    assert len(gemini_msgs) == 1 and gemini_msgs[0]["content"] == "for gemini"


def test_multiple_messages_preserve_order(session_dir):
    for i in range(5):
        mb.deliver(session_dir, "claude", "codex", f"msg-{i}", round_n=1)
    msgs = mb.fetch_inbox(session_dir, "codex")
    assert [m["content"] for m in msgs] == [f"msg-{i}" for i in range(5)]
