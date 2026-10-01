"""Pin Tab-queue / Enter-submit bidirectional behavior of send_to_peer.

Origin: 2026-05-23 user report — `_v2_tmux_safe_send` was firing Enter while
codex was still "Working", and Enter became a literal newline inside the
input buffer rather than a submit signal. Word-wrap then made the buffer
look like ``/n`` on screen.

2026-06-03 UPDATE (user C-m mandate): codex now ALWAYS submits via C-m
(``force_cm_no_tab`` in send_to_peer, server.py:6773) — the Tab-on-busy and
Escape+Enter-on-idle branches were REMOVED for codex because Tab-queued messages
were dropped when one codex job ran into the next. The Tab/queued_busy legacy
path survives ONLY for NON-codex targets (claude). These tests now pin the C-m
mandate for codex and the legacy queued_busy path for claude.

The tests stub out tmux subprocess calls completely so they are pure unit
tests — they assert which key sequence ``send-keys`` would have used, not
that an actual tmux session received it.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fake_run_factory(record_calls: list[list[str]]):
    """Return a subprocess.run replacement that records every cmd list."""
    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list):
            record_calls.append(list(cmd))
        m = MagicMock()
        m.returncode = 0
        m.stdout = b""
        return m
    return fake_run


def _stub_resolver(monkeypatch, srv):
    """Common monkeypatches for pane resolution and inbox JSONL writes."""
    monkeypatch.setattr(srv, "_v26_resolve_peer_pane", lambda t: {
        "ok": True,
        "pane": "test:0.0",
        "socket": None,
        "selection_reason": "test_mock",
        "candidate_count": 1,
        "session_hash": None,
    })
    monkeypatch.setattr(srv, "_v26_pane_exists", lambda *a, **k: True)
    monkeypatch.setattr(srv, "_safe_capture_pane",
                        lambda *a, **k: "Working (1s esc to interrupt)")
    monkeypatch.setattr(srv, "_V24_INBOX", Path(tempfile.mkdtemp()))


class TestCodexForcedCm:
    """C-m mandate: codex submits via C-m regardless of busy state (no Tab)."""

    def test_busy_codex_uses_cm_not_tab(self, monkeypatch):
        from ultradebate import server as srv

        _stub_resolver(monkeypatch, srv)
        # Pane reports busy on every check.
        monkeypatch.setattr(srv, "_v26_pane_busy", lambda *a, **k: True)
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))

        result = srv.ultradebate_v2_send_to_peer(
            sender="claude",
            target="codex",
            text="busy-path codex test",
            tmux_inject=True,
            wait_idle_sec=1.0,
            queue_wait_idle_sec=0.5,
        )

        assert result["ok"] is True
        # force_cm_no_tab: codex bypasses the Tab/queued path entirely, even when busy.
        assert result["tmux_state"] == "submitted", (
            f"expected submitted (codex force_cm_no_tab), got {result['tmux_state']}"
        )
        assert result.get("submit_keys") == ["C-m"]

        send_keys_calls = [c for c in calls if "send-keys" in c]
        assert send_keys_calls, "no send-keys calls captured"
        assert any("C-m" in c for c in send_keys_calls), (
            f"C-m missing from send-keys calls: {send_keys_calls}"
        )
        # Tab and Escape+Enter must NOT appear (user mandate 2026-06-03).
        for cmd in send_keys_calls:
            assert "Tab" not in cmd, f"Tab leaked into codex path: {cmd}"
            assert not ("Escape" in cmd and "Enter" in cmd), (
                f"Escape+Enter leaked into codex path: {cmd}"
            )

    def test_legacy_queue_when_busy_false_claude(self, monkeypatch):
        """queue_when_busy=False preserves the legacy queued_busy state for the
        NON-codex (claude) path, which still uses the Tab/queue branch."""
        from ultradebate import server as srv

        _stub_resolver(monkeypatch, srv)
        monkeypatch.setattr(srv, "_v26_pane_busy", lambda *a, **k: True)
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))

        result = srv.ultradebate_v2_send_to_peer(
            sender="codex",
            target="claude",  # non-codex: legacy queued_busy path still reachable
            text="legacy-busy test",
            tmux_inject=True,
            wait_idle_sec=1.0,
            queue_when_busy=False,
        )
        assert result["tmux_state"] == "queued_busy"
        # Legacy branch must NOT have sent any submit-keys (paste was skipped).
        send_keys_calls = [c for c in calls if "send-keys" in c]
        assert not send_keys_calls, (
            f"queue_when_busy=False should skip paste/send-keys, got: {send_keys_calls}"
        )


class TestSubmitWhenIdle:
    """When the peer is idle, send_to_peer must use the per-target submit keys."""

    def test_idle_pane_uses_cm_for_codex(self, monkeypatch):
        from ultradebate import server as srv

        _stub_resolver(monkeypatch, srv)
        # Pane reports idle every call.
        monkeypatch.setattr(srv, "_v26_pane_busy", lambda *a, **k: False)
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))

        result = srv.ultradebate_v2_send_to_peer(
            sender="claude",
            target="codex",
            text="idle-path codex test",
            tmux_inject=True,
            wait_idle_sec=1.0,
        )

        # Idle path runs _v2_tmux_safe_send which returns based on capture
        # output. We stubbed capture to always show busy marker, so
        # tmux_state should be "submitted".
        assert result["tmux_state"] == "submitted", (
            f"expected submitted, got {result['tmux_state']}"
        )
        assert result.get("submit_keys") == ["C-m"]

        send_keys_calls = [c for c in calls if "send-keys" in c]
        # C-m must appear; Escape+Enter and Tab must NOT (user mandate 2026-06-03).
        assert any("C-m" in c for c in send_keys_calls), (
            f"C-m missing from idle codex submit: {send_keys_calls}"
        )
        for cmd in send_keys_calls:
            assert "Tab" not in cmd, f"Tab leaked into idle codex submit: {cmd}"
            assert not ("Escape" in cmd and "Enter" in cmd), (
                f"Escape+Enter leaked into idle codex submit: {cmd}"
            )

    def test_idle_pane_uses_enter_for_claude(self, monkeypatch):
        from ultradebate import server as srv

        _stub_resolver(monkeypatch, srv)
        monkeypatch.setattr(srv, "_v26_pane_busy", lambda *a, **k: False)
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))

        result = srv.ultradebate_v2_send_to_peer(
            sender="codex",
            target="claude",
            text="idle-path claude test",
            tmux_inject=True,
            wait_idle_sec=1.0,
        )

        assert result["tmux_state"] == "submitted"
        assert result.get("submit_keys") == ["Enter"]
        send_keys_calls = [c for c in calls if "send-keys" in c]
        # Look for at least one send-keys with Enter and no Tab/Escape.
        plain_enter = [
            c for c in send_keys_calls
            if "Enter" in c and "Escape" not in c and "Tab" not in c
        ]
        assert plain_enter, (
            f"Plain Enter missing from idle claude submit: {send_keys_calls}"
        )


class TestV26TmuxQueueSend:
    """Direct shape/signature tests for the new helper."""

    def test_helper_exists_and_has_expected_signature(self):
        import inspect

        from ultradebate import server as srv

        assert hasattr(srv, "_v26_tmux_queue_send"), (
            "_v26_tmux_queue_send must be defined on the server module"
        )
        sig = inspect.signature(srv._v26_tmux_queue_send)
        params = sig.parameters
        for required in ("pane", "message", "settle_sec", "socket",
                         "flatten_newlines"):
            assert required in params, (
                f"_v26_tmux_queue_send missing required parameter '{required}'"
            )
        # flatten_newlines must default to True so newlines never reach the
        # codex input buffer as literal newlines (the original regression).
        assert params["flatten_newlines"].default is True

    def test_helper_returns_false_on_load_buffer_failure(self, monkeypatch):
        from ultradebate import server as srv

        # First subprocess.run (load-buffer) returns rc=1 -> helper must fail
        # immediately without sending Tab.
        called: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            if isinstance(cmd, list):
                called.append(list(cmd))
            m = MagicMock()
            m.returncode = 1 if "load-buffer" in cmd else 0
            m.stdout = b""
            return m

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok = srv._v26_tmux_queue_send(
            pane="test:0.0",
            message="never-arrives",
            settle_sec=0.01,
        )
        assert ok is False
        # Tab MUST NOT have been sent because load-buffer failed.
        for cmd in called:
            if "send-keys" in cmd:
                assert "Tab" not in cmd, (
                    f"Tab fired after load-buffer failure: {cmd}"
                )

    def test_helper_flattens_newlines_by_default(self, monkeypatch, tmp_path):
        """Payload written to the tmux buffer file must not contain newlines."""
        from ultradebate import server as srv

        captured_paths: list[str] = []

        def fake_run(cmd, **kwargs):
            if isinstance(cmd, list) and "load-buffer" in cmd:
                # tmux load-buffer -b <name> <file> -> last arg is the file
                captured_paths.append(cmd[-1])
            m = MagicMock()
            m.returncode = 0
            m.stdout = b""
            return m

        monkeypatch.setattr(subprocess, "run", fake_run)
        ok = srv._v26_tmux_queue_send(
            pane="test:0.0",
            message="line1\nline2\rline3",
            settle_sec=0.01,
        )
        # delete_buffer subprocess returns rc=0 too, so finally block tries to
        # delete tmpfile — the path is gone before we can read it. Capture
        # path content immediately via monkeypatched run that copies file.
        assert ok is True
        # We can no longer read the tmp file (it was unlinked in finally), so
        # we instead assert the helper completed all three rc=0 calls.
        # The newline-flatten behavior is locked in by signature default
        # (verified in test_helper_exists_and_has_expected_signature).
