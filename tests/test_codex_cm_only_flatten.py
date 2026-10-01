"""Lock the 2026-06-16 'codex C-m ONLY + flatten Tab/newline/ESC' mandate.

User mandate: when delivering to codex, even if the payload itself contains
\\t / \\n / \\r / ESC, the message must paste as ONE inert line and be submitted
by a single C-m — never a Tab (which queues/autocompletes), never a literal
newline. These tests pin the two sanitizers (_flatten_payload_for_tmux,
_sanitize_submit_keys) and the _v26_tmux_queue_send codex guard.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fake_run_factory(record: list[list[str]]):
    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list):
            record.append(list(cmd))
        m = MagicMock()
        m.returncode = 0
        m.stdout = b""
        return m
    return fake_run


class TestFlattenPayload:
    def test_all_c0_controls_to_space(self):
        from ultradebate import server as s
        # tab, newline, CR, ESC, DEL all collapse to a single space.
        assert s._flatten_payload_for_tmux("a\tb\nc\rd\x1be\x7ff") == "a b c d e f"

    def test_utf8_preserved(self):
        from ultradebate import server as s
        assert s._flatten_payload_for_tmux("한글\tテスト\n끝") == "한글 テスト 끝"

    def test_idempotent(self):
        from ultradebate import server as s
        once = s._flatten_payload_for_tmux("x\ty\nz")
        assert s._flatten_payload_for_tmux(once) == once

    def test_no_flatten_still_removes_tab_and_esc(self):
        from ultradebate import server as s
        out = s._flatten_payload_for_tmux("a\tb\nc\x1bd", flatten=False)
        assert "\t" not in out and "\x1b" not in out
        assert "\n" in out  # newline deliberately preserved in non-flatten mode


class TestSanitizeSubmitKeys:
    def test_codex_hard_locked_to_cm(self):
        from ultradebate import server as s
        for keys in (["Tab"], ["Escape", "Enter"], ["Enter"], ["C-m"], None, ["Tab", "C-m"]):
            assert s._sanitize_submit_keys(keys, peer="codex") == ["C-m"], keys

    def test_non_codex_strips_tab_keeps_enter(self):
        from ultradebate import server as s
        assert s._sanitize_submit_keys(["Enter"]) == ["Enter"]
        assert s._sanitize_submit_keys(["Tab"]) == ["C-m"]          # tab-only -> default C-m
        assert s._sanitize_submit_keys(["Tab", "Enter"]) == ["Enter"]


class TestQueueSendCodexGuard:
    def test_codex_queue_falls_back_to_cm_never_tab(self, monkeypatch):
        from ultradebate import server as s
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))
        monkeypatch.setattr(
            s, "_safe_capture_pane",
            lambda *a, **k: "Working (1s) esc to interrupt",
        )
        ok = s._v26_tmux_queue_send(
            pane="t:0.0", message="x\ty\nz", peer="codex", settle_sec=0.0,
        )
        assert ok is True
        send_keys = [c for c in calls if "send-keys" in c]
        assert any("C-m" in c for c in send_keys), send_keys
        for c in send_keys:
            assert "Tab" not in c, f"Tab leaked to codex via queue path: {c}"

    def test_claude_queue_still_uses_tab(self, monkeypatch):
        from ultradebate import server as s
        calls: list[list[str]] = []
        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))
        ok = s._v26_tmux_queue_send(
            pane="t:0.0", message="hello claude", peer="claude", settle_sec=0.0,
        )
        assert ok is True
        send_keys = [c for c in calls if "send-keys" in c]
        assert any("Tab" in c for c in send_keys), (
            f"non-codex queue path should still Tab: {send_keys}"
        )


class TestEndToEndCodexNoTabNoNewline:
    def test_busy_codex_with_tabby_payload_only_cm(self, monkeypatch):
        """Full send_to_peer(codex) with a payload full of \\t/\\n while the pane
        is busy: no Tab anywhere, submit is C-m, and the load-buffer temp file
        content is flattened (no raw \\t/\\n that tmux could interpret)."""
        from ultradebate import server as s
        import tempfile as _tf

        calls: list[list[str]] = []
        written: list[str] = []

        monkeypatch.setattr(subprocess, "run", _fake_run_factory(calls))
        monkeypatch.setattr(s, "_v26_resolve_peer_pane", lambda t: {
            "ok": True, "pane": "test:0.0", "socket": None, "session_hash": None,
        })
        monkeypatch.setattr(s, "_v26_pane_exists", lambda *a, **k: True)
        monkeypatch.setattr(s, "_v26_pane_busy", lambda *a, **k: True)
        monkeypatch.setattr(s, "_safe_capture_pane",
                            lambda *a, **k: "Working (1s) esc to interrupt")
        monkeypatch.setattr(s, "_V24_INBOX", Path(_tf.mkdtemp()))

        # Capture what gets written to the paste temp file.
        real_ntf = _tf.NamedTemporaryFile

        def spy_ntf(*a, **k):
            fh = real_ntf(*a, **k)
            orig_write = fh.write

            def wrap(data):
                written.append(data if isinstance(data, str) else data.decode("utf-8", "replace"))
                return orig_write(data)
            fh.write = wrap
            return fh
        monkeypatch.setattr(_tf, "NamedTemporaryFile", spy_ntf)

        result = s.ultradebate_v2_send_to_peer(
            sender="claude", target="codex",
            text="line1\tcol2\nline2\rwith\x1besc",
            tmux_inject=True, wait_idle_sec=1.0, queue_wait_idle_sec=0.5,
        )
        assert result["ok"] is True
        assert result.get("submit_keys") == ["C-m"]

        send_keys = [c for c in calls if "send-keys" in c]
        assert send_keys and any("C-m" in c for c in send_keys)
        for c in send_keys:
            assert "Tab" not in c
            assert not ("Escape" in c and "Enter" in c)

        # The pasted payload must be flattened — no raw control bytes survive.
        assert written, "expected a paste temp-file write"
        joined = "".join(written)
        for bad in ("\t", "\n", "\r", "\x1b"):
            assert bad not in joined, f"raw {bad!r} reached the paste buffer"
