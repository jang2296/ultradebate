"""Regression tests for v2.6 audit fix #1 — send-wrapper verify behavior.

Pin the contract that ``_v2_tmux_safe_send`` performs *active verification*:
  * paste-buffer settle scales with payload length (avoid premature submit
    on long 2500-char payloads).
  * After submit-keys, capture-pane checks for the Codex/Claude busy marker
    ("esc to interrupt" / "working" / "thinking").
  * If no busy marker on first capture, an additional Enter is retried
    (up to 2 retries).

The test mocks ``subprocess.run`` so we never touch a real tmux socket.
All paths exercise the in-process verify loop.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ultradebate import server as srv


def _fake_proc(returncode=0, stdout=""):
    """Mimic subprocess.CompletedProcess minimally."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


class _Recorder:
    """Capture subprocess.run invocations and return scripted results.

    Each entry in ``results`` is consumed FIFO. Falls back to a success stub
    with empty stdout when results are exhausted, matching real tmux behavior
    where housekeeping calls (delete-buffer) succeed silently.
    """

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, *args, **kwargs):
        cmd = list(args[0]) if args else list(kwargs.get("args", []))
        self.calls.append(cmd)
        if self.results:
            return self.results.pop(0)
        return _fake_proc(returncode=0, stdout="")


def _classify(cmd):
    if "load-buffer" in cmd:
        return "load"
    if "paste-buffer" in cmd:
        return "paste"
    if "send-keys" in cmd:
        return "submit"
    if "capture-pane" in cmd:
        return "capture"
    if "delete-buffer" in cmd:
        return "delete"
    return "other"


def test_short_payload_busy_marker_seen_returns_true(monkeypatch):
    """Short message + busy marker after submit -> single capture, return True."""
    rec = _Recorder([
        _fake_proc(0),                                  # load-buffer
        _fake_proc(0),                                  # paste-buffer
        _fake_proc(0),                                  # send-keys (first)
        _fake_proc(0, stdout="codex> Working... esc to interrupt"),  # capture-pane
    ])
    monkeypatch.setattr(srv.subprocess, "run", rec)
    monkeypatch.setattr(srv.time, "sleep", lambda _x: None)

    ok = srv._v2_tmux_safe_send(
        pane="0:1.1",
        message="hello codex",
        submit_keys=["Escape", "Enter"],
        settle_sec=0.01,
    )
    assert ok is True
    kinds = [_classify(c) for c in rec.calls]
    # Must contain at least: load, paste, submit, capture. No retry needed.
    assert "load" in kinds
    assert "paste" in kinds
    assert kinds.count("submit") == 1
    assert kinds.count("capture") == 1


def test_no_busy_marker_triggers_retry(monkeypatch):
    """No busy marker on first 2 captures -> retry Enter, finally succeed."""
    rec = _Recorder([
        _fake_proc(0),                                  # load
        _fake_proc(0),                                  # paste
        _fake_proc(0),                                  # submit #1
        _fake_proc(0, stdout="codex> "),                # capture #1 (no busy)
        _fake_proc(0),                                  # retry Enter
        _fake_proc(0, stdout="codex> Thinking..."),     # capture #2 (busy!)
    ])
    monkeypatch.setattr(srv.subprocess, "run", rec)
    monkeypatch.setattr(srv.time, "sleep", lambda _x: None)

    ok = srv._v2_tmux_safe_send(
        pane="0:1.1",
        message="needs retry",
        submit_keys=["Escape", "Enter"],
        settle_sec=0.01,
    )
    assert ok is True
    kinds = [_classify(c) for c in rec.calls]
    # Initial submit + at least 1 retry submit
    assert kinds.count("submit") >= 2
    assert kinds.count("capture") >= 2


def test_long_payload_scales_settle_proportional(monkeypatch):
    """2500-char message -> effective settle approx len/500 == 5s (capped at 8s).

    We patch time.sleep to record requested delays and verify the largest
    sleep arg is in the 5-8s window.
    """
    sleeps: list[float] = []
    rec = _Recorder([
        _fake_proc(0),                                  # load
        _fake_proc(0),                                  # paste
        _fake_proc(0),                                  # submit
        _fake_proc(0, stdout="esc to interrupt"),       # capture (busy)
    ])
    monkeypatch.setattr(srv.subprocess, "run", rec)
    monkeypatch.setattr(srv.time, "sleep", lambda x: sleeps.append(x))

    payload = "x" * 2500
    ok = srv._v2_tmux_safe_send(
        pane="0:1.1",
        message=payload,
        submit_keys=["Escape", "Enter"],
        settle_sec=2.0,
    )
    assert ok is True
    # Largest settle must be >= 5.0 (2500/500) and <= 8.0 (cap)
    assert max(sleeps) >= 5.0
    assert max(sleeps) <= 8.0


def test_load_buffer_failure_returns_false(monkeypatch):
    """tmux load-buffer rc != 0 -> propagate False (no submit attempted)."""
    rec = _Recorder([_fake_proc(1)])                    # load fails
    monkeypatch.setattr(srv.subprocess, "run", rec)
    monkeypatch.setattr(srv.time, "sleep", lambda _x: None)

    ok = srv._v2_tmux_safe_send(
        pane="0:1.1",
        message="anything",
        submit_keys=["Enter"],
        settle_sec=0.01,
    )
    assert ok is False
    kinds = [_classify(c) for c in rec.calls]
    assert "submit" not in kinds
    assert "capture" not in kinds


def test_paste_buffer_failure_returns_false(monkeypatch):
    """tmux paste-buffer rc != 0 -> propagate False."""
    rec = _Recorder([
        _fake_proc(0),                                  # load
        _fake_proc(1),                                  # paste fails
    ])
    monkeypatch.setattr(srv.subprocess, "run", rec)
    monkeypatch.setattr(srv.time, "sleep", lambda _x: None)

    ok = srv._v2_tmux_safe_send(
        pane="0:1.1",
        message="x",
        submit_keys=["Enter"],
        settle_sec=0.01,
    )
    assert ok is False


def test_paste_buffer_uses_bracketed_paste_flag(monkeypatch):
    """paste-buffer 호출에 ``-p`` 플래그가 반드시 포함되어야 한다 (bracketed paste).

    Codex/Claude TUI 가 paste-buffer 내용을 typed input으로 해석하지 않게
    하는 핵심 안전장치. 이 플래그가 빠지면 Escape+Enter가 in-input newline으로
    소비되어 submit 이 실패한다 (audit fix #1 의 1차 원인).
    """
    rec = _Recorder([
        _fake_proc(0),
        _fake_proc(0),
        _fake_proc(0),
        _fake_proc(0, stdout="esc to interrupt"),
    ])
    monkeypatch.setattr(srv.subprocess, "run", rec)
    monkeypatch.setattr(srv.time, "sleep", lambda _x: None)

    srv._v2_tmux_safe_send(
        pane="0:1.1",
        message="probe",
        submit_keys=["Escape", "Enter"],
        settle_sec=0.01,
    )
    paste_calls = [c for c in rec.calls if "paste-buffer" in c]
    assert paste_calls, "paste-buffer 호출이 없음"
    assert "-p" in paste_calls[0], (
        "paste-buffer 호출에 ``-p`` (bracketed paste) 플래그가 없음 — "
        "Codex submit 실패 회귀 위험"
    )


# --- Round-2 audit (2026-05-22) regression pins ----------------------------

def test_round2_no_busy_after_all_retries_returns_false(monkeypatch):
    """Codex round-2 verdict: load/paste/send all rc=0 + capture always idle
    must return *False*, not True.

    Round-1 returned True here, masking submit-failed-no-busy cases as
    success. Round-2 contract: only verified busy marker → True; everything
    else (after 3 capture retries) → False so the caller can downgrade
    tmux_state to ``submit_unverified`` and re-poll inbox.
    """
    # 1 load + 1 paste + 1 first submit + (capture + retry submit) x 3 captures
    rec = _Recorder([
        _fake_proc(0),                              # load
        _fake_proc(0),                              # paste
        _fake_proc(0),                              # submit #1
        _fake_proc(0, stdout="codex> "),            # capture #1 (idle)
        _fake_proc(0),                              # retry submit
        _fake_proc(0, stdout="codex> "),            # capture #2 (idle)
        _fake_proc(0),                              # retry submit
        _fake_proc(0, stdout="codex> "),            # capture #3 (idle)
    ])
    monkeypatch.setattr(srv.subprocess, "run", rec)
    monkeypatch.setattr(srv.time, "sleep", lambda _x: None)

    ok = srv._v2_tmux_safe_send(
        pane="0:1.1",
        message="never-confirmed",
        submit_keys=["Escape", "Enter"],
        settle_sec=0.01,
    )
    assert ok is False, (
        "Round-2 audit fix #1: capture-pane idle on all 3 attempts must "
        "return False (not the round-1 lenient True). caller classifies "
        "this as submit_unverified."
    )
    kinds = [_classify(c) for c in rec.calls]
    assert kinds.count("capture") == 3
    assert kinds.count("submit") >= 3  # initial + 2 retries


def test_round2_send_to_peer_classifies_submit_unverified(monkeypatch, tmp_path):
    """send_to_peer must map _v2_tmux_safe_send=False + live pane to
    tmux_state='submit_unverified' (round-2 classification).

    This pins the wrapper-level state granularity demanded by the round-2
    audit: a False from the inner wrapper is not "submit_failed" in general
    — it's the conservative ``submit_unverified`` only when the pane is
    still alive. paste_failed is reserved for cases where the pane itself
    is gone (tmux subprocess error path).
    """
    # Patch inbox dependencies + pane resolution so the test focuses purely
    # on the post-_v2_tmux_safe_send classification branch.
    monkeypatch.setattr(
        srv, "ultradebate_v2_priority_send",
        lambda **kw: {
            "ok": True, "msg_id": "test-msg-1",
            "target_inbox_path": str(tmp_path / "codex.jsonl"),
            "ts": "2026-05-22T00:00:00+00:00",
        },
    )
    # _v2_tmux_safe_send returns False (the round-2 strict path).
    monkeypatch.setattr(srv, "_v2_tmux_safe_send", lambda **kw: False)
    # Pane is alive (display-message returns rc=0 with id).
    monkeypatch.setattr(srv, "_v26_pane_exists", lambda *a, **kw: True)
    # Busy marker not present after the False return.
    monkeypatch.setattr(srv, "_v26_pane_busy", lambda *a, **kw: False)
    # Stub the capture-pane tail call inside send_to_peer.
    monkeypatch.setattr(
        srv.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout="codex> idle prompt", stderr=""
        ),
    )
    # Bypass the idle-wait loop quickly.
    monkeypatch.setattr(srv.time, "sleep", lambda _x: None)
    monkeypatch.setattr(srv.time, "time", lambda: 0.0)

    out = srv.ultradebate_v2_send_to_peer(
        sender="claude", target="codex",
        text="round-2 unverified probe",
        pane="0:1.1",
        wait_idle_sec=0.0,
    )
    assert out["ok"] is True
    assert out["tmux_state"] == "submit_unverified", (
        f"expected 'submit_unverified', got {out.get('tmux_state')!r}. "
        "Round-2 contract: wrapper-False + live pane = submit_unverified."
    )
    # The inbox line must have landed even though tmux submit was unverified.
    assert out.get("msg_id") == "test-msg-1"


def test_round2_send_to_peer_classifies_paste_failed(monkeypatch, tmp_path):
    """send_to_peer must map _v2_tmux_safe_send=False + dead pane to
    tmux_state='paste_failed' (true subprocess error, retry-safe)."""
    monkeypatch.setattr(
        srv, "ultradebate_v2_priority_send",
        lambda **kw: {
            "ok": True, "msg_id": "test-msg-2",
            "target_inbox_path": str(tmp_path / "codex.jsonl"),
            "ts": "2026-05-22T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(srv, "_v2_tmux_safe_send", lambda **kw: False)
    monkeypatch.setattr(srv, "_v26_pane_exists", lambda *a, **kw: False)
    monkeypatch.setattr(srv, "_v26_pane_busy", lambda *a, **kw: False)
    # capture-pane returns empty stdout (pane gone).
    monkeypatch.setattr(
        srv.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout="", stderr="no pane"),
    )
    monkeypatch.setattr(srv.time, "sleep", lambda _x: None)
    monkeypatch.setattr(srv.time, "time", lambda: 0.0)

    out = srv.ultradebate_v2_send_to_peer(
        sender="claude", target="codex",
        text="probe against dead pane",
        pane="9:9.9",
        wait_idle_sec=0.0,
    )
    assert out["tmux_state"] == "paste_failed", (
        f"expected 'paste_failed', got {out.get('tmux_state')!r}"
    )
