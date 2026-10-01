"""Publication regressions: JSON alerts and one controlled tmux delivery."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from ultradebate import server as srv


@pytest.mark.parametrize("inject", [False, True])
def test_urgent_wrapper_controls_delivery(tmp_path, monkeypatch, inject):
    for name in ("_V24_BASE", "_V24_INBOX", "_V24_HEARTBEAT", "_V24_CYCLES"):
        monkeypatch.setattr(srv, name, tmp_path / name)
    monkeypatch.setattr(srv, "_V24_EVENTS", tmp_path / "events.jsonl")
    wake = Mock()
    send = Mock(return_value=True)
    monkeypatch.setattr(srv, "_v25_attempt_codex_tmux_wake", wake)
    monkeypatch.setattr(srv, "_v2_tmux_safe_send", send)
    monkeypatch.setattr(srv, "_v26_pane_busy", lambda *a, **kw: False)
    monkeypatch.setattr(srv, "_safe_capture_pane", lambda *a, **kw: "Working")
    result = srv.ultradebate_v2_send_to_codex(
        sender="claude", text="Review the patch", priority="urgent",
        pane="%1", tmux_inject=inject,
    )
    assert result["ok"]
    assert result["tmux_state"] == ("submitted" if inject else "skipped")
    assert json.loads((srv._V24_INBOX / "codex.jsonl").read_text())["payload"]["text"] == "Review the patch"
    wake.assert_not_called()
    assert send.call_count == int(inject)


@pytest.mark.parametrize("reason", ["cwd_match", "common_prefix", "session_json_fallback"])
def test_urgent_wake_requires_matching_project(monkeypatch, reason):
    monkeypatch.setattr(srv, "_v26_resolve_peer_pane", lambda target: {
        "ok": True, "pane": "%1", "socket": "/tmp/example.sock",
        "selection_reason": reason,
    })
    send = Mock(return_value=True)
    monkeypatch.setattr(srv, "_v2_tmux_safe_send", send)
    result = srv._v25_attempt_codex_tmux_wake("message", "review", "claude", "urgent")
    assert result["attempted"] == (reason == "cwd_match")
    if reason == "cwd_match":
        assert send.call_args.kwargs["socket"] == "/tmp/example.sock"
    else:
        send.assert_not_called()


def test_watcher_emits_json_events(tmp_path):
    # Feed a controlled stream through the real shell/Python pipeline.
    tail = tmp_path / "tail"
    tail.write_text("#!/bin/sh\nprintf '%s\\n' 'invalid' '[]' "
                    "'{\"from\":\"codex\",\"priority\":\"urgent\",\"type\":\"review\",\"msg_id\":\"0123456789abcdef\"}'\n")
    tail.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "scripts" / "claude_inbox_watcher.sh"
    # Isolate the inbox path without altering the shipped script.
    isolated = tmp_path / "watcher.sh"
    isolated.write_text(script.read_text().replace(
        "INBOX_JSONL=/tmp/ultradebate_v2/inbox/${PEER}.jsonl",
        f'INBOX_JSONL="{tmp_path}/inbox/${{PEER}}.jsonl"',
    ))
    result = subprocess.run(["bash", str(isolated), "claude"], capture_output=True,
                            text=True, timeout=10, env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "event": "V2_INBOX_NEW", "peer": "claude", "from": "codex",
        "priority": "urgent", "type": "review", "msg_id": "0123456789abcdef",
    }
    invalid = subprocess.run(["bash", str(isolated), "../other"], capture_output=True, text=True, timeout=10)
    assert invalid.returncode == 2
