"""Regression tests for MCP JSON safety around malformed Unicode.

The Claude/Codex bridge may ingest tmux pane text or user-hook payloads that
contain lone UTF-16 surrogates after truncation.  Those strings are legal inside
Python but invalid for downstream JSON/API transport.  Pin the server-side
sanitizer so inbox reads/writes never re-emit such payloads.
"""
from __future__ import annotations

import json

from ultradebate import server as srv


def _assert_utf8_json_encodable(value):
    json.dumps(value, ensure_ascii=False).encode("utf-8")


def test_json_safe_obj_replaces_lone_surrogate():
    obj = {"payload": {"text": "bad-\ud83d-tail"}}

    safe = srv._json_safe_obj(obj)

    assert safe["payload"]["text"] == "bad-\ufffd-tail"
    _assert_utf8_json_encodable(safe)


def test_priority_send_sanitizes_payload_before_jsonl_write(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_V24_BASE", tmp_path)
    monkeypatch.setattr(srv, "_V24_INBOX", tmp_path / "inbox")
    monkeypatch.setattr(srv, "_V24_EVENTS", tmp_path / "events.jsonl")

    result = srv.ultradebate_v2_priority_send(
        sender="codex",
        target="claude",
        priority="urgent",
        msg_type="bad-\ud83d-type",
        payload={"text": "payload-\ud83d"},
        ref="ref-\ud83d",
    )

    assert result["ok"] is True
    _assert_utf8_json_encodable(result)

    inbox = tmp_path / "inbox" / "claude.jsonl"
    stored = json.loads(inbox.read_text().splitlines()[-1])
    assert stored["type"] == "bad-\ufffd-type"
    assert stored["ref"] == "ref-\ufffd"
    assert stored["payload"]["text"] == "payload-\ufffd"
    _assert_utf8_json_encodable(stored)


def test_priority_inbox_read_sanitizes_legacy_surrogate_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_V24_BASE", tmp_path)
    monkeypatch.setattr(srv, "_V24_INBOX", tmp_path / "inbox")
    (tmp_path / "inbox").mkdir(parents=True)

    legacy = {
        "ts": "2026-05-22T00:00:00+09:00",
        "from": "claude",
        "to": "codex",
        "priority": "normal",
        "type": "legacy",
        "msg_id": "m1",
        "payload": {"text": "legacy-\ud83d"},
    }
    # ensure_ascii=True mirrors legacy JSONL that can carry escaped surrogates.
    (tmp_path / "inbox" / "codex.jsonl").write_text(
        json.dumps(legacy, ensure_ascii=True) + "\n"
    )

    result = srv.ultradebate_v2_priority_inbox_read(agent="codex")

    assert result["ok"] is True
    assert result["messages"][0]["payload"]["text"] == "legacy-\ufffd"
    _assert_utf8_json_encodable(result)
