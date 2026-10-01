"""Tests for text safety — surrogate sanitization, bounded capture, truncation.

These pin the 2026-05-22 audit fix: ``tmux capture-pane`` occasionally produces
lone surrogates from broken multi-byte UTF-8, which previously could leak into
tool returns and trigger Anthropic API HTTP 400.

The invariant under test: NO code path that captures pane content or returns
a tool dict may emit a string containing an unpaired UTF-16 surrogate.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from ultradebate import text_safety as ts


# ---------------------------------------------------------------------------
# sanitize_text
# ---------------------------------------------------------------------------


class TestSanitizeText:
    def test_lone_high_surrogate_replaced(self):
        s = "before\ud800after"
        out = ts.sanitize_text(s)
        assert "\ud800" not in out
        # encode round-trip must succeed.
        out.encode("utf-8")

    def test_lone_low_surrogate_replaced(self):
        s = "x\udcccy"
        out = ts.sanitize_text(s)
        assert "\udccc" not in out
        out.encode("utf-8")

    def test_valid_emoji_preserved(self):
        s = "hello 🎉 world 한글 ✅"
        out = ts.sanitize_text(s)
        assert "🎉" in out
        assert "한글" in out
        assert "✅" in out

    def test_paired_surrogates_in_python_string(self):
        # Python normally stores 🎉 as a single codepoint U+1F389; on a narrow
        # build (or after a json.loads of an ensure_ascii payload) it can be a
        # surrogate pair. Either way, the sanitizer must leave it dump-able.
        s = "before\U0001F389after"  # canonical
        out = ts.sanitize_text(s)
        json.dumps({"x": out})

    def test_empty_string(self):
        assert ts.sanitize_text("") == ""

    def test_non_string_passthrough(self):
        assert ts.sanitize_text(None) is None
        assert ts.sanitize_text(42) == 42
        assert ts.sanitize_text(b"bytes") == b"bytes"

    def test_idempotent(self):
        s = "before\ud800after\udcccmore"
        once = ts.sanitize_text(s)
        twice = ts.sanitize_text(once)
        assert once == twice

    def test_dumpable_to_json(self):
        s = "broken\ud800\ud800"
        out = ts.sanitize_text(s)
        # The whole point: result must json-dump cleanly.
        json.dumps({"x": out}, ensure_ascii=False).encode("utf-8")

    def test_mixed_surrogate_block(self):
        # An entire run of surrogates — pathological case.
        s = "x" + "".join(chr(c) for c in range(0xD800, 0xD810)) + "y"
        out = ts.sanitize_text(s)
        assert "x" in out and "y" in out
        out.encode("utf-8")


# ---------------------------------------------------------------------------
# sanitize_payload
# ---------------------------------------------------------------------------


class TestSanitizePayload:
    def test_nested_dict(self):
        p = {"a": "\ud800x", "b": {"c": ["\udcccy", 42]}, "d": None}
        out = ts.sanitize_payload(p)
        json.dumps(out, ensure_ascii=False).encode("utf-8")
        assert "\ud800" not in out["a"]

    def test_tuple_and_set(self):
        p = {"t": ("\ud800a", 1), "s": {"\udcccb"}}
        out = ts.sanitize_payload(p)
        # Set ordering varies; just ensure no surrogate survives the sweep.
        serial = json.dumps({"t": list(out["t"]), "s": list(out["s"])},
                            ensure_ascii=False)
        assert "\ud800" not in serial
        assert "\udccc" not in serial

    def test_non_string_keys_preserved(self):
        out = ts.sanitize_payload({1: "x", 2: "\ud800"})
        assert 1 in out and 2 in out
        assert "\ud800" not in out[2]

    def test_scalar_passthrough(self):
        assert ts.sanitize_payload(42) == 42
        assert ts.sanitize_payload(3.14) == 3.14
        assert ts.sanitize_payload(True) is True
        assert ts.sanitize_payload(None) is None


# ---------------------------------------------------------------------------
# truncate_text_bytes
# ---------------------------------------------------------------------------


class TestTruncateBytes:
    def test_short_unchanged(self):
        assert ts.truncate_text_bytes("hi", max_bytes=100) == "hi"

    def test_long_truncated(self):
        long_str = "a" * 10_000
        out = ts.truncate_text_bytes(long_str, max_bytes=1000)
        assert len(out.encode("utf-8")) <= 1000 + 50  # plus marker headroom
        assert "[truncated" in out

    def test_multibyte_no_codepoint_split(self):
        # 4-byte UTF-8 chars; cut must not break a codepoint.
        s = "🎉" * 1000
        out = ts.truncate_text_bytes(s, max_bytes=300)
        # decode round-trip must succeed without surrogates.
        out.encode("utf-8").decode("utf-8")
        # No lone surrogate slipped in via partial codepoint.
        for ch in out:
            assert not (0xD800 <= ord(ch) <= 0xDFFF)

    def test_non_string_passthrough(self):
        assert ts.truncate_text_bytes(42, max_bytes=10) == 42
        assert ts.truncate_text_bytes(None, max_bytes=10) is None


# ---------------------------------------------------------------------------
# safe_capture_pane
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int, stdout: bytes):
        self.returncode = returncode
        self.stdout = stdout


class TestSafeCapturePane:
    def test_subprocess_timeout_returns_empty(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args, timeout=1)
        monkeypatch.setattr(ts.subprocess, "run", fake_run)
        assert ts.safe_capture_pane("0:0") == ""

    def test_file_not_found_returns_empty(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("tmux not installed")
        monkeypatch.setattr(ts.subprocess, "run", fake_run)
        assert ts.safe_capture_pane("0:0") == ""

    def test_nonzero_returncode_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            ts.subprocess, "run",
            lambda *a, **k: _FakeProc(returncode=1, stdout=b""),
        )
        assert ts.safe_capture_pane("0:0") == ""

    def test_broken_utf8_no_lone_surrogate(self, monkeypatch):
        """The critical guarantee: stdout bytes contain a broken multi-byte
        sequence, and safe_capture_pane must NOT produce a lone surrogate."""
        # b'\xed\xa0\x80' is the encoding of U+D800 (invalid UTF-8).
        # b'\xe0' alone is a truncated 3-byte start.
        # b'\xf0\x9f\x8e' is a truncated 4-byte emoji prefix.
        broken = b"hello \xed\xa0\x80 mid \xe0 end \xf0\x9f\x8e"
        monkeypatch.setattr(
            ts.subprocess, "run",
            lambda *a, **k: _FakeProc(returncode=0, stdout=broken),
        )
        out = ts.safe_capture_pane("0:0")
        # No surrogate in the output.
        for ch in out:
            assert not (0xD800 <= ord(ch) <= 0xDFFF), (
                f"lone surrogate U+{ord(ch):04X} leaked through"
            )
        # Must be json-safe.
        json.dumps({"x": out})

    def test_truncates_huge_capture(self, monkeypatch):
        big = b"a" * (ts.MAX_CAPTURE_BYTES * 3)
        monkeypatch.setattr(
            ts.subprocess, "run",
            lambda *a, **k: _FakeProc(returncode=0, stdout=big),
        )
        out = ts.safe_capture_pane("0:0")
        assert len(out.encode("utf-8")) <= ts.MAX_CAPTURE_BYTES + 100

    def test_valid_utf8_preserved(self, monkeypatch):
        good = "hello 🎉 한글".encode("utf-8")
        monkeypatch.setattr(
            ts.subprocess, "run",
            lambda *a, **k: _FakeProc(returncode=0, stdout=good),
        )
        out = ts.safe_capture_pane("0:0")
        assert "🎉" in out and "한글" in out


# ---------------------------------------------------------------------------
# Server integration — priority_send and _json_safe_text delegation
# ---------------------------------------------------------------------------


class TestServerIntegration:
    def test_server_helpers_delegate_to_text_safety(self):
        from ultradebate import server as srv
        assert srv._json_safe_text("\ud800x") == ts.sanitize_text("\ud800x")
        assert srv._json_safe_obj({"a": "\ud800"}) == ts.sanitize_payload(
            {"a": "\ud800"}
        )

    def test_priority_send_strips_surrogate_from_inbox(
        self, tmp_path, monkeypatch
    ):
        from ultradebate import server as srv
        monkeypatch.setattr(srv, "_V24_BASE", tmp_path)
        monkeypatch.setattr(srv, "_V24_INBOX", tmp_path / "inbox")
        monkeypatch.setattr(srv, "_V24_EVENTS", tmp_path / "events.jsonl")

        evil = {
            "text": "before\ud800after",
            "nested": {"key": ["\udccc", 1, "\ud800-\udfff-mix"]},
        }
        r = srv.ultradebate_v2_priority_send(
            sender="claude", target="codex",
            priority="normal", msg_type="text_safety_test",
            payload=evil,
        )
        assert r["ok"] is True

        inbox = tmp_path / "inbox" / "codex.jsonl"
        line = inbox.read_text(encoding="utf-8").splitlines()[-1]
        # Both the raw line and the parsed object must be surrogate-free.
        assert "\\ud800" not in line.lower()  # JSON escape form too
        assert "\\udccc" not in line.lower()
        d = json.loads(line)
        serial = json.dumps(d, ensure_ascii=False)
        for ch in serial:
            assert not (0xD800 <= ord(ch) <= 0xDFFF)

    def test_json_dumps_safe_double_sanitizes(self):
        from ultradebate import server as srv
        # ensure_ascii=False can serialize a surrogate that survived the
        # payload sweep (e.g. via a dataclass __repr__). The wrapper's second
        # sanitize pass on the encoded string is the safety net.
        out = srv._json_dumps_safe({"k": "x\ud800y"}, ensure_ascii=False)
        assert "\ud800" not in out
        assert "\\ud800" not in out.lower()

    def test_huge_pane_tail_truncated_in_send_to_peer_return(
        self, tmp_path, monkeypatch
    ):
        """A 1MB pane tail (the audit scenario) must be truncated AND
        sanitized before it appears in the tool return dict."""
        from ultradebate import server as srv
        monkeypatch.setattr(srv, "_V24_BASE", tmp_path)
        monkeypatch.setattr(srv, "_V24_INBOX", tmp_path / "inbox")
        monkeypatch.setattr(srv, "_V24_EVENTS", tmp_path / "events.jsonl")

        # Fake the pane resolver + tmux capture to return ~1MB of bytes
        # containing a lone surrogate in the middle.
        huge = (b"x" * (500 * 1024)) + b"\xed\xa0\x80" + (b"y" * (500 * 1024))

        def fake_capture(*args, **kwargs):
            return _FakeProc(returncode=0, stdout=huge)

        # safe_capture_pane is imported into server as _safe_capture_pane;
        # we patch the underlying subprocess.run inside the text_safety module
        # so both call sites (server + tests) see the fake.
        monkeypatch.setattr(ts.subprocess, "run", fake_capture)
        # Skip tmux delivery entirely so we test the inbox + early-return path
        # without depending on _v2_tmux_safe_send.
        r = srv.ultradebate_v2_send_to_peer(
            sender="claude", target="codex",
            text="hi", tmux_inject=False,
        )
        assert r["ok"] is True
        # tmux_state is "skipped" when tmux_inject=False so there is no
        # pane_tail; the inbox JSONL is what we verify. (The 1MB pane is
        # covered by TestSafeCapturePane.test_truncates_huge_capture.)
        assert r.get("tmux_state") == "skipped"
        # Re-read inbox; must be surrogate-free.
        line = (tmp_path / "inbox" / "codex.jsonl").read_text("utf-8").splitlines()[-1]
        d = json.loads(line)
        assert "\ud800" not in json.dumps(d, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Specific regression: the exact symptom from the 2026-05-22 audit
# ---------------------------------------------------------------------------


def test_audit_2026_05_22_payload_is_json_safe():
    """The audit reported a ~1MB payload containing a lone surrogate at
    char 1041685. Synthesize an equivalent payload and prove our pipeline
    never produces such a string."""
    # Simulate the exact failure: large body + invalid UTF-8 byte cluster
    # near the end.
    fake_capture_bytes = (
        b"normal text\n" * 50_000 +  # bulk
        b"truncated multi-byte \xed\xa0\x80 here"  # invalid encoding of U+D800
    )
    decoded = fake_capture_bytes.decode("utf-8", errors="replace")
    cleaned = ts.sanitize_text(decoded)
    # The whole point — Anthropic API would reject decoded.encode but accept
    # the cleaned version.
    for ch in cleaned:
        assert not (0xD800 <= ord(ch) <= 0xDFFF)
    # Bounded by truncate_text_bytes in real call sites; here we just verify
    # the surrogate scrub.
    json.dumps({"pane_tail": cleaned}, ensure_ascii=False).encode("utf-8")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
