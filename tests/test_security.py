"""Security guard tests — path traversal, OAuth-token rejection, key redaction."""
from __future__ import annotations

import pytest

from ultradebate.security import (
    SecurityError,
    assert_no_oauth_token,
    redact,
    safe_session_dir,
    validate_session_id,
)


def test_validate_ok():
    validate_session_id("abc12345")
    validate_session_id("a" * 16)
    validate_session_id("abc-def-12345678")


def test_validate_rejects_path_traversal():
    with pytest.raises(SecurityError):
        validate_session_id("../etc/passwd")
    with pytest.raises(SecurityError):
        validate_session_id("ABCDEFGH")  # uppercase not allowed
    with pytest.raises(SecurityError):
        validate_session_id("a")  # too short
    with pytest.raises(SecurityError):
        validate_session_id("x" * 100)  # too long


def test_validate_rejects_non_str():
    with pytest.raises(SecurityError):
        validate_session_id(123)  # type: ignore[arg-type]


def test_safe_session_dir_traversal_blocked(tmp_path):
    with pytest.raises(SecurityError):
        safe_session_dir(tmp_path, "../escape12")


def test_safe_session_dir_creates_under_base(tmp_path):
    target = safe_session_dir(tmp_path, "abcdef12")
    target.mkdir()
    assert tmp_path.resolve() in target.resolve().parents


def test_redact_openai_key():
    s = "key=sk-abcdefghijklmnopqrstuvwxyz1234 done"
    out = redact(s)
    assert "sk-abc" not in out
    assert "[REDACTED]" in out


def test_redact_anthropic_key():
    s = "ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuv"
    out = redact(s)
    assert "[REDACTED]" in out
    assert "ant-api03" not in out


def test_redact_google_key():
    s = "key=AIzaSyA_abcdefghijklmnopqrstuvwxyz1234567"
    out = redact(s)
    assert "AIza" not in out


def test_oauth_token_in_api_key_env_rejected():
    with pytest.raises(SecurityError):
        assert_no_oauth_token({"GEMINI_API_KEY": "ya29.something_long"})


def test_normal_api_keys_pass():
    # No exception expected
    assert_no_oauth_token(
        {
            "ANTHROPIC_API_KEY": "sk-ant-something",
            "OPENAI_API_KEY": "sk-something",
            "GEMINI_API_KEY": "AIzaSomething",
        }
    )
