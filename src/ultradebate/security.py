"""Security guards: session_id validation, path traversal prevention, key redaction."""
from __future__ import annotations

import os
import re
from pathlib import Path

# 16 hex chars by default (secrets.token_hex(8)); allow 8-32 alphanumeric+hyphen
_SESSION_RE = re.compile(r"^[a-z0-9-]{8,32}$")

# Heuristic: redact obvious vendor key shapes. Defense in depth — keys should never
# be passed to redact() in the first place; this catches accidental leaks.
_KEY_PATTERNS = [
    re.compile(r"sk-(?:ant-)?[A-Za-z0-9_\-]{20,}"),  # OpenAI / Anthropic
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),          # Google API key
    re.compile(r"ya29\.[A-Za-z0-9_\-]{20,}"),        # Google OAuth (forbidden, but redact if seen)
]


class SecurityError(Exception):
    """Raised on invalid input that could cause path traversal or injection."""


def validate_session_id(session_id: str) -> str:
    """Return the session_id if valid; raise SecurityError otherwise.

    Valid format: 8-32 chars of [a-z0-9-]. This blocks any path-traversal-capable
    input like '../etc/passwd' or absolute paths.
    """
    if not isinstance(session_id, str):
        raise SecurityError(f"session_id must be str, got {type(session_id).__name__}")
    if not _SESSION_RE.fullmatch(session_id):
        raise SecurityError(
            f"Invalid session_id format: {session_id!r} (expected 8-32 chars of [a-z0-9-])"
        )
    return session_id


def safe_session_dir(base: Path, session_id: str) -> Path:
    """Resolve {base}/{session_id} ensuring no traversal escapes the base."""
    sid = validate_session_id(session_id)
    base_resolved = Path(base).resolve()
    target = (base_resolved / sid).resolve()
    # Defense in depth: even though session_id is regex-restricted, double-check.
    try:
        target.relative_to(base_resolved)
    except ValueError as e:
        raise SecurityError(f"Path traversal attempt for session_id={sid!r}") from e
    return target


def redact(text: str) -> str:
    """Replace any visible API-key-shaped substring with '[REDACTED]'."""
    if not isinstance(text, str):
        return text
    out = text
    for pat in _KEY_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def assert_no_oauth_token(env: dict[str, str]) -> None:
    """Refuse to start if any OAuth-shaped token is in env where API keys belong.

    OAuth tokens (ya29.*, sbp_*) signal subscription-session abuse — that's the exact
    pattern Anthropic/OpenAI/Google ban for. Force the user to use proper API keys.
    """
    suspicious = []
    for k, v in env.items():
        if not v:
            continue
        if k.endswith("_API_KEY") and (v.startswith("ya29.") or v.startswith("sbp_")):
            suspicious.append(k)
    if suspicious:
        raise SecurityError(
            f"OAuth-shaped tokens detected in {suspicious} — "
            "use vendor-issued API keys, NOT subscription session tokens. "
            "See README §Security."
        )


def collect_api_keys() -> dict[str, str | None]:
    """Read all relevant API key env vars (does not log values)."""
    keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    snapshot = {k: os.environ.get(k) for k in keys}
    assert_no_oauth_token({k: v or "" for k, v in snapshot.items()})
    return snapshot
