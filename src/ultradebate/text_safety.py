"""Text safety: surrogate sanitization, safe tmux capture, payload size cap.

Single source of truth for any text that crosses an MCP/IPC/API boundary.
After 2026-05-22 audit, ``tmux capture-pane`` was identified as the primary
source of lone-surrogate corruption — multi-byte UTF-8 characters truncated
at the pane wrap or capture boundary, then decoded with surrogateescape,
yielding U+D800-U+DBFF without a paired low surrogate. The Anthropic API
(and other JSON consumers) reject those with HTTP 400.

This module guarantees:
  * Every captured tmux text is sanitized BEFORE it becomes a return value.
  * Every payload going to inbox JSONL or tool return dict is recursively
    sanitized.
  * No captured text exceeds ``MAX_CAPTURE_BYTES`` (256 KB by default).
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Any

# Surrogate range — see RFC 3629 / UTF-16. These code points are valid only
# as part of a pair; alone they're invalid UTF-8 (an encoder must reject).
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")
_REPLACEMENT = "�"

# Caps. ``tmux capture-pane`` can return MB-scale output; we limit to keep
# JSON payloads under sane API limits AND avoid the surrogate issue entirely
# (more content = more chance of a boundary cut).
MAX_CAPTURE_BYTES: int = int(
    os.environ.get("ULTRADEBATE_MAX_CAPTURE_BYTES", str(256 * 1024))
)
MAX_PAYLOAD_BYTES: int = int(
    os.environ.get("ULTRADEBATE_MAX_PAYLOAD_BYTES", str(512 * 1024))
)
DEFAULT_CAPTURE_LINES: int = int(
    os.environ.get("ULTRADEBATE_CAPTURE_LINES", "30")
)


def sanitize_text(value: Any) -> Any:
    """Replace lone UTF-16 surrogates with U+FFFD. Idempotent.

    Non-string inputs are returned unchanged so callers can pass mixed payloads
    without pre-checking. The encode/decode round-trip is a defensive belt-and-
    braces step that catches *any* other UTF-8 oddity (truncated 4-byte cuts,
    bare continuation bytes mishandled by an upstream decoder).
    """
    if not isinstance(value, str):
        return value
    if not value:
        return value
    if _SURROGATE_RE.search(value) is None:
        try:
            return value.encode("utf-8", "replace").decode("utf-8", "replace")
        except Exception:
            return value
    cleaned = _SURROGATE_RE.sub(_REPLACEMENT, value)
    try:
        return cleaned.encode("utf-8", "replace").decode("utf-8", "replace")
    except Exception:
        # 병적 케이스 — ASCII fallback.
        return cleaned.encode("ascii", "replace").decode("ascii", "replace")


def sanitize_payload(value: Any) -> Any:
    """Recursively sanitize all str values in a dict/list/tuple/set."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {
            sanitize_text(str(k)) if isinstance(k, str) else k: sanitize_payload(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_payload(v) for v in value)
    if isinstance(value, set):
        return {sanitize_payload(v) for v in value}
    return value


def truncate_text_bytes(
    value: Any,
    *,
    max_bytes: int = MAX_PAYLOAD_BYTES,
    marker: str = "...[truncated]",
) -> Any:
    """Truncate a string at a UTF-8 byte boundary. Preserves codepoint integrity.

    Non-string inputs pass through unchanged. The cut-point is computed in
    bytes, then ``decode('utf-8', errors='replace')`` cleans any partial
    codepoint at the cut (turns the orphan bytes into U+FFFD instead of
    leaving them to surface as a lone surrogate later).
    """
    if not isinstance(value, str):
        return value
    b = value.encode("utf-8", "replace")
    if len(b) <= max_bytes:
        return value
    marker_bytes = marker.encode("utf-8")
    cut_to = max(0, max_bytes - len(marker_bytes))
    safe = b[:cut_to].decode("utf-8", "replace")
    return safe + marker


def safe_capture_pane(
    pane: str,
    *,
    lines: int = DEFAULT_CAPTURE_LINES,
    socket: str | None = None,
    timeout: float = 3.0,
    max_bytes: int = MAX_CAPTURE_BYTES,
    join_wrapped: bool = True,
) -> str:
    """Capture tmux pane with bounded size + guaranteed-clean UTF-8.

    Behavior:
      * ``text=False`` (raw bytes from subprocess) so we control the decode
        policy.
      * Explicit ``decode('utf-8', errors='replace')`` — invalid bytes →
        U+FFFD, never surrogateescape.
      * ``tmux -J`` joins wrapped lines, reducing the chance of a multi-byte
        cut at the wrap column.
      * ``-S -lines`` limits to the last N lines (default 30, env-tunable).
      * Final truncation at ``max_bytes`` — safety net against runaway
        captures.
      * Sanitize as a last layer (paranoid).

    Returns ``""`` on any subprocess failure (timeout, missing tmux, bad
    pane). Never raises.
    """
    sock_args = ["-L", socket] if socket else []
    extra: list[str] = ["-J"] if join_wrapped else []
    try:
        proc = subprocess.run(
            [
                "tmux",
                *sock_args,
                "capture-pane",
                "-p",
                *extra,
                "-t",
                pane,
                "-S",
                f"-{int(lines)}",
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    raw = proc.stdout if proc.stdout else b""
    # CRITICAL: decode bytes ourselves with errors='replace' (never strict or
    # surrogateescape — both can produce/leak lone surrogates).
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        # Defensive: subprocess returned str (older monkeypatched envs).
        text = raw
    text = sanitize_text(text)
    text = truncate_text_bytes(
        text,
        max_bytes=max_bytes,
        marker=f"...[truncated_at_{max_bytes}B]",
    )
    return text


def safe_subprocess_decode(
    stdout_bytes: bytes | str,
    *,
    max_bytes: int | None = None,
) -> str:
    """Decode + sanitize subprocess output that was already collected.

    For call sites that already ran ``subprocess.run`` and want a safe string
    representation without re-running. Accepts both ``bytes`` (preferred) and
    pre-decoded ``str`` (sanitized as-is).
    """
    if isinstance(stdout_bytes, str):
        out = stdout_bytes
    elif stdout_bytes:
        out = stdout_bytes.decode("utf-8", errors="replace")
    else:
        out = ""
    out = sanitize_text(out)
    if max_bytes is not None:
        out = truncate_text_bytes(out, max_bytes=max_bytes)
    return out
