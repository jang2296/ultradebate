"""Live quota probe — read actual 5h/weekly limit percentages from peer CLI.

Why this exists:
  The allocator MLP (src/ultradebate/allocator.py + scripts/extract_features.py)
  uses features like 'claude_5h_used', 'codex_5h_used', etc. Until now those
  were derived from *message-count proxy* (estimate_usage in extract_features).
  Proxy is OK for training a *pattern*, but for *live recommendations* the MLP
  should consume the actual quota percentages reported by the peer CLI itself.

What this module does:
  * ``probe_codex_quota(pane, ...)``: send ``/status`` to a codex tmux pane,
    parse the output, return percentages. (Cooperative — pane must be idle or
    willing to queue.)
  * ``parse_status_output(text)``: pure-function regex parser for the
    ``5h limit: [▓▓░░] N% left (resets HH:MM)`` and ``Weekly limit: ... N% left``
    lines. Tolerant to whitespace, unicode bar variations, BMP/non-BMP chars.
  * ``probe_claude_quota()``: returns a structured 'not_directly_probable'
    result that explains *why* (we run inside the Claude Code agent loop and
    cannot introspect the outer host's /status). Includes recipe for manual
    snapshot.
  * ``snapshot_to_features(snapshots)``: convert
    ``{claude:{...}, codex:{...}}`` snapshots into MLP feature dict slots
    (claude_5h_used = 1 - pct_left/100, etc).

This module never blocks the agent loop; subprocess timeouts ≤5s.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Persistent snapshot location (JSONL append-only). Override via env when needed.
QUOTA_LOG = Path(os.environ.get("ULTRADEBATE_QUOTA_LOG", "/tmp/ultradebate_v2/quota_log.jsonl"))

# Regex patterns. Tolerant to unicode bar chars and varying whitespace.
# Examples of input we accept:
#   "5h limit:                    [█████░] 85% left (resets 00:43)"
#   "Weekly limit:                [██░] 79% left (resets 12:34 on 27 May)"
#   "  5h limit: [#####] 50% left"
_FIVE_H_RE = re.compile(
    r"5h\s+limit\s*:\s*\[[^\]]*\]\s*(\d{1,3})\s*%\s*left"
    r"(?:\s*\(resets\s+([^)]+)\))?",
    re.IGNORECASE,
)
_WEEKLY_RE = re.compile(
    r"weekly\s+limit\s*:\s*\[[^\]]*\]\s*(\d{1,3})\s*%\s*left"
    r"(?:\s*\(resets\s+([^)]+)\))?",
    re.IGNORECASE,
)


def parse_status_output(text: str) -> dict[str, Any]:
    """Pure function: extract 5h/weekly %left from a ``/status`` capture.

    Handles multiple occurrences (codex outputs two pairs — main model + Spark).
    We take the *first* occurrence of each (primary model). Returns empty
    dict when no match found or input is invalid.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    out: dict[str, Any] = {}
    m5 = _FIVE_H_RE.search(text)
    if m5:
        # %left clamped to [0,100] just in case.
        out["5h_pct_left"] = max(0, min(100, int(m5.group(1))))
        if m5.group(2):
            out["5h_resets"] = m5.group(2).strip()
    mw = _WEEKLY_RE.search(text)
    if mw:
        out["weekly_pct_left"] = max(0, min(100, int(mw.group(1))))
        if mw.group(2):
            out["weekly_resets"] = mw.group(2).strip()
    return out


def _fallback_capture_pane(
    pane: str,
    *,
    lines: int,
    socket: str | None = None,
    timeout: float = 5.0,
) -> str:
    """Lightweight pane capture used when text_safety.safe_capture_pane unavailable."""
    sock_args = ["-L", socket] if socket else []
    try:
        proc = subprocess.run(
            ["tmux", *sock_args, "capture-pane", "-p", "-J", "-t", pane, "-S", f"-{int(lines)}"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode("utf-8", errors="replace")


def probe_codex_quota(
    pane: str,
    *,
    socket: str | None = None,
    send_status_cmd: bool = True,
    wait_after_send: float = 4.0,
    capture_lines: int = 80,
    timeout: float = 5.0,
    submit_key: str = "C-m",
) -> dict[str, Any]:
    """Probe live codex quota by reading ``/status`` output from its tmux pane.

    Two modes:
      send_status_cmd=True (default): paste ``/status`` + submit (``submit_key``,
        default C-m per the 2026-06-03 mandate) and capture after
        ``wait_after_send`` seconds.
        ⚠ This DOES briefly interrupt codex if it's busy — caller should check
        idle first OR accept the small disturbance.
      send_status_cmd=False: just capture-pane and parse whatever ``/status``
        output is already visible. Non-intrusive but only works when
        ``/status`` was recently invoked.

    Returns dict with shape::

      {
        ok: bool,
        agent: 'codex',
        ts: <iso ts>,
        '5h_pct_left'?: int,
        'weekly_pct_left'?: int,
        '5h_resets'?: str,
        'weekly_resets'?: str,
        'raw_capture_tail'?: str,
        'error'?: str,
      }
    """
    # Lazy import so this module remains usable in isolation.
    try:
        from .text_safety import safe_capture_pane as _capture
    except ImportError:
        _capture = _fallback_capture_pane

    sock_args = ["-L", socket] if socket else []

    if send_status_cmd:
        try:
            # paste-buffer + submit. Use a tempfile for the buffer payload to
            # avoid shell quoting issues with the slash character.
            buf = f"udb_probe_{int(time.time() * 1000) % 1_000_000}"
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write("/status")
                tmp = f.name
            try:
                subprocess.run(
                    ["tmux", *sock_args, "load-buffer", "-b", buf, tmp],
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
                subprocess.run(
                    ["tmux", *sock_args, "paste-buffer", "-b", buf, "-p", "-d", "-t", pane],
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
                # Settle before submit so codex registers the slash command.
                time.sleep(1.0)
                # Submit. User mandate 2026-06-03: C-m ONLY (the 2026-05 Escape+Enter
                # path was reverted — see server.py CODEX_SUBMIT_KEYS notes). submit_key
                # may be space-separated for legacy multi-key fallback.
                subprocess.run(
                    ["tmux", *sock_args, "send-keys", "-t", pane, *submit_key.split()],
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            return {
                "ok": False,
                "agent": "codex",
                "ts": _iso_now(),
                "error": f"send_status_failed: {e!r}",
            }
        time.sleep(max(0.0, float(wait_after_send)))

    captured = _capture(pane, lines=capture_lines, socket=socket)
    if not captured:
        return {
            "ok": False,
            "agent": "codex",
            "ts": _iso_now(),
            "error": "capture_pane_empty",
            "note": "tmux capture-pane returned empty — pane wrong or tmux down",
        }

    parsed = parse_status_output(captured)
    has_any = ("5h_pct_left" in parsed) or ("weekly_pct_left" in parsed)
    out: dict[str, Any] = {
        "ok": bool(has_any),
        "agent": "codex",
        "ts": _iso_now(),
        **parsed,
    }
    if not has_any:
        out["error"] = "no_status_lines_found_in_capture"
        # Bounded tail for debugging — sanitizer at server boundary will scrub it.
        out["raw_capture_tail"] = captured[-1500:]
    return out


def probe_claude_quota(**_kwargs: Any) -> dict[str, Any]:
    """Claude quota cannot be probed from inside the Claude Code agent loop.

    We run *inside* the host Claude session — the outer Claude's quota is
    not exposed to our tools (no ``/status`` capture path, no API endpoint
    accessible from this MCP). This function returns a structured
    'not_directly_probable' result with the recipe for manual snapshot.
    """
    return {
        "ok": False,
        "agent": "claude",
        "ts": _iso_now(),
        "error": "not_directly_probable",
        "note": (
            "Claude Code's /status is rendered in the host UI, not in a tmux "
            "pane this MCP can capture. Manual workflow: user runs /status, "
            "pastes the output to ultradebate_v2_quota_record_manual"
            "(claude, '<paste here>') which calls parse_status_output and "
            "records it."
        ),
        "manual_recipe": {
            "step1": "User runs `/status` in the Claude session.",
            "step2": "User copies the '5h limit' + 'Weekly limit' lines.",
            "step3": "Call ultradebate_v2_quota_record_manual(agent='claude', text=...).",
        },
    }


def probe_claude_credentials_meta(
    credentials_path: str | None = None,
) -> dict[str, Any]:
    """Read claude plan metadata from ``~/.claude/.credentials.json``.

    What this returns (auto, no user interaction):
      * subscription_type — e.g. "max", "pro", "free"
      * rate_limit_tier   — e.g. "default_claude_max_20x"
      * oauth_expires_at_iso — when the OAuth token expires
      * 5h/weekly_pct_left  — NOT available here (server-tracked, not in this file)

    What this CANNOT do: produce current usage % — that data lives on
    Anthropic's side and is rendered only in the host UI's /status. Use
    ``ultradebate_v2_quota_record_manual`` for the % values.

    Security: only metadata is returned. Access/refresh tokens are *never*
    surfaced. The file is read-only opened.
    """
    from datetime import datetime, timezone

    path = credentials_path or os.path.expanduser("~/.claude/.credentials.json")
    if not os.path.exists(path):
        return {
            "ok": False,
            "agent": "claude",
            "ts": _iso_now(),
            "error": "credentials_not_found",
            "path_checked": path,
        }
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "agent": "claude",
            "ts": _iso_now(),
            "error": "credentials_read_failed",
            "detail": repr(exc)[:200],
        }

    oauth = (data.get("claudeAiOauth") or data.get("oauth") or {})
    sub = oauth.get("subscriptionType")
    tier = oauth.get("rateLimitTier")
    expires_at_ms = oauth.get("expiresAt")
    expires_iso: str | None = None
    if isinstance(expires_at_ms, (int, float)):
        try:
            expires_iso = datetime.fromtimestamp(
                expires_at_ms / 1000.0, tz=timezone.utc,
            ).isoformat()
        except (OSError, OverflowError, ValueError):
            expires_iso = None

    return {
        "ok": bool(sub or tier),
        "agent": "claude",
        "ts": _iso_now(),
        "subscription_type": sub,
        "rate_limit_tier": tier,
        "oauth_expires_at_iso": expires_iso,
        "note": (
            "Plan metadata only. Current 5h/weekly usage % is server-tracked "
            "and not exposed in the local credentials file — use "
            "ultradebate_v2_quota_record_manual after running /status in the host UI."
        ),
    }


def record_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Append a snapshot to the persistent quota log (JSONL)."""
    QUOTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(snapshot)
    snapshot.setdefault("ts", _iso_now())
    with QUOTA_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    return {"ok": True, "logged": True, "ts": snapshot["ts"]}


def read_recent_snapshots(
    *,
    agent: str | None = None,
    max_n: int = 50,
) -> list[dict[str, Any]]:
    """Read last N quota snapshots (newest last). Filters by agent if given."""
    if not QUOTA_LOG.exists():
        return []
    try:
        raw_lines = QUOTA_LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    # If filtering, scan from the end so we surface the most recent matches.
    out: list[dict[str, Any]] = []
    for ln in raw_lines:
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if agent is not None and d.get("agent") != agent:
            continue
        out.append(d)
    if max_n is not None and len(out) > max_n:
        out = out[-int(max_n):]
    return out


def snapshot_to_features(snapshots: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Convert ``{'claude': {...}, 'codex': {...}}`` snapshots into MLP slots.

    Mapping (all clamped to [0,1]):
      ``*_5h_used``    = 1 - 5h_pct_left/100      (if available, else 0.0)
      ``*_week_used``  = 1 - weekly_pct_left/100  (if available, else 0.0)

    Caller fills the remaining FEATURE_ORDER slots (cyclic time, queue_len, etc.).
    """
    out: dict[str, float] = {
        "claude_5h_used": 0.0,
        "codex_5h_used": 0.0,
        "claude_week_used": 0.0,
        "codex_week_used": 0.0,
    }
    if not isinstance(snapshots, dict):
        return out
    cl = snapshots.get("claude") or {}
    cx = snapshots.get("codex") or {}
    if cl.get("ok") and "5h_pct_left" in cl:
        out["claude_5h_used"] = _clamp01(1.0 - float(cl["5h_pct_left"]) / 100.0)
    if cl.get("ok") and "weekly_pct_left" in cl:
        out["claude_week_used"] = _clamp01(1.0 - float(cl["weekly_pct_left"]) / 100.0)
    if cx.get("ok") and "5h_pct_left" in cx:
        out["codex_5h_used"] = _clamp01(1.0 - float(cx["5h_pct_left"]) / 100.0)
    if cx.get("ok") and "weekly_pct_left" in cx:
        out["codex_week_used"] = _clamp01(1.0 - float(cx["weekly_pct_left"]) / 100.0)
    return out


def _clamp01(v: float) -> float:
    if v != v:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, float(v)))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
