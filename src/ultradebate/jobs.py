"""Background-job runner for ultradebate MCP — agent-loop non-blocking by design.

Critical invariant: any tool call that would block the agent loop for more than
a few seconds MUST be wrapped here so the loop returns immediately and the work
runs detached. Polling is done via short ``check_job`` calls.

This module is the structural fix for the "Codex foreground-runs replay smoke →
inbox queue stalls → looks like messages were dropped" pattern (design doc §2.1,
§2.5, §4.1). Do not bypass this by calling subprocess.run() inline from a tool.

Design notes:
    * The module is intentionally free of MCP / FastMCP imports. The server-side
      tool wrapper is the only place that knows about ``priority_send``; this
      keeps the job runner reusable and unit-testable in isolation.
    * Jobs are spawned with ``start_new_session=True`` so the OS detaches the
      child from the agent process group — if the MCP server exits, jobs keep
      running and can be reaped on next boot via the meta files on disk.
    * Logs are file-backed only; we never accumulate stdout/stderr in memory.
    * ``exit_code`` is best-effort. Once the PID is gone we cannot reliably
      ``waitpid`` (the child is in a new session and not our direct descendant
      in some launch paths), so we record ``finished_no_exit_code`` and persist
      a terminal-state snapshot to avoid re-checking forever.
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JOBS_DIR = Path("/tmp/ultradebate_v2/jobs")


# ---------------------------------------------------------------------------
# Internal helpers (private — not part of the public surface)
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _now_epoch() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _meta_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _log_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.log"


def _load_meta(job_id: str) -> dict[str, Any] | None:
    p = _meta_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_meta(meta: dict[str, Any]) -> None:
    """Atomic write: tmp file + rename (avoids torn reads from concurrent
    ``check_job`` polls)."""
    job_id = meta["job_id"]
    final = _meta_path(job_id)
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(final)


def _pid_alive(pid: int) -> bool:
    """True iff /proc/{pid} exists and is not a zombie.

    We treat zombies as 'not running' because the job is effectively done and
    only waiting to be reaped; the agent caller should see ``running=False``.
    """
    if pid <= 0:
        return False
    proc_dir = Path(f"/proc/{pid}")
    if not proc_dir.exists():
        return False
    try:
        status = (proc_dir / "status").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    for line in status.splitlines():
        if line.startswith("State:"):
            # State: Z (zombie) → not running for our purposes
            return "Z" not in line
    return True


def _tail_log(path: Path, n: int) -> str:
    """Read last *n* lines from *path*. Bounded by a short timeout so a wedged
    fs doesn't stall the agent loop."""
    if n <= 0 or not path.exists():
        return ""
    try:
        proc = subprocess.run(
            ["tail", "-n", str(int(n)), str(path)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        return proc.stdout or ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Fallback: pure-python tail. Read up to 64 KiB from the end.
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                read = min(size, 64 * 1024)
                f.seek(size - read)
                data = f.read(read).decode("utf-8", errors="replace")
            lines = data.splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""


def _finalize_if_done(meta: dict[str, Any]) -> dict[str, Any]:
    """If pid is no longer alive and we haven't cached a terminal snapshot yet,
    persist ``finished_at`` + best-effort ``exit_code`` so future ``check_job``
    calls don't repeat the /proc scan."""
    if meta.get("finished_at"):
        return meta
    pid = int(meta.get("pid", 0))
    if pid <= 0 or _pid_alive(pid):
        return meta
    # Process is gone. Try to reap (will raise ChildProcessError if not our
    # direct child or already reaped — that's fine; we record best-effort).
    exit_code: int | None = None
    try:
        _, status = os.waitpid(pid, os.WNOHANG)
        if os.WIFEXITED(status):
            exit_code = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            exit_code = -os.WTERMSIG(status)
    except (ChildProcessError, OSError):
        exit_code = None
    meta["finished_at"] = _now_iso()
    meta["exit_code"] = exit_code
    meta["finished_no_exit_code"] = exit_code is None
    _save_meta(meta)
    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit_job(
    cmd: str | list[str],
    *,
    est_minutes: float | None = None,
    peer: str | None = None,
    kind: str = "long_job",
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Spawn a detached background job and return its metadata immediately.

    Args:
        cmd: Either a shell command string (routed through ``bash -lc`` so the
            user's PATH/aliases work) or an argv list (executed directly with
            ``shell=False`` — preferred for untrusted inputs).
        est_minutes: Optional estimate, recorded as metadata only.
        peer: Optional peer name (``claude`` / ``codex`` / ``gemini``). Recorded
            in metadata so the server tool layer can send an ACK via
            ``priority_send``. **This module never sends ACKs itself** — that
            keeps it free of MCP infrastructure imports.
        kind: Free-form tag (e.g. ``"replay_smoke"``, ``"feature_extract"``).
        extra_meta: Arbitrary additional metadata to persist alongside the job.

    Returns:
        ``{job_id, pid, cmd, log, started_at, started_at_epoch, est_minutes,
          peer, kind, ...}``
    """
    _ensure_dirs()

    job_id = uuid.uuid4().hex[:12]
    log_path = _log_path(job_id)

    if isinstance(cmd, str):
        argv: list[str] = ["bash", "-lc", cmd]
        cmd_str = cmd
    else:
        argv = [str(x) for x in cmd]
        cmd_str = " ".join(shlex.quote(x) for x in argv)

    # Open log file write-only; the child inherits this fd via stdout/stderr
    # redirection. We intentionally do NOT use buffering=0 — line-buffered is
    # plenty for tail polling, and unbuffered slows the child for noisy jobs.
    log_fh = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 — closed below
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from agent process group
            close_fds=True,
        )
    finally:
        # Parent doesn't need the fd; child has dup'd it via fork/exec.
        try:
            log_fh.close()
        except OSError:
            pass

    started_epoch = _now_epoch()
    meta: dict[str, Any] = {
        "job_id": job_id,
        "pid": proc.pid,
        "cmd": cmd_str,
        "argv": argv,
        "log": str(log_path),
        "started_at": _now_iso(),
        "started_at_epoch": started_epoch,
        "est_minutes": est_minutes,
        "peer": peer,
        "kind": kind,
        "exit_code": None,
        "finished_at": None,
        "finished_no_exit_code": False,
    }
    if extra_meta:
        # Don't allow callers to overwrite the canonical job fields.
        for k, v in extra_meta.items():
            meta.setdefault(k, v)
    _save_meta(meta)
    return meta


def check_job(job_id: str, *, log_tail_lines: int = 20) -> dict[str, Any]:
    """Inspect a job's state. Non-blocking, returns immediately.

    Returns a dict with::

        {
          "ok": bool,
          "job_id": str,
          "running": bool,            # True iff PID is alive and not zombie
          "pid": int,
          "elapsed_s": float,
          "exit_code": int | None,    # None while running
          "finished_no_exit_code": bool,
          "log_tail": str,            # last N lines of stdout+stderr
          "log": str,
          "cmd": str,
          "kind": str,
          "peer": str | None,
          "started_at": str,
          "finished_at": str | None,
        }

    If the job is unknown, returns ``{"ok": False, "error": "...", ...}``.
    """
    meta = _load_meta(job_id)
    if not meta:
        return {"ok": False, "error": f"unknown job_id: {job_id}", "job_id": job_id}

    meta = _finalize_if_done(meta)
    pid = int(meta.get("pid", 0))
    running = _pid_alive(pid)
    elapsed = max(0.0, _now_epoch() - float(meta.get("started_at_epoch") or 0))
    log_tail = _tail_log(Path(meta["log"]), int(log_tail_lines))

    return {
        "ok": True,
        "job_id": job_id,
        "running": running,
        "pid": pid,
        "elapsed_s": round(elapsed, 3),
        "exit_code": meta.get("exit_code"),
        "finished_no_exit_code": bool(meta.get("finished_no_exit_code", False)),
        "log_tail": log_tail,
        "log": meta["log"],
        "cmd": meta.get("cmd", ""),
        "kind": meta.get("kind", "long_job"),
        "peer": meta.get("peer"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "est_minutes": meta.get("est_minutes"),
    }


def list_jobs(*, only_running: bool = False, max_items: int = 50) -> dict[str, Any]:
    """List recent jobs, newest first.

    Sort key: ``started_at_epoch`` descending. Bounded by *max_items* so a
    long-lived agent doesn't accidentally pull tens of thousands of entries
    into a single tool response.
    """
    _ensure_dirs()
    metas: list[dict[str, Any]] = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta = _finalize_if_done(meta)
        pid = int(meta.get("pid", 0))
        running = _pid_alive(pid)
        if only_running and not running:
            continue
        metas.append({
            "job_id": meta.get("job_id"),
            "running": running,
            "pid": pid,
            "kind": meta.get("kind", "long_job"),
            "peer": meta.get("peer"),
            "started_at": meta.get("started_at"),
            "started_at_epoch": meta.get("started_at_epoch") or 0,
            "finished_at": meta.get("finished_at"),
            "exit_code": meta.get("exit_code"),
            "cmd": meta.get("cmd", ""),
            "log": meta.get("log"),
        })
    metas.sort(key=lambda m: -float(m.get("started_at_epoch") or 0))
    return {"ok": True, "jobs": metas[: int(max_items)], "total": len(metas)}


def cancel_job(job_id: str, *, sig: str = "TERM") -> dict[str, Any]:
    """Send a signal to the job's process group.

    Args:
        job_id: Identifier returned by :func:`submit_job`.
        sig: ``"TERM"`` (default, graceful) or ``"KILL"`` (forced). Other names
            from :mod:`signal` are accepted as well (e.g. ``"INT"``, ``"HUP"``).

    We signal the *process group* (negative pid via :func:`os.killpg`) because
    jobs were started with ``start_new_session=True`` and may have forked their
    own children (e.g. ``bash -lc`` → python → workers). Signalling only the
    leader would orphan grandchildren.
    """
    meta = _load_meta(job_id)
    if not meta:
        return {"ok": False, "error": f"unknown job_id: {job_id}"}
    pid = int(meta.get("pid", 0))
    if pid <= 0:
        return {"ok": False, "error": "invalid pid in job meta", "job_id": job_id}
    if not _pid_alive(pid):
        return {
            "ok": True, "job_id": job_id, "pid": pid,
            "signaled": False, "reason": "already exited",
        }
    signo = getattr(signal, f"SIG{sig.upper()}", None)
    if signo is None:
        return {"ok": False, "error": f"unknown signal: {sig}"}
    try:
        # Negative pid → process group. PG id == leader pid since we used
        # start_new_session=True at spawn.
        os.killpg(pid, signo)
    except ProcessLookupError:
        return {
            "ok": True, "job_id": job_id, "pid": pid,
            "signaled": False, "reason": "process gone between check and kill",
        }
    except PermissionError as exc:
        return {"ok": False, "error": f"permission denied: {exc}", "job_id": job_id}
    except OSError as exc:
        return {"ok": False, "error": f"signal failed: {exc}", "job_id": job_id}
    return {
        "ok": True, "job_id": job_id, "pid": pid,
        "signaled": True, "signal": sig.upper(),
    }
