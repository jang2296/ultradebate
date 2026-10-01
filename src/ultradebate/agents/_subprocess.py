"""Subprocess helpers shared across worker implementations.

Two flavors:
  * `run_pipe()` — standard stdin/stdout/stderr pipes. Works for `claude -p` and `gemini -p`.
  * `run_pty()`  — pseudo-tty wrapper to dodge the Codex CLI TTY-detach bug
                   (openai/codex#19945) where stdio detached from a controlling TTY
                   causes `codex exec` to silently exit with empty stdout.
"""
from __future__ import annotations

import asyncio
import errno
import os
import shutil
import time

DEFAULT_PTY_TIMEOUT = 180


def find_executable(name: str) -> str | None:
    """Return absolute path to `name` on PATH, or None."""
    return shutil.which(name)


def estimate_tokens(text: str) -> int:
    """Rough cross-vendor estimate: Korean/CJK ≈ 1.5 chars/tok, English ≈ 4. Use 3 as middle."""
    if not text:
        return 0
    return max(1, len(text) // 3)


async def run_pipe(
    cmd: list[str],
    stdin_text: str,
    *,
    env: dict | None = None,
    timeout: int = 120,
) -> tuple[int, str, str, int]:
    """Run a subprocess with PIPE stdio. Returns (returncode, stdout, stderr, latency_ms)."""
    t0 = time.monotonic()
    full_env = {**os.environ, **(env or {})}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=full_env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_text.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        elapsed_ms,
    )


async def run_pty(
    cmd: list[str],
    stdin_text: str,
    *,
    env: dict | None = None,
    timeout: int = DEFAULT_PTY_TIMEOUT,
) -> tuple[int, str, int]:
    """Run a subprocess attached to a pseudo-tty.

    The child sees a real TTY on fd 0/1/2, so CLIs that gate behavior on
    `isatty()` (looking at you, codex exec) emit output normally. Output from
    stdout and stderr is interleaved (single PTY fd) — not a problem for our
    parser since the four-tag protocol is robust to interleaving.
    """
    import fcntl as _fcntl
    import pty

    t0 = time.monotonic()
    master_fd, slave_fd = pty.openpty()
    full_env = {**os.environ, **(env or {})}

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=full_env,
            start_new_session=True,
        )
    finally:
        os.close(slave_fd)

    # Non-blocking master so we can poll without hanging.
    flags = _fcntl.fcntl(master_fd, _fcntl.F_GETFL)
    _fcntl.fcntl(master_fd, _fcntl.F_SETFL, flags | os.O_NONBLOCK)

    if stdin_text:
        # Many CLIs require a trailing newline to terminate input.
        try:
            os.write(master_fd, stdin_text.encode("utf-8") + b"\n")
        except OSError:
            pass

    output = bytearray()
    deadline = t0 + timeout

    while True:
        if time.monotonic() > deadline:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            break
        try:
            chunk = os.read(master_fd, 8192)
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                # No data ready; if the child has exited, drain once and stop.
                if proc.returncode is not None:
                    break
                await asyncio.sleep(0.05)
                continue
            # EIO typically means the slave side closed — clean exit.
            break
        if not chunk:
            break
        output.extend(chunk)

    try:
        os.close(master_fd)
    except OSError:
        pass

    try:
        rc = await asyncio.wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
        rc = await proc.wait()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return rc, output.decode("utf-8", errors="replace"), elapsed_ms
