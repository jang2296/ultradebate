"""Regression tests for ``scripts/metrics_ack_gap.sh`` (round-2 audit, 2026-05-22).

Pins:
  * ACK and silent-gap counts match a controlled JSONL inbox fixture.
  * ``--since`` window defaults to NOW-3600s; ``--all-time`` overrides to epoch.
  * ``--json`` output emits the round-2 fields (window_start, window_end,
    msg_count_in_window, interpretation).

These tests shell out to ``bash`` and substitute a temporary inbox root via
``/tmp/ultradebate_v2_test_<pid>`` so they never touch the live inbox at
``/tmp/ultradebate_v2``. The script targets a hard-coded path, so we work
around that by copying the script into a tmp dir and sed-replacing the
V2 root with the test root.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "metrics_ack_gap.sh"


def _make_test_script(test_root: Path, dest: Path) -> Path:
    """Patch the hard-coded V2 root inside metrics_ack_gap.sh to a tmp path.

    The original script does ``V2 = "/tmp/ultradebate_v2"`` inside its
    embedded python heredoc. We substitute that literal so the test never
    pollutes (or is polluted by) the live inbox tree.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    patched = src.replace(
        'V2 = "/tmp/ultradebate_v2"',
        f'V2 = "{test_root}"',
    )
    dest.write_text(patched, encoding="utf-8")
    dest.chmod(0o755)
    return dest


def _write_jsonl_msg(
    inbox_dir: Path, peer: str, msg: dict, ts_dt: datetime | None = None
):
    """Append a single JSONL line to {inbox_dir}/{peer}.jsonl."""
    if ts_dt is None:
        ts_dt = datetime.now(timezone.utc)
    msg.setdefault("ts", ts_dt.isoformat())
    inbox_dir.mkdir(parents=True, exist_ok=True)
    path = inbox_dir / f"{peer}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg) + "\n")


def _run_script(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True, text=True, timeout=30, check=False,
    )


@pytest.fixture
def patched_script(tmp_path):
    """Provide a fresh test-rooted copy of metrics_ack_gap.sh per test."""
    test_root = tmp_path / "udb_root"
    (test_root / "inbox").mkdir(parents=True, exist_ok=True)
    script = _make_test_script(test_root, tmp_path / "metrics_ack_gap.sh")
    return script, test_root


def test_default_window_is_one_hour(patched_script):
    """--since absent and --all-time absent → window starts at NOW-3600s."""
    script, _root = patched_script
    proc = _run_script(script, "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    start = datetime.fromisoformat(data["window_start"])
    end = datetime.fromisoformat(data["window_end"])
    delta = (end - start).total_seconds()
    # Allow generous slack: script start-time vs subprocess clock can drift
    # a few seconds in CI.
    assert 3500 <= delta <= 3700, (
        f"default window must be ~3600s, got {delta:.1f}s "
        f"(start={start}, end={end})"
    )


def test_all_time_uses_epoch_window(patched_script):
    """--all-time → window_start is the literal 'epoch' (no since filter)."""
    script, _root = patched_script
    proc = _run_script(script, "--all-time", "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["window_start"] == "epoch"
    assert data["since"] == "epoch"


def test_zero_messages_emits_zero_gap_zero_ack(patched_script):
    """Empty inbox → ACK_OK=0, SILENT_GAPS=0, MSGS_IN_WINDOW=0."""
    script, _root = patched_script
    proc = _run_script(script, "--all-time", "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["ack_ok"] == 0
    assert data["silent_gaps"] == 0
    assert data["msg_count_in_window"] == 0
    # Empty-window interpretation must be explicit, not silently "conformant".
    assert "empty window" in data["interpretation"].lower() or \
           "does not prove" in data["interpretation"].lower()


def test_ack_pair_counts_correctly(patched_script):
    """One outbound message + one job_started ACK → ACK_OK=1, SILENT_GAPS=0."""
    script, root = patched_script
    inbox = root / "inbox"
    now = datetime.now(timezone.utc)
    # Outbound message from claude → codex (lives on codex's inbox).
    _write_jsonl_msg(
        inbox, "codex",
        {"from": "claude", "to": "codex", "type": "message",
         "msg_id": "m1", "payload": {"text": "hello"}},
        ts_dt=now,
    )
    # Codex acks via job_started referencing m1.
    _write_jsonl_msg(
        inbox, "claude",
        {"from": "codex", "to": "claude", "type": "job_started",
         "msg_id": "a1", "payload": {"in_reply_to": "m1"}},
        ts_dt=now + timedelta(seconds=5),
    )
    proc = _run_script(script, "--all-time", "--json", "--source", "jsonl")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["ack_ok"] == 1, f"expected ACK_OK=1, data={data}"
    assert data["silent_gaps"] == 0
    assert data["msg_count_in_window"] == 2


def test_silent_gap_detected_when_no_ack(patched_script):
    """Outbound message older than threshold + no reply → SILENT_GAPS>=1."""
    script, root = patched_script
    inbox = root / "inbox"
    # Message from 1 hour ago, no ack anywhere.
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    _write_jsonl_msg(
        inbox, "codex",
        {"from": "claude", "to": "codex", "type": "message",
         "msg_id": "lonely", "payload": {"text": "lonely msg"}},
        ts_dt=old,
    )
    # Use --all-time to ensure the old message is in scope.
    proc = _run_script(
        script, "--all-time", "--json", "--source", "jsonl", "--threshold", "30"
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["silent_gaps"] >= 1, (
        f"expected silent_gaps>=1 for orphan message, got {data}"
    )
    # Gap detail must reference the orphan target.
    targets = {g["target"] for g in data["gap_details"]}
    assert "codex" in targets
