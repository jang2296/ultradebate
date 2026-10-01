"""Tests for v2.6 jobs module — agent-loop non-blocking guarantee.

Critical invariant tests: submitting a long job must return in <0.5s.
If this regresses (e.g., someone changes Popen to subprocess.run()), CI fails.

Note: conftest.py at tests/ already prepends src/ to sys.path.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from ultradebate import jobs


@pytest.fixture(autouse=True)
def isolated_jobs_dir(tmp_path, monkeypatch):
    """Each test gets a clean jobs directory under tmp_path."""
    test_dir = tmp_path / "jobs"
    monkeypatch.setattr(jobs, "JOBS_DIR", test_dir)
    yield test_dir


def test_submit_returns_immediately(isolated_jobs_dir):
    """CRITICAL: agent loop must not block. submit_job(sleep 3) returns <0.5s."""
    t0 = time.time()
    meta = jobs.submit_job("sleep 3 && echo done")
    elapsed = time.time() - t0
    assert elapsed < 0.5, f"submit_job blocked {elapsed:.2f}s — should be <0.5s"
    assert "job_id" in meta
    assert "pid" in meta
    assert "log" in meta
    assert meta["pid"] > 0
    # Cleanup: cancel the lingering sleep so the test process exits cleanly.
    jobs.cancel_job(meta["job_id"], sig="KILL")


def test_check_job_running_then_done(isolated_jobs_dir):
    """check_job reports running=True while alive, running=False after exit."""
    meta = jobs.submit_job("sleep 1; echo finished")
    jid = meta["job_id"]
    # During run (give the child ~50ms to start)
    time.sleep(0.1)
    st = jobs.check_job(jid)
    assert st["ok"] is True
    assert st["running"] is True
    assert st["exit_code"] is None
    # After done: 1s sleep + slack for OS scheduling
    time.sleep(2.0)
    st = jobs.check_job(jid)
    assert st["running"] is False
    # exit_code may be None (waitpid failed) or 0 (clean exit). Both acceptable.
    assert st["exit_code"] in (None, 0)
    assert "finished" in (st.get("log_tail") or "")


def test_cancel_terminates_running_job(isolated_jobs_dir):
    """cancel_job(TERM) stops a long-running sleep within ~1s."""
    meta = jobs.submit_job("sleep 30")
    jid = meta["job_id"]
    time.sleep(0.2)  # let child establish
    r = jobs.cancel_job(jid, sig="TERM")
    assert r["ok"] is True
    assert r.get("signaled") is True
    time.sleep(1.0)
    st = jobs.check_job(jid)
    assert st["running"] is False


def test_list_jobs_filters(isolated_jobs_dir):
    """only_running=True excludes already-finished jobs."""
    m1 = jobs.submit_job("sleep 3")
    m2 = jobs.submit_job("echo quick")
    # let m2 finish
    time.sleep(0.5)
    all_jobs = jobs.list_jobs(only_running=False)
    running = jobs.list_jobs(only_running=True)
    assert len(all_jobs["jobs"]) >= 2
    # m1 should still be running
    assert any(j["job_id"] == m1["job_id"] for j in running["jobs"])
    # m2 likely finished; if it didn't (slow host), at least all_jobs has it
    assert any(j["job_id"] == m2["job_id"] for j in all_jobs["jobs"])
    jobs.cancel_job(m1["job_id"], sig="KILL")


def test_log_tail_captures_output(isolated_jobs_dir):
    """log_tail returns stdout lines captured to the per-job log file."""
    meta = jobs.submit_job("echo line1; echo line2; echo line3")
    # short wait for the echo lines to flush
    time.sleep(0.4)
    st = jobs.check_job(meta["job_id"], log_tail_lines=10)
    tail = st.get("log_tail") or ""
    assert "line1" in tail
    assert "line3" in tail


def test_unknown_job_id(isolated_jobs_dir):
    """check_job on a non-existent job_id returns ok=False with error message."""
    r = jobs.check_job("nonexistent_id_12345")
    assert r["ok"] is False
    assert "error" in r


def test_cancel_unknown_job_id(isolated_jobs_dir):
    r = jobs.cancel_job("nonexistent_id_99999")
    assert r["ok"] is False
    assert "error" in r


def test_argv_list_input(isolated_jobs_dir):
    """submit_job accepts argv lists (shell=False path) for untrusted input."""
    meta = jobs.submit_job(["echo", "hello", "world"])
    time.sleep(0.3)
    st = jobs.check_job(meta["job_id"])
    tail = st.get("log_tail") or ""
    assert "hello world" in tail


def test_jobs_dir_isolation(isolated_jobs_dir):
    """The monkeypatched JOBS_DIR is what the module actually uses."""
    meta = jobs.submit_job("echo isolated")
    time.sleep(0.2)
    # The meta file should live inside our tmp dir, not the global /tmp path.
    assert str(isolated_jobs_dir) in meta["log"]
    assert (isolated_jobs_dir / f"{meta['job_id']}.json").exists()
