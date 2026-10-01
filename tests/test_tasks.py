"""Task list atomicity & lifecycle."""
from __future__ import annotations

import asyncio

import pytest

from ultradebate.state.tasks import (
    add_task,
    block_task,
    claim_task,
    complete_task,
    list_tasks,
)


def test_add_and_list(session_dir):
    tid = add_task(session_dir, "design the parser", created_by="claude")
    assert tid.startswith("T-")
    tasks = list_tasks(session_dir)
    assert len(tasks) == 1
    assert tasks[0].title == "design the parser"
    assert tasks[0].status == "pending"
    assert tasks[0].created_by == "claude"


def test_claim_then_complete(session_dir):
    tid = add_task(session_dir, "do X", "claude")
    assert claim_task(session_dir, "codex", tid)
    # Re-claim by another agent should fail.
    assert not claim_task(session_dir, "gemini", tid)
    # Only owner can complete.
    assert not complete_task(session_dir, "claude", tid)
    assert complete_task(session_dir, "codex", tid)
    final = list_tasks(session_dir)
    assert final[0].status == "done"
    assert final[0].owner == "codex"


def test_block_pending(session_dir):
    tid = add_task(session_dir, "X", "claude")
    assert block_task(session_dir, "any", tid)
    assert list_tasks(session_dir)[0].status == "blocked"


def test_concurrent_claim_only_one_wins(session_dir):
    """50 concurrent claim attempts on the same task — exactly one must succeed."""
    tid = add_task(session_dir, "race", "claude")

    async def attempt(agent_idx: int) -> bool:
        # Run synchronous claim_task in a thread to truly parallelize.
        return await asyncio.to_thread(claim_task, session_dir, f"agent{agent_idx % 3}", tid)

    async def race():
        results = await asyncio.gather(*(attempt(i) for i in range(50)))
        return results

    results = asyncio.run(race())
    assert sum(1 for r in results if r) == 1
    assert list_tasks(session_dir)[0].status == "in_progress"


def test_empty_title_raises(session_dir):
    with pytest.raises(ValueError):
        add_task(session_dir, "  ", "claude")


def test_list_when_no_file(session_dir):
    assert list_tasks(session_dir) == []
