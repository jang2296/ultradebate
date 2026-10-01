"""Full pipeline test using MockWorker — no external CLIs / API keys required."""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def mock_server(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTRADEBATE_MOCK", "1")
    monkeypatch.setenv("ULTRADEBATE_BASE", str(tmp_path / ".ultradebate"))
    # Reload modules so Config.from_env() picks up the test env.
    from ultradebate import config as _cfg
    importlib.reload(_cfg)
    from ultradebate import server as _srv
    importlib.reload(_srv)
    return _srv


@pytest.mark.asyncio
async def test_check_environment_mock(mock_server):
    env = await mock_server.ultradebate_check_environment()
    assert "clis_available" in env
    assert env["config"]["mock"] is True


@pytest.mark.asyncio
async def test_full_round_flow_mock(mock_server):
    start = await mock_server.ultradebate_start(
        topic="Test debate topic", rounds=2, mode="lenient"
    )
    assert "session_id" in start, start
    sid = start["session_id"]

    # Round 1
    r1 = await mock_server.ultradebate_round(session_id=sid, round_n=1)
    assert "responses" in r1, r1
    assert len(r1["responses"]) == 3
    assert "consensus" in r1
    assert r1["consensus"]["score"] >= 0.0

    # Round 2 — should also work; mailbox messages from r1 are delivered
    r2 = await mock_server.ultradebate_round(session_id=sid, round_n=2)
    assert "responses" in r2
    assert r2["round"] == 2

    # Status check
    status = await mock_server.ultradebate_status(session_id=sid)
    assert status["transcript_entries"] == 6  # 3 agents * 2 rounds

    # Synthesize
    syn = await mock_server.ultradebate_synthesize(session_id=sid)
    assert "synthesis" in syn
    assert syn["rounds_completed"] == 2


@pytest.mark.asyncio
async def test_round_out_of_range_rejected(mock_server):
    start = await mock_server.ultradebate_start(topic="x", rounds=2, mode="lenient")
    sid = start["session_id"]
    r = await mock_server.ultradebate_round(session_id=sid, round_n=5)
    assert "error" in r


@pytest.mark.asyncio
async def test_send_then_status_shows_routing(mock_server):
    start = await mock_server.ultradebate_start(topic="x", rounds=1, mode="lenient")
    sid = start["session_id"]
    res = await mock_server.ultradebate_send(
        session_id=sid, from_agent="host", to_agent="claude", content="hello claude"
    )
    assert res["ok"]
    status = await mock_server.ultradebate_status(session_id=sid)
    assert status["inbox_sizes"]["claude"] == 1


@pytest.mark.asyncio
async def test_invalid_session_id_rejected(mock_server):
    r = await mock_server.ultradebate_round(session_id="../escape", round_n=1)
    assert "error" in r


@pytest.mark.asyncio
async def test_cleanup_force_removes_dir(mock_server):
    start = await mock_server.ultradebate_start(topic="x", rounds=1, mode="lenient")
    sid = start["session_id"]
    res = await mock_server.ultradebate_cleanup(session_id=sid, force=True)
    assert res["ok"]
    # status after cleanup should error since meta.json gone
    status = await mock_server.ultradebate_status(session_id=sid)
    assert "error" in status
