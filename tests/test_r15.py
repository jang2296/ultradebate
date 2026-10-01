"""R15 tests — model-pinning in MaxModePolicy + cost estimate / abort gate."""
from __future__ import annotations

import os

import pytest

os.environ["ULTRADEBATE_MOCK"] = "1"

from ultradebate.config import Config, MaxModePolicy  # noqa: E402
from ultradebate.agents.claude_worker import ClaudeWorker  # noqa: E402
from ultradebate.agents.codex_worker import CodexWorker  # noqa: E402
from ultradebate.agents.gemini_worker import GeminiWorker  # noqa: E402
from ultradebate import server as srv  # noqa: E402


def test_max_mode_policy_has_model_fields():
    """All four model fields exist on MaxModePolicy (R15-A spec compliance).

    Defaults are empty strings — actual injection requires user opt-in via
    ULTRADEBATE_<AGENT>_MODEL env, since CLI versions / account tiers vary.
    """
    p = MaxModePolicy()
    assert hasattr(p, "claude_model")
    assert hasattr(p, "codex_model")
    assert hasattr(p, "gemini_model")
    assert hasattr(p, "grok_model")
    # Empty default = no auto-injection (verified by skip tests below)
    assert p.claude_model == ""
    assert p.codex_model == ""
    assert p.gemini_model == ""


def test_workers_skip_model_when_default_empty():
    """With empty defaults, --model is NOT injected (CLI uses its own default)."""
    assert "--model" not in ClaudeWorker()._final_args()
    assert "--model" not in CodexWorker()._final_args()
    assert "--model" not in GeminiWorker()._final_args()


def test_claude_worker_injects_model_when_set():
    policy = MaxModePolicy(claude_model="claude-opus-4-7")
    args = ClaudeWorker(max_policy=policy)._final_args()
    assert "--model" in args
    assert args[args.index("--model") + 1] == "claude-opus-4-7"


def test_codex_worker_injects_model_when_set():
    policy = MaxModePolicy(codex_model="gpt-5-codex")
    args = CodexWorker(max_policy=policy)._final_args()
    assert "--model" in args
    assert args[args.index("--model") + 1] == "gpt-5-codex"
    assert any("model_reasoning_effort" in a for a in args)


def test_gemini_worker_injects_model_when_set():
    policy = MaxModePolicy(gemini_model="gemini-2.5-pro")
    args = GeminiWorker(max_policy=policy)._final_args()
    assert "--model" in args
    assert args[args.index("--model") + 1] == "gemini-2.5-pro"


def test_workers_skip_model_injection_when_max_mode_off():
    policy = MaxModePolicy(
        enabled=False,
        claude_model="claude-opus-4-7",
        codex_model="gpt-5-codex",
        gemini_model="gemini-2.5-pro",
    )
    assert "--model" not in ClaudeWorker(max_policy=policy)._final_args()
    assert "--model" not in CodexWorker(max_policy=policy)._final_args()
    assert "--model" not in GeminiWorker(max_policy=policy)._final_args()


def test_estimate_session_cost_basic():
    roster = {"claude": 1, "codex": 1, "gemini": 1, "grok": 0}
    est = srv.estimate_session_cost(roster, rounds=2, max_mode=True)
    assert est["active_agents"] == 3
    assert est["rounds"] == 2
    assert est["per_agent_per_round_tokens"] == 20_000
    # 3 agents × 2 rounds × 20K × 1.2 = 144K tokens
    assert est["estimated_tokens"] == 144_000
    assert est["estimated_cost_usd"] > 0
    assert est["abort_threshold_usd"] >= est["warn_threshold_usd"]


def test_estimate_session_cost_default_mode():
    roster = {"claude": 1, "codex": 1, "gemini": 1, "grok": 0}
    est = srv.estimate_session_cost(roster, rounds=5, max_mode=False)
    assert est["per_agent_per_round_tokens"] == 5_000


@pytest.mark.asyncio
async def test_session_create_returns_cost_estimate():
    sess = await srv.ultradebate_session_create(
        topic="cost test",
        roster={"claude": 1, "codex": 1, "gemini": 1, "grok": 0},
        rounds=2,
        mode="lenient",
    )
    assert "cost_estimate" in sess
    ce = sess["cost_estimate"]
    assert ce["active_agents"] == 3
    assert "estimated_cost_usd" in ce
    assert "abort_threshold_usd" in ce


@pytest.mark.asyncio
async def test_session_create_aborts_on_excessive_cost():
    """Force abort by setting roster + rounds large enough to exceed threshold."""
    # Default abort=$20.00, warn=$5.00. With 3 agents × 20 rounds × 20K × 1.2 ≈ 1.44M tokens
    # × ~$30/M ≈ $43 → should abort.
    sess = await srv.ultradebate_session_create(
        topic="cost abort test",
        roster={"claude": 1, "codex": 1, "gemini": 1, "grok": 0},
        rounds=20,
        mode="lenient",
    )
    assert sess.get("error") == "cost_abort"
    assert "cost_estimate" in sess
    assert sess["cost_estimate"]["estimated_cost_usd"] > sess["cost_estimate"]["abort_threshold_usd"]


def test_config_loads_abort_cost_from_env(monkeypatch):
    monkeypatch.setenv("ULTRADEBATE_ABORT_COST_USD", "50.00")
    monkeypatch.setenv("ULTRADEBATE_WARN_COST_USD", "10.00")
    cfg = Config.from_env()
    assert cfg.abort_cost_usd == 50.00
    assert cfg.warn_cost_usd == 10.00
