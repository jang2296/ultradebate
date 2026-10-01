"""Standalone E2E test runner.

Run this once Codex CLI is installed to verify the entire pipeline end-to-end:

    # Mock test (no API costs, no CLIs required) — ~5 sec
    python -m ultradebate.cli

    # Live test (real CLIs + API keys required) — ~$0.20-0.50
    python -m ultradebate.cli --live --rounds 2 \\
        --topic "Python의 GIL 제거 PEP 703의 장단점은?"

The runner prints:
    1. Environment check (which CLIs / keys present)
    2. Session start with available agents
    3. Each round's consensus + per-agent vote
    4. Final synthesis report
    5. Stats (tokens, latency, errors)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


async def run_e2e(topic: str, rounds: int, mode: str, live: bool) -> int:
    if not live:
        os.environ["ULTRADEBATE_MOCK"] = "1"

    # Lazy import so env vars are picked up by Config.from_env()
    from . import server as srv

    sep = "=" * 72
    sub = "-" * 72
    print(sep)
    print(f"🧪 ultradebate-mcp E2E test  (mode={mode}, rounds={rounds}, live={live})")
    print(sep)

    print("\n[1/4] Environment check")
    print(sub)
    env = await srv.ultradebate_check_environment()
    print(json.dumps(env, indent=2, ensure_ascii=False))

    if live and not env["ready_for_strict_mode"]:
        missing = env["missing_clis"]
        print(f"\n⚠️  Missing CLIs in --live mode: {missing}")
        if "codex" in missing:
            print("   → Codex CLI not installed yet. Run: bash scripts/install_codex.sh")
        if "claude" in missing:
            print("   → Claude Code CLI: https://claude.com/code (npm i -g @anthropic-ai/claude-code)")
        if "gemini" in missing:
            print("   → Gemini CLI: https://github.com/google-gemini/gemini-cli")
        if mode == "strict":
            print("\n   strict mode requires all 3. Aborting.")
            return 1
        print("   Proceeding in lenient mode anyway...")

    print(f"\n[2/4] Starting session — topic preview: {topic[:80]}...")
    print(sub)
    start = await srv.ultradebate_start(topic=topic, rounds=rounds, mode=mode)
    print(json.dumps(start, indent=2, ensure_ascii=False))
    if "error" in start:
        return 1
    session_id = start["session_id"]

    print(f"\n[3/4] Running up to {rounds} rounds")
    print(sub)
    last_round = 0
    for r in range(1, rounds + 1):
        print(f"\n  ┌── Round {r} dispatch ──┐")
        result = await srv.ultradebate_round(session_id=session_id, round_n=r)
        if "error" in result and "responses" not in result:
            print(f"  ❌ Round {r} failed: {result.get('error')}")
            for e in result.get("errors", []):
                print(f"     - {e['agent']}: {e['error']}")
            break
        last_round = r
        cons = result["consensus"]
        print(f"  consensus={cons['score']:.3f}  vote_agreement={cons['vote_agreement']:.2f}  "
              f"text_agreement={cons['text_agreement']:.2f}  converged={cons['converged']}")
        print(f"  vote distribution: {cons['distribution']}")
        for agent, info in result["responses"].items():
            v = info["vote"]
            print(f"    [{agent}]  vote={v['position']:>8} ({v['confidence']:.2f})  "
                  f"latency={info['latency_ms']:>5}ms  tokens={info['tokens']}  "
                  f"msgs={info['messages_sent']}  tasks={info['task_actions_count']}")
        for evt in result.get("task_events", []):
            print(f"    📋 task: {evt}")
        if result.get("errors"):
            for e in result["errors"]:
                print(f"    ⚠️  {e['agent']}: {e['error']}")
        if result.get("should_stop"):
            print(f"\n  ⏹  Stopping early — reason: {result.get('stop_reason')}")
            break

    print(f"\n[4/4] Synthesizing final report (after {last_round} rounds)")
    print(sub)
    if last_round == 0:
        print("  ⚠️  No completed rounds — skipping synthesis")
        return 1
    syn = await srv.ultradebate_synthesize(session_id=session_id)

    print("\n" + sep)
    print("📋 FINAL SYNTHESIS")
    print(sep)
    print(syn.get("synthesis", "[no synthesis]"))

    print("\n" + sub)
    print("📊 Stats")
    print(sub)
    print(f"  rounds_completed:  {syn['rounds_completed']}")
    print(f"  final_consensus:   {syn.get('final_consensus')}")
    print(f"  consensus_history: {syn.get('consensus_history')}")
    print(f"  tokens_total_est:  {syn['tokens_total']}")
    print(f"  tasks_total:       {len(syn['tasks_summary'])}")
    if syn.get("synthesis_error"):
        print(f"  synthesis_error:   {syn['synthesis_error']}")

    print("\n✅ E2E test complete.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ultradebate-test",
        description="End-to-end smoke test for ultradebate-mcp.",
    )
    parser.add_argument(
        "--topic",
        default="Python의 GIL을 제거하는 PEP 703에 찬성하는가? 장단점을 토론하라.",
        help="Debate topic.",
    )
    parser.add_argument("--rounds", type=int, default=2, help="Max rounds (default 2).")
    parser.add_argument(
        "--mode",
        choices=["strict", "lenient"],
        default="lenient",
        help="strict requires all 3 agents; lenient proceeds with 2/3.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real CLIs (requires claude, codex, gemini on PATH + API keys).",
    )
    args = parser.parse_args()
    rc = asyncio.run(run_e2e(args.topic, args.rounds, args.mode, args.live))
    sys.exit(rc)


if __name__ == "__main__":
    main()
