#!/usr/bin/env python3
"""Standalone CLI wrapper for ultradebate v1.1 auto-debate orchestration.

Usage:
    PYTHONPATH=src python3 scripts/auto_debate_cli.py auto "topic here" --rounds 3
    PYTHONPATH=src python3 scripts/auto_debate_cli.py cross-verify SESSION_ID --round 2
    PYTHONPATH=src python3 scripts/auto_debate_cli.py parallel TASKS_JSON_FILE

Bypasses the long-running MCP server (which has cached old tool list) by directly
invoking the tool functions from the updated server.py module. Useful for:
- Testing v1.1 tools before MCP restart
- Scripted/cron-driven debates from outside Claude Code
- Integration tests for the auto-debate flow
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _setup_path() -> None:
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))


def _print_result(result: dict, *, json_only: bool = False) -> None:
    text = json.dumps(result, indent=2, default=str, ensure_ascii=False)
    if json_only:
        print(text)
    else:
        print("=" * 80)
        print(text)
        print("=" * 80)


async def cmd_auto(args: argparse.Namespace) -> dict:
    from ultradebate.server import ultradebate_auto_debate
    return await ultradebate_auto_debate(
        topic=args.topic,
        max_rounds=args.rounds,
        roster=json.loads(args.roster) if args.roster else None,
        mode=args.mode,
        finalize=not args.no_finalize,
        early_stop_on_converge=not args.no_early_stop,
    )


async def cmd_cross_verify(args: argparse.Namespace) -> dict:
    from ultradebate.server import ultradebate_cross_verify
    pairs = json.loads(args.pairs) if args.pairs else None
    return await ultradebate_cross_verify(
        session_id=args.session_id,
        round_n=args.round,
        verifier_pairs=pairs,
    )


async def cmd_parallel(args: argparse.Namespace) -> dict:
    from ultradebate.server import ultradebate_parallel_tasks
    payload = json.loads(Path(args.tasks_file).read_text())
    if not isinstance(payload, list):
        raise SystemExit("tasks file must contain a JSON list of {agent, prompt} objects")
    return await ultradebate_parallel_tasks(
        tasks=payload,
        timeout_per_task_sec=args.timeout,
    )


def main() -> int:
    _setup_path()

    p = argparse.ArgumentParser(description="ultradebate v1.1 auto-debate CLI")
    p.add_argument("--json-only", action="store_true",
                   help="Print only JSON output (no banner lines)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("auto", help="Run auto_debate (session→rounds→synthesize→finalize)")
    pa.add_argument("topic", help="Debate topic (English)")
    pa.add_argument("--rounds", type=int, default=3)
    pa.add_argument("--mode", default="lenient", choices=["lenient", "strict"])
    pa.add_argument("--roster", default=None,
                    help='JSON, e.g. {"claude":1,"codex":1,"gemini":0}')
    pa.add_argument("--no-finalize", action="store_true",
                    help="Skip ultradebate_finalize archive step")
    pa.add_argument("--no-early-stop", action="store_true",
                    help="Run all max_rounds even after convergence")
    pa.set_defaults(func=cmd_auto)

    pc = sub.add_parser("cross-verify", help="Cross-verify a round across roster")
    pc.add_argument("session_id")
    pc.add_argument("--round", type=int, required=True)
    pc.add_argument("--pairs", default=None,
                    help='JSON, e.g. {"claude":"codex","codex":"claude"}')
    pc.set_defaults(func=cmd_cross_verify)

    pp = sub.add_parser("parallel", help="Dispatch parallel tasks")
    pp.add_argument("tasks_file",
                    help="Path to JSON file with list of {agent,prompt,task_id} dicts")
    pp.add_argument("--timeout", type=int, default=240)
    pp.set_defaults(func=cmd_parallel)

    args = p.parse_args()
    result = asyncio.run(args.func(args))
    _print_result(result, json_only=args.json_only)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
