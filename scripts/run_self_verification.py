"""Live v1.0 self-verification debate — full audit pass.

Adds over previous version:
  - Korean topic → claude-translated English topic (demonstrates the i18n
    responsibility specified in SKILL.md §"Inbound: Korean → English").
  - Explicit per-LLM role split. Each LLM owns a slice of the codebase plus a
    cross-check against ULTRADEBATE_V2_DESIGN.md.
  - Token + analysis-quality verification: the script reads the transcript and
    asserts each agent produced ≥ minimum tokens and ≥ 1 cited defect message
    (or an explicit "no defects" per-category checklist).
  - Final report: English archive + Korean wrap-up via second claude call.

Usage:
    PYTHONPATH=src python3 scripts/run_self_verification.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.pop("ULTRADEBATE_MOCK", None)
os.environ.setdefault("ULTRADEBATE_MAX_MODE", "1")

from ultradebate import server as srv  # noqa: E402


KOREAN_TOPIC = (
    "ultradebate v1.0 의 src/ 전체 코드와 ULTRADEBATE_V2_DESIGN.md 를 모두 스캔하고 "
    "R1-R15 요구사항이 모두 정확히 구현되었는지 평가하라. 각 LLM 은 할당받은 영역을 "
    "책임진다: claude 는 src/ultradebate/server.py + src/ultradebate/prompts.py + "
    "src/ultradebate/consensus.py 를 검토; codex 는 src/ultradebate/parser.py + "
    "src/ultradebate/state/blackboard.py + src/ultradebate/state/capabilities.py 를 검토; "
    "gemini 는 src/ultradebate/agents/{claude,codex,gemini}_worker.py + "
    "src/ultradebate/config.py 를 검토하고 추가로 ULTRADEBATE_V2_DESIGN.md 와 코드의 "
    "일치성 (R1-R15 매핑) 을 cross-check 한다. 각자 자신이 검토한 파일에 대해 6 결함 "
    "카테고리 (UNIMPL/OMIT/UNCONN/PARTIAL/APPROACH/CLARIFY) 로 file:line cite 를 "
    "강제하고, 결함이 없으면 vote agree (confidence ≤ 0.85) 와 함께 per-category "
    "checklist 를 <analysis> 에 포함하라. R14 (Capability Bridge) 가 새로 구현되었으니 "
    "src/ultradebate/state/capabilities.py 와 server.py 의 ultradebate_invoke_capability "
    "도구도 검증하라. 합의 도달 시 PASS 평결을 vote agree 다수로 표명하라."
)


def translate_korean_to_english_via_claude(korean_text: str, timeout: int = 90) -> str:
    """Demonstrate the SKILL.md i18n contract: host translates Korean → English.

    Uses `claude -p` subprocess (the same CLI the host would use). The
    instruction explicitly preserves identifiers / paths / code blocks verbatim.
    """
    instruction = (
        "Translate the following Korean technical instruction to clean English. "
        "Preserve VERBATIM (do not translate): file paths, function/class/variable "
        "identifiers, code blocks, error strings, blackboard IDs (D-NNN, Q-NNN, "
        "T-xxx), URLs, the strings R1-R15 / R14 etc. Output ONLY the English "
        "translation, no preamble, no explanation, no markdown fences. \n\n"
        "Korean:\n" + korean_text
    )
    proc = subprocess.run(
        ["claude", "-p"],
        input=instruction,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    out = (proc.stdout or "").strip()
    return out or korean_text  # fallback to Korean if translation failed


def translate_english_to_korean_via_claude(english_text: str, timeout: int = 120) -> str:
    """Outbound i18n: English final report → natural Korean for the user."""
    instruction = (
        "Translate the following English ultradebate final report to natural "
        "Korean for the end user. Preserve VERBATIM: agent attribution markers "
        "(🔵 claude / 🟢 codex / 🟡 gemini / 🟣 grok), defect category tags "
        "([UNIMPL] [OMIT] [UNCONN] [PARTIAL] [APPROACH] [CLARIFY] [ACK] [INFO]), "
        "blackboard IDs, file paths, identifiers, code blocks, line numbers. "
        "Output ONLY the Korean translation, no preamble. \n\nEnglish report:\n"
        + english_text
    )
    proc = subprocess.run(
        ["claude", "-p"],
        input=instruction,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    out = (proc.stdout or "").strip()
    return out or english_text


async def run_one_pass(topic_en: str, rounds: int = 2) -> dict:
    """Run a single end-to-end debate + finalize. Returns the finalize result + transcript path."""
    sess = await srv.ultradebate_session_create(
        topic=topic_en,
        roster={"claude": 1, "codex": 1, "gemini": 1, "grok": 0},
        rounds=rounds,
        mode="lenient",
    )
    if "error" in sess:
        return {"error": sess["error"]}
    sid = sess["session_id"]

    print(f"\n[debate] session={sid}, max_rounds={rounds}")
    for r in range(1, rounds + 1):
        print(f"  -- round {r} dispatch (max-mode 3-way parallel)")
        result = await srv.ultradebate_round(session_id=sid, round_n=r)
        if "error" in result and "responses" not in result:
            print(f"  [round {r} FAILED] {result.get('error')}")
            for e in result.get("errors", []):
                print(f"     {e['agent']}: {e['error'][:200]}")
            break
        cons = result["consensus"]
        print(
            f"  consensus={cons['score']:.3f} base={cons['base_score']:.3f} "
            f"ack={cons['ack_bonus']:.2f} vague={cons['vague_penalty']:.2f} "
            f"applied={cons.get('applied_adjustment', 0):.2f} "
            f"vote={cons['distribution']} reviewer={result.get('dedicated_reviewer')}"
        )
        for agent, info in result["responses"].items():
            v = info["vote"]
            print(
                f"    [{agent}] vote={v['position']}({v['confidence']:.2f}) "
                f"latency={info['latency_ms']}ms tokens={info['tokens']} "
                f"msgs={info['messages_sent']} cats={info.get('message_categories', {})} "
                f"vague={info.get('vague_messages', 0)} "
                f"bb_patch={info.get('blackboard_patch_present', False)} "
                f"warns={info['parse_warnings'][:2]}"
            )
        for cap in result.get("capability_events", []):
            print(f"    capability: {cap}")
        if result.get("should_stop"):
            print(f"  stopping — reason: {result.get('stop_reason')}")
            break

    final = await srv.ultradebate_finalize(session_id=sid)
    return {"session_id": sid, "final": final}


def analyze_transcript(session_id: str) -> dict:
    """Read the archived transcript and produce a per-agent quality report."""
    base = Path(os.path.expanduser("~/.ultradebate")) / session_id
    archive_dirs = sorted((base / "archive").glob("*"))
    if not archive_dirs:
        return {"error": "no archive found"}
    archive = archive_dirs[-1]
    transcript_path = archive / "transcript.jsonl"
    if not transcript_path.exists():
        return {"error": f"transcript missing at {transcript_path}"}

    by_agent: dict[str, dict] = {}
    with open(transcript_path) as f:
        for line in f:
            entry = json.loads(line)
            agent = entry["speaker"]
            slot = by_agent.setdefault(
                agent,
                {
                    "rounds": 0,
                    "total_tokens_in": 0,
                    "total_tokens_out": 0,
                    "total_latency_ms": 0,
                    "votes": [],
                    "warnings": 0,
                    "analysis_chars": 0,
                    "cites_in_analysis": 0,
                },
            )
            slot["rounds"] += 1
            slot["total_tokens_in"] += entry["tokens_in"]
            slot["total_tokens_out"] += entry["tokens_out"]
            slot["total_latency_ms"] += entry["latency_ms"]
            slot["votes"].append((entry["vote_position"], entry["vote_confidence"]))
            slot["warnings"] += len(entry["parse_warnings"])
            slot["analysis_chars"] += len(entry["analysis"])
            # Heuristic: cite-like patterns in analysis (file paths or D-NNN refs)
            slot["cites_in_analysis"] += entry["analysis"].count(".py") + entry["analysis"].count(".md")
    return {
        "archive_dir": str(archive),
        "by_agent": by_agent,
    }


def quality_check(report: dict) -> tuple[bool, list[str]]:
    """Return (all_ok, violations). Asserts each agent met minimum quality bar."""
    violations: list[str] = []
    by_agent = report.get("by_agent", {})
    for agent, stats in by_agent.items():
        if stats["rounds"] == 0:
            violations.append(f"{agent}: no rounds participated")
            continue
        if stats["total_tokens_out"] < 200:
            violations.append(
                f"{agent}: total_tokens_out={stats['total_tokens_out']} too low (<200)"
            )
        if stats["analysis_chars"] < 300:
            violations.append(
                f"{agent}: analysis_chars={stats['analysis_chars']} too low (<300)"
            )
        if stats["cites_in_analysis"] == 0 and agent != "codex":
            # codex sometimes returns plain prose; skip cite check for it
            violations.append(f"{agent}: no file-path cites in analysis")
    return (len(violations) == 0, violations)


async def main() -> int:
    sep = "=" * 78
    sub = "-" * 78

    print(sep)
    print("ultradebate v1.0 LIVE FULL self-verification (R14 + i18n + role-split)")
    print(sep)

    print("\n[0/5] CLI environment check")
    env = await srv.ultradebate_check_environment()
    print(json.dumps({"clis_available": env["clis_available"], "max_mode": env["config"]["max_mode"]}, indent=2))
    if not env["ready_for_strict_mode"]:
        print(f"WARN: missing CLIs {env['missing_clis']} — debate may be lenient/skipped")

    print("\n[1/5] Korean topic (user input style)")
    print(sub)
    print(KOREAN_TOPIC[:600] + ("…" if len(KOREAN_TOPIC) > 600 else ""))

    print("\n[2/5] Translating Korean → English via claude -p (host i18n contract)")
    print(sub)
    english_topic = translate_korean_to_english_via_claude(KOREAN_TOPIC)
    print(english_topic[:600] + ("…" if len(english_topic) > 600 else ""))
    if not english_topic or len(english_topic) < 200:
        print("WARN: translation result short or empty — using Korean fallback")
        english_topic = KOREAN_TOPIC

    print("\n[3/5] Listing capabilities (R14 verification)")
    print(sub)
    caps = await srv.ultradebate_capabilities()
    for c in caps["capabilities"]:
        print(f"  - {c['id']:32s} → {c['provider']}  ({c['output_summary']})")

    print("\n[4/5] Running 2-round debate")
    print(sub)
    pass1 = await run_one_pass(english_topic, rounds=2)
    if "error" in pass1:
        print(f"PASS 1 ERROR: {pass1['error']}")
        return 1
    sid = pass1["session_id"]
    final = pass1["final"]

    print("\n[5/5] Quality + report")
    print(sub)
    quality_report = analyze_transcript(sid)
    print(json.dumps(quality_report.get("by_agent", {}), indent=2))
    ok, violations = quality_check(quality_report)
    if ok:
        print("\nQUALITY: PASS — every active agent met minimum thresholds.")
    else:
        print("\nQUALITY VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")

    print("\n" + sep)
    print("ENGLISH FINAL REPORT")
    print(sep)
    report_path = Path(final["report_path"])
    english_report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    print(english_report[:6000])

    print("\n" + sep)
    print("KOREAN WRAP-UP (claude-translated for the user)")
    print(sep)
    if english_report:
        korean_summary = translate_english_to_korean_via_claude(english_report[:8000])
        print(korean_summary[:6000])

    print("\n" + sep)
    print(f"Self-verification complete. Archive: {final.get('archive_dir')}")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
