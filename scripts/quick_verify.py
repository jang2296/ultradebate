"""Compact 1-round live verification — fits in <10min bash window.

Same fact pattern as run_self_verification.py but:
  - rounds=1 (still produces a vote per agent)
  - Korean→English translation step preserved
  - English-only debate
  - English→Korean wrap-up preserved
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
    "ultradebate v1.0 코드가 R1-R15 요구사항 모두 정확히 구현되었는지 1라운드 평가하라. "
    "특히 R14 (Capability Bridge — src/ultradebate/state/capabilities.py + "
    "src/ultradebate/server.py 의 ultradebate_invoke_capability 도구) 와 "
    "R15-A (MaxModePolicy 의 claude_model/codex_model/gemini_model 필드 + 워커 --model 주입) 와 "
    "R15-B (estimate_session_cost + cost_estimate 응답 필드 + ULTRADEBATE_ABORT_COST_USD env) 가 "
    "올바르게 구현되었는지 확인하라. "
    "claude 는 server.py + consensus.py, codex 는 parser.py + state/capabilities.py, "
    "gemini 는 config.py + agents/* 워커 3개 검토. "
    "결함 없으면 vote agree (confidence ≤ 0.85), 결함 있으면 6 카테고리 + cite 강제. "
    "한 라운드에서 PASS/FAIL 평결을 명확히 표명하라."
)


def claude_translate(text: str, direction: str, timeout: int = 90) -> str:
    if direction == "ko2en":
        instr = (
            "Translate the following Korean instruction to clean technical English. "
            "Preserve VERBATIM: file paths, identifiers, code blocks, R1-R15. "
            "Output ONLY the English translation.\n\nKorean:\n" + text
        )
    else:
        instr = (
            "Translate the following English report to natural Korean. Preserve "
            "VERBATIM: agent markers (🔵🟢🟡🟣), defect tags ([UNIMPL] etc.), "
            "blackboard IDs, file paths, identifiers. Output ONLY the Korean.\n\n"
            "English:\n" + text
        )
    proc = subprocess.run(
        ["claude", "-p"], input=instr, text=True, capture_output=True,
        timeout=timeout, check=False,
    )
    return (proc.stdout or "").strip() or text


async def main() -> int:
    sep = "=" * 78
    print(sep)
    print("ultradebate v1.0 QUICK 1-round verification (R14 + R15-A + R15-B fixes)")
    print(sep)

    print("\n[1] Korean topic")
    print(KOREAN_TOPIC[:500])

    print("\n[2] Translating Korean → English via claude -p")
    english_topic = claude_translate(KOREAN_TOPIC, "ko2en")
    print(english_topic[:500])

    print("\n[3] Listing capabilities (R14 verification)")
    caps_res = await srv.ultradebate_capabilities()
    for c in caps_res["capabilities"]:
        print(f"  - {c['id']:32s} → {c['provider']:6s} ({c['output_summary']})")

    print("\n[4] Creating session (cost-gate test)")
    sess = await srv.ultradebate_session_create(
        topic=english_topic,
        roster={"claude": 1, "codex": 1, "gemini": 1, "grok": 0},
        rounds=1,
        mode="lenient",
    )
    if "error" in sess:
        print(f"ERROR: {sess['error']}")
        return 1
    print(f"  session_id: {sess['session_id']}")
    print(f"  cost_estimate: {sess['cost_estimate']}")
    sid = sess["session_id"]

    print("\n[5] Round 1 (max-mode 3-way parallel)")
    result = await srv.ultradebate_round(session_id=sid, round_n=1)
    if "error" in result and "responses" not in result:
        print(f"  ERROR: {result.get('error')}")
        for e in result.get("errors", []):
            print(f"    {e['agent']}: {e['error'][:200]}")
        return 2
    cons = result["consensus"]
    print(f"  consensus={cons['score']:.3f} base={cons['base_score']:.3f}")
    print(f"  votes: {cons['distribution']}")
    print(f"  reviewer={result.get('dedicated_reviewer')} target={result.get('reviewer_target')}")
    for agent, info in result["responses"].items():
        v = info["vote"]
        print(
            f"  [{agent}] vote={v['position']}({v['confidence']:.2f}) "
            f"latency={info['latency_ms']}ms tokens={info['tokens']} "
            f"msgs={info['messages_sent']} cats={info.get('message_categories', {})} "
            f"warns={info['parse_warnings'][:2]}"
        )

    print("\n[6] Finalize (synthesize + archive)")
    final = await srv.ultradebate_finalize(session_id=sid)
    if "error" in final:
        print(f"  finalize ERROR: {final['error']}")
        return 3
    print(f"  archive_dir: {final['archive_dir']}")
    print(f"  rounds_completed: {final['rounds_completed']}")
    print(f"  final_consensus: {final['final_consensus']}")

    print("\n" + sep)
    print("ENGLISH FINAL REPORT")
    print(sep)
    report_path = Path(final["report_path"])
    eng = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    print(eng[:5000])

    print("\n" + sep)
    print("KOREAN WRAP-UP")
    print(sep)
    if eng:
        print(claude_translate(eng[:6000], "en2ko")[:5000])

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
