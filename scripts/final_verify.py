"""Bounded verification — focuses on the 3 iter5 fixes + R12 cross-review demo.

The previous open-ended topic ("evaluate R1-R15") let LLMs find unbounded
new gaps each round (correct behavior, but never converges). This script
asks a CLOSED yes/no question about specific lines + uses ultradebate_review
to demonstrate split-work / R12 explicitly.
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
    "iter5 에서 LLM 들이 발견한 3 개 결함이 정확히 fix 되었는지 단일 라운드로 확인하라. "
    "FIX-1: src/ultradebate/prompts.py 의 CAPABILITY_PROMPT 에 {rationale} 슬롯이 추가되었고, "
    "src/ultradebate/server.py 의 _invoke_capability_internal 가 args 에서 __rationale__ 를 "
    "꺼내 prompt 에 전달하는가? "
    "FIX-2: src/ultradebate/state/capabilities.py 의 validate_args 가 unknown 키를 거부하는가? "
    "(allowed = input_keys ∪ {__rationale__}) "
    "FIX-3: src/ultradebate/parser.py 가 Vote.confidence 의 NaN/Inf 를 0.5 로 sanitize 하는가? "
    "(math.isnan / math.isinf 명시 분기) "
    "추가로 src/ultradebate/parser.py 에 template_echo 감지 로직이 추가되어 'REPLACE_WITH' 또는 "
    "'agree|disagree|abstain' echo 시 vote 를 abstain 으로 자동 downgrade 하는지도 확인하라. "
    "각 fix 에 대해 PASS/FAIL 을 명시하고, 모두 PASS 면 최종 vote agree (confidence ≤ 0.85), "
    "하나라도 FAIL 이면 vote disagree + 6 카테고리 + cite_path:line 강제. "
    "역할 분담: claude 는 prompts.py + server.py, codex 는 capabilities.py, "
    "gemini 는 parser.py 를 검토. 각자 자신 영역의 fix 만 책임지면 충분."
)


def claude_translate(text: str, direction: str, timeout: int = 90) -> str:
    if direction == "ko2en":
        instr = (
            "Translate the Korean instruction below to clean technical English. "
            "Preserve VERBATIM: file paths, identifiers, code blocks, FIX-N labels. "
            "Output ONLY the English translation.\n\nKorean:\n" + text
        )
    else:
        instr = (
            "Translate the English report to natural Korean. Preserve VERBATIM: "
            "🔵🟢🟡 markers, [UNIMPL] etc tags, IDs, file paths, identifiers. "
            "Output ONLY the Korean.\n\nEnglish:\n" + text
        )
    proc = subprocess.run(
        ["claude", "-p"], input=instr, text=True, capture_output=True,
        timeout=timeout, check=False,
    )
    return (proc.stdout or "").strip() or text


async def main() -> int:
    sep = "=" * 78
    print(sep)
    print("ultradebate v1.0 FINAL bounded verification (iter6 — closed question + R12 demo)")
    print(sep)

    print("\n[1] Korean bounded topic")
    print(KOREAN_TOPIC[:400] + "…")

    print("\n[2] Translating to English via claude -p (host i18n)")
    en_topic = claude_translate(KOREAN_TOPIC, "ko2en")
    print(en_topic[:400] + "…")
    if not en_topic or len(en_topic) < 100:
        en_topic = KOREAN_TOPIC

    print("\n[3] Creating session")
    sess = await srv.ultradebate_session_create(
        topic=en_topic,
        roster={"claude": 1, "codex": 1, "gemini": 1, "grok": 0},
        rounds=1,
        mode="lenient",
    )
    if "error" in sess:
        print(f"ERROR: {sess['error']}")
        return 1
    sid = sess["session_id"]
    print(f"  session_id={sid}")
    print(f"  cost_estimate=${sess['cost_estimate']['estimated_cost_usd']}")

    print("\n[4] Round 1 dispatch (3 LLMs, max-mode)")
    result = await srv.ultradebate_round(session_id=sid, round_n=1)
    if "error" in result and "responses" not in result:
        print(f"  ROUND ERROR: {result.get('error')}")
        return 2
    cons = result["consensus"]
    print(f"  consensus={cons['score']:.3f}  votes={cons['distribution']}")
    for agent, info in result["responses"].items():
        v = info["vote"]
        echo = "[ECHO!]" if "template-echo" in (info.get("parse_warnings", []) or [""])[0] else ""
        print(
            f"  [{agent}] vote={v['position']}({v['confidence']:.2f}) "
            f"latency={info['latency_ms']}ms tokens={info['tokens']} "
            f"msgs={info['messages_sent']} cats={info.get('message_categories', {})} "
            f"vague={info.get('vague_messages', 0)} {echo}"
        )

    print("\n[5] R12 cross-review demonstration — claude reviews codex's parser.py")
    review_result = await srv.ultradebate_review(
        session_id=sid,
        reviewer="claude",
        reviewee="codex",
        artifact_path="src/ultradebate/parser.py",
    )
    print(json.dumps(review_result, indent=2, ensure_ascii=False))

    print("\n[6] Listing reviews (R12)")
    listing = await srv.ultradebate_list_reviews(session_id=sid)
    print(json.dumps(listing, indent=2, ensure_ascii=False))

    print("\n[7] Pipeline stage check (R4)")
    stage = await srv.ultradebate_pipeline_stage(session_id=sid)
    print(json.dumps(stage, indent=2, ensure_ascii=False))

    print("\n[8] Capability bridge check (R14) — listing catalog")
    caps_res = await srv.ultradebate_capabilities()
    for c in caps_res["capabilities"]:
        print(f"  - {c['id']:32s} → {c['provider']}")

    print("\n[9] Finalize")
    final = await srv.ultradebate_finalize(session_id=sid)
    print(f"  archive_dir={final['archive_dir']}")
    print(f"  final_consensus={final['final_consensus']}")

    report_path = Path(final["report_path"])
    eng = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    print("\n" + sep)
    print("ENGLISH FINAL REPORT (excerpt)")
    print(sep)
    print(eng[:5000])

    print("\n" + sep)
    print("KOREAN WRAP-UP (claude-translated)")
    print(sep)
    if eng:
        print(claude_translate(eng[:6000], "en2ko")[:5000])

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
