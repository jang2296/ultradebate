"""Deterministic mock worker — no API calls. Used for unit tests and dry-runs.

Each MockWorker takes one of three personas (claude/codex/gemini) and emits a
distinct opinion so the consensus engine has real signal to work with.
"""
from __future__ import annotations

import asyncio
import random
import re

from ..parser import parse_response
from .base import AgentResponse

# Each persona has a flavor and a tendency. Different vote distributions exercise
# the consensus engine across rounds.
_PERSONAS = {
    "claude": {
        "flavor": "체계적이고 신중한 분석",
        "vote_bias": "agree",
    },
    "codex": {
        "flavor": "구현 중심의 실무적 관점",
        "vote_bias": "disagree",
    },
    "gemini": {
        "flavor": "균형 잡힌 절충안",
        "vote_bias": "abstain",
    },
}


class MockWorker:
    def __init__(self, name: str, *, seed: int | None = None):
        if name not in _PERSONAS:
            raise ValueError(f"unknown persona: {name}")
        self.name = name
        self._rng = random.Random(seed)

    async def is_available(self) -> bool:
        return True

    async def invoke(self, prompt: str, timeout: int = 60) -> AgentResponse:
        await asyncio.sleep(0.02 + self._rng.random() * 0.05)

        topic = "(주제 미상)"
        m = re.search(r"\[토론 주제\]\s*\n(.+?)(?:\n\n|\n\[)", prompt, re.DOTALL)
        if m:
            topic = m.group(1).strip()[:120]

        round_n = 1
        m = re.search(r"현재 라운드:\s*(\d+)", prompt)
        if m:
            round_n = int(m.group(1))

        persona = _PERSONAS[self.name]
        confidence = round(0.4 + self._rng.random() * 0.5, 2)
        opening = f"[mock-{self.name} r{round_n}] {persona['flavor']} 관점에서 보면, "
        body = f"주제 '{topic[:60]}...' 에 대해 다음 근거로 입장을 정합니다."
        analysis = opening + body

        # On round >= 2, occasionally send a P2P message to simulate debate
        messages = "[]"
        if round_n >= 2 and self._rng.random() < 0.5:
            others = [p for p in _PERSONAS if p != self.name]
            target = self._rng.choice(others)
            messages = f'[{{"to":"{target}","content":"r{round_n}에서 당신 의견에 추가 질문 있습니다."}}]'

        text = (
            f"<analysis>\n{analysis}\n</analysis>\n"
            f"<messages>{messages}</messages>\n"
            f"<tasks>[]</tasks>\n"
            f'<vote>{{"position":"{persona["vote_bias"]}","confidence":{confidence}}}</vote>'
        )
        parsed = parse_response(text)
        return AgentResponse(
            agent=self.name,
            text=parsed.analysis,
            parsed=parsed,
            tokens_in=len(prompt) // 4,
            tokens_out=len(text) // 4,
            latency_ms=50,
            raw=text,
        )

    async def shutdown(self) -> None:
        return None
