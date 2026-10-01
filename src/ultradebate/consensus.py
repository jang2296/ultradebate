"""Consensus scoring + convergence detection (v1.0).

Adds two adjustments on top of v0.1's vote/text Jaccard score:
  1. ACK bonus (+0.10 each, capped) — a sender's ACK reply to a defect message
     in the SAME round signals a healthy critique→accept cycle.
  2. Vague penalty (-0.05 each, capped) — messages flagged `vague=True` by the
     parser (missing category, or defect category without cite_path) reduce
     score because they degrade signal quality.

Caps: net adjustment is bounded to [-0.20, +0.20] so a flood of ACKs can't
fake convergence and a flood of vague messages can't fake collapse.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from .parser import ParsedResponse

_WORD_RE = re.compile(r"[\w가-힣]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "")}


def jaccard(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class ConsensusReport:
    score: float
    vote_agreement: float
    text_agreement: float
    distribution: dict = field(default_factory=dict)
    most_common_vote: str = "abstain"
    converged: bool = False
    delta: float | None = None
    # Raw counts × per-message weight (BEFORE the ±0.20 cap). Useful for telemetry.
    ack_bonus: float = 0.0
    vague_penalty: float = 0.0
    # The (capped, signed) net adjustment that was actually applied to base_score.
    # Always |applied_adjustment| ≤ bonus_cap. This is what moved the score.
    applied_adjustment: float = 0.0
    base_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "vote_agreement": round(self.vote_agreement, 4),
            "text_agreement": round(self.text_agreement, 4),
            "distribution": self.distribution,
            "most_common_vote": self.most_common_vote,
            "converged": self.converged,
            "delta": None if self.delta is None else round(self.delta, 4),
            "ack_bonus": round(self.ack_bonus, 4),
            "vague_penalty": round(self.vague_penalty, 4),
            "applied_adjustment": round(self.applied_adjustment, 4),
            "base_score": round(self.base_score, 4),
        }


def consensus_score(
    parsed_list: list[ParsedResponse],
    *,
    prev_score: float | None = None,
    delta_threshold: float = 0.05,
    min_score_for_convergence: float = 0.6,
    ack_bonus_per_message: float = 0.10,
    vague_penalty_per_message: float = 0.05,
    bonus_cap: float = 0.20,
) -> ConsensusReport:
    """Compute weighted consensus with category-aware adjustments."""
    if not parsed_list:
        return ConsensusReport(
            score=0.0,
            vote_agreement=0.0,
            text_agreement=0.0,
            distribution={},
            most_common_vote="abstain",
            converged=False,
        )

    votes = [p.vote.position for p in parsed_list]
    distribution = {v: votes.count(v) for v in ("agree", "disagree", "abstain")}
    most_common = max(distribution, key=lambda k: (distribution[k], k == "agree"))
    vote_agreement = distribution[most_common] / len(votes)

    pairs = list(itertools.combinations(parsed_list, 2))
    if pairs:
        text_agreement = sum(jaccard(a.analysis, b.analysis) for a, b in pairs) / len(pairs)
    else:
        text_agreement = 1.0

    base_score = 0.4 * vote_agreement + 0.6 * text_agreement

    # Category adjustments
    ack_count = 0
    vague_count = 0
    for parsed in parsed_list:
        for msg in parsed.messages:
            if msg.is_ack:
                ack_count += 1
            if msg.vague:
                vague_count += 1

    raw_bonus = ack_count * ack_bonus_per_message
    raw_penalty = vague_count * vague_penalty_per_message
    net_adj = raw_bonus - raw_penalty
    if net_adj > bonus_cap:
        net_adj = bonus_cap
    elif net_adj < -bonus_cap:
        net_adj = -bonus_cap

    score = max(0.0, min(1.0, base_score + net_adj))

    delta = None if prev_score is None else abs(score - prev_score)
    converged = (
        prev_score is not None
        and delta is not None
        and delta < delta_threshold
        and score >= min_score_for_convergence
    )

    return ConsensusReport(
        score=score,
        vote_agreement=vote_agreement,
        text_agreement=text_agreement,
        distribution=distribution,
        most_common_vote=most_common,
        converged=converged,
        delta=delta,
        ack_bonus=raw_bonus,
        vague_penalty=raw_penalty,
        applied_adjustment=net_adj,
        base_score=base_score,
    )
