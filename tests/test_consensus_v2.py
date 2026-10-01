"""v1.0 consensus tests: ACK bonus, vague penalty, score clamping."""
from __future__ import annotations

from ultradebate.consensus import consensus_score
from ultradebate.parser import OutgoingMessage, ParsedResponse, Vote


def _resp(*, vote: str, conf: float, analysis: str, msgs=None) -> ParsedResponse:
    return ParsedResponse(
        analysis=analysis,
        messages=list(msgs or []),
        vote=Vote(position=vote, confidence=conf),
    )


def _msg(category: str, vague: bool = False) -> OutgoingMessage:
    return OutgoingMessage(
        to="claude",
        content="x",
        category=category,
        cite_path="D-1" if not vague else None,
        vague=vague,
    )


def test_ack_messages_increase_score():
    """Use partially-overlapping analyses so base_score < 1.0 and bonus can show."""
    base = [
        _resp(vote="agree", conf=0.8, analysis="we should do X for security"),
        _resp(vote="agree", conf=0.8, analysis="we should do X for performance"),
    ]
    cons_no_ack = consensus_score(base)
    assert cons_no_ack.base_score < 1.0  # leave headroom

    with_acks = [
        _resp(vote="agree", conf=0.8, analysis="we should do X for security", msgs=[_msg("ACK")]),
        _resp(vote="agree", conf=0.8, analysis="we should do X for performance", msgs=[_msg("ACK")]),
    ]
    cons_acks = consensus_score(with_acks)
    assert cons_acks.ack_bonus == 0.20
    assert cons_acks.score > cons_no_ack.score


def test_vague_messages_decrease_score():
    with_vague = [
        _resp(
            vote="agree", conf=0.8, analysis="X",
            msgs=[_msg("INFO", vague=True), _msg("INFO", vague=True)],
        ),
    ]
    cons = consensus_score(with_vague)
    assert cons.vague_penalty == 0.10
    assert cons.score < cons.base_score


def test_net_adjustment_capped():
    """20 ACKs shouldn't push score adjustment beyond +0.20."""
    spam = [
        _resp(
            vote="agree", conf=0.5, analysis="x",
            msgs=[_msg("ACK") for _ in range(20)],
        ),
    ]
    cons = consensus_score(spam)
    # raw bonus 2.0, but applied adjustment must be ≤ 0.20
    assert cons.ack_bonus == 2.0  # raw still recorded
    assert cons.score - cons.base_score <= 0.20 + 1e-9


def test_score_clamped_to_one():
    """High base + high bonus shouldn't exceed 1.0."""
    perfect = [
        _resp(
            vote="agree", conf=1.0, analysis="identical text",
            msgs=[_msg("ACK"), _msg("ACK")],
        ),
        _resp(
            vote="agree", conf=1.0, analysis="identical text",
            msgs=[_msg("ACK"), _msg("ACK")],
        ),
    ]
    cons = consensus_score(perfect)
    assert cons.score <= 1.0


def test_convergence_requires_high_score_and_small_delta():
    """prev_score must be close enough that |Δ| < 0.05."""
    parsed = [
        _resp(vote="agree", conf=0.9, analysis="aligned text"),
        _resp(vote="agree", conf=0.9, analysis="aligned text"),
    ]
    # current score will be 1.0; prev close to it
    cons = consensus_score(parsed, prev_score=0.98)
    assert cons.delta is not None and cons.delta < 0.05
    assert cons.score >= 0.6
    assert cons.converged is True


def test_no_convergence_when_score_low():
    parsed = [
        _resp(vote="disagree", conf=0.5, analysis="A"),
        _resp(vote="agree", conf=0.5, analysis="completely different B C D"),
    ]
    cons = consensus_score(parsed, prev_score=0.30)
    assert cons.converged is False


def test_distribution_keys_present():
    parsed = [
        _resp(vote="agree", conf=0.5, analysis="x"),
        _resp(vote="disagree", conf=0.5, analysis="y"),
    ]
    cons = consensus_score(parsed)
    d = cons.to_dict()
    assert set(d["distribution"]) >= {"agree", "disagree", "abstain"}
    assert "ack_bonus" in d
    assert "vague_penalty" in d
    assert "applied_adjustment" in d
    assert "base_score" in d


def test_applied_adjustment_reflects_capping():
    """Raw bonus 2.0 (20 ACKs × 0.10) capped to +0.20 applied."""
    spam = [
        _resp(
            vote="agree", conf=0.5, analysis="x",
            msgs=[_msg("ACK") for _ in range(20)],
        ),
    ]
    cons = consensus_score(spam)
    assert cons.ack_bonus == 2.0  # raw recorded
    assert cons.applied_adjustment == 0.20  # capped applied
