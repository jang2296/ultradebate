"""Consensus engine tests."""
from __future__ import annotations

from ultradebate.consensus import consensus_score, jaccard
from ultradebate.parser import ParsedResponse, Vote


def _parsed(analysis: str, position: str, confidence: float = 0.7) -> ParsedResponse:
    return ParsedResponse(analysis=analysis, vote=Vote(position=position, confidence=confidence))


def test_jaccard_basic():
    assert jaccard("a b c", "a b") > 0.5
    assert jaccard("x y z", "p q r") == 0
    assert jaccard("", "anything") == 0
    assert jaccard("일치하는 한국어", "일치하는") > 0


def test_unanimous_agree():
    parsed = [_parsed("python is great and fast", "agree")] * 3
    r = consensus_score(parsed)
    assert r.vote_agreement == 1.0
    assert r.most_common_vote == "agree"
    assert r.score > 0.9


def test_three_way_split():
    parsed = [
        _parsed("X is true", "agree"),
        _parsed("X is false", "disagree"),
        _parsed("uncertain", "abstain"),
    ]
    r = consensus_score(parsed)
    assert r.vote_agreement < 0.5
    assert r.score < 0.5


def test_convergence_only_above_floor():
    """A flat-low score should NOT be marked as converged."""
    parsed = [_parsed("totally different word salad alpha", "agree"),
              _parsed("completely orthogonal beta gamma", "disagree"),
              _parsed("third unrelated delta epsilon", "abstain")]
    r = consensus_score(parsed, prev_score=0.1, delta_threshold=0.05)
    # Score is low; even though delta is small, we don't call this convergence.
    assert not r.converged


def test_convergence_fires_when_high_and_stable():
    parsed = [_parsed("identical strong text identical strong text", "agree")] * 3
    # current score will be 1.0; prev=0.96 makes delta=0.04 < threshold=0.05
    r = consensus_score(parsed, prev_score=0.96, delta_threshold=0.05)
    assert r.converged, f"expected convergence; got delta={r.delta}, score={r.score}"


def test_distribution_keys_complete():
    """All three vote categories should appear in distribution even if zero."""
    parsed = [_parsed("text", "agree")] * 3
    r = consensus_score(parsed)
    assert set(r.distribution.keys()) == {"agree", "disagree", "abstain"}
    assert r.distribution["disagree"] == 0


def test_to_dict_serializable():
    import json
    parsed = [_parsed("text", "agree")] * 3
    r = consensus_score(parsed)
    json.dumps(r.to_dict())  # must not raise


def test_empty_list_safe():
    r = consensus_score([])
    assert r.score == 0.0
    assert r.most_common_vote == "abstain"
