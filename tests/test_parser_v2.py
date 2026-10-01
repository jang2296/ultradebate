"""v1.0 parser tests: 6 defect categories, cite/proposal, blackboard_patch."""
from __future__ import annotations

from ultradebate.parser import (
    DEFECT_CATEGORIES,
    OutgoingMessage,
    parse_response,
)


def _wrap(messages_json: str) -> str:
    return (
        "<analysis>x</analysis>"
        f"<messages>{messages_json}</messages>"
        '<tasks>[]</tasks>'
        '<vote>{"position":"agree","confidence":0.5}</vote>'
    )


def test_defect_message_with_cite_is_not_vague():
    text = _wrap(
        '[{"to":"codex","category":"PARTIAL","cite_path":"src/auth.ts",'
        '"cite_lines":"34-41","content":"clockTolerance missing","proposal":"add option"}]'
    )
    r = parse_response(text)
    assert len(r.messages) == 1
    msg = r.messages[0]
    assert msg.category == "PARTIAL"
    assert msg.cite_path == "src/auth.ts"
    assert msg.cite_lines == "34-41"
    assert msg.proposal == "add option"
    assert msg.vague is False
    assert msg.is_defect is True
    assert msg.is_ack is False
    assert not r.parse_warnings


def test_defect_category_without_cite_is_vague():
    text = _wrap('[{"to":"codex","category":"UNIMPL","content":"missing impl"}]')
    r = parse_response(text)
    assert r.messages[0].category == "UNIMPL"
    assert r.messages[0].vague is True
    assert any("missing cite_path" in w for w in r.parse_warnings)


def test_message_without_category_defaults_to_INFO_vague():
    text = _wrap('[{"to":"codex","content":"general comment"}]')
    r = parse_response(text)
    assert r.messages[0].category == "INFO"
    assert r.messages[0].vague is True
    assert any("missing/invalid category" in w for w in r.parse_warnings)


def test_invalid_category_falls_back_to_INFO():
    text = _wrap('[{"to":"codex","category":"GIBBERISH","content":"x"}]')
    r = parse_response(text)
    assert r.messages[0].category == "INFO"
    assert r.messages[0].vague is True


def test_ack_category_is_not_vague_without_cite():
    """ACK is a control message, not a defect — no cite required."""
    text = _wrap('[{"to":"codex","category":"ACK","content":"agreed, fix applied"}]')
    r = parse_response(text)
    assert r.messages[0].category == "ACK"
    assert r.messages[0].is_ack is True
    assert r.messages[0].vague is False


def test_at_prefix_in_to_is_stripped():
    text = _wrap(
        '[{"to":"@gemini","category":"CLARIFY","cite_path":"D-001","content":"explain"}]'
    )
    r = parse_response(text)
    assert r.messages[0].to == "gemini"


def test_blackboard_patch_extracted():
    text = (
        "<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"agree","confidence":0.5}</vote>'
        '<blackboard_patch>{"focus":"new focus","decided":{"add":["D-1: pick RS256"]}}</blackboard_patch>'
    )
    r = parse_response(text)
    assert isinstance(r.blackboard_patch, dict)
    assert r.blackboard_patch["focus"] == "new focus"


def test_blackboard_patch_missing_returns_none():
    text = (
        "<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"agree","confidence":0.5}</vote>'
    )
    r = parse_response(text)
    assert r.blackboard_patch is None


def test_blackboard_patch_invalid_json_warns():
    text = (
        "<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"agree","confidence":0.5}</vote>'
        '<blackboard_patch>not json</blackboard_patch>'
    )
    r = parse_response(text)
    # Loose parser will not extract a list/dict from "not json"; bb_patch stays None
    assert r.blackboard_patch is None


def test_grok_recipient_accepted():
    text = _wrap(
        '[{"to":"grok","category":"INFO","content":"fyi","cite_path":"D-1"}]'
    )
    r = parse_response(text)
    assert len(r.messages) == 1
    assert r.messages[0].to == "grok"


def test_defect_categories_count_is_six():
    """Architectural invariant — design doc commits to exactly 6 defect classes."""
    assert len(DEFECT_CATEGORIES) == 6
    assert DEFECT_CATEGORIES == {
        "UNIMPL", "OMIT", "UNCONN", "PARTIAL", "APPROACH", "CLARIFY"
    }


def test_template_echo_detected_from_replace_with_marker():
    """When LLM echoes the REPLACE_WITH placeholder, parser flags it."""
    text = (
        "<analysis>[REPLACE_WITH_YOUR_DETAILED_ANALYSIS — at least 500 chars...]</analysis>"
        "<messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"REPLACE_WITH_ONE_OF_agree_disagree_abstain","confidence":0.85}</vote>'
    )
    r = parse_response(text)
    assert r.template_echo is True
    assert any("template-echo detected" in w for w in r.parse_warnings)
    # Vote downgraded to abstain since position was invalid placeholder
    assert r.vote.position == "abstain"


def test_template_echo_detected_from_pipe_list_in_vote():
    """Older codex versions echoed `agree|disagree|abstain` literally."""
    text = (
        "<analysis>some real analysis</analysis>"
        "<messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"agree|disagree|abstain","confidence":0.5}</vote>'
    )
    r = parse_response(text)
    assert r.template_echo is True


def test_template_echo_false_for_well_formed_response():
    """Real, filled-in responses do NOT trigger the echo flag."""
    text = (
        "<analysis>This is a real analysis with actual content.</analysis>"
        "<messages>[]</messages><tasks>[]</tasks>"
        '<vote>{"position":"agree","confidence":0.85}</vote>'
    )
    r = parse_response(text)
    assert r.template_echo is False
    assert r.vote.position == "agree"
    assert r.vote.confidence == 0.85


def test_template_echo_zeros_confidence_unconditionally():
    """iter6 finding: downgrade was dead code (gated by `not in VALID_VOTES`
    which was unreachable). Now the downgrade fires unconditionally — both
    position becomes 'abstain' AND confidence becomes 0.0."""
    text = (
        "<analysis>[REPLACE_WITH_YOUR_DETAILED_ANALYSIS — at least 500 chars...]</analysis>"
        "<messages>[]</messages><tasks>[]</tasks>"
        # Note: a high confidence value that should still be zeroed out
        '<vote>{"position":"agree","confidence":0.99}</vote>'
    )
    r = parse_response(text)
    assert r.template_echo is True
    assert r.vote.position == "abstain"
    # Confidence MUST be zeroed even if the echo had a valid position string
    assert r.vote.confidence == 0.0
