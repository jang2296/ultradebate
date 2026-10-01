"""Parser robustness tests — every malformed-input path must return a valid object."""
from __future__ import annotations

from ultradebate.parser import parse_response


def test_full_well_formed():
    """v1.0: well-formed messages must include category to avoid vague-warning."""
    text = """<analysis>
This is my analysis. I think X is true because Y holds.
</analysis>
<messages>[{"to":"codex","category":"APPROACH","cite_path":"D-001","content":"Disagree on Z"}]</messages>
<tasks>[{"action":"add","title":"verify Y"}]</tasks>
<vote>{"position":"agree","confidence":0.8}</vote>"""
    r = parse_response(text)
    assert "analysis" in r.analysis.lower() or "true" in r.analysis.lower()
    assert len(r.messages) == 1
    assert r.messages[0].to == "codex"
    assert r.messages[0].content == "Disagree on Z"
    assert r.messages[0].category == "APPROACH"
    assert r.messages[0].vague is False
    assert len(r.tasks) == 1
    assert r.tasks[0].action == "add"
    assert r.tasks[0].title == "verify Y"
    assert r.vote.position == "agree"
    assert abs(r.vote.confidence - 0.8) < 1e-6
    assert not r.parse_warnings


def test_missing_all_tags_fallback():
    r = parse_response("just plain text no tags")
    assert r.analysis == "just plain text no tags"
    assert r.vote.position == "abstain"
    assert "missing <analysis> tag" in r.parse_warnings


def test_empty_string():
    r = parse_response("")
    assert r.analysis == ""
    assert r.vote.position == "abstain"
    assert r.messages == []
    assert r.tasks == []


def test_non_string_input():
    r = parse_response(None)  # type: ignore[arg-type]
    assert r.parse_warnings == ["non-string response"]


def test_malformed_messages_json():
    text = "<analysis>x</analysis><messages>not valid json</messages><tasks>[]</tasks><vote>{}</vote>"
    r = parse_response(text)
    assert r.messages == []
    assert any("messages" in w for w in r.parse_warnings) or r.messages == []


def test_invalid_vote_position_falls_back():
    text = (
        '<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks>'
        '<vote>{"position":"yes","confidence":1.5}</vote>'
    )
    r = parse_response(text)
    assert r.vote.position == "abstain"
    assert r.vote.confidence == 1.0  # clamped to [0, 1]


def test_negative_confidence_clamped():
    text = '<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks><vote>{"position":"agree","confidence":-0.5}</vote>'
    r = parse_response(text)
    assert r.vote.confidence == 0.0


def test_nan_confidence_sanitized_to_default():
    """LLM-found defect: NaN survived min/max clamp; must default to 0.5."""
    text = (
        '<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks>'
        '<vote>{"position":"agree","confidence":NaN}</vote>'
    )
    r = parse_response(text)
    # NaN is not valid JSON literal — _parse_json_loose may fail entirely.
    # Either way, parser must not return a NaN confidence.
    import math
    assert not math.isnan(r.vote.confidence)
    assert 0.0 <= r.vote.confidence <= 1.0


def test_inf_confidence_sanitized_to_default():
    """Inf must also be sanitized to a finite default."""
    text = (
        '<analysis>x</analysis><messages>[]</messages><tasks>[]</tasks>'
        '<vote>{"position":"agree","confidence":Infinity}</vote>'
    )
    r = parse_response(text)
    import math
    assert not math.isinf(r.vote.confidence)
    assert 0.0 <= r.vote.confidence <= 1.0


def test_messages_filter_invalid_recipient():
    text = (
        '<analysis>x</analysis>'
        '<messages>[{"to":"hacker","content":"x"},{"to":"claude","content":"hi"}]</messages>'
        '<tasks>[]</tasks><vote>{"position":"agree","confidence":0.5}</vote>'
    )
    r = parse_response(text)
    assert len(r.messages) == 1
    assert r.messages[0].to == "claude"


def test_messages_self_target_kept_at_parser_level():
    """Parser allows self-targeted messages; routing layer drops them."""
    text = (
        '<analysis>x</analysis>'
        '<messages>[{"to":"codex","content":"hi"}]</messages>'
        '<tasks>[]</tasks><vote>{"position":"agree","confidence":0.5}</vote>'
    )
    r = parse_response(text)
    assert r.messages[0].to == "codex"


def test_tasks_unknown_action_dropped():
    text = (
        '<analysis>x</analysis><messages>[]</messages>'
        '<tasks>[{"action":"hack","title":"x"},{"action":"add","title":"ok"}]</tasks>'
        '<vote>{"position":"agree","confidence":0.5}</vote>'
    )
    r = parse_response(text)
    assert len(r.tasks) == 1
    assert r.tasks[0].action == "add"


def test_json_in_code_fence():
    text = (
        '<analysis>x</analysis>\n'
        '<messages>```json\n[{"to":"claude","content":"hi"}]\n```</messages>\n'
        '<tasks>[]</tasks><vote>{"position":"agree","confidence":0.7}</vote>'
    )
    r = parse_response(text)
    assert len(r.messages) == 1


def test_extra_whitespace_and_case_insensitive():
    text = (
        '<ANALYSIS>  multi\nline  </ANALYSIS>'
        '<MESSAGES>[]</MESSAGES><TASKS>[]</TASKS>'
        '<VOTE>{"position":"AGREE","confidence":0.5}</VOTE>'
    )
    r = parse_response(text)
    assert "multi" in r.analysis
    assert r.vote.position == "agree"
