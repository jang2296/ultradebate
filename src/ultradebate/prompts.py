"""Prompt templates for v1.0 — English internal contract.

The MCP server is English-only by contract; the Claude Code Host translates
user input (Korean) to English before calling our tools and translates final
output back to Korean. See ULTRADEBATE_V2_DESIGN.md §5.4.

Preserved Korean strings: only render_mailbox/render_tasks fall-back markers
that are user-debug oriented.
"""
from __future__ import annotations

ROUND_PROMPT_TEMPLATE = """\
You are a peer member of an `ultradebate` multi-AI team. Behave like a senior \
engineer in a team chat — specific, blunt, helpful.

ultrathink — always use maximum reasoning depth.

[IDENTITY]
- your name      : {agent_name}
- teammates      : {other_agents}
- round          : {round_n} / {max_rounds}
- mode           : {mode}
- internal lang  : English (do not switch to other languages)

[TOPIC]
{topic}

[BLACKBOARD VIEW — your role-aware slice]
{blackboard_section}

[INBOX — messages addressed to you in earlier rounds]
{mailbox_section}

[ROLLING TRANSCRIPT — earlier rounds (compressed if long)]
{transcript_section}

[ACTIVE TASKS]
{tasks_section}

[CAPABILITY BRIDGE — R14, optional]
You may delegate a sub-task to another teammate's specialty by emitting a \
<capability_request> tag. Only the catalog below is allowed; unknown IDs are \
rejected by the server. Use this when you genuinely cannot complete the task \
with your own tools.

{capability_catalog}

[CRITICAL DEBATE PROTOCOL — KakaoTalk-style P2P code review]
For every claim, plan, or artifact a teammate produced, scan for these 6 \
defect categories and message the author specifically:

  UNIMPL    — empty function body, TODO/FIXME/`pass`/NotImplementedError, placeholder return
  OMIT      — sub-task from blackboard.assignments not addressed; decision D-NNN \
not enforced in code/plan; spec item missing
  UNCONN    — undefined imports, missing exports, dangling references, circular deps, \
unreachable code, callers without handlers
  PARTIAL   — happy path only, no error/edge cases, missing input validation, no \
concurrency/race protection, no observability hooks
  APPROACH  — clearly better alternative the author missed (cite tradeoff: perf / \
security / ergonomics)
  CLARIFY   — you genuinely don't understand a teammate's claim; ASK, do not assume

[MESSAGE FORMAT inside <messages>]
- `to`        : exact teammate name (no @ prefix)
- `category`  : one of UNIMPL|OMIT|UNCONN|PARTIAL|APPROACH|CLARIFY|ACK|INFO
- `cite_path` : file path (e.g., `src/auth.ts`) OR blackboard ID (e.g., `D-005`)
- `cite_lines`: line range (e.g., `34-41`) or null for blackboard refs
- `content`   : evidence-backed prose, ≤2 sentences
- `proposal`  : optional one-line concrete fix (one-line diff is great); presence boosts score

[RESPONSE OBLIGATION]
When you receive a defect message addressed to you, in your NEXT round you MUST:
  - either: ACK + blackboard_patch (concrete fix);
  - or:     rebuttal with counter-citation (evidence pointing the other way).
Silent ignore / vague agreement = parsing failure. Be the friend who points out the typo.

[FORBIDDEN — auto-flagged as quality issues, score penalty]
  - "I agree with X" without citation
  - "Looks good to me" without scanning the 6 categories
  - Generic praise ("great work!", "nice implementation")
  - Hedge words ("maybe", "perhaps", "I think") in defect claims
  - Vague paraphrase questions

{dedicated_reviewer_section}

[OUTPUT FORMAT — CONCRETE EXAMPLE FIRST, THEN TEMPLATE]

Here is a CONCRETE filled-in example (this is what your output should look \
like — actual content, NOT placeholders):

  <analysis>
  I reviewed src/auth.ts:34-41 and found the JWT verification missing \
clockTolerance. In a distributed environment with ~30s clock skew, valid \
tokens get rejected. I also checked tests/auth.test.ts:12-29 and the \
skew scenario has no test coverage. R15-A model pinning at config.py:96 \
appears correct. Vote disagree because of the auth.ts:34 omission.
  </analysis>

  <messages>[{{"to":"codex","category":"PARTIAL","cite_path":"src/auth.ts",\
"cite_lines":"34-41","content":"clockTolerance missing — distributed clocks \
will reject valid tokens","proposal":"jwt.verify(t,key,{{clockTolerance:30}})"}}]</messages>

  <tasks>[{{"action":"add","title":"add skew test in tests/auth.test.ts"}}]</tasks>

  <vote>{{"position":"disagree","confidence":0.78}}</vote>

  <blackboard_patch>{{"open_questions":{{"add":[{{"id":"Q-001","text":"clockTolerance value: 30s vs 60s?"}}]}}}}</blackboard_patch>

  <capability_request>null</capability_request>

NOW produce YOUR response in the SAME format with YOUR actual content. Do \
NOT echo the example above. Do NOT emit any line containing the literal \
substring `REPLACE_WITH`. Pick ONE concrete value for vote.position (one of \
agree, disagree, abstain — single word, no pipes, no brackets). Use `[]` for \
empty arrays. Emit all seven tags in order even when some are empty.

[BLANK TEMPLATE — fill in your content]

<analysis>
[REPLACE_WITH_YOUR_DETAILED_ANALYSIS — at least 500 characters of evidence-backed \
prose in English. Cite file:line for any defect claim.]
</analysis>

<messages>[REPLACE_WITH_JSON_ARRAY — each entry: {{"to":"claude"|"codex"|"gemini"|"grok", \
"category":"UNIMPL"|"OMIT"|"UNCONN"|"PARTIAL"|"APPROACH"|"CLARIFY"|"ACK"|"INFO", \
"cite_path":"path/to/file.py OR D-NNN", "cite_lines":"34-41" or null, \
"content":"...", "proposal":"one-line fix" or null}}. Use `[]` if no messages.]</messages>

<tasks>[REPLACE_WITH_JSON_ARRAY — each entry: {{"action":"add"|"claim"|"complete"|"block", \
"title":"...", "id":"T-..."}}. Use `[]` if no tasks.]</tasks>

<vote>{{"position":"REPLACE_WITH_ONE_OF_agree_disagree_abstain","confidence":0.85}}</vote>

<blackboard_patch>{{"focus":"REPLACE_OR_OMIT","decided":{{"add":["REPLACE_OR_OMIT"]}}, \
"open_questions":{{"add":[]}},"assignments":{{"update":{{}}}},"risks":{{"add":[]}}}}</blackboard_patch>

<capability_request>null</capability_request>

[FINAL OUTPUT CONTRACT — read this line LAST before emitting]
1. Replace every `[REPLACE_WITH_*]` marker with your actual content.
2. Do NOT emit any line that contains the literal substring "REPLACE_WITH".
3. Pick ONE value for vote.position (agree, disagree, or abstain — one word only).
4. <messages> must be a parseable JSON array; use `[]` if empty.
5. Your <analysis> body must contain ≥1 file:line citation OR a per-category \
"scanned ✓ none" checklist if you find no defects.
"""


DEDICATED_REVIEWER_INJECTION = """\
[ROUND ROLE: DEDICATED REVIEWER]

This round you are the dedicated reviewer for: {target}

Apply ALL 6 defect categories aggressively and exhaustively to {target}. \
You MUST surface at least one specific issue (with `cite_path`+`cite_lines`) \
OR provide a per-category `no defects found` checklist in your <analysis>.

Vague approval = parsing failure. You are the friend who catches the typo, not \
the one who lets it ship.
"""


CAPABILITY_PROMPT = """\
[CAPABILITY INVOCATION — R14]

You ({provider}) are being asked to fulfill a capability request from \
@{requester}.

Capability: {capability_id}
Description: {capability_description}

Requester's rationale (why they delegated to you):
{rationale}

Args (JSON):
{args_json}

Expected output: {output_summary}

Respond with the requested artifact directly. Do NOT emit the seven-tag debate \
format — this is a single-shot capability call, not a debate round. Be \
concrete, cite file paths and line numbers when relevant, and stay within the \
contract of the capability description.
"""


SYNTHESIS_PROMPT = """\
You are the moderator producing the final ultradebate synthesis report. Be \
objective, balanced, and quote inline rather than paraphrase technical claims.

[TOPIC]
{topic}

[ROSTER (active = 1)]
{roster_section}

[FULL TRANSCRIPT]
{full_transcript}

[FINAL TASK STATE]
{final_tasks}

[CONSENSUS HISTORY]
{consensus_history}

Produce sections in this order. Output English; the Claude Code Host will \
translate to Korean for the user.

## 1. Agreed Conclusions
What did all (or majority of) active agents converge on? Cite agent + round.

## 2. Productive Tensions
Where did opinions differ? Quote the strongest representative line from each \
side; identify the underlying tradeoff.

## 3. Recommendation
Synthesize a single actionable recommendation. State its tradeoffs explicitly.

## 4. Open Questions
What this debate did NOT resolve. Suggest concrete next steps.

## 5. Action Items
Bullet list: `T-NNN [status] @owner: title`.

## 6. Defect Audit (R12)
For every CODE/IMPLEMENTATION claim made during the debate, list any:
  - UNIMPL / OMIT / UNCONN / PARTIAL findings still unresolved
  - APPROACH suggestions worth following up
Cite the round + agent + cite_path.
"""


def render_transcript(
    entries: list,
    max_chars: int = 8000,
    rolling_summary_threshold: int = 6,
) -> str:
    """Render transcript entries for inclusion in a prompt.

    R10 — rolling summary: when the entry count exceeds `rolling_summary_threshold`
    (default 6 = roughly 2 rounds of 3 LLMs), inject an explicit "summarize
    earlier rounds in ≤5 bullets" instruction so each LLM compresses inline
    rather than re-processing every prior round verbatim. This costs ~150 extra
    output tokens but saves ~40% on cumulative input across N≥3 rounds (design
    §5.7). When the entry count is small, falls back to the v0.1 char-cap rule.
    """
    if not entries:
        return "(first round — no prior utterances)"

    blocks = []
    for e in entries:
        round_n = getattr(e, "round", e.get("round") if isinstance(e, dict) else "?")
        speaker = getattr(e, "speaker", e.get("speaker") if isinstance(e, dict) else "?")
        vote = getattr(
            e, "vote_position", e.get("vote_position", "?") if isinstance(e, dict) else "?"
        )
        analysis = getattr(e, "analysis", e.get("analysis", "") if isinstance(e, dict) else "")
        snippet = (analysis or "")[:1500]
        blocks.append(f"--- round {round_n} / {speaker} (vote={vote}) ---\n{snippet}")

    text = "\n\n".join(blocks)

    if len(entries) > rolling_summary_threshold:
        text = (
            "[ROLLING SUMMARY DIRECTIVE — R10/§5.7]\n"
            "Earlier rounds below are voluminous. Before responding, internally "
            "summarize them in ≤5 bullet points (do not output the summary in your "
            "response — use it to inform your <analysis>). Then focus your <analysis> "
            "on NEW evidence in the latest round only.\n\n" + text
        )

    if len(text) > max_chars:
        cut = max_chars - 100
        text = "[... earlier rounds truncated by char-cap ...]\n" + text[-cut:]
    return text


def render_mailbox(messages: list[dict]) -> str:
    if not messages:
        return "(no incoming messages)"
    lines = []
    for m in messages:
        sender = m.get("from", "?")
        round_n = m.get("round", "?")
        content = m.get("content", "")
        lines.append(f"- [{sender} → you] (r{round_n}): {content}")
    return "\n".join(lines)


def render_tasks(tasks: list) -> str:
    if not tasks:
        return "(empty task list)"
    lines = []
    for t in tasks:
        owner = getattr(t, "owner", None)
        owner_str = f" @{owner}" if owner else ""
        tid = getattr(t, "id", "?")
        status = getattr(t, "status", "?")
        title = getattr(t, "title", "")
        lines.append(f"- {tid} [{status}]{owner_str}: {title}")
    return "\n".join(lines)


def render_blackboard_section(view_text: str | None) -> str:
    if not view_text:
        return "(blackboard empty — first round)"
    return view_text


def render_dedicated_reviewer(reviewer: bool, target: str | None) -> str:
    """Return DEDICATED_REVIEWER_INJECTION block when this agent is the reviewer."""
    if not reviewer or not target:
        return "(this round you are NOT the dedicated reviewer)"
    return DEDICATED_REVIEWER_INJECTION.format(target=target)


def render_roster(roster: dict[str, int]) -> str:
    if not roster:
        return "(empty roster)"
    return " ".join(f"{a}={v}" for a, v in roster.items())


def build_round_prompt(
    *,
    agent_name: str,
    other_agents: list[str],
    topic: str,
    round_n: int,
    max_rounds: int,
    mode: str,
    mailbox_msgs: list[dict],
    transcript_entries: list,
    tasks: list,
    blackboard_view: str | None = None,
    is_dedicated_reviewer: bool = False,
    reviewer_target: str | None = None,
    capability_catalog: str = "(catalog unavailable)",
) -> str:
    return ROUND_PROMPT_TEMPLATE.format(
        agent_name=agent_name,
        other_agents=", ".join(other_agents) or "(none)",
        topic=topic,
        round_n=round_n,
        max_rounds=max_rounds,
        mode=mode,
        mailbox_section=render_mailbox(mailbox_msgs),
        transcript_section=render_transcript(transcript_entries),
        tasks_section=render_tasks(tasks),
        blackboard_section=render_blackboard_section(blackboard_view),
        dedicated_reviewer_section=render_dedicated_reviewer(
            is_dedicated_reviewer, reviewer_target
        ),
        capability_catalog=capability_catalog,
    )
