---
name: ultradebate
description: Coordinate running Claude Code and Codex CLI sessions through the Ultradebate MCP, with inbox monitoring and peer review.
---

# Ultradebate

Use when the user requests Claude–Codex collaboration. Keep existing CLI sessions and their configured models. Do not start a second CLI for routine collaboration.

1. Confirm both CLIs run in the same project and Codex runs inside tmux. Register each with `ultradebate_v2_session_attach` and check `ultradebate_v2_session_status`.
2. Read pending messages with `ultradebate_v2_priority_inbox_read`. On Claude, keep a persistent Monitor subscription to `scripts/claude_inbox_watcher.sh claude` if the host provides Monitor. Otherwise, keep `ultradebate_v2_wait_for_peer_message` active during collaboration and rearm after each result. A background log alone cannot wake the conversation.
3. Divide work into distinct file scopes. Send objectives, paths and acceptance checks with `ultradebate_v2_send_to_codex` or `ultradebate_v2_send_to_claude`. For Claude outside tmux, use `tmux_inject=false` and its inbox alert subscription.
4. Use MCP paste-buffer delivery and the actual submission keystroke. A literal `\n` cannot substitute for Enter. Codex uses separate `C-m`; do not use Tab or Escape+Enter.
5. Acknowledge received messages by `msg_id`. Distinguish inbox write, tmux submission and the recipient's actual response. Avoid duplicate execution when submission is unverified.
6. Use `ultradebate_v2_run_long_job` and its returned `job_id` for long commands. Poll the existing job before restarting it.
7. Review peer changes, resolve findings and run relevant checks. Report in the user's language with file paths and evidence.

Do not exchange credentials, personal records or unrelated project content. Keep CLI permission settings enabled. The current v2 inbox is shared within the local user; use one collaboration project at a time.
