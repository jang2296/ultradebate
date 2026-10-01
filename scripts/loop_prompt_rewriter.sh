#!/usr/bin/env bash
# ultradebate v2.3 — /loop prompt staleness rewriter (B12 rule)
#
# Accepts a /loop prompt on stdin (or as first arg). Emits an annotated prompt
# that prepends a CURRENT-STATE banner from shared_plan.json. Claude reads the
# rewritten prompt and uses the live state instead of the (potentially stale)
# user-typed content.
#
# Usage:
#   echo "$ORIG_PROMPT" | bash loop_prompt_rewriter.sh
#   bash loop_prompt_rewriter.sh "$ORIG_PROMPT"

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [ "$#" -ge 1 ]; then
  ORIG="$1"
else
  ORIG="$(cat)"
fi

SUMMARY="$("$SCRIPT_DIR/update_shared_plan.sh" summary 2>/dev/null || echo '(plan unavailable)')"

cat <<EOF
[B12 SHARED PLAN CURRENT STATE]
$SUMMARY

[user_loop_prompt_below — may be stale, defer to shared plan if mismatch]
$ORIG
EOF
