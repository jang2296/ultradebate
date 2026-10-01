#!/usr/bin/env bash
# ultradebate v2.3 — heartbeat helper (B10 rule)
#
# Convenience wrapper for update_shared_plan.sh heartbeat ...
# Intended to be called at every turn by both Claude and Codex.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
actor="${1:-claude}"
action="${2:-heartbeat tick}"
exec "$SCRIPT_DIR/update_shared_plan.sh" heartbeat "$actor" "$action"
