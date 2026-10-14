#!/bin/bash
# Stop an external agent started by ext-launch.sh (its whole tmux session).
#   scripts/agents/ext-stop.sh <name>
set -u
runs="${AGENT_RUNS:-$HOME/.cache/cuteafd/builds/codex-runs}"
name="${1:?task name}"
session="ext-$name"
tmux kill-session -t "$session" 2>/dev/null && echo "stopped $session" || echo "$session was not running"
rm -f "$runs/$name.tmux"
