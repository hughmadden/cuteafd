#!/bin/bash
# Stop a Codex task started by codex-launch.sh, including every child process.
# Killing only the launcher leaves Codex running as an orphan that keeps editing.
runs="${CODEX_RUNS:-$HOME/.cache/cuteafd/builds/codex-runs}"
name="${1:?task name}"
pid="$(cat "$runs/$name.pid" 2>/dev/null)" || { echo "no running task $name" >&2; exit 1; }
kill -TERM -- "-$pid" 2>/dev/null
sleep 3
kill -KILL -- "-$pid" 2>/dev/null
rm -f "$runs/$name.pid"
echo "stopped $name ($pid)"
