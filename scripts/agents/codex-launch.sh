#!/bin/bash
# Run one Codex task with full access (GPU, SSH, git push) and wait for it.
#   scripts/agents/codex-launch.sh <name> [effort]     effort: high (default) | xhigh | ...
# The prompt is scripts/agents/codex-preamble.md + $CODEX_RUNS/<name>.md.
# CODEX_MODEL overrides the model (default: ~/.codex/config.toml).
# Output: <name>.report.md (final message), <name>.log, <name>.pid while running.
# Start it as a background command; its exit wakes the orchestrator.
set -u
repo="$(cd "$(dirname "$0")/../.." && pwd)"
runs="${CODEX_RUNS:-$HOME/.cache/cuteafd/builds/codex-runs}"
name="${1:?task name}"
effort="${2:-high}"
[[ -f "$runs/$name.md" ]] || { echo "missing $runs/$name.md" >&2; exit 2; }
if [[ -f "$runs/$name.pid" ]] && kill -0 "$(cat "$runs/$name.pid")" 2>/dev/null; then
  echo "$name is already running (pid $(cat "$runs/$name.pid")); stop it first" >&2
  exit 2
fi
cat "$repo/scripts/agents/codex-preamble.md" "$runs/$name.md" > "$runs/$name.prompt"
cd "$repo" || exit 2
model_args=()
[[ -n "${CODEX_MODEL:-}" ]] && model_args=(-m "$CODEX_MODEL")
# setsid: the whole Codex process tree is one group, so codex-stop.sh can end all of it.
setsid codex exec "${model_args[@]}" -s danger-full-access -c model_reasoning_effort="$effort" \
  -o "$runs/$name.report.md" - < "$runs/$name.prompt" > "$runs/$name.log" 2>&1 &
pid=$!
echo "$pid" > "$runs/$name.pid"
wait "$pid"
rc=$?
rm -f "$runs/$name.pid"
echo "codex $name exited $rc"
