#!/bin/bash
# Run one task on an external CLI agent (Grok, or Gemini via Antigravity) in a
# detached tmux session, so the orchestrator can watch, steer and stop it.
#   scripts/agents/ext-launch.sh grok|agy <name> [model] [effort]
# agy models carry their effort in the name (gemini-4-argon-high); effort applies only without a model.
# The prompt is scripts/agents/codex-preamble.md + $AGENT_RUNS/<name>.md.
# Output: <name>.report.md (the transcript tail), <name>.log (full transcript).
# The script waits until the agent exits, so start it as a background command;
# its exit wakes the orchestrator.
#   Peek:  tail <name>.log        Stop: scripts/agents/ext-stop.sh <name>
set -u
repo="${AGENT_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
runs="${AGENT_RUNS:-$HOME/.cache/cuteafd/builds/codex-runs}"
tool="${1:?grok or agy}"
name="${2:?task name}"
model="${3:-}"
effort="${4:-high}"
[[ -f "$runs/$name.md" ]] || { echo "missing $runs/$name.md" >&2; exit 2; }
session="ext-$name"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "$session is already running; stop it first" >&2
  exit 2
fi
prompt="$runs/$name.prompt" log="$runs/$name.log"
cat "$repo/scripts/agents/codex-preamble.md" "$runs/$name.md" > "$prompt"
# A per-run runner script: the prompt never passes through tmux's command line.
runner="$runs/$name.run.sh"
{
  echo '#!/bin/bash'
  echo "cd $(printf %q "$repo")"
  case "$tool" in
    grok)
      echo "grok --always-approve --output-format plain --cwd $(printf %q "$repo") ${model:+-m $(printf %q "$model")} --reasoning-effort $(printf %q "$effort") --prompt-file $(printf %q "$prompt")"
      echo 'rc=$?' ;;
    agy)
      # Antigravity's eligibility check returns transient 503s; retry those.
      echo "for attempt in 1 2 3 4 5; do"
      echo "  agy --dangerously-skip-permissions --add-dir $(printf %q "$repo") $(if [[ -n "$model" ]]; then printf -- '--model %q' "$model"; else printf -- '--effort %q' "$effort"; fi) -p \"\$(cat $(printf %q "$prompt"))\" 2>&1 | tee /tmp/agy-\$\$.out"
      echo '  rc=${PIPESTATUS[0]}'
      echo "  grep -q 'Eligibility check failed: UNAVAILABLE' /tmp/agy-\$\$.out || break"
      echo '  echo "agy 503, retry $attempt in 60 s"; sleep 60'
      echo "done; rm -f /tmp/agy-\$\$.out" ;;
    *) echo "unknown tool $tool (grok or agy)" >&2; exit 2 ;;
  esac
  echo 'echo "exit $rc"'
} > "$runner"
chmod +x "$runner"
tmux new-session -d -s "$session" -x 220 -y 50 -c "$repo" "bash -c '$runner 2>&1 | tee $(printf %q "$log")'"
while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
tail -n 200 "$log" > "$runs/$name.report.md"
echo "$tool $name finished: $(tail -n 1 "$log")"
