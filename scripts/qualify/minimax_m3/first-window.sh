#!/usr/bin/env bash
# Prints the complete qualification schedule by default. No device probe in dry mode.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
self="$root/scripts/qualify/minimax_m3/first-window.sh"
checks="$root/scripts/qualify/minimax_m3/first-window-check.py"
python="${MINIMAX_PYTHON:-$root/.venv/bin/python}"
[[ -x "$python" ]] || python=python3
run=0 card=all internal='' stage_config=''
usage() {
  cat <<'EOF'
Usage: first-window.sh [--run] [--card all|pro6000|5090]

Default: print the ordered schedule, then check local files only. No Docker,
SSH, CUDA, device probes, engine launches or hardware locks are invoked.
--run requires CUTEAFD_MINIMAX_HARDWARE_WINDOW=authorized in this shell.
Use --card to qualify the physically available card; other cards stay unverified.
Every command and numerical gate must pass before the next step starts.

Required environment:
  MINIMAX_SNAPSHOT      Full pinned release snapshot; header-only files are refused.
  MINIMAX_GOLDEN        Complete Python --activation served-cache --layers 3 5 output.
                       Every window needs >=66 tokens for 63/64/65 prefix checks;
                       one needs >=2177 tokens to exercise MSA pruning.
  MINIMAX_CONFIG_DIR    Private complete copies of the six example configurations.
                       Use explicit rank HOST/LANE_A entries and EXPERT_PORT;
                       TP2/3 prefix TP4. --card all uses the same rank pool.
  MINIMAX_PLAN_BIN      Existing CPU-capable cuteafd executable for plan admission.
  MINIMAX_RUN_ID        Fresh identifier (1..20 letters, digits, dot, dash, underscore).
  MINIMAX_GPU_PRO6000   Explicit local device index 0 or 1, when selected.
  MINIMAX_GPU_5090      Explicit local device index 0 or 1, when selected.
Run-only environment:
  MINIMAX_BUILD_SEED    Explicit member of the selected TP4 rank list.
  MINIMAX_URL           API URL matching MINIMAX_ADDR; use an unused local port.
  MINIMAX_ADDR          Server listen address on that unused port.
Optional: MINIMAX_PYTHON, MINIMAX_TIMEOUT (default 7200 seconds per step),
MINIMAX_LOCK_WAIT (default 21600 seconds). No dependencies are installed.
Prepare the reference with python/reference/families/minimax_m3/golden.py
--snapshot SNAPSHOT --windows SET.json --out REFERENCE --activation served-cache
--layers 3 5; the reference's own prefix gate must pass before this runbook.

Outputs stay under the minimax-qualify build task. Each fresh run has isolated
WIP and serving instance names. Existing output or containers cause refusal.
Interrupted output is retained; choose a new run identifier after inspection.
The golden command itself supports --resume for a separately reviewed rerun.
EOF
}
while (($#)); do
  case "$1" in
    --run) run=1; shift ;;
    --card) card="${2:?--card needs a value}"; shift 2 ;;
    --internal-serve) internal=serve; stage_config="${2:?configuration required}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "first-window: unknown argument $1" >&2; exit 2 ;;
  esac
done
case "$card" in all|pro6000|5090) ;; *) echo 'first-window: invalid card' >&2; exit 2 ;; esac
if ((run)) || [[ -n "$internal" ]]; then
  [[ "${CUTEAFD_MINIMAX_HARDWARE_WINDOW:-}" == authorized ]] || {
    echo 'first-window: --run requires CUTEAFD_MINIMAX_HARDWARE_WINDOW=authorized' >&2; exit 2;
  }
fi

show() { printf '  '; printf '%q ' "$@"; printf '\n'; }
step() { printf '\n%s\n  PASS: %s\n' "$1" "$2"; }
schedule() {
  local selected tp
  for selected in pro6000 5090; do
    [[ "$card" == all || "$card" == "$selected" ]] || continue
    step "$selected: 1. Plan all configurations" 'placement supported; all memory budgets fit'
    show run.sh --config "CONFIG_DIR/minimax-m31-flash-nvfp4-$selected-tp4.config" --dry-run
    step "$selected: 2. Build WIP coordinator and TP2/TP3/TP4 experts" 'sealed artifacts; matching source; native NVFP4 W4A4 and MXFP8'
    show wip.sh --slot qualification --role both --config CONFIG --seed BUILD_SEED
    step "$selected: 3. Single-layer diagnostic on local RTX, no Sparks" 'MSA selection and router ids equal; all intermediates finite and within tolerance'
    show minimax-golden --last-layer 3 --expert-layers 3 --window DIAGNOSTIC_WINDOW
    show python/reference/families/minimax_m3/diagnostic.py --snapshot SNAPSHOT --dump DUMP
    step "$selected: 4. Golden through layer 5 with local RTX experts, no Sparks" 'cosine >=0.99; relative L2 <=0.05; byte-exact prefix restore (partial-model diagnostic, no NLL claim)'
    show minimax-golden --last-layer 5 --expert-layers 3,4,5 --check-prefix
    for tp in 4 2 3; do
      step "$selected: Spark TP$tp full golden" 'top-1 >=90%; KL <=0.06 nat; byte-exact prefix restore'
      show minimax-golden --peers '{peers}' --check-prefix
      step "$selected: TP$tp serve and api-smoke" 'HTTP success; nonempty completion; no API error; owned containers stopped before lock release'
      show run.sh --wip qualification --config "CONFIG_DIR/minimax-m31-flash-nvfp4-$selected-tp$tp.config"
      show scripts/qualify/api-smoke.sh URL MiniMaxAI/MiniMax-M3.1-Flash
    done
  done
  printf '\nFull goldens gate the model; subset diagnostics do not establish full-model fidelity.\n'
}

if [[ -z "$internal" ]]; then
  schedule
  "$python" "$checks" preflight "$card"
  ((run)) || exit 0
fi

[[ -n "${MINIMAX_BUILD_SEED:-}" && -n "${MINIMAX_URL:-}" && -n "${MINIMAX_ADDR:-}" ]] || {
  echo 'first-window: --run needs MINIMAX_BUILD_SEED, MINIMAX_URL and MINIMAX_ADDR' >&2; exit 2;
}
"$python" "$checks" runtime "$card"
for command in docker ssh nvidia-smi curl jq timeout; do command -v "$command" >/dev/null; done
timeout_s="${MINIMAX_TIMEOUT:-7200}"
wait_s="${MINIMAX_LOCK_WAIT:-21600}"
[[ "$timeout_s" =~ ^[1-9][0-9]*$ && "$wait_s" =~ ^[1-9][0-9]*$ ]] || exit 2
build="$HOME/.cache/cuteafd/builds/minimax-qualify"
output="$build/first-window/$MINIMAX_RUN_ID"
instance="minimax-qualify-$MINIMAX_RUN_ID"
export WIP_INSTANCE="$instance" WIP_ROOT="$build/wip-$MINIMAX_RUN_ID"
export TMPDIR="$build/tmp" CARGO_TARGET_DIR="$build/target" CUTEAFD_BUILD_CACHES=off
export CUTEAFD_PLAN_BIN="$MINIMAX_PLAN_BIN"

serve_cleanup() {
  local host status=0 name
  if docker container inspect "$container" >/dev/null 2>&1; then
    timeout 120 docker stop "$container" && docker rm "$container" || status=1
  fi
  for host in "${serve_hosts[@]}"; do
    name="cuteafd-spark-expert-$host-$expert_port"
    timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
      "if docker container inspect '$name' >/dev/null 2>&1; then docker stop '$name' && docker rm '$name'; fi" || status=1
  done
  return "$status"
}
if [[ "$internal" == serve ]]; then
  [[ -n "${CUTEAFD_HW_LOCKS_HELD:-}" && "$stage_config" == "$output/"* ]] || {
    echo 'first-window: internal serve requires the hardware lock and owned configuration' >&2; exit 2;
  }
  container="cuteafd-coordinator-$instance"
  if docker container inspect "$container" >/dev/null 2>&1; then
    echo 'first-window: serving instance already exists; refusing ownership' >&2; exit 2
  fi
  expert_port="$("$python" "$checks" expert-port "$stage_config")"
  serve_host_csv="$("$python" "$checks" hosts "$stage_config")"
  IFS=, read -r -a serve_hosts <<<"$serve_host_csv"
  for host in "${serve_hosts[@]}"; do
    name="cuteafd-spark-expert-$host-$expert_port"
    timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
      "if docker container inspect '$name' >/dev/null 2>&1; then exit 1; fi"
  done
  trap serve_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  "$root/run.sh" --wip qualification --config "$stage_config"
  deadline=$((SECONDS + timeout_s))
  until curl --max-time 5 -fsS "$MINIMAX_URL/health" >/dev/null; do
    ((SECONDS < deadline)) || { echo 'first-window: readiness timeout' >&2; exit 1; }
    sleep 2
  done
  "$root/scripts/qualify/api-smoke.sh" "$MINIMAX_URL" MiniMaxAI/MiniMax-M3.1-Flash
  "$python" "$checks" api "$MINIMAX_URL"
  exit 0
fi

"$python" "$root/scripts/build/assert-build-filesystem.py" "$build" "$WIP_ROOT"
[[ ! -e "$output" ]] || { echo 'first-window: run output already exists' >&2; exit 2; }
mkdir -p "$output" "$TMPDIR"
build_hosts='' build_owned=0 built=0
build_cleanup() {
  local host status=0 name
  ((build_owned)) || return 0
  name="cuteafd-coordinator-wip-$instance"
  if docker container inspect "$name" >/dev/null 2>&1; then timeout 120 docker stop "$name" || status=1; fi
  name="cuteafd-spark-expert-wip-$instance"
  IFS=, read -r -a owned_hosts <<<"$build_hosts"
  for host in "${owned_hosts[@]}"; do
    timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
      "if docker container inspect '$name' >/dev/null 2>&1; then docker stop '$name'; fi" || status=1
  done
  return "$status"
}
trap build_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

golden() {
  local name="$1" cfg="$2" gpu="$3" peers="$4"; shift 4
  local budgets gpu_budget worker_budget
  budgets="$("$python" "$checks" budgets "$cfg")"
  read -r gpu_budget worker_budget <<<"$budgets"
  local -a command=("$python" "$root/scripts/bench/golden-run.py" --task minimax-qualify
    --output-dir "$output/$name" --family minimax --model "$MINIMAX_SNAPSHOT"
    --arm "native=$instance:qualification" --config "$cfg" --gpus "$gpu"
    --timeout "$timeout_s" --wait "$wait_s" --preflight "$preflight")
  if [[ -n "$peers" ]]; then command+=(--spark-hosts "$peers" --expert-layers 3-59 --worker-budget-bytes "$worker_budget"); fi
  if [[ "$name" == *-layer3 ]]; then
    command+=(--env 'CUTEAFD_MINIMAX_DUMP_DIR={out}/diagnostic' --env CUTEAFD_MINIMAX_DUMP_LAYER=3)
  fi
  command+=(-- --snapshot '{snapshot}' --golden "$MINIMAX_GOLDEN" --save-logits '{out}/scores'
    --prefill-rows 128 --coordinator-gpu-budget-gib "$gpu_budget")
  if [[ "$name" != *-layer3 ]]; then command+=(--check-prefix); fi
  if [[ "$name" == *-layer3 ]]; then command+=(--window "$diagnostic_window"); fi
  if [[ -n "$peers" ]]; then command+=(--peers '{peers}'); fi
  command+=("$@")
  show "${command[@]}"
  "${command[@]}"
  local kind=full
  if [[ "$name" == *-layer3 ]]; then kind=diagnostic; elif [[ "$name" == *-layer5 ]]; then kind=partial; fi
  "$python" "$checks" result "$output/$name/native/r1/scores/summary.json" --kind "$kind"
}

printf -v preflight '%q %q preflight %q' "$python" "$checks" "$card"
diagnostic_window="$("$python" "$checks" window "$MINIMAX_GOLDEN")"
for selected in pro6000 5090; do
  [[ "$card" == all || "$card" == "$selected" ]] || continue
  gpu_key="MINIMAX_GPU_${selected^^}"; gpu="${!gpu_key}"
  for tp in 4 2 3; do
    cfg="$output/$selected-tp$tp.config"
    cp "$MINIMAX_CONFIG_DIR/minimax-m31-flash-nvfp4-$selected-tp$tp.config" "$cfg"
    printf '\nMODEL_ID=MiniMaxAI/MiniMax-M3.1-Flash\nINSTANCE=%s\nWIP_INSTANCE=%s\nWIP_ROOT="%s"\nADDR=%s\n' \
      "$instance" "$instance" "$WIP_ROOT" "$MINIMAX_ADDR" >>"$cfg"
    "$root/run.sh" --config "$cfg" --set "MODEL_ID=$MINIMAX_SNAPSHOT" --dry-run >"$output/$selected-tp$tp-plan.json"
    "$python" "$checks" plan "$output/$selected-tp$tp-plan.json"
  done
  device="$(nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader)"
  if [[ "$selected" == 5090 && "$device" != *5090* ]] || [[ "$selected" == pro6000 && "$device" != *PRO*6000* ]]; then
    echo 'first-window: selected physical GPU does not match the requested card' >&2; exit 2
  fi
  cfg="$output/$selected-tp4.config"
  build_hosts="$("$python" "$checks" hosts "$cfg")"
  [[ ",$build_hosts," == *",$MINIMAX_BUILD_SEED,"* ]] || {
    echo 'first-window: build seed must be one of the configured TP4 ranks' >&2; exit 2;
  }
  if ((built == 0)); then
  if docker container inspect "cuteafd-coordinator-wip-$instance" >/dev/null 2>&1; then
    echo 'first-window: WIP instance already exists' >&2; exit 2
  fi
  IFS=, read -r -a owned_hosts <<<"$build_hosts"
  for host in "${owned_hosts[@]}"; do
    timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" \
      "if docker container inspect 'cuteafd-spark-expert-wip-$instance' >/dev/null 2>&1; then exit 1; fi"
  done
  build_owned=1
  CUTEAFD_WIP_SPARK_TP_ROLES='tp2;tp3' CUTEAFD_WIP_GENERIC_SPARK_COUNTS='minimax=2;minimax=3;minimax=4' \
    timeout "$timeout_s" "$root/wip.sh" --slot qualification --role both --config "$cfg" --seed "$MINIMAX_BUILD_SEED"
  build_cleanup
  build_owned=0
  built=1
  fi
  golden "$selected-layer3" "$cfg" "$gpu" '' --last-layer 3 --expert-layers 3
  "$python" "$root/python/reference/families/minimax_m3/diagnostic.py" --snapshot "$MINIMAX_SNAPSHOT" \
    --dump "$output/$selected-layer3/native/r1/diagnostic" --out "$output/$selected-layer3/diagnostic.json"
  golden "$selected-layer5" "$cfg" "$gpu" '' --last-layer 5 --expert-layers 3,4,5
  for tp in 4 2 3; do
    cfg="$output/$selected-tp$tp.config"
    peers="$("$python" "$checks" hosts "$cfg")"
    golden "$selected-tp$tp" "$cfg" "$gpu" "$peers"
    locks="$("$python" "$checks" locks "$cfg" --gpu "$gpu")"
    "$python" "$root/scripts/hw/hwlock.py" --locks "$locks" --wait "$wait_s" --preflight "$preflight" -- \
      timeout "$timeout_s" "$self" --card "$card" --internal-serve "$cfg"
  done
done
printf '\nSelected configurations passed; evidence: %s\n' "$output"
