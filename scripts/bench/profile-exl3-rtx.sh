#!/bin/bash
# Single RTX-local measurement, serialized with other cluster users.
# Usage: profile-exl3-rtx.sh peak
#        profile-exl3-rtx.sh timing|phases|ncu|nsys qwen4|glmf SNAPSHOT LAYER [ROWS...]
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
build_root=${CUTEAFD_EXL3_PROFILE_BUILD_ROOT:-$HOME/.cache/cuteafd/builds/exl3-a8-rtx}
image=${CUTEAFD_EXL3_PROFILE_IMAGE:-cuteafd-coordinator-dev:latest}
gpu=${CUTEAFD_EXL3_PROFILE_GPU:-1}
mode=${1:?expected peak, timing, phases, ncu or nsys}
shift
case "$mode" in peak|timing|phases|ncu|nsys) ;; *) exit 2 ;; esac
[[ $gpu == 1 ]] || { echo 'This runner holds gpu1.lock and requires GPU 1.' >&2; exit 2; }
python3 "$repo/scripts/build/assert-build-filesystem.py" "$build_root" "$build_root/cache" "$build_root/tmp"
mkdir -p "$build_root/cache" "$build_root/tmp"
build_root=$(cd "$build_root" && pwd)

if [[ $mode == peak ]]; then
  [[ $# == 0 ]] || exit 2
  # The real workspace path and its git metadata must both be visible inside
  # the container: submodule gitdirs in a worktree are relative to that path.
  flock "$HOME/.cache/cuteafd/build.lock" timeout --kill-after=30 180 \
    docker run --rm --entrypoint nvcc -e TMPDIR=/w/tmp \
      -v "$repo:$repo:ro" -v "$build_root:/w" "$image" \
      -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -lineinfo \
      -o /w/mma_peak "$repo/python/tools/bench/exl3_mma_peak.cu"
  command=(/w/mma_peak)
else
  [[ $# -ge 3 ]] || { echo 'expected GEOMETRY SNAPSHOT LAYER [ROWS...]' >&2; exit 2; }
  geometry=$1 snapshot=$2 layer=$3
  shift 3
  [[ $geometry == qwen4 || $geometry == glmf ]] || exit 2
  command=(python "$repo/python/tools/bench/profile_exl3_rtx.py"
    --geometry "$geometry" --snapshot "$snapshot" --layer "$layer"
    --output "/w/$geometry-$mode.json")
  if [[ $# -gt 0 ]]; then command+=(--rows "$@"); fi
  case "$mode" in
    phases) command+=(--phase-clock) ;;
    ncu)
      command+=(--profile)
      command=(ncu --profile-from-start off --replay-mode application
        --section SpeedOfLight --section InstructionStats --section WarpStateStats
        --section SourceCounters --section MemoryWorkloadAnalysis_Tables
        --force-overwrite -o "/w/$geometry-ncu" "${command[@]}") ;;
    nsys)
      command+=(--profile)
      command=(nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx
        --capture-range=cudaProfilerApi --capture-range-end=repeat
        -f true -o "/w/$geometry-nsys" "${command[@]}") ;;
  esac
fi

# Match locked2.sh: avoid monopolizing sparks.lock while a GPU1 run is long.
while true; do
  exec 9>"$HOME/.cache/cuteafd/sparks.lock" 8>"$HOME/.cache/cuteafd/gpu1.lock"
  flock 9
  if flock -w 600 8; then break; fi
  flock -u 9
  exec 8>&- 9>&-
done
name="cuteafd-exl3-rtx-$BASHPID"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
rdma link
for rate in /sys/class/infiniband/*/ports/*/rate; do echo "$rate: $(cat "$rate")"; done
nvidia-smi --query-gpu=index,power.limit,clocks.sm,clocks.mem,temperature.gpu,memory.used --format=csv
capabilities=()
launcher=(docker)
if [[ $mode == ncu ]]; then
  capabilities=(--cap-add SYS_ADMIN)
  launcher=(agent-sudo --agent-context "Run Nsight Compute with profiling-counter access in a temporary SM120 container; no host settings change" docker)
fi
# NCU needs the profiler capability in its container on hosts with restricted
# counters. No host driver settings are changed. No workers or servers start.
timeout --signal=TERM --kill-after=30 1200 "${launcher[@]}" run --rm --name "$name" \
  --gpus device=1 "${capabilities[@]}" --workdir /w --entrypoint "${command[0]}" \
  -e PYTHONDONTWRITEBYTECODE=1 -e "PYTHONPATH=$repo/third_party/sparkinfer" \
  -e "CUTEAFD_SPARKINFER_SOURCE_DIR=$repo/third_party/sparkinfer" \
  -e "CUTEAFD_SPARKINFER_LOCK_FILE=$repo/third_party/sparkinfer.lock.json" \
  -e B12X_COMPILE_CACHE_DIR=/w/cache/b12x -e CUDA_CACHE_PATH=/w/cache/cuda \
  -e TRITON_CACHE_DIR=/w/cache/triton -e TMPDIR=/w/tmp \
  -v "$(dirname "$repo"):$(dirname "$repo"):ro" -v "$build_root:/w" \
  -v "$build_root/cache:/root/.cache" -v /mnt/sparknest:/mnt/sparknest:ro \
  "$image" "${command[@]:1}"
