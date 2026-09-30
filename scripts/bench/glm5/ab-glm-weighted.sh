#!/usr/bin/env bash
# GLM 5.3 decode A/B on the Sparks: glmrt v9 vs CuteAFD arms over glmrt v9's
# weighted mix (scripts/bench/glm5/bench-glm-weighted.py), one model served at a time,
# arms interleaved (ABC, CBA, ...), plus the branch's Spark-side gates.
#
#   scripts/bench/glm5/ab-glm-weighted.sh gates  OUT BUILD      # NLL, decode agreement, verify-step table (branch build)
#   scripts/bench/glm5/ab-glm-weighted.sh image  BUILD TAG [NATIVE]  # coordinator image = p7 + BUILD's cuteafd/lib/manifest
#   scripts/bench/glm5/ab-glm-weighted.sh ab     OUT ROUNDS ARM...   # ARM: glmrt | IMAGE[+KEY=VALUE,...]
#   scripts/bench/glm5/ab-glm-weighted.sh stop                  # stop every arm's containers
#
# BUILD holds target/release/cuteafd and coord/native/{libcuteafd_native.so,
# dsv4_programs/dsv4_programs.json} (~/.cache/cuteafd/builds/glm-parity layout;
# NATIVE names another native build directory). An arm's +KEY=VALUE list adds
# run-family.sh config keys, e.g. IMAGE+L2_PREFETCH=auto or IMAGE+SPECULATOR_FP8=off:
# the L2 prefetch A/B is `ab OUT 3 IMAGE IMAGE+L2_PREFETCH=auto`.
# Needs raptor GPU0 and ostrich..kiwi free; stop other models first. The power
# cap is whatever raptor has (glmrt v9 published 33.42 at 400 W; compare arms
# at the same cap). glmrt v9 images come from ghcr by digest and are tagged to
# the local names glmrt-release's run.sh expects; rail B moved from
# 10.55.0.5-8 to 10.55.1.1-4 since v9, so the glmrt config is rewritten.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
glmrt="${GLMRT_RELEASE:-$HOME/Developer/glmrt-release}"
py="${PYTHON:-$HOME/Developer/cuteafd-work/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/mnt/sparknest/hf-home}"
hub=/mnt/sparknest/hf-home/hub
k4=models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1/snapshots/47af23347db743b4666d952e2eb48f2b01c3fede
dflash=models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c
golden="${GOLDEN:-$HOME/Developer/cuteafd-work/runs/glm-golden}"
sparks=(ostrich dodo emu kiwi)
p7=ghcr.io/tpurtell/cuteafd-coordinator:p7
p7_spark=ghcr.io/tpurtell/cuteafd-spark-expert:p7
glmrt_coord=ghcr.io/tpurtell/glmrt-5.3-coordinator@sha256:71a2f65d43bac97bde85bca8bb1e99ae86bfb5f70ea8da18d6bfb62a7e15f972
glmrt_spark=ghcr.io/tpurtell/glmrt-5.3-spark-expert@sha256:dcd90cf8a1461d0dabc40c0634e1266060cd40bb36769d7c60d1baea168d88ca

stop_all() {
  (cd "$glmrt" && ./stop.sh --profile "${glmrt_config:-$glmrt/glmrt.config}" >/dev/null 2>&1) || true
  docker rm -f cuteafd-coordinator >/dev/null 2>&1 || true
  for h in "${sparks[@]}"; do ssh "$h" "docker rm -f cuteafd-spark-expert-$h-19461 >/dev/null 2>&1 || true"; done
}

# A run-family.sh config for coordinator IMAGE (p7 Spark workers, GPU0:8200, DFlash2).
cuteafd_config() {
  local image=${1%%+*} keys= trace=$2
  [[ $1 == *+* ]] && keys=${1#*+}
  sed -e "s#^COORDINATOR_DOCKER_INFERENCE=.*#COORDINATOR_DOCKER_INFERENCE=$image#" \
      -e "s#^SPARK_EXPERT_DOCKER_INFERENCE=.*#SPARK_EXPERT_DOCKER_INFERENCE=$p7_spark#" \
      "$HOME/.cache/cuteafd/builds/sparkrun/glm-p7.config"
  echo "SPECULATION_TRACE=$trace"
  [[ -z $keys ]] || tr ',' '\n' <<< "$keys"
}

glmrt_prepare() {
  glmrt_config="$1/glmrt-ab.config"
  { grep -v '^SPARK_[0-3]_LANE_B=' "$glmrt/glmrt.config"
    for i in 0 1 2 3; do echo "SPARK_${i}_LANE_B=10.55.1.$((i + 1))"; done; } > "$glmrt_config"
  docker image inspect glrmt-coordinator >/dev/null 2>&1 ||
    { docker pull "$glmrt_coord" && docker tag "$glmrt_coord" glrmt-coordinator; }
  for h in "${sparks[@]}"; do
    ssh "$h" "docker image inspect glrmt-spark-expert >/dev/null 2>&1 || \
      { docker pull '$glmrt_spark' && docker tag '$glmrt_spark' glrmt-spark-expert; }" &
  done
  wait
}

# Serve ARM; prints its base URL.
launch() {
  local arm=$1 out=$2 tag=$3
  stop_all
  nest drop-caches $(printf -- '--host %s ' "${sparks[@]}") >/dev/null 2>&1 || true
  if [[ $arm == glmrt ]]; then
    (cd "$glmrt" && ./run.sh --profile "$glmrt_config" --restart) > "$out/$tag.launch.log" 2>&1
    echo http://127.0.0.1:8000
  else
    cuteafd_config "$arm" "$out/$tag.trace.jsonl" > "$out/$tag.config"
    "$repo/scripts/launch/run-family.sh" --config "$out/$tag.config" --restart > "$out/$tag.launch.log" 2>&1
    echo http://127.0.0.1:8200
  fi
}

logs() {
  local arm=$1 out=$2 tag=$3
  if [[ $arm == glmrt ]]; then docker logs glrmt-coordinator > "$out/$tag.serve.log" 2>&1 || true
  else docker logs cuteafd-coordinator 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$out/$tag.serve.log" || true; fi
}

case "${1:-}" in
  stop) stop_all ;;
  image)
    build=$2 tag=$3 native=${4:-$2/coord/native} ctx="$(mktemp -d)"
    cp "$build/target/release/cuteafd" "$native/libcuteafd_native.so" \
      "$native/dsv4_programs/dsv4_programs.json" "$ctx/"
    printf 'FROM %s\nCOPY cuteafd /opt/cuteafd/bin/cuteafd\nCOPY libcuteafd_native.so /opt/cuteafd/lib/libcuteafd_native.so\nCOPY dsv4_programs.json /opt/cuteafd/share/PROGRAMS.json\n' \
      "$p7" > "$ctx/Dockerfile"
    docker build -q -t "$tag" "$ctx" && rm -rf "$ctx" ;;
  gates)
    out=$2 build=$3; mkdir -p "$out"
    stop_all
    cuteafd_config "$p7" "" | grep -v '^SPECULATION_TRACE' > "$out/workers.config"
    # p7 Spark workers only: run-family.sh starts them, then its coordinator is removed.
    "$repo/scripts/launch/run-family.sh" --config "$out/workers.config" --restart > "$out/workers.log" 2>&1
    docker rm -f cuteafd-coordinator >/dev/null
    peers=10.55.0.1:19461,10.55.0.2:19461,10.55.0.3:19461,10.55.0.4:19461
    golden_run() {
      docker run --rm --gpus device=0 --network host --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband \
        -v "$build:$build:ro" -v "$golden:$golden:ro" -v /mnt/sparknest:/mnt/sparknest:ro -e RUST_LOG=warn \
        --entrypoint bash "$p7" -c "export LD_LIBRARY_PATH=\$(python -m cutlass.cute.export.aot_config --libdir 2>/dev/null):\$LD_LIBRARY_PATH; \
        $build/target/release/cuteafd glm-golden --snapshot $hub/$k4 --native-lib $build/coord/native/libcuteafd_native.so \
        --manifest $build/coord/native/dsv4_programs/dsv4_programs.json --peers $peers --golden $golden $*"
    }
    # Gates vs runs/glm-golden (p7 reference: NLL 2.4629, golden 2.4677; decode top-1 vs golden).
    golden_run --nll | tee "$out/nll.txt"
    golden_run --prefill 1024 --step-rows 8 --no-layer-compare | tee "$out/decode-agreement.txt"
    # The DFlash2 policy's K4_TP4_STEP_MS: verify steps by rows, then the drafter.
    golden_run --draft "$hub/$dflash" --bench-verify 64 --bench-rows 1,2,3,4,5,6,7,8,10,12,16,24,32,48,64 \
      | tee "$out/verify-table.txt"
    golden_run --bench-verify 8 --bench-context 6000 | tee "$out/verify-6k.txt"
    stop_all ;;
  ab)
    out=$2 rounds=$3; shift 3; arms=("$@"); mkdir -p "$out"
    for arm in "${arms[@]}"; do [[ $arm == glmrt ]] && glmrt_prepare "$out"; done
    for round in $(seq 1 "$rounds"); do
      order=("${arms[@]}"); (( round % 2 == 0 )) && order=($(printf '%s\n' "${arms[@]}" | tac))
      for arm in "${order[@]}"; do
        tag="$(echo "$arm" | tr '/:@+=,' '______')-r$round"
        url="$(launch "$arm" "$out" "$tag")"
        # Warm-up: lazily allocated workspaces and graphs, then the timed corpus.
        "$py" "$repo/scripts/bench/glm5/bench-glm-weighted.py" --base-url "$url" --repeats 1 --warmup 3 \
          --output "$out/$tag.warmup.json" --label "$tag-warmup" > /dev/null
        "$py" "$repo/scripts/bench/glm5/bench-glm-weighted.py" --base-url "$url" --repeats 5 \
          --output "$out/$tag.json" --label "$tag" | tail -1 | sed "s/^/$tag: /"
        if [[ $arm == glmrt && $round == 1 ]]; then
          # glmrt's own server-timed number (its published metric) once.
          "$py" "$repo/scripts/bench/glm5/bench-glm-weighted.py" --base-url "$url" --repeats 5 --timing server \
            --output "$out/$tag.server.json" --label "$tag-server" | tail -1 | sed "s/^/$tag server: /"
        fi
        logs "$arm" "$out" "$tag"
        [[ -s "$out/$tag.trace.jsonl" ]] && "$py" "$repo/scripts/qualify/glm5/glm-draft-trace.py" "$out/$tag.trace.jsonl" \
          > "$out/$tag.policy.txt" 2>&1 || true
      done
    done
    stop_all
    "$py" - "$out" <<'PY'
import json, sys, glob, collections, statistics
by = collections.defaultdict(list)
for f in sorted(glob.glob(f'{sys.argv[1]}/*-r*.json')):
    if f.endswith(('.warmup.json', '.server.json')): continue
    s = json.load(open(f))['summary']; by[s['label'].rsplit('-r', 1)[0]].append(s)
for arm, runs in by.items():
    w = [r['weighted_tps'] for r in runs]
    cases = {k: statistics.median(r['case_tps'][k] for r in runs) for k in runs[0]['case_tps']}
    print(f"{arm}: weighted median {statistics.median(w):.2f} {[round(x, 2) for x in w]} | "
          + ' '.join(f'{k} {v:.1f}' for k, v in cases.items()))
PY
    ;;
  *) sed -n 2,22p "$0"; exit 2 ;;
esac
