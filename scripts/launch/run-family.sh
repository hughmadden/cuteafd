#!/usr/bin/env bash
# Launch a DeepSeek V4, GLM 5.x, GLM 5.3 Flash, MiMo V2 or Qwen 3.8 Flash Next
# checkpoint (the family's serve command on one RTX, routed experts on the first
# SPARK_COUNT Sparks) from the release images named in the config. ./run.sh
# starts this for every family but DeepSeek V4.1; the family comes from the
# snapshot's config.json (scripts/lib/checkpoint-family.py), or --family.
# Containers use run.sh's names, so ./stop.sh stops them.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$repo_root/cuteafd.config"
restart=0
family=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?--config requires FILE}"; shift 2 ;;
    --family) family="${2:?--family requires ID}"; shift 2 ;;
    --restart) restart=1; shift ;;
    *) echo "usage: $0 [--config FILE] [--family ID] [--restart]" >&2; exit 2 ;;
  esac
done
# Plain KEY=VALUE lines; the launch reads only the keys below.
declare -A cfg
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[A-Z_0-9]+$ ]] && cfg[$key]="$value"
done < <(grep -E '^[A-Z_0-9]+=' "$config")
get() { printf '%s' "${cfg[$1]:-${2:-}}"; }
# key NEW OLD [DEFAULT]: a renamed key; the pre-rename spelling works for one
# release with a warning.
key() {
  if [[ -n "${cfg[$1]:-}" ]]; then printf '%s' "${cfg[$1]}"
  elif [[ -n "${cfg[$2]:-}" ]]; then echo "warning: config key $2 is deprecated; use $1" >&2; printf '%s' "${cfg[$2]}"
  else printf '%s' "${3:-}"; fi
}
model="$(get MODEL_ID)"
revision="$(get MODEL_REVISION)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
hub="$(readlink -f "$hf_home/hub")"
root="$hub/models--${model//\//--}"
[[ -n "$revision" ]] || revision="$(<"$root/refs/main")"
snapshot="/root/.cache/huggingface/hub/models--${model//\//--}/snapshots/$revision"
[[ -d "$root/snapshots/$revision" ]] || { echo "missing snapshot $model@$revision" >&2; exit 1; }
# Family, the serving command and the layers with routed experts.
described="$(python3 "$repo_root/scripts/lib/checkpoint-family.py" "$root/snapshots/$revision/config.json")" || exit 2
read -r detected model_type first_layer last_layer <<<"$described"
[[ -z "$family" || "$family" == "$detected" ]] ||
  { echo "--family $family does not match the checkpoint ($detected, model_type $model_type)" >&2; exit 2; }
family="$detected"
case "$family" in
  deepseek_v4) serve=serve-dsv4 ;;
  glm5) serve=serve-glm ;;
  glm5_flash) serve=serve-glmf ;;
  mimo_v2) serve=serve-mimo ;;
  qwen4) serve=serve-qwen4 ;;
  *) echo "run-family.sh serves DeepSeek V4, GLM 5.x, GLM 5.3 Flash, MiMo V2 and Qwen 3.8 checkpoints, not $family (./run.sh serves DeepSeek V4.1)" >&2; exit 2 ;;
esac
layer_args="--first-layer $first_layer"
[[ "$last_layer" == -1 ]] || layer_args+=" --last-layer $last_layer"
# Snapshot of a model id (and optional revision) inside the containers.
snapshot_of() {
  local id="$1" rev="$2" dir="$hub/models--${1//\//--}"
  [[ -n "$rev" ]] || rev="$(<"$dir/refs/main")"
  [[ -d "$dir/snapshots/$rev" ]] || { echo "missing snapshot $id@$rev" >&2; return 1; }
  printf '%s' "/root/.cache/huggingface/hub/models--${id//\//--}/snapshots/$rev"
}
# SPECULATOR picks the drafter (default off):
#   dflash2  GLM 5.x / GLM 5.3 Flash: the DFlash2 checkpoint SPECULATOR_MODEL_ID
#            (e.g. incoai/GLM-5.3-DFlash2, incoai/GLM-5.3-Flash-DFlash2);
#            MiMo V2.6 Pro: the snapshot's own dflash/ drafter unless
#            SPECULATOR_MODEL_ID names one (V2.6 Pro needs SPARK_COUNT=6)
#   mtp      MiMo V2 Flash, Qwen 3.8: the checkpoint's native MTP layers,
#            SPECULATOR_DEPTH drafts (default 1)
#   dspark   DeepSeek V4
# SPECULATOR_FP8=off drafts in BF16. Pre-rename keys (DRAFT_MODEL_ID, DFLASH,
# MTP, DSPARK, DRAFT_FP8) still work for one release.
speculator="$(get SPECULATOR)"
if [[ -z "$speculator" ]]; then
  if [[ -n "$(get DRAFT_MODEL_ID)" || "$(get DFLASH off)" == on ]]; then speculator=dflash2
  elif [[ "$(get MTP 0)" != 0 ]]; then speculator=mtp
  elif [[ $family == deepseek_v4 && "$(get DSPARK off)" == on ]]; then speculator=dspark
  else speculator=off; fi
  [[ $speculator == off ]] || echo "warning: DRAFT_MODEL_ID/DFLASH/MTP/DSPARK are deprecated; use SPECULATOR=$speculator" >&2
fi
case "$family:$speculator" in
  qwen4:mtp)
    # The MTP layer's experts run on the coordinator: only with local experts (SPARK_COUNT=0).
    [[ "$(get SPARK_COUNT 4)" == 0 ]] ||
      { echo "SPECULATOR=mtp for Qwen needs SPARK_COUNT=0 (local experts); the Spark ranks do not serve the MTP layer's experts" >&2; exit 2; } ;;
  *:off|glm5:dflash2|glm5_flash:dflash2|mimo_v2:dflash2|mimo_v2:mtp|deepseek_v4:dspark) ;;
  *) echo "SPECULATOR=$speculator does not apply to $family" >&2; exit 2 ;;
esac
draft_args=()
family_args=()
dspark_args=()
case "$speculator" in
  dflash2)
    drafter="$(key SPECULATOR_MODEL_ID DRAFT_MODEL_ID)"
    if [[ -n "$drafter" ]]; then
      draft_snapshot="$(snapshot_of "$drafter" "$(key SPECULATOR_MODEL_REVISION DRAFT_MODEL_REVISION)")" || exit 1
      draft_args=(--draft "$draft_snapshot")
    elif [[ $family == mimo_v2 ]]; then
      draft_args=(--draft "$snapshot")
    else
      echo "SPECULATOR=dflash2 needs SPECULATOR_MODEL_ID (a DFlash2 checkpoint)" >&2; exit 2
    fi ;;
  mtp) family_args+=(--mtp "$(key SPECULATOR_DEPTH MTP 1)") ;;
  dspark) dspark_args=(--dspark) ;;
esac
# SPECULATOR_DRAFTS: adaptive (default) or a fixed draft count per cycle
# (DFlash2 on GLM 5.x, GLM 5.3 Flash and MiMo V2), for policy A/B runs.
drafts="$(get SPECULATOR_DRAFTS adaptive)"
if [[ "$drafts" != adaptive ]]; then
  [[ "$drafts" =~ ^[0-9]+$ && $speculator == dflash2 ]] ||
    { echo "SPECULATOR_DRAFTS must be adaptive or a draft count, with SPECULATOR=dflash2" >&2; exit 2; }
  draft_args+=(--draft-fixed "$drafts")
fi
# Prefix cache (MiMo, GLM 5.3, GLM 5.3 Flash, Qwen 3.8, DeepSeek V4): PREFIX_CACHE_ENTRIES
# snapshots per bank (prompts, turns; 0 = off), HOST_CACHE_BYTES of pinned host memory for
# snapshots the device evicts (e.g. 64GiB; 0 = off). POOL_TOKENS: paged KV tokens shared by
# live sequences and retained snapshots.
case $family in
  mimo_v2|glm5|glm5_flash|qwen4|deepseek_v4)
    family_args+=(--prefix-cache-entries "$(get PREFIX_CACHE_ENTRIES 20)")
    [[ "$(get HOST_CACHE_BYTES 0)" == 0 ]] || family_args+=(--host-cache-bytes "$(get HOST_CACHE_BYTES)") ;;
esac
if [[ $family == mimo_v2 ]]; then
  family_args+=(--pool-tokens "$(get POOL_TOKENS 131072)")
  # PREFIX_PARTIAL=on: V4.1-style partial reuse (approximate; off = exact restores only).
  family_args+=(--prefix-partial "$(get PREFIX_PARTIAL off)")
  # KV_CACHE: int8 (the engine default: 8-bit full-attention records with FP32 scales per 32
  # dims; SWA rings stay BF16) or bf16.
  [[ -z "$(get KV_CACHE)" ]] || family_args+=(--kv-cache "$(get KV_CACHE)")
  # DECODE_GRAPHS=on: decode/verify steps replay per-layer CUDA graph segments captured at startup
  # (default off: neutral against the host-driven Spark exchange, 0.4-1.2 GiB of graphs).
  case "$(get DECODE_GRAPHS)" in
    "") ;;
    on) family_args+=(--decode-graphs true) ;;
    off) family_args+=(--decode-graphs false) ;;
    *) echo "DECODE_GRAPHS must be on or off" >&2; exit 2 ;;
  esac
fi
[[ ! $family =~ ^(glm5|qwen4|deepseek_v4)$ || -z "$(get POOL_TOKENS)" ]] || family_args+=(--pool-tokens "$(get POOL_TOKENS)")
# GLM 5.3 Flash: the MLA, dense and shared-expert projections are FP8 only,
# from the official FP8 release (GLM5_FLASH_FP8_MODEL_ID; "off" quantizes the
# BF16 checkpoint's at load); KDA projections get per-row FP8 decode copies
# (GLM5_FLASH_KDA_FP8: row128, channel or off) and optionally an FP8 LM head
# (GLM5_FLASH_FP8_HEAD=on); its MLA pools hold POOL_TOKENS tokens (a key every
# family with a paged KV pool reads). GLM5_FLASH_FP8_PREFILL lists the prefill
# projections that run W8A8 (E4M3 activations per 128-K block): unset = the
# engine default mla,ffn (the official FP8 tensors), a list of
# mla,ffn,kda-in,kda-o / all, or off (MLA/FFN W8A16, KDA BF16). The GLMF_*
# spellings still work for one release.
if [[ $family == glm5_flash ]]; then
  fp8_model="$(key GLM5_FLASH_FP8_MODEL_ID GLMF_FP8_MODEL_ID zai-org/GLM-5.3-Flash)"
  if [[ "$fp8_model" != off ]]; then
    fp8_snapshot="$(snapshot_of "$fp8_model" "$(key GLM5_FLASH_FP8_MODEL_REVISION GLMF_FP8_MODEL_REVISION)")" || exit 1
    family_args+=(--fp8-decode --fp8-snapshot "$fp8_snapshot")
  fi
  family_args+=(--kda-fp8 "$(key GLM5_FLASH_KDA_FP8 GLMF_KDA_FP8 row128)" --pool-tokens "$(get POOL_TOKENS 65536)")
  [[ "$(key GLM5_FLASH_FP8_HEAD GLMF_FP8_HEAD off)" != on ]] || family_args+=(--fp8-head)
  fp8_prefill="$(key GLM5_FLASH_FP8_PREFILL GLMF_FP8_PREFILL)"
  case "$fp8_prefill" in
    "") ;;
    off) family_args+=(--fp8-prefill none) ;;
    *) family_args+=(--fp8-prefill "$fp8_prefill") ;;
  esac
fi
# COPY_DRAFTS=off: decode without copy-window drafts (serve-glm, serve-glmf, serve-mimo, serve-qwen4).
if [[ "$(get COPY_DRAFTS on)" == off ]]; then
  [[ $serve != serve-dsv4 ]] || { echo "COPY_DRAFTS applies to GLM, MiMo and Qwen checkpoints" >&2; exit 2; }
  family_args+=(--no-copy-drafts)
fi
# DECODE_SHARE: the share of the time running requests keep while prompts
# prefill (the engine's default 0.2; 0 prefills whole prompts before the next
# step). Keys left unset pass nothing (images older than the options run).
[[ -z "$(get DECODE_SHARE)" ]] || family_args+=(--decode-share "$(get DECODE_SHARE)")
# GLM, GLM Flash, MiMo, Qwen: L2_PREFETCH (off, auto = 3/4 of the L2, or MiB;
# unset: auto for GLM 5.3 and GLM 5.3 Flash, off for MiMo and Qwen) pulls the
# next layer's weights into L2 during each one-lane decode step's Spark exchange; FP8_SCALES (amax, pow2, best) is the scale rule of the FP8
# copies made from BF16 weights at load.
if [[ $serve != serve-dsv4 ]]; then
  [[ -z "$(get L2_PREFETCH)" ]] || family_args+=(--l2-prefetch "$(get L2_PREFETCH)")
  [[ -z "$(get FP8_SCALES)" ]] || family_args+=(--fp8-scales "$(get FP8_SCALES)")
  if [[ ${#draft_args[@]} -gt 0 && "$(key SPECULATOR_FP8 DRAFT_FP8 on)" == off ]]; then family_args+=(--draft-fp8 false); fi
fi
# Opt-in engine switches set in the launching environment reach the coordinator
# (CUTEAFD_SPARK_DEVICE=1: decode/verify waves on the device-driven Spark exchange).
switch_args=()
for switch_name in CUTEAFD_SPARK_DEVICE CUTEAFD_SPARK_DEVICE_STEP CUTEAFD_SPARK_WRITE CUTEAFD_MIMO_DECODE_GRAPHS; do
  [[ -z "${!switch_name:-}" ]] || switch_args+=(-e "$switch_name=${!switch_name}")
done
# SERVED_MODEL_ID: the public model id (default: the checkpoint's Hugging Face id).
served_args=()
served="$(get SERVED_MODEL_ID)"
[[ -z "$served" ]] || served_args=(--model-id "$served")
coordinator_image="$(get COORDINATOR_DOCKER_INFERENCE)"
# SPECULATION_TRACE=/abs/host/file.jsonl: the per-cycle speculation trace
# (CUTEAFD_SPECULATION_TRACE, written by serve-glm and serve-qwen4; read by
# scripts/qualify/glm5/glm-draft-trace.py and qualify/qwen4/qwen4-draft-trace.py).
# COORDINATOR_TRACE is its pre-rename key; images older than the rename read
# the family variables, which are set too for one release.
trace_args=()
trace="$(key SPECULATION_TRACE COORDINATOR_TRACE)"
if [[ -n "$trace" ]]; then
  mkdir -p "$(dirname "$trace")"
  trace_args=(-v "$(dirname "$trace"):$(dirname "$trace")" -e "CUTEAFD_SPECULATION_TRACE=$trace"
    -e "CUTEAFD_GLM_TRACE=$trace" -e "CUTEAFD_QWEN4_TRACE=$trace")
fi
spark_image="$(get SPARK_EXPERT_DOCKER_INFERENCE)"
port="$(get EXPERT_PORT 19441)"
addr="$(get ADDR 0.0.0.0:8000)"
ranks="$(get SPARK_COUNT 4)"
budget="$(get SPARK_DEVICE_BUDGET_BYTES 107374182400)"
gpu="$(get COORDINATOR_GPU 0)"
# Two coordinator GPUs (RTX_GPUS=auto/2 with COORDINATOR_GPU as V4.1's two-RTX config picks
# the other card, or an explicit COORDINATOR_GPUS=0,1): families with a
# head split (MiMo V2.6 Pro, GLM 5.x, DeepSeek V4) split every layer's attention heads and dense /
# shared-expert MLPs over both by default (COORDINATOR_SPLIT=auto or heads), one hidden
# all-reduce per layer over peer memory; experts, router, head and drafter stay on the
# first GPU. COORDINATOR_SPLIT=off serves from the first GPU alone. Families without a
# split use the first GPU (the serve command logs it for MiMo V2 Flash). The container
# sees both GPUs in host order. COORDINATOR_SPLIT_GPU names the second GPU when only
# COORDINATOR_GPU is set (default the other of 0/1).
coordinator_gpus="$(get COORDINATOR_GPUS)"
if [[ -z "$coordinator_gpus" ]]; then
  coordinator_gpus="$gpu"
  case "$(get RTX_GPUS auto)" in
    1) ;;
    2|auto)
      other="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -vx "$gpu" | head -n 1)"
      [[ -z "$other" ]] || coordinator_gpus="$gpu,$other" ;;
    *) echo "RTX_GPUS must be auto, 1 or 2" >&2; exit 2 ;;
  esac
fi
IFS=, read -r -a coordinator_gpus <<<"$coordinator_gpus"
gpu="${coordinator_gpus[0]}"
split="$(get COORDINATOR_SPLIT auto)"
case "$split" in auto|heads|off) ;; *) echo "COORDINATOR_SPLIT must be auto, heads or off" >&2; exit 2 ;; esac
second=""
if [[ ${#coordinator_gpus[@]} -ge 2 ]]; then
  second="${coordinator_gpus[1]}"
elif [[ "$split" == heads ]]; then
  second="$(get COORDINATOR_SPLIT_GPU $((1 - gpu)))"
fi
gpus="device=$gpu"
if [[ -n "$second" && "$split" != off ]]; then
  [[ "$second" != "$gpu" ]] || { echo "the second coordinator GPU must differ from the first" >&2; exit 2; }
  case "$family" in
    mimo_v2|glm5|deepseek_v4)
      lower=$((gpu < second ? gpu : second)) upper=$((gpu < second ? second : gpu))
      gpus="\"device=$lower,$upper\""
      family_args+=(--device $((gpu == lower ? 0 : 1)) --split-device $((gpu == lower ? 1 : 0))) ;;
    *) echo "note: $family has no head split; serving from GPU $gpu alone" >&2 ;;
  esac
fi
# INSTANCE names a launch that runs beside others on disjoint hardware
# (`cuteafd bench smoke` sets it): its coordinator container is
# cuteafd-coordinator-INSTANCE; empty keeps the one cuteafd-coordinator.
instance="$(get INSTANCE)"
[[ -z "$instance" || "$instance" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$ ]] || { echo "INSTANCE must be [A-Za-z0-9_.-]" >&2; exit 2; }
coordinator_name="cuteafd-coordinator${instance:+-$instance}"
# SPARK_COUNT=0: the routed experts run on the coordinator GPU (--local-experts;
# GLM 5.3 Flash, MiMo V2 and Qwen 3.8), the natural minimum for checkpoints
# that fit one RTX.
if [[ "$ranks" == 0 ]]; then
  case "$family" in
    glm5_flash|mimo_v2|qwen4) family_args+=(--local-experts) ;;
    *) echo "SPARK_COUNT=0 (local experts) serves GLM 5.3 Flash, MiMo V2 and Qwen 3.8, not $family" >&2; exit 2 ;;
  esac
fi
peers=()
# --restart removes this launcher's containers (stop.sh's release parser rejects
# the keys above, e.g. SPECULATOR).
# One model is served at a time: every expert worker on these hosts goes, whatever
# its port (a leftover worker of another model holds Spark memory and OOMs the next).
if [[ "$restart" == 1 ]]; then
  docker rm -f "$coordinator_name" >/dev/null 2>&1 || true
  for ((rank = 0; rank < ranks; rank++)); do
    host="$(get "SPARK_${rank}_HOST")"
    ssh "$host" 'ids=$(docker ps -aq --filter name=^cuteafd-spark-expert-); [ -z "$ids" ] || docker rm -f $ids >/dev/null 2>&1 || true'
  done
fi
# FP8_EXPERT_PREFILL: how FP8 expert packages run prefill row counts: auto
# (default: wire rows W8A8 with E4M3 x E4M3 gate/up, BF16 rows W8A16), w8a16
# (the former programs) or w8a8 (also quantizes the BF16 rows of experts on the
# coordinator GPU). Spark workers and the coordinator both read it.
fp8_prefill="$(get FP8_EXPERT_PREFILL auto)"
case "$fp8_prefill" in auto|w8a8|w8a16) ;; *) echo "FP8_EXPERT_PREFILL must be auto, w8a8 or w8a16" >&2; exit 2 ;; esac
# GB10 CUDA allocations cannot reclaim page cache: drop it on the expert hosts first.
spark_hosts=()
for ((rank = 0; rank < ranks; rank++)); do spark_hosts+=(--host "$(get "SPARK_${rank}_HOST")"); done
((ranks == 0)) || nest drop-caches "${spark_hosts[@]}" >/dev/null || echo "warning: could not drop Spark page caches" >&2
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  lane="$(get "SPARK_${rank}_LANE_A")"
  peers+=("$lane:$port")
  ssh "$host" "docker run -d --name cuteafd-spark-expert-$host-$port --restart no --gpus all --network host \
    --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -e CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill \
    -v \$(readlink -f \$HOME/.cache/huggingface/hub):/root/.cache/huggingface/hub:ro '$spark_image' \
    cuteafd expertd-native --snapshot '$snapshot' --native-lib /opt/cuteafd/lib/libcuteafd_native.so \
    --rank $rank --world $ranks --capacity 4096 --device-budget-bytes $budget $layer_args \
    --listen 0.0.0.0:$port >/dev/null" &
done
wait
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  until ssh "$host" "docker logs cuteafd-spark-expert-$host-$port 2>&1 | grep -q 'worker ready'"; do
    ssh "$host" "docker ps -q -f name=cuteafd-spark-expert-$host-$port | grep -q ." ||
      { echo "$host expert worker exited:" >&2; ssh "$host" "docker logs --tail 20 cuteafd-spark-expert-$host-$port" >&2; exit 1; }
    sleep 2
  done
done
peer_csv="$(IFS=,; echo "${peers[*]}")"
peer_args=()
[[ -z "$peer_csv" ]] || peer_args=(--peers "$peer_csv")
# The in-server benchmark keeps its history (SQLite) on the host; the image
# name labels its reports.
bench_dir="$HOME/.cache/cuteafd/bench"
mkdir -p "$bench_dir"
# SPARK_INTAKE: how routed partials reach the coordinator GPU (auto, gpu, pinned
# or host; see rust/crates/cuteafd-daemon/src/shared/spark_intake.rs).
intake="$(get SPARK_INTAKE auto)"
case "$intake" in auto|gpu|pinned|host) ;; *) echo "SPARK_INTAKE must be auto, gpu, pinned or host" >&2; exit 2 ;; esac
# CONSOLE_TEXT=on lets the live console at / stream generated token text (anyone who
# can reach the API port can then read every session's output).
console_text="$(get CONSOLE_TEXT off)"
case "$console_text" in on|off) ;; *) echo "CONSOLE_TEXT must be on or off" >&2; exit 2 ;; esac
docker run -d --name "$coordinator_name" --restart no --gpus "$gpus" --network host --ipc host \
  --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -e "CUTEAFD_SPARK_INTAKE=$intake" \
  -e "CUTEAFD_CONSOLE_TEXT=$([[ $console_text == on ]] && echo true || echo false)" \
  -e "CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill" -e "CUTEAFD_IMAGE=$coordinator_image" "${switch_args[@]}" \
  -v "$hub:/root/.cache/huggingface/hub:ro" -v "$bench_dir:/root/.cache/cuteafd/bench" \
  "${trace_args[@]}" "$coordinator_image" cuteafd $serve --snapshot "$snapshot" \
  --native-lib /opt/cuteafd/lib/libcuteafd_native.so "${peer_args[@]}" --listen "$addr" \
  --max-sequences "$(get CONCURRENCY 8)" --max-context "$(get MAX_CONTEXT_TOKENS 8192)" \
  --max-output "$(get MAX_OUTPUT_TOKENS 4096)" "${dspark_args[@]}" \
  "${family_args[@]}" "${draft_args[@]}" "${served_args[@]}" >/dev/null
url="http://127.0.0.1:${addr##*:}"
until curl -sf "$url/health" >/dev/null; do
  docker ps -q -f "name=^$coordinator_name\$" | grep -q . ||
    { echo "coordinator exited:" >&2; docker logs --tail 30 "$coordinator_name" >&2; exit 1; }
  sleep 2
done
echo "API ready at $url/v1/ ($(curl -s "$url/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'))"
