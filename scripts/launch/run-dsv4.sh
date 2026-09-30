#!/usr/bin/env bash
# Launch a DeepSeek V4, GLM 5.x, GLM 5.3 Flash, MiMo V2 or Qwen 3.8 Flash Next
# checkpoint (serve-dsv4 / serve-glm / serve-glmf / serve-mimo / serve-qwen4 on
# one RTX, routed experts on the first SPARK_COUNT Sparks) from the release images
# named in the config. The family comes from the snapshot's config.json.
# Containers use run.sh's names, so ./stop.sh stops them.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config="$repo_root/cuteafd.config"
restart=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?--config requires FILE}"; shift 2 ;;
    --restart) restart=1; shift ;;
    *) echo "usage: $0 [--config FILE] [--restart]" >&2; exit 2 ;;
  esac
done
# Plain KEY=VALUE lines; the launch reads only the keys below.
declare -A cfg
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[A-Z_0-9]+$ ]] && cfg[$key]="$value"
done < <(grep -E '^[A-Z_0-9]+=' "$config")
get() { printf '%s' "${cfg[$1]:-${2:-}}"; }
model="$(get MODEL_ID)"
revision="$(get MODEL_REVISION)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
hub="$(readlink -f "$hf_home/hub")"
root="$hub/models--${model//\//--}"
[[ -n "$revision" ]] || revision="$(<"$root/refs/main")"
snapshot="/root/.cache/huggingface/hub/models--${model//\//--}/snapshots/$revision"
[[ -d "$root/snapshots/$revision" ]] || { echo "missing snapshot $model@$revision" >&2; exit 1; }
# Family: the serving command and the layers with routed experts (the Spark
# ranks serve [first, last]; MTP/nextn layers past the decoder stay off them
# for GLM 5.3 Flash and Qwen 3.8, whose serve paths do not run their MTP heads).
read -r model_type first_layer last_layer < <(python3 - "$root/snapshots/$revision/config.json" <<'PY'
import json, sys
top = json.load(open(sys.argv[1]))
c = top.get("text_config", top)
kind = top.get("model_type", "?")
first, last = 0, -1
if kind in ("glm_moe_dsa", "glm5_next"):
    types = c.get("mlp_layer_types")
    first = types.index("sparse") if types else c.get("first_k_dense_replace", 0)
elif kind in ("mimo_v2_flash", "mimo_v2"):
    first = c["moe_layer_freq"].index(1)
if kind in ("glm5_next", "qwen4_exp"):
    last = c["num_hidden_layers"] - 1
print(kind, first, last)
PY
)
case "$model_type" in
  deepseek_v4) serve=serve-dsv4 ;;
  glm_moe_dsa) serve=serve-glm ;;
  glm5_next) serve=serve-glmf ;;
  mimo_v2_flash|mimo_v2) serve=serve-mimo ;;
  qwen4_exp) serve=serve-qwen4 ;;
  *) echo "run-dsv4.sh serves deepseek_v4, glm_moe_dsa, glm5_next, mimo_v2_flash, mimo_v2 and qwen4_exp checkpoints, not $model_type" >&2; exit 2 ;;
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
# DRAFT_MODEL_ID (serve-glm, serve-glmf): a DFlash2 drafter checkpoint, e.g.
# incoai/GLM-5.3-DFlash2 or incoai/GLM-5.3-Flash-DFlash2.
draft_args=()
draft="$(get DRAFT_MODEL_ID)"
if [[ -n "$draft" ]]; then
  [[ $serve == serve-glm || $serve == serve-glmf ]] ||
    { echo "DRAFT_MODEL_ID applies to GLM checkpoints (DeepSeek V4 uses DSPARK=on)" >&2; exit 2; }
  draft_snapshot="$(snapshot_of "$draft" "$(get DRAFT_MODEL_REVISION)")" || exit 1
  draft_args=(--draft "$draft_snapshot")
fi
# serve-glmf: decode rows read FP8 copies of the dense projections from the
# official FP8 release (GLMF_FP8_MODEL_ID, "off" for BF16), KDA projections
# as per-row FP8 (GLMF_KDA_FP8: row128, channel or off) and optionally an FP8
# LM head (GLMF_FP8_HEAD=on); its MLA pools hold POOL_TOKENS tokens.
# GLMF_FP8_PREFILL (off, or a list of mla,ffn,kda-in,kda-o / all) runs those
# prefill projections as block-FP8 GEMMs (E4M3 activations per 128-K block).
family_args=()
# serve-mimo on MiMo V2.6 Pro (mimo_v2): its experts need six Spark ranks
# (SPARK_COUNT=6, TP6 MXFP4 slices, ~93 GiB each); DFLASH=on drafts with the
# snapshot's own dflash/ drafter.
if [[ $model_type == mimo_v2 && "$(get DFLASH off)" == on ]]; then
  draft_args=(--draft "$snapshot")
fi
# MTP=N: serve-mimo drafts with the checkpoint's first N native MTP layers
# (MiMo V2 Flash; V2.6 Pro drafts better with DFLASH=on).
if [[ $serve == serve-mimo && "$(get MTP 0)" != 0 ]]; then
  family_args+=(--mtp "$(get MTP 0)")
fi
if [[ $serve == serve-glmf ]]; then
  fp8_model="$(get GLMF_FP8_MODEL_ID zai-org/GLM-5.3-Flash)"
  if [[ "$fp8_model" != off ]]; then
    fp8_snapshot="$(snapshot_of "$fp8_model" "$(get GLMF_FP8_MODEL_REVISION)")" || exit 1
    family_args+=(--fp8-decode --fp8-snapshot "$fp8_snapshot")
  fi
  family_args+=(--kda-fp8 "$(get GLMF_KDA_FP8 row128)" --pool-tokens "$(get POOL_TOKENS 65536)")
  [[ "$(get GLMF_FP8_HEAD off)" != on ]] || family_args+=(--fp8-head)
  fp8_prefill="$(get GLMF_FP8_PREFILL off)"
  [[ "$fp8_prefill" == off ]] || family_args+=(--fp8-prefill "$fp8_prefill")
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
# copies made from BF16 weights at load; DRAFT_FP8=off drafts in BF16.
if [[ $serve != serve-dsv4 ]]; then
  [[ -z "$(get L2_PREFETCH)" ]] || family_args+=(--l2-prefetch "$(get L2_PREFETCH)")
  [[ -z "$(get FP8_SCALES)" ]] || family_args+=(--fp8-scales "$(get FP8_SCALES)")
  if [[ ${#draft_args[@]} -gt 0 && "$(get DRAFT_FP8 on)" == off ]]; then family_args+=(--draft-fp8 false); fi
fi
# SERVED_MODEL_ID: the public model id (default: the checkpoint's Hugging Face id).
served_args=()
served="$(get SERVED_MODEL_ID)"
[[ -z "$served" ]] || served_args=(--model-id "$served")
coordinator_image="$(get COORDINATOR_DOCKER_INFERENCE)"
# COORDINATOR_TRACE=/abs/host/file.jsonl: serve-glm's per-cycle speculation
# trace (CUTEAFD_GLM_TRACE; scripts/qualify/glm5/glm-draft-trace.py reads it).
trace_args=()
trace="$(get COORDINATOR_TRACE)"
if [[ -n "$trace" ]]; then
  mkdir -p "$(dirname "$trace")"
  trace_args=(-v "$(dirname "$trace"):$(dirname "$trace")" -e "CUTEAFD_GLM_TRACE=$trace")
fi
spark_image="$(get SPARK_EXPERT_DOCKER_INFERENCE)"
port="$(get EXPERT_PORT 19441)"
addr="$(get ADDR 0.0.0.0:8000)"
ranks="$(get SPARK_COUNT 4)"
budget="$(get SPARK_DEVICE_BUDGET_BYTES 107374182400)"
gpu="$(get COORDINATOR_GPU 0)"
peers=()
# --restart removes this launcher's containers (stop.sh's release parser rejects
# the keys above, e.g. DRAFT_MODEL_ID).
if [[ "$restart" == 1 ]]; then
  docker rm -f cuteafd-coordinator >/dev/null 2>&1 || true
  for ((rank = 0; rank < ranks; rank++)); do
    host="$(get "SPARK_${rank}_HOST")"
    ssh "$host" "docker rm -f cuteafd-spark-expert-$host-$port >/dev/null 2>&1 || true"
  done
fi
# GB10 CUDA allocations cannot reclaim page cache: drop it on the expert hosts first.
spark_hosts=()
for ((rank = 0; rank < ranks; rank++)); do spark_hosts+=(--host "$(get "SPARK_${rank}_HOST")"); done
nest drop-caches "${spark_hosts[@]}" >/dev/null || echo "warning: could not drop Spark page caches" >&2
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  lane="$(get "SPARK_${rank}_LANE_A")"
  peers+=("$lane:$port")
  ssh "$host" "docker run -d --name cuteafd-spark-expert-$host-$port --restart no --gpus all --network host \
    --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info \
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
# SPARK_INTAKE: how routed partials reach the coordinator GPU (auto, gpu, pinned
# or host; see rust/crates/cuteafd-daemon/src/shared/spark_intake.rs).
intake="$(get SPARK_INTAKE auto)"
case "$intake" in auto|gpu|pinned|host) ;; *) echo "SPARK_INTAKE must be auto, gpu, pinned or host" >&2; exit 2 ;; esac
docker run -d --name cuteafd-coordinator --restart no --gpus "device=$gpu" --network host --ipc host \
  --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -e "CUTEAFD_SPARK_INTAKE=$intake" \
  -v "$hub:/root/.cache/huggingface/hub:ro" \
  "${trace_args[@]}" "$coordinator_image" cuteafd $serve --snapshot "$snapshot" \
  --native-lib /opt/cuteafd/lib/libcuteafd_native.so --peers "$peer_csv" --listen "$addr" \
  --max-sequences "$(get CONCURRENCY 8)" --max-context "$(get MAX_CONTEXT_TOKENS 8192)" \
  --max-output "$(get MAX_OUTPUT_TOKENS 4096)" $([[ $serve == serve-dsv4 && "$(get DSPARK off)" == on ]] && echo --dspark) \
  "${family_args[@]}" "${draft_args[@]}" "${served_args[@]}" >/dev/null
url="http://127.0.0.1:${addr##*:}"
until curl -sf "$url/health" >/dev/null; do
  docker ps -q -f name=cuteafd-coordinator | grep -q . ||
    { echo "coordinator exited:" >&2; docker logs --tail 30 cuteafd-coordinator >&2; exit 1; }
  sleep 2
done
echo "API ready at $url/v1/ ($(curl -s "$url/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'))"
