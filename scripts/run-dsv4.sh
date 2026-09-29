#!/usr/bin/env bash
# Launch a DeepSeek V4, GLM 5.x or GLM 5.3 Flash checkpoint (serve-dsv4 /
# serve-glm / serve-glmf on one RTX, routed experts on the first SPARK_COUNT
# Sparks) from the release images
# named in the config. The family comes from the snapshot's config.json.
# Containers use run.sh's names, so ./stop.sh stops them.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
# Family: the serving command, the first layer with routed experts and the
# backbone's layer count (GLM 5.3 Flash checkpoints also carry an MTP layer
# after it, which the Spark workers do not serve).
read -r model_type first_layer layers < <(python3 - "$root/snapshots/$revision/config.json" <<'PY'
import json, sys
top = json.load(open(sys.argv[1]))
c = top.get("text_config", top)
kind = top.get("model_type", "?")
types = c.get("mlp_layer_types")
first = types.index("sparse") if types else c.get("first_k_dense_replace", 0)
print(kind, first if kind in ("glm_moe_dsa", "glm5_next") else 0, c.get("num_hidden_layers", 0))
PY
)
case "$model_type" in
  deepseek_v4) serve=serve-dsv4 ;;
  glm_moe_dsa) serve=serve-glm ;;
  glm5_next) serve=serve-glmf ;;
  *) echo "run-dsv4.sh serves deepseek_v4, glm_moe_dsa and glm5_next checkpoints, not $model_type" >&2; exit 2 ;;
esac
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
family_args=()
worker_args=()
if [[ $serve == serve-glmf ]]; then
  fp8_model="$(get GLMF_FP8_MODEL_ID zai-org/GLM-5.3-Flash)"
  if [[ "$fp8_model" != off ]]; then
    fp8_snapshot="$(snapshot_of "$fp8_model" "$(get GLMF_FP8_MODEL_REVISION)")" || exit 1
    family_args+=(--fp8-decode --fp8-snapshot "$fp8_snapshot")
  fi
  family_args+=(--kda-fp8 "$(get GLMF_KDA_FP8 row128)" --pool-tokens "$(get POOL_TOKENS 65536)")
  [[ "$(get GLMF_FP8_HEAD off)" != on ]] || family_args+=(--fp8-head)
  worker_args+=(--last-layer $((layers - 1)))
fi
coordinator_image="$(get COORDINATOR_DOCKER_INFERENCE)"
spark_image="$(get SPARK_EXPERT_DOCKER_INFERENCE)"
port="$(get EXPERT_PORT 19441)"
addr="$(get ADDR 0.0.0.0:8000)"
ranks="$(get SPARK_COUNT 4)"
budget="$(get SPARK_DEVICE_BUDGET_BYTES 107374182400)"
gpu="$(get COORDINATOR_GPU 0)"
peers=()
[[ "$restart" == 0 ]] || "$repo_root/stop.sh" --config "$config" >/dev/null
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  lane="$(get "SPARK_${rank}_LANE_A")"
  peers+=("$lane:$port")
  ssh "$host" "docker run -d --name cuteafd-spark-expert-$host-$port --restart no --gpus all --network host \
    --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info \
    -v \$(readlink -f \$HOME/.cache/huggingface/hub):/root/.cache/huggingface/hub:ro '$spark_image' \
    cuteafd expertd-native --snapshot '$snapshot' --native-lib /opt/cuteafd/lib/libcuteafd_native.so \
    --rank $rank --world $ranks --capacity 4096 --device-budget-bytes $budget --first-layer $first_layer \
    ${worker_args[*]} --listen 0.0.0.0:$port >/dev/null" &
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
docker run -d --name cuteafd-coordinator --restart no --gpus "device=$gpu" --network host --ipc host \
  --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -v "$hub:/root/.cache/huggingface/hub:ro" \
  "$coordinator_image" cuteafd $serve --snapshot "$snapshot" \
  --native-lib /opt/cuteafd/lib/libcuteafd_native.so --peers "$peer_csv" --listen "$addr" \
  --max-sequences "$(get CONCURRENCY 8)" --max-context "$(get MAX_CONTEXT_TOKENS 8192)" \
  --max-output "$(get MAX_OUTPUT_TOKENS 4096)" $([[ $serve == serve-dsv4 && "$(get DSPARK off)" == on ]] && echo --dspark) \
  "${family_args[@]}" "${draft_args[@]}" >/dev/null
url="http://127.0.0.1:${addr##*:}"
until curl -sf "$url/health" >/dev/null; do
  docker ps -q -f name=cuteafd-coordinator | grep -q . ||
    { echo "coordinator exited:" >&2; docker logs --tail 30 cuteafd-coordinator >&2; exit 1; }
  sleep 2
done
echo "API ready at $url/v1/ ($(curl -s "$url/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'))"
