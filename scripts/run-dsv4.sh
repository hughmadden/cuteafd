#!/usr/bin/env bash
# Launch a DeepSeek V4 or GLM 5.x checkpoint (serve-dsv4 / serve-glm on one
# RTX, routed experts on the first SPARK_COUNT Sparks) from the release images
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
# Family: the serving command and the first layer with routed experts.
read -r model_type first_layer < <(python3 - "$root/snapshots/$revision/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c = c.get("text_config", c)
types = c.get("mlp_layer_types")
first = types.index("sparse") if types else c.get("first_k_dense_replace", 0)
print(c.get("model_type", "?"), first if c.get("model_type") == "glm_moe_dsa" else 0)
PY
)
case "$model_type" in
  deepseek_v4) serve=serve-dsv4 ;;
  glm_moe_dsa) serve=serve-glm ;;
  *) echo "run-dsv4.sh serves deepseek_v4 and glm_moe_dsa checkpoints, not $model_type" >&2; exit 2 ;;
esac
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
docker run -d --name cuteafd-coordinator --restart no --gpus "device=$gpu" --network host --ipc host \
  --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -v "$hub:/root/.cache/huggingface/hub:ro" \
  "$coordinator_image" cuteafd $serve --snapshot "$snapshot" \
  --native-lib /opt/cuteafd/lib/libcuteafd_native.so --peers "$peer_csv" --listen "$addr" \
  --max-sequences "$(get CONCURRENCY 8)" --max-context "$(get MAX_CONTEXT_TOKENS 8192)" \
  --max-output "$(get MAX_OUTPUT_TOKENS 4096)" $([[ $serve == serve-dsv4 && "$(get DSPARK off)" == on ]] && echo --dspark) >/dev/null
url="http://127.0.0.1:${addr##*:}"
until curl -sf "$url/health" >/dev/null; do
  docker ps -q -f name=cuteafd-coordinator | grep -q . ||
    { echo "coordinator exited:" >&2; docker logs --tail 30 cuteafd-coordinator >&2; exit 1; }
  sleep 2
done
echo "DeepSeek V4 API ready at $url/v1/ ($(curl -s "$url/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'))"
