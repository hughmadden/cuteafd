#!/usr/bin/env bash
# Launch a DeepSeek V4, GLM 5.x, GLM 5.3 Flash, MiMo V2 or Qwen 3.8 Flash Next
# checkpoint (the family's serve command on one RTX, routed experts on the first
# SPARK_COUNT Sparks, or SPARK_HOSTS in explicit rank order) from the release images named in the config. ./run.sh
# starts this for every family but DeepSeek V4.1; the family comes from the
# snapshot's config.json (scripts/lib/checkpoint-family.py), or --family.
# Containers use run.sh's names, so ./stop.sh stops them.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo_root/scripts/lib/release-common.sh"
config="$repo_root/cuteafd.config"
restart=0
family=""
embedding_override=""
wip_slot=""
# GLM5_FLASH_MEMORY=auto re-runs this launch with the arguments it was given (below).
launch_args=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?--config requires FILE}"; shift 2 ;;
    --family) family="${2:?--family requires ID}"; shift 2 ;;
    --embedding-placement) embedding_override="${2:?--embedding-placement requires host or gpu}"; shift 2 ;;
    --restart) restart=1; shift ;;
    --wip) wip_slot="${2:?--wip requires SLOT}"; shift 2 ;;
    *) echo "usage: $0 [--config FILE] [--family ID] [--embedding-placement host|gpu] [--restart] [--wip SLOT]" >&2; exit 2 ;;
  esac
done
# Plain KEY=VALUE lines; the launch reads only the keys below.
declare -A cfg
while IFS='=' read -r key value; do
  release_known_key "$key" || release_die "unknown configuration key: $key"
  cfg[$key]="$value"
done < <(grep -E '^[A-Z_0-9]+=' "$config")
[[ -z "$embedding_override" ]] || cfg[EMBEDDING]="$embedding_override"
get() { printf '%s' "${cfg[$1]:-${2:-}}"; }
vision="$(get VISION off)"
audio="$(get AUDIO off)"
vision_replicas="$(get VISION_REPLICAS 1)"
[[ "$vision_replicas" =~ ^[1-6]$ ]] || release_die "VISION_REPLICAS must be 1..6"
[[ "$vision" =~ ^(auto|off|rtx|spark)(:[0-9]+)?$ && ( "$vision" != auto:* && "$vision" != off:* ) ]] || release_die "VISION must be auto, off, rtx[:gpu] or spark[:rank]"
case "$audio" in auto|off) ;; *) release_die "AUDIO must be auto or off" ;; esac
coordinator_budget="$(get COORDINATOR_GPU_BUDGET_GIB)"
release_validate_coordinator_gpu_budget "$coordinator_budget"
coordinator_budget_args=()
[[ -z "$coordinator_budget" ]] || coordinator_budget_args=(--coordinator-gpu-budget-gib "$coordinator_budget")
# RDMA_BOND_BALANCE: the coordinator's expert QPs connect with RoCE v2 flow labels it chooses,
# so a coordinator port that is an LACP bond carries as many of them on each member: off
# (default: the kernel's per-QP labels, re-rolled at every start), labels (fixed labels, the
# same placement at every start) or probe (labels measured onto alternating members; see
# rust/crates/cuteafd-transport/src/bond.rs). Workers need no setting.
bond_balance="$(get RDMA_BOND_BALANCE off)"
case "$bond_balance" in off|labels|probe) ;; *) release_die "RDMA_BOND_BALANCE must be off, labels or probe" ;; esac
bond_args=()
[[ "$bond_balance" == off ]] || bond_args=(-e "CUTEAFD_RDMA_BOND_BALANCE=$bond_balance")
# Validate the name before it is used to identify allocations during admission.
instance="$(get INSTANCE)"
[[ -z "$instance" || "$instance" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$ ]] || { echo "INSTANCE must be [A-Za-z0-9_.-]" >&2; exit 2; }
coordinator_name="cuteafd-coordinator${instance:+-$instance}"
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
# Qualified MiMo and GLM Flash encoders use Spark-first auto unless explicitly off.
# Other generic families keep off until their towers are qualified.
if [[ ( "$family" == mimo_v2 || "$family" == glm5_flash ) && -z "$(get VISION)" ]]; then vision=auto; fi
# Auto/spark placement is resolved by the encoder plan below.
# EXPERT_BACKEND=auto prefers qualified local experts when the planner admits
# their weights plus serving reservations on the selected GPU. SPARK_COUNT is
# the fallback topology; EXPERT_BACKEND=spark explicitly keeps it.
ranks="$(get SPARK_COUNT 4)"
configured_ranks="$ranks"
if [[ -n "$(get SPARK_HOSTS)" ]]; then
  host_rows="$(release_spark_host_rows "$(get SPARK_HOSTS)" "$ranks")" || exit 2
  while read -r rank host lane_a lane_b; do
    cfg[SPARK_${rank}_HOST]="$host"
    cfg[SPARK_${rank}_LANE_A]="$lane_a"
    cfg[SPARK_${rank}_LANE_B]="$lane_b"
  done <<<"$host_rows"
fi
backend="$(get EXPERT_BACKEND auto)"
case "$backend" in
  auto|spark) ;;
  local) ranks=0 ;;
  *) echo "EXPERT_BACKEND must be auto, local or spark" >&2; exit 2 ;;
esac
# GLM5_FLASH_MEMORY (GLM 5.3 Flash): standard (the default: the settings as configured), compact or
# auto. compact is the profile measured on 1 RTX 5090 + 4 DGX Sparks at 16 sequences, each setting
# gated: the compact DSA index and a BF16 KDA state over the checkpoint-precision KDA projections and
# head (every compact measurement ran at checkpoint precision), prefix marks in the pool with a host
# tier of half this host's RAM (at most 64 GiB: the measured 64 GiB on a 125 GiB host), the embedding
# in host RAM, 1 GiB of headroom, a 512 MiB graph budget with row buckets, replay records in the
# prefill scratch, 128-row decode steps, the gb10 Spark schedule, the probed bond split where an RDMA
# port of this host is a bond, the tensor-core W8A8 drafter and the tensor-core target head past 8 rows
# (1,683,456 KV tokens beside 131,072-token requests, and one 1,048,576-token request). auto lays the standard
# settings out with `cuteafd plan --layout` on the coordinator GPU's free memory for CONCURRENCY
# sequences and MAX_CONTEXT_TOKENS, keeps them when that pool holds one MAX_CONTEXT_TOKENS request and
# 65,536 tokens for each other sequence, and takes compact when it cannot. GLM5_FLASH_PROFILE=rtx5090
# names compact. A key the config sets keeps its value: the profile fills in the others (the BF16 KDA
# state and the tensor-core target head only over the BF16 KDA projections and head they run on, the
# graph budget only when the config does not capture every decode graph at startup), and the launch
# notes each value it sets, keeps or leaves unset. The graph budget captures decode graphs lazily: the
# whole startup set (GLM5_FLASH_STARTUP_GRAPHS, the engine's default otherwise) would take 2.4 GB of a
# 5090 at 131,072 tokens. VISION keeps its own default (auto: a checkpoint's tower on a Spark).
glmf_compact=(GLM5_FLASH_KDA_FP8=off GLM5_FLASH_FP8_HEAD=off GLM5_FLASH_FP8_PREFILL=off GLM5_FLASH_INDEX_CACHE=compact
  GLM5_FLASH_KDA_STATE=bf16 GLM5_FLASH_PREFIX_MARKS=pool HOST_CACHE_BYTES=64GiB EMBEDDING=host GLM5_FLASH_HEADROOM_GIB=1
  GLM5_FLASH_GRAPH_BUDGET_MIB=512 GLM5_FLASH_DECODE_ROW_BUCKETS=on GLM5_FLASH_REPLAY_RECORDS=shared
  GLM5_FLASH_DECODE_ROWS=128 GLM5_FLASH_EXL3_SCHEDULE=gb10 GLM5_FLASH_EXL3_WORKER_PATH=async RDMA_BOND_BALANCE=probe
  GLM5_FLASH_DRAFT_HEAD=tensor GLM5_FLASH_DRAFT_LINEAR=w8a8 GLM5_FLASH_TARGET_HEAD=tensor)
# Whether an RDMA port of this host is a bond (two or more members): a GID's netdev, or the one device
# under a VLAN, with bonding members, as the engine's bond balance finds it (sysfs: CUTEAFD_SYSFS_ROOT).
glmf_rdma_bond() {
  local root="${CUTEAFD_SYSFS_ROOT:-/sys}" ndev name hop slaves lowers
  for ndev in "$root"/class/infiniband/*/ports/*/gid_attrs/ndevs/*; do
    name="$(cat "$ndev" 2>/dev/null || true)"
    for hop in 1 2; do
      [[ -n "$name" && -d "$root/class/net/$name" ]] || break
      if [[ -r "$root/class/net/$name/bonding/slaves" ]]; then
        slaves=()
        read -r -a slaves < "$root/class/net/$name/bonding/slaves" || true
        if ((${#slaves[@]} >= 2)); then return 0; fi
        break
      fi
      lowers=("$root/class/net/$name"/lower_*)
      [[ ${#lowers[@]} == 1 && -e "${lowers[0]}" ]] || break
      name="${lowers[0]##*/lower_}"
    done
  done
  return 1
}
# KEY=VALUE of a GLM 5.3 Flash precision key as the config sets it, under its current or pre-rename name.
glmf_configured() {
  local old="GLMF_${1#GLM5_FLASH_}"
  if [[ -n "${cfg[$1]:-}" ]]; then printf '%s=%s' "$1" "${cfg[$1]}"; else printf '%s=%s' "$old" "${cfg[$old]:-}"; fi
}
glmf_memory="$(get GLM5_FLASH_MEMORY)"
glmf_profile="$(get GLM5_FLASH_PROFILE)"
if [[ -n "$glmf_memory$glmf_profile" && "$family" != glm5_flash ]]; then
  echo "GLM5_FLASH_MEMORY and GLM5_FLASH_PROFILE apply to GLM 5.3 Flash checkpoints, not $family" >&2
  exit 2
fi
case "$glmf_profile" in
  "") ;;
  rtx5090)
    [[ -z "$glmf_memory" || "$glmf_memory" == compact ]] ||
      { echo "GLM5_FLASH_PROFILE=rtx5090 is GLM5_FLASH_MEMORY=compact; set one of them, not GLM5_FLASH_MEMORY=$glmf_memory" >&2; exit 2; }
    glmf_memory=compact ;;
  *) echo "GLM5_FLASH_PROFILE must be rtx5090" >&2; exit 2 ;;
esac
case "$glmf_memory" in
  "") glmf_memory=standard ;;
  standard|compact|auto) ;;
  *) echo "GLM5_FLASH_MEMORY must be auto, compact or standard" >&2; exit 2 ;;
esac
if [[ "$glmf_memory" == auto && "${CUTEAFD_GLMF_MEMORY_CHOSEN:-}" == compact ]]; then
  echo "note: GLM5_FLASH_MEMORY=auto runs compact, as planned" >&2
  glmf_memory=compact
fi
if [[ "$glmf_memory" == compact ]]; then
  [[ "$ranks" != 0 ]] || { echo "GLM5_FLASH_MEMORY=compact runs the routed experts on Sparks, as measured; this launch" \
    "runs them on the GPU (SPARK_COUNT=0 or EXPERT_BACKEND=local)" >&2; exit 2; }
  # The host tier: half of this host's RAM (MemTotal, as the engine reads it; CUTEAFD_PROC_ROOT), whole
  # GiB, at most 64 GiB.
  glmf_mem_kib="$(awk '/^MemTotal:/ {print $2; exit}' "${CUTEAFD_PROC_ROOT:-/proc}/meminfo" 2>/dev/null || true)"
  glmf_host_cache=""
  if [[ "$glmf_mem_kib" =~ ^[1-9][0-9]*$ ]]; then
    glmf_host_gib=$((glmf_mem_kib / 2 / 1048576))
    if ((glmf_host_gib > 64)); then glmf_host_gib=64; fi
    if ((glmf_host_gib >= 1)); then glmf_host_cache="${glmf_host_gib}GiB"; fi
  fi
  for glmf_setting in "${glmf_compact[@]}"; do
    glmf_key="${glmf_setting%%=*}" glmf_value="${glmf_setting#*=}" glmf_old=""
    if [[ "$glmf_key" == HOST_CACHE_BYTES && -n "$glmf_host_cache" ]]; then glmf_value="$glmf_host_cache"; fi
    case "$glmf_key" in GLM5_FLASH_KDA_FP8|GLM5_FLASH_FP8_HEAD|GLM5_FLASH_FP8_PREFILL) glmf_old="GLMF_${glmf_key#GLM5_FLASH_}" ;; esac
    if [[ -n "${cfg[$glmf_key]:-}" ]]; then
      echo "note: GLM5_FLASH_MEMORY=compact keeps $glmf_key=${cfg[$glmf_key]} as configured (compact: $glmf_value)" >&2
    elif [[ -n "$glmf_old" && -n "${cfg[$glmf_old]:-}" ]]; then
      echo "note: GLM5_FLASH_MEMORY=compact keeps $glmf_old=${cfg[$glmf_old]} as configured (compact: $glmf_key=$glmf_value)" >&2
    elif [[ "$glmf_key" == GLM5_FLASH_KDA_STATE && "$(glmf_configured GLM5_FLASH_KDA_FP8)" != *=off ]]; then
      echo "note: GLM5_FLASH_MEMORY=compact leaves $glmf_key unset (compact: $glmf_value): the BF16 state runs over the" \
        "BF16 KDA projections, and the config keeps $(glmf_configured GLM5_FLASH_KDA_FP8)" >&2
    elif [[ "$glmf_key" == GLM5_FLASH_TARGET_HEAD && "$(glmf_configured GLM5_FLASH_FP8_HEAD)" != *=off ]]; then
      echo "note: GLM5_FLASH_MEMORY=compact leaves $glmf_key unset (compact: $glmf_value): the tensor-core target head" \
        "runs the BF16 head, and the config keeps $(glmf_configured GLM5_FLASH_FP8_HEAD)" >&2
    elif [[ "$glmf_key" == HOST_CACHE_BYTES && -z "$glmf_host_cache" ]]; then
      echo "note: GLM5_FLASH_MEMORY=compact leaves $glmf_key unset (compact: half of this host's RAM, at most 64 GiB):" \
        "no MemTotal in ${CUTEAFD_PROC_ROOT:-/proc}/meminfo; pool marks then take the engine's automatic host tier" >&2
    elif [[ "$glmf_key" == RDMA_BOND_BALANCE ]] && ! glmf_rdma_bond; then
      echo "note: GLM5_FLASH_MEMORY=compact leaves $glmf_key unset (compact: $glmf_value): no RDMA port of this host" \
        "is a bond" >&2
    elif [[ "$glmf_key" == GLM5_FLASH_GRAPH_BUDGET_MIB && "${cfg[GLM5_FLASH_STARTUP_GRAPHS]:-}" == on ]]; then
      echo "note: GLM5_FLASH_MEMORY=compact leaves $glmf_key unset (compact: $glmf_value): the budget bounds lazily" \
        "captured decode graphs, and the config captures every one at startup (GLM5_FLASH_STARTUP_GRAPHS=on)" >&2
    else
      cfg[$glmf_key]="$glmf_value"
      echo "note: GLM5_FLASH_MEMORY=compact sets $glmf_key=$glmf_value" >&2
      if [[ "$glmf_key" == HOST_CACHE_BYTES ]]; then
        echo "note: HOST_CACHE_BYTES=$glmf_value is half of this host's $((glmf_mem_kib / 1048576)) GiB of RAM, at most 64 GiB" >&2
      fi
    fi
  done
fi
# RDMA_BOND_BALANCE: the coordinator's expert QPs connect with RoCE v2 flow labels it chooses,
# so a coordinator port that is an LACP bond carries as many of them on each member: off
# (default: the kernel's per-QP labels, re-rolled at every start), labels (fixed labels, the
# same placement at every start) or probe (labels measured onto alternating members; see
# rust/crates/cuteafd-transport/src/bond.rs). Workers need no setting.
bond_balance="$(get RDMA_BOND_BALANCE off)"
case "$bond_balance" in off|labels|probe) ;; *) release_die "RDMA_BOND_BALANCE must be off, labels or probe" ;; esac
bond_args=()
[[ "$bond_balance" == off ]] || bond_args=(-e "CUTEAFD_RDMA_BOND_BALANCE=$bond_balance")
# The free memory of coordinator GPU $1 in GiB as an admission would see it: nvidia-smi's free MiB,
# with --restart crediting this launch's own coordinator on it (removed after validation; never
# another launch's memory), within COORDINATOR_GPU_BUDGET_GIB when set (0 without a total to charge
# against). Empty when nvidia-smi gives no sample. While the coordinator is the GPU's only compute
# process the credit is the device's used memory: nvidia-smi attributes part of a process's device
# memory to no process (142 MiB of a GLM 5.3 Flash coordinator on an RTX 5090), and idle the device's
# free memory is its total less the reserved memory alone.
selected_gpu_free_gib() {
  local selected="$1" free_mib used_mib own_pids own_mib free_gib total_mib
  free_mib="$(nvidia-smi --id="$selected" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
  [[ "$free_mib" =~ ^[0-9]+$ ]] || return 0
  if [[ "$restart" == 1 ]]; then
    own_pids="$(docker top "$coordinator_name" -eo pid 2>/dev/null | tail -n +2 || true)"
    if [[ -n "$own_pids" ]]; then
      used_mib="$(nvidia-smi --id="$selected" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
      own_mib="$(nvidia-smi --id="$selected" --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null \
        | python3 -c 'import csv,sys
p, used = set(sys.argv[1].split()), sys.argv[2]
rows = [(r[0].strip(), r[1].strip()) for r in csv.reader(sys.stdin) if len(r) == 2 and r[0].strip()]
own = sum(int(m) for pid, m in rows if pid in p and m.isdigit())
alone = all(pid in p for pid, _ in rows)
print(int(used) if own and alone and used.isdigit() else own)' "$own_pids" "$used_mib" || true)"
      [[ "$own_mib" =~ ^[0-9]+$ ]] && free_mib=$((free_mib + own_mib))
    fi
  fi
  free_gib="$(python3 -c 'import sys; print(int(sys.argv[1])/1024)' "$free_mib")"
  if [[ -n "$coordinator_budget" ]]; then
    total_mib="$(nvidia-smi --id="$selected" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
    # Credit this launch's restart above, but charge other physical usage
    # against the simulated smaller card, just as runtime admission does.
    if [[ "$total_mib" =~ ^[0-9]+$ ]]; then
      free_gib="$(python3 -c 'import sys; free,total,budget=map(float,sys.argv[1:]); print(max(0,min(free,budget-total+free)))' \
        "$free_gib" "$(python3 -c 'import sys; print(int(sys.argv[1])/1024)' "$total_mib")" "$coordinator_budget")"
    else
      free_gib=0 # No trustworthy sample.
    fi
  fi
  printf '%s' "$free_gib"
}
qwen_exl3=0
qwen_mtp=0
if [[ "$family" == qwen4 ]]; then
  qwen_features="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); q=c.get("quantization_config", {}); print(int(q.get("quant_method", q.get("method")) == "exl3"), c.get("text_config", c).get("mtp_num_hidden_layers", 0))' "$root/snapshots/$revision/config.json")"
  read -r qwen_exl3 qwen_mtp <<<"$qwen_features"
fi
if [[ "$qwen_exl3" == 1 && "$backend" == auto && "$ranks" != 0 ]]; then
  selected="$(get COORDINATOR_GPUS "$(get COORDINATOR_GPU 0)")"; selected="${selected%%,*}"
  # Without a total under a budget the free memory reads 0: the Spark fallback stays.
  free_gib="$(selected_gpu_free_gib "$selected")"
  if [[ -n "$free_gib" ]]; then
    pool="$(get POOL_TOKENS 32768)"
    if [[ "$pool" =~ ^[1-9][0-9]*$ ]]; then
      # CPU-only preflight reads checkpoint headers in the selected serving image.
      # Older images that do not qualify auto placement keep the Spark fallback.
      preferred="$(docker run --rm --network none -v "$hub:/root/.cache/huggingface/hub:ro" \
        "$(get COORDINATOR_DOCKER_INFERENCE)" cuteafd plan "$snapshot" --vision "$vision" --audio "$audio" --json --layout \
        --rtx 1 --rtx-gib "$free_gib" --coordinator-budget-gib "$free_gib" --pool-tokens "$pool" \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["spark_ranks"])' 2>/dev/null || true)"
      if [[ "$preferred" == 0 ]]; then
        echo "note: Qwen EXL3 auto selected resident local experts on GPU $selected; EXPERT_BACKEND=spark forces Spark ranks" >&2
        ranks=0
      fi
    fi
  fi
fi
layer_args="--first-layer $first_layer"
[[ "$last_layer" == -1 ]] || layer_args+=" --last-layer $last_layer"
# Options only the Spark expert workers take: their arguments, and their docker environment and mounts.
spark_worker_args=""
spark_worker_env=""
# Snapshot of a model id (and optional revision) inside the containers.
snapshot_of() {
  local id="$1" rev="$2" dir="$hub/models--${1//\//--}"
  [[ -n "$rev" ]] || rev="$(<"$dir/refs/main")"
  [[ -d "$dir/snapshots/$rev" ]] || { echo "missing snapshot $id@$rev" >&2; return 1; }
  printf '%s' "/root/.cache/huggingface/hub/models--${id//\//--}/snapshots/$rev"
}
# SPECULATOR picks the drafter (Flash MOPD: bundled DFlash; Qwen local EXL3:
# MTP3; GLM 5.3 Flash: its measured best; otherwise off):
#   dflash2  GLM 5.x / GLM 5.3 Flash: the DFlash2 checkpoint SPECULATOR_MODEL_ID
#            (e.g. incoai/GLM-5.3-DFlash2, incoai/GLM-5.3-Flash-DFlash2);
#            MiMo V2.6 Flash/Pro: the snapshot's own dflash/ drafter unless
#            SPECULATOR_MODEL_ID names one (Pro needs SPARK_COUNT=6)
#   mtp      MiMo V2 Flash, Qwen 3.8: the checkpoint's native MTP layers,
#            SPECULATOR_DEPTH drafts (qualified local Qwen default 3; otherwise 1)
#   dspark   DeepSeek V4 (its own drafter); GLM 5.3 Flash: the dSpark
#            checkpoint SPECULATOR_MODEL_ID (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview)
# Official Flash MOPD's bundled drafter defaults to single-copy FP8 (measured
# separately from its checkpoint BF16 target head/O); other MiMo defaults follow
# runtime weight policy. SPECULATOR_FP8=auto keeps drafter checkpoint format. GLM auto/unset
# drafts in single-copy FP8. on converts, off selects BF16. Pre-rename keys
# (DRAFT_MODEL_ID, DFLASH,
# MTP, DSPARK, DRAFT_FP8) still work for one release.
# GLM 5.3 Flash always drafts with an external speculator: the one measured
# fastest (emitted tok/s: C1/C4 code and an agentic reasoning session, 1 RTX +
# 2 Sparks and 2 RTX + 4 Sparks) for each checkpoint. DFlash2 led on every one
# measured (2026-10-04: wrldsuksgo2mars EXL3 K3.25, nvidia NVFP4, brandonmusic
# tr3 4bpw; agentic 1.3-1.5x dSpark); SPECULATOR=dspark selects the RedHat dSpark.
glm5_flash_speculator() {
  case "$1" in
    *) echo "dflash2 incoai/GLM-5.3-Flash-DFlash2" ;;
  esac
}
# Qualification is for this official checkpoint, not a shape-compatible sibling.
mimo_flash_mopd=0
[[ "$family:$model" != mimo_v2:XiaomiMiMo/MiMo-V2.6-Flash-MOPD ]] || mimo_flash_mopd=1
speculator="$(get SPECULATOR)"
default_drafter=""
if [[ -z "$speculator" ]]; then
  if [[ -n "$(get DRAFT_MODEL_ID)" || "$(get DFLASH off)" == on ]]; then speculator=dflash2
  elif [[ "$(get MTP 0)" != 0 ]]; then speculator=mtp
  elif [[ $family == deepseek_v4 && "$(get DSPARK off)" == on ]]; then speculator=dspark
  else speculator=off; fi
  [[ $speculator == off ]] || echo "warning: DRAFT_MODEL_ID/DFLASH/MTP/DSPARK are deprecated; use SPECULATOR=$speculator" >&2
  if [[ $speculator == off && $family == glm5_flash ]]; then
    read -r speculator default_drafter <<<"$(glm5_flash_speculator "$model")"
    if [[ -d "$hub/models--${default_drafter//\//--}" ]]; then
      echo "note: GLM 5.3 Flash drafts with $speculator ($default_drafter) for $model; SPECULATOR=off disables it" >&2
    else
      echo "warning: GLM 5.3 Flash drafts with $speculator by default but $default_drafter is not downloaded" \
        "(hf download $default_drafter); serving without a drafter" >&2
      speculator=off default_drafter=""
    fi
  fi
  if [[ $speculator == off && $mimo_flash_mopd == 1 && -z ${cfg[MTP]+set} && -z ${cfg[DFLASH]+set} ]]; then
    speculator=dflash2
    echo "note: MiMo V2.6 Flash MOPD drafts with its bundled DFlash; SPECULATOR=off disables it" >&2
  fi
  # Only the resident EXL3 path is qualified. Spark workers serve backbone
  # layers, not mtp.layers.0; other expert formats keep their opt-in status.
  # An explicit SPECULATOR=off or legacy MTP=0 disables the family default.
  if [[ $speculator == off && $qwen_exl3 == 1 && $qwen_mtp == 1 && $ranks == 0 && -z ${cfg[MTP]+set} ]]; then
    speculator=mtp
    echo "note: Qwen local EXL3 drafts with native MTP (default depth 3); SPECULATOR=off disables it" >&2
  fi
fi
case "$family:$speculator" in
  qwen4:mtp) ;; # The MTP layer's experts stay local even with Spark backbone experts.
  *:off|glm5:dflash2|glm5_flash:dflash2|glm5_flash:dspark|mimo_v2:dflash2|mimo_v2:mtp|deepseek_v4:dspark) ;;
  *) echo "SPECULATOR=$speculator does not apply to $family" >&2; exit 2 ;;
esac
draft_args=()
embedding="$(get EMBEDDING gpu)"
case "$embedding" in host|gpu) ;; *) echo "EMBEDDING must be host or gpu" >&2; exit 2 ;; esac
family_args=(--embedding-placement "$embedding")
chat_template_mounts=()
# Vision-only override: no template inference and no changes to text-only prompts.
chat_template_from="$(get CHAT_TEMPLATE_FROM)"
if [[ -n "$chat_template_from" && "$vision" != off ]]; then
  [[ "$family" == glm5_flash ]] || release_die "CHAT_TEMPLATE_FROM currently applies only to GLM Flash vision"
  if [[ -d "$chat_template_from" ]]; then
    chat_template_from="$(readlink -f "$chat_template_from")"
    release_validate_path_setting CHAT_TEMPLATE_FROM "$chat_template_from"
    if release_path_within "$chat_template_from" "$hub"; then
      chat_template_from="/root/.cache/huggingface/hub${chat_template_from#"$hub"}"
    else
      chat_template_mounts=(-v "$chat_template_from:$chat_template_from:ro")
    fi
  else
    [[ "$chat_template_from" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] &&
      release_path_has_no_dot_segment "$chat_template_from" || release_die "CHAT_TEMPLATE_FROM must be an existing snapshot or ORG/MODEL HF id"
    snapshot_of "$chat_template_from" "" >/dev/null || exit 2
  fi
  family_args+=(--chat-template-from "$chat_template_from")
fi
family_args+=(--vision "$vision" --audio "$audio")
dspark_args=()
if [[ $family == mimo_v2 ]]; then
  case "$(get MIMO_WEIGHT_POLICY auto)" in
    auto) ;; # The runtime and planner share the metadata qualifier.
    checkpoint) family_args+=(--weight-policy checkpoint) ;;
    *) echo "MIMO_WEIGHT_POLICY must be auto or checkpoint" >&2; exit 2 ;;
  esac
  for projection in HEAD O_PROJ; do
    mode="$(get "MIMO_FP8_$projection")"
    option=--fp8-head
    [[ $projection != O_PROJ ]] || option=--fp8-o-proj
    case "$mode" in
      ""|auto) ;;
      on) family_args+=("$option" true) ;;
      off) family_args+=("$option" false) ;;
      *) echo "MIMO_FP8_$projection must be auto, on or off" >&2; exit 2 ;;
    esac
  done
fi
case "$speculator" in
  dflash2|dspark)
    drafter="$(key SPECULATOR_MODEL_ID DRAFT_MODEL_ID)"
    [[ -n "$drafter" ]] || drafter="$default_drafter"
    if [[ $speculator == dspark && $family == deepseek_v4 ]]; then
      dspark_args=(--dspark)
    elif [[ -n "$drafter" ]]; then
      draft_snapshot="$(snapshot_of "$drafter" "$(key SPECULATOR_MODEL_REVISION DRAFT_MODEL_REVISION)")" ||
        { [[ -z "$default_drafter" ]] || echo "download it (hf download $drafter) or set SPECULATOR=off" >&2; exit 1; }
      draft_args=(--draft "$draft_snapshot")
    elif [[ $family == mimo_v2 ]]; then
      draft_args=(--draft "$snapshot")
    else
      echo "SPECULATOR=$speculator needs SPECULATOR_MODEL_ID (a ${speculator/dflash2/DFlash2} checkpoint)" >&2; exit 2
    fi ;;
  mtp)
    mtp_depth=1
    [[ $qwen_exl3 != 1 || $qwen_mtp != 1 ]] || mtp_depth=3
    family_args+=(--mtp "$(key SPECULATOR_DEPTH MTP "$mtp_depth")") ;;
esac
# SPECULATOR_DRAFTS: adaptive (default) or a fixed draft count per cycle
# (DFlash2 on GLM 5.x, GLM 5.3 Flash and MiMo V2; dSpark on GLM 5.3 Flash), for policy A/B runs.
drafts="$(get SPECULATOR_DRAFTS adaptive)"
if [[ "$drafts" != adaptive ]]; then
  [[ "$drafts" =~ ^[0-9]+$ && ($speculator == dflash2 || $family:$speculator == glm5_flash:dspark) ]] ||
    { echo "SPECULATOR_DRAFTS must be adaptive or a draft count, with SPECULATOR=dflash2 (or dspark on GLM 5.3 Flash)" >&2; exit 2; }
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
if [[ ( $family == mimo_v2 || $family == qwen4 || $family == glm5_flash ) && $vision != off ]]; then
  # Text-only frozen daemons can predate the optional media-cache flag.
  [[ -z "$(get MEDIA_CACHE_BYTES)" ]] || family_args+=(--media-cache-bytes "$(get MEDIA_CACHE_BYTES)")
fi
if [[ $family == mimo_v2 ]]; then
  # POOL_TOKENS=auto: the largest pool every GPU admits after all fixed costs (up to 2M tokens).
  # Default auto (measured 2026-10-03, MiMo V2.6 Pro 2 RTX + 6: 131072 -> 2,097,152 tokens, C1/C4/8K
  # prefill unchanged); a number pins the pool.
  mimo_pool="$(get POOL_TOKENS auto)"; [[ "$mimo_pool" != auto ]] || mimo_pool=0
  family_args+=(--pool-tokens "$mimo_pool")
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
# EXPERT_INPUT is opt-in for MiMo's Spark exchange; unset preserves image defaults.
expert_input="$(get EXPERT_INPUT)"
if [[ -n "$expert_input" ]]; then
  [[ "$family" == mimo_v2 ]] || { echo "EXPERT_INPUT applies to MiMo checkpoints" >&2; exit 2; }
  case "$expert_input" in
    fp8|bf16|bf16-decode) family_args+=(--expert-input "$expert_input") ;;
    *) echo "EXPERT_INPUT must be fp8, bf16 or bf16-decode" >&2; exit 2 ;;
  esac
fi
# Qwen 3.8: QWEN_FP8_DECODE=on|off converts the GDN/attention projections to one
# resident E4M3 copy; QWEN_FP8_HEAD=on|off does the same for the head target and
# MTP share. Unset keeps the engine defaults.
if [[ $family == qwen4 ]]; then
  for key in FP8_DECODE:--fp8-decode FP8_HEAD:--mtp-fp8-head; do
    mode="$(get "QWEN_${key%%:*}")"
    case "$mode" in
      "") ;;
      on) family_args+=("${key#*:}" true) ;;
      off) family_args+=("${key#*:}" false) ;;
      *) echo "QWEN_${key%%:*} must be on or off" >&2; exit 2 ;;
    esac
  done
fi
# POOL_TOKENS=auto (GLM 5.3, GLM 5.3 Flash, MiMo, Qwen, DeepSeek V4): the largest pool the GPUs hold after the
# planner's remaining costs (up to 2M tokens).
# GLM 5.3 and Qwen default to auto; Qwen's former 32768-token pool admitted
# only seven 4096-output requests, below the default eight serving lanes.
# DeepSeek V4 keeps its engine default.
glm_default=""; [[ ! $family =~ ^(glm5|qwen4)$ ]] || glm_default=auto
if [[ $family =~ ^(glm5|qwen4|deepseek_v4)$ && -n "$(get POOL_TOKENS "$glm_default")" ]]; then
  pool="$(get POOL_TOKENS "$glm_default")"
  if [[ "$pool" == auto ]]; then
    pool=0
  fi
  family_args+=(--pool-tokens "$pool")
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
# DeepSeek V4 keeps complete expert layers local while memory permits. Honor
# an explicit limit; zero leaves the backbone experts on the Sparks.
if [[ $serve == serve-dsv4 ]]; then
  local_layers="$(get RTX_EXPERT_LAYERS auto)"
  case "$local_layers" in
    ""|auto) ;;
    *[!0-9]*) echo "RTX_EXPERT_LAYERS must be auto or a nonnegative integer" >&2; exit 2 ;;
    *) family_args+=(--local-expert-layers "$local_layers") ;;
  esac
fi
# GLM, GLM Flash, MiMo, Qwen: L2_PREFETCH (off, auto = 3/4 of the L2, or MiB;
# unset: auto for GLM 5.3 and GLM 5.3 Flash, off for MiMo and Qwen) pulls the
# next layer's weights into L2 during each one-lane decode step's Spark exchange; FP8_SCALES (amax, pow2, best) is the scale rule of the FP8
# copies made from BF16 weights at load.
if [[ $serve != serve-dsv4 ]]; then
  [[ -z "$(get L2_PREFETCH)" ]] || family_args+=(--l2-prefetch "$(get L2_PREFETCH)")
  [[ -z "$(get FP8_SCALES)" ]] || family_args+=(--fp8-scales "$(get FP8_SCALES)")
  if [[ ${#draft_args[@]} -gt 0 ]]; then
    if [[ $family == mimo_v2 ]]; then
      mimo_draft_default=""
      if [[ $mimo_flash_mopd == 1 && $speculator == dflash2 && -z "$drafter" ]]; then
        mimo_draft_default=on
      fi
      case "$(key SPECULATOR_FP8 DRAFT_FP8 "$mimo_draft_default")" in
        "") ;;
        auto) family_args+=(--draft-representation checkpoint) ;;
        on) family_args+=(--draft-fp8 true) ;;
        off) family_args+=(--draft-fp8 false) ;;
        *) echo "MiMo SPECULATOR_FP8/DRAFT_FP8 must be auto, on or off" >&2; exit 2 ;;
      esac
    elif [[ $family == glm5 || $family == glm5_flash ]]; then
      case "$(key SPECULATOR_FP8 DRAFT_FP8 auto)" in
        auto) ;; # The engine default: single-copy FP8 drafter weights.
        on) family_args+=(--draft-fp8 true) ;;
        off) family_args+=(--draft-fp8 false) ;;
        *) echo "SPECULATOR_FP8 must be auto, on or off" >&2; exit 2 ;;
      esac
      [[ -z "$(get DRAFT_CONTEXT_SLOTS)" ]] || family_args+=(--draft-context-slots "$(get DRAFT_CONTEXT_SLOTS)")
      [[ -z "$(get DRAFT_SEQUENCES)" ]] || family_args+=(--draft-sequences "$(get DRAFT_SEQUENCES)")
    elif [[ "$(key SPECULATOR_FP8 DRAFT_FP8 on)" == off ]]; then
      family_args+=(--draft-fp8 false)
    fi
  fi
fi
# SERVED_MODEL_ID: the public model id (default: the checkpoint's Hugging Face id).
served_args=()
served="$(get SERVED_MODEL_ID)"
[[ -z "$served" ]] || served_args=(--model-id "$served")
coordinator_image="$(get COORDINATOR_DOCKER_INFERENCE)"
# --wip SLOT serves a ./wip.sh slot: the development images run its artifacts, staged from
# the WIP containers into a release-shaped /opt/cuteafd layout per host (as ./run.sh --wip
# does for DeepSeek V4.1).
wip_layout="" wip_mount_args=() wip_worker_args=""
if [[ -n "$wip_slot" ]]; then
  [[ "$wip_slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || { echo "invalid WIP slot name: $wip_slot" >&2; exit 2; }
  coordinator_image="$(get COORDINATOR_DOCKER_DEV cuteafd-coordinator-dev)"
  wip_layout="$HOME/.cache/cuteafd/wip-run/$wip_slot"
  wip_mount_args=(-v "$wip_layout/bin:/opt/cuteafd/bin:ro" -v "$wip_layout/lib:/opt/cuteafd/lib:ro"
    -v "$wip_layout/share:/opt/cuteafd/share:ro"
    -e "PATH=/opt/cuteafd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    -e CUTEAFD_NATIVE_LIB=/opt/cuteafd/lib/libcuteafd_native.so
    --entrypoint /opt/cuteafd/share/release-entrypoint.sh)
  wip_worker_args="-v \$HOME/.cache/cuteafd/wip-run/$wip_slot/bin:/opt/cuteafd/bin:ro \
    -v \$HOME/.cache/cuteafd/wip-run/$wip_slot/lib:/opt/cuteafd/lib:ro \
    -v \$HOME/.cache/cuteafd/wip-run/$wip_slot/share:/opt/cuteafd/share:ro \
    -e PATH=/opt/cuteafd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e CUTEAFD_NATIVE_LIB=/opt/cuteafd/lib/libcuteafd_native.so --entrypoint /opt/cuteafd/share/release-entrypoint.sh"
fi
# SPECULATION_TRACE=/abs/host/file.jsonl: the per-cycle speculation trace
# (CUTEAFD_SPECULATION_TRACE, written by serve-glm and serve-qwen4; read by
# scripts/qualify/glm5/glm-draft-trace.py and qualify/qwen4/qwen4-draft-trace.py).
# COORDINATOR_TRACE is its pre-rename key; images older than the rename read
# the family variables, which are set too for one release.
# PROBE_DUMP_ROOT=/abs/host/dir: benchmark probes (POST /v1/bench/probe) may stream
# full-vocabulary rows (dump_rows) into new leaves under it (CUTEAFD_PROBE_DUMP_ROOT, mounted
# at the same path); unset, remote probes cannot write rows.
probe_args=()
probe_root="$(get PROBE_DUMP_ROOT)"
if [[ -n "$probe_root" ]]; then
  [[ "$probe_root" == /* && -d "$probe_root" ]] || { echo "PROBE_DUMP_ROOT must be an existing absolute directory" >&2; exit 2; }
  probe_args=(-v "$probe_root:$probe_root" -e "CUTEAFD_PROBE_DUMP_ROOT=$probe_root")
fi
trace_args=()
trace="$(key SPECULATION_TRACE COORDINATOR_TRACE)"
if [[ -n "$trace" ]]; then
  mkdir -p "$(dirname "$trace")"
  trace_args=(-v "$(dirname "$trace"):$(dirname "$trace")" -e "CUTEAFD_SPECULATION_TRACE=$trace"
    -e "CUTEAFD_GLM_TRACE=$trace" -e "CUTEAFD_QWEN4_TRACE=$trace")
fi
# GLM 5.3 Flash serving captures every decode graph at startup (CUTEAFD_GLMF_STARTUP_GRAPHS) unless a
# graph budget (GLM5_FLASH_GRAPH_BUDGET_MIB) asks for lazily captured ones; preserve explicit overrides.
if [[ $family == glm5_flash ]]; then
  case "$(get GLM5_FLASH_STARTUP_GRAPHS)" in
    "") ;;
    on)
      [[ -z "$(get GLM5_FLASH_GRAPH_BUDGET_MIB)" ]] || { echo "GLM5_FLASH_STARTUP_GRAPHS=on holds every decode graph;" \
        "GLM5_FLASH_GRAPH_BUDGET_MIB bounds lazily captured ones: set one of them" >&2; exit 2; }
      trace_args+=(-e CUTEAFD_GLMF_STARTUP_GRAPHS=1) ;;
    off) trace_args+=(-e CUTEAFD_GLMF_STARTUP_GRAPHS=0) ;;
    *) echo "GLM5_FLASH_STARTUP_GRAPHS must be on or off" >&2; exit 2 ;;
  esac
fi
# Qwen serving defaults to qualified startup graphs; preserve explicit overrides.
if [[ $family == qwen4 ]]; then
  case "$(get QWEN_STARTUP_GRAPHS)" in
    "") ;;
    on) trace_args+=(-e CUTEAFD_QWEN4_STARTUP_GRAPHS=1) ;;
    off) trace_args+=(-e CUTEAFD_QWEN4_STARTUP_GRAPHS=0) ;;
    *) echo "QWEN_STARTUP_GRAPHS must be on or off" >&2; exit 2 ;;
  esac
fi
spark_image="$(get SPARK_EXPERT_DOCKER_INFERENCE)"
[[ -z "$wip_slot" ]] || spark_image="$(get SPARK_EXPERT_DOCKER_DEV cuteafd-spark-expert-dev)"
port="$(get EXPERT_PORT 19441)"
addr="$(get ADDR 0.0.0.0:8000)"
budget="$(get SPARK_DEVICE_BUDGET_BYTES 107374182400)"
gpu="$(get COORDINATOR_GPU 0)"
# Two coordinator GPUs (RTX_GPUS=auto/2 with COORDINATOR_GPU as V4.1's two-RTX config picks
# the other card, or an explicit COORDINATOR_GPUS=0,1): families with a
# head split (MiMo V2 Flash/Pro, GLM 5.x, GLM 5.3 Flash, DeepSeek V4) split every layer's
# attention heads and dense / shared-expert MLPs over both by default (COORDINATOR_SPLIT=auto
# or heads), one hidden all-reduce per layer over peer memory; experts, router, head and
# drafter stay on the first GPU. COORDINATOR_SPLIT=off serves from the first GPU alone. Auto
# selection uses one GPU for checkpoints without a split and for splits that are opt-in
# (`split_opt_in`: measured not to pay at the reference layouts); an explicit split request
# (RTX_GPUS=2, COORDINATOR_GPUS=a,b or COORDINATOR_SPLIT=heads) fails before containers
# start when the checkpoint has none. The container sees both GPUs in host order.
# COORDINATOR_SPLIT_GPU names the second GPU when only
# COORDINATOR_GPU is set (default the other of 0/1).
rtx_gpus="$(get RTX_GPUS auto)"
case "$rtx_gpus" in auto|1|2) ;; *) echo "RTX_GPUS must be auto, 1 or 2" >&2; exit 2 ;; esac
physical_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true)"
coordinator_gpus="$(get COORDINATOR_GPUS)"
explicit_coordinator_gpus="$coordinator_gpus"
if [[ -z "$coordinator_gpus" ]]; then
  coordinator_gpus="$gpu"
  case "$rtx_gpus" in
    1) ;;
    2|auto)
      other="$(awk -v first="$gpu" '/^[0-9]+$/ && $0 != first {print; exit}' <<<"$physical_gpus")"
      [[ -z "$other" ]] || coordinator_gpus="$gpu,$other" ;;
  esac
fi
[[ "$coordinator_gpus" =~ ^[0-9]+(,[0-9]+)?$ ]] ||
  { echo "COORDINATOR_GPUS must name one or two GPU indices" >&2; exit 2; }
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
explicit_split=0
if [[ "$rtx_gpus" == 2 || "$explicit_coordinator_gpus" == *,* || "$split" == heads ]]; then
  explicit_split=1
fi
split_hint=""
split_opt_in=""
case "$family:$model_type" in
  deepseek_v4:*|glm5:*|glm5_flash:*|mimo_v2:mimo_v2|mimo_v2:mimo_v2_flash) ;;
  qwen4:*) split_hint="add Qwen head-split GDN/GQA/shared-expert kernels and sharded recurrent/KV state" ;;
  mimo_v2:*) split_hint="add MiMo head-split attention/projection kernels for $model_type" ;;
  *) split_hint="add coordinator head-split kernels for $model_type" ;;
esac
if [[ -n "$split_opt_in" && "$split" == auto && "$explicit_split" == 0 && -z "$split_hint" ]]; then
  echo "note: $family head split is opt-in ($split_opt_in); auto selected GPU $gpu alone" >&2
  second=""
fi
if [[ "$split" != off && "$explicit_split" == 1 ]]; then
  [[ -z "$split_hint" ]] ||
    echo "note: $family ($model_type) has no head split yet ($split_hint); serving from GPU $gpu alone" >&2
  [[ -n "$second" ]] ||
    { echo "RTX_GPUS=2 requires two physical coordinator GPUs; only GPU $gpu was selected" >&2; exit 2; }
fi
gpus="device=$gpu"
head_split=0
if [[ -n "$second" && "$split" != off ]]; then
  [[ "$second" =~ ^[0-9]+$ ]] || { echo "COORDINATOR_SPLIT_GPU must be a GPU index" >&2; exit 2; }
  [[ "$second" != "$gpu" ]] || { echo "the second coordinator GPU must differ from the first" >&2; exit 2; }
  if [[ -z "$split_hint" ]]; then
    for selected in "$gpu" "$second"; do
      grep -qx "$selected" <<<"$physical_gpus" ||
        { echo "two-GPU head split requires physical GPU $selected, but nvidia-smi did not report it" >&2; exit 2; }
    done
    lower=$((gpu < second ? gpu : second)) upper=$((gpu < second ? second : gpu))
    gpus="\"device=$lower,$upper\""
    head_split=1
    family_args+=(--device $((gpu == lower ? 0 : 1)) --split-device $((gpu == lower ? 1 : 0)))
  elif [[ "$explicit_split" != 1 ]]; then
    echo "note: $family ($model_type) has no head split; auto selected GPU $gpu alone" >&2
  fi
fi
if [[ "$glmf_memory" == compact && "$head_split" == 1 ]]; then
  echo "GLM5_FLASH_MEMORY=compact serves from one GPU, and this launch splits heads over GPUs $gpu and $second;" \
    "set RTX_GPUS=1 (or COORDINATOR_SPLIT=off)" >&2
  exit 2
fi
# GLM 5.3 Flash: the MLA, dense and shared-expert projections are FP8 only,
# from the official FP8 release (GLM5_FLASH_FP8_MODEL_ID; "off" requires native
# FP8 block tensors in the primary checkpoint, else BF16 ones are quantized to
# 128x128 blocks at load). Resolve precision after the serving split: both
# layouts default to row128 KDA and an FP8 head; the two-GPU head split adds
# token-row KDA output ownership (--kda-output-shard --kda-prefill-expanded),
# which avoids rounding half-K BF16 partials (2026-10-05: C1 +7.6%, paired
# top-1 453 -> 460/512, KL +0.0008). GLM5_FLASH_KDA_SPLIT=partials keeps the
# old split path. Explicit current or legacy keys override either default.
# Each weight has one resident copy; the DFlash2 drafter stays FP8 by default.
# Its MLA pools hold POOL_TOKENS tokens (a key every
# family with a paged KV pool reads). GLM5_FLASH_FP8_PREFILL lists the prefill
# projections that run W8A8 (E4M3 activations per 128-K block): unset = the
# engine default mla,ffn, a list of mla,ffn,kda-in,kda-o / all (kda-* need
# GLM5_FLASH_KDA_FP8 row128/channel), or off (every FP8 weight W8A16). The
# GLMF_* spellings still work for one release.
if [[ $family == glm5_flash ]]; then
  fp8_model="$(key GLM5_FLASH_FP8_MODEL_ID GLMF_FP8_MODEL_ID zai-org/GLM-5.3-Flash)"
  if [[ "$fp8_model" != off ]]; then
    fp8_snapshot="$(snapshot_of "$fp8_model" "$(key GLM5_FLASH_FP8_MODEL_REVISION GLMF_FP8_MODEL_REVISION)")" || exit 1
    family_args+=(--fp8-decode --fp8-snapshot "$fp8_snapshot")
  fi
  kda_fp8="$(key GLM5_FLASH_KDA_FP8 GLMF_KDA_FP8 auto)"
  case "$kda_fp8" in
    ""|auto) kda_fp8=row128 ;;
    off|row128|channel) ;;
    *) echo "GLM5_FLASH_KDA_FP8 must be auto, off, row128 or channel" >&2; exit 2 ;;
  esac
  # Default auto (GLM 5.3 Flash 1 RTX + 2: 65536 -> 2,097,152 tokens, 44 GiB still free, speed unchanged).
  glmf_pool="$(get POOL_TOKENS auto)"; [[ "$glmf_pool" != auto ]] || glmf_pool=0
  family_args+=(--kda-fp8 "$kda_fp8" --pool-tokens "$glmf_pool")
  kda_split="$(get GLM5_FLASH_KDA_SPLIT auto)"
  case "$kda_split" in
    ""|auto) [[ $head_split == 0 || $kda_fp8 == off ]] || family_args+=(--kda-output-shard --kda-prefill-expanded) ;;
    partials) ;;
    *) echo "GLM5_FLASH_KDA_SPLIT must be auto or partials" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_PREFILL_LANES (1-4) lanes of GLM5_FLASH_PREFILL_LANE_ROWS rows (whole 64-row
  # pages up to 4096) take each Spark prefill chunk, every lane with its own exchange in flight;
  # unset keeps the engine's two lanes of 4096.
  prefill_lanes="$(get GLM5_FLASH_PREFILL_LANES)"
  if [[ -n "$prefill_lanes" ]]; then
    [[ "$prefill_lanes" =~ ^[1-4]$ ]] || { echo "GLM5_FLASH_PREFILL_LANES must be 1 to 4" >&2; exit 2; }
    family_args+=(--prefill-lanes "$prefill_lanes")
  fi
  lane_rows="$(get GLM5_FLASH_PREFILL_LANE_ROWS)"
  if [[ -n "$lane_rows" ]]; then
    if ! [[ "$lane_rows" =~ ^[1-9][0-9]*$ ]] || ((lane_rows > 4096 || lane_rows % 64 != 0)); then
      echo "GLM5_FLASH_PREFILL_LANE_ROWS must be a multiple of 64 up to 4096" >&2; exit 2
    fi
    family_args+=(--prefill-lane-rows "$lane_rows")
  fi
  # GLM5_FLASH_HEADROOM_GIB: GPU memory an automatic pool leaves free for runtime growth when
  # every other allocation precedes it (unset: the engine's 2 GiB; a 32 GB card takes 1).
  headroom="$(get GLM5_FLASH_HEADROOM_GIB)"
  if [[ -n "$headroom" ]]; then
    [[ "$headroom" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "GLM5_FLASH_HEADROOM_GIB must be a non-negative size in GiB" >&2; exit 2; }
    family_args+=(--headroom-gib "$headroom")
  fi
  # GLM5_FLASH_GRAPH_BUDGET_MIB: device memory the captured decode graphs may hold (the least
  # recently launched leave past it); unset: unbounded, with the planner's allowance reserved.
  graph_budget="$(get GLM5_FLASH_GRAPH_BUDGET_MIB)"
  if [[ -n "$graph_budget" ]]; then
    [[ "$graph_budget" =~ ^[1-9][0-9]*$ ]] || { echo "GLM5_FLASH_GRAPH_BUDGET_MIB must be a positive whole number of MiB" >&2; exit 2; }
    family_args+=(--graph-budget-mib "$graph_budget")
  fi
  # GLM5_FLASH_INDEX_CACHE: the DSA index cache, keys (default: every token's BF16 key | gate
  # row beside its latent record, 11,804 B per token) or compact (the pooled keys plus each
  # sequence's open pool, 6,172 B per token, the same pooled keys bit for bit; one GPU only,
  # a head split keeps keys).
  index_cache="$(get GLM5_FLASH_INDEX_CACHE keys)"
  case "$index_cache" in
    ""|keys) ;;
    compact) family_args+=(--index-cache compact) ;;
    *) echo "GLM5_FLASH_INDEX_CACHE must be keys or compact" >&2; exit 2 ;;
  esac
  fp8_head="$(key GLM5_FLASH_FP8_HEAD GLMF_FP8_HEAD auto)"
  case "$fp8_head" in
    ""|auto) fp8_head=on ;;
    on|off) ;;
    *) echo "GLM5_FLASH_FP8_HEAD must be auto, on or off" >&2; exit 2 ;;
  esac
  case "$fp8_head" in
    on) family_args+=(--fp8-head true) ;;
    off) family_args+=(--fp8-head false) ;;
  esac
  fp8_prefill="$(key GLM5_FLASH_FP8_PREFILL GLMF_FP8_PREFILL)"
  if [[ " ${family_args[*]} " == *" --kda-output-shard "* ]]; then
    case ",$fp8_prefill," in
      *,all,*|*,kda-o,*)
        echo "GLM5_FLASH_FP8_PREFILL=$fp8_prefill (KDA output W8A8) does not combine with the split's token-row KDA output; set GLM5_FLASH_KDA_SPLIT=partials or drop kda-o" >&2
        exit 2 ;;
    esac
  fi
  case ",$fp8_prefill," in
    *,all,*|*,kda-in,*|*,kda-o,*)
      if [[ $kda_fp8 == off ]]; then
        echo "GLM5_FLASH_FP8_PREFILL=$fp8_prefill runs KDA W8A8 over FP8 KDA weights; set GLM5_FLASH_KDA_FP8=row128 or channel" >&2
        exit 2
      fi ;;
  esac
  case "$fp8_prefill" in
    "") ;;
    off) family_args+=(--fp8-prefill none) ;;
    *) family_args+=(--fp8-prefill "$fp8_prefill") ;;
  esac
  # GLM5_FLASH_PREFIX_MARKS: where prefix-cache snapshots keep their KDA state marks: unset or
  # arena (the engine default: a 2C + 2 device arena beside the KV pool) or pool (units of the
  # KV pool itself, evicted like any snapshot's rows; no arena to reserve). Pool marks turn the
  # pinned host tier on (HOST_CACHE_BYTES=auto) unless HOST_CACHE_BYTES is set (0 keeps it off):
  # snapshots the pool evicts move to RAM instead of being lost.
  prefix_marks="$(get GLM5_FLASH_PREFIX_MARKS)"
  case "$prefix_marks" in
    "") ;;
    arena|pool) family_args+=(--prefix-marks "$prefix_marks") ;;
    *) echo "GLM5_FLASH_PREFIX_MARKS must be arena or pool" >&2; exit 2 ;;
  esac
  [[ "$prefix_marks" != pool || -n "$(get HOST_CACHE_BYTES)" ]] || family_args+=(--host-cache-bytes auto)
  # GLM5_FLASH_KDA_STATE: the KDA recurrent state, f32 (default) or bf16: half the state and
  # prefix-mark bytes, computed in FP32 and rounded after every decode/verify/commit row and at
  # each chunked-prefill window end (bf16-tile: after every 16-row prefill tile). It runs the
  # BF16-projection KDA programs on one GPU (GLM5_FLASH_KDA_FP8=off, no head split).
  kda_state="$(get GLM5_FLASH_KDA_STATE f32)"
  case "$kda_state" in
    ""|f32) ;;
    bf16|bf16-tile)
      if [[ $kda_fp8 != off || $head_split != 0 ]]; then
        echo "GLM5_FLASH_KDA_STATE=$kda_state runs the BF16-projection KDA programs on one GPU; set GLM5_FLASH_KDA_FP8=off without a head split" >&2
        exit 2
      fi
      family_args+=(--kda-state "$kda_state") ;;
    *) echo "GLM5_FLASH_KDA_STATE must be f32, bf16 or bf16-tile" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_PREFILL_BATCH: on (the engine's default: the prompts that wait together prefill in
  # one pass, each sequence's own programs over its rows, the bits of its own pass, one Spark wave
  # per MoE layer for all of them) or off (one prefill pass per prompt). Unset passes nothing.
  prefill_batch="$(get GLM5_FLASH_PREFILL_BATCH)"
  case "$prefill_batch" in
    "") ;;
    on) family_args+=(--prefill-batch true) ;;
    off) family_args+=(--prefill-batch false) ;;
    *) echo "GLM5_FLASH_PREFILL_BATCH must be on or off" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_VERIFY_POLICY: which drafts a speculative step verifies under its verify budget
  # (GLM5_FLASH_DECODE_ROWS: 64 rows, or 127 on an RTX 5090 at 128): auto (the engine's default:
  # chain from GLM5_FLASH_VERIFY_CHAIN_MIN_SEQUENCES active sequences, 9 at 64 rows and 16 at 127
  # unless set, cost below), cost (the same room for every sequence, the cost model's depth within
  # it) or chain (each sequence's drafts cut at GLM5_FLASH_SPEC_TAU, default 0.7, of cumulative
  # draft probability, then the least likely drafts across sequences dropped first). Unset passes
  # nothing.
  verify_policy="$(get GLM5_FLASH_VERIFY_POLICY)"
  case "$verify_policy" in
    "") ;;
    auto|cost|chain) family_args+=(--verify-policy "$verify_policy") ;;
    *) echo "GLM5_FLASH_VERIFY_POLICY must be auto, cost or chain" >&2; exit 2 ;;
  esac
  chain_from="$(get GLM5_FLASH_VERIFY_CHAIN_MIN_SEQUENCES)"
  if [[ -n "$chain_from" ]]; then
    [[ "$chain_from" =~ ^[1-9][0-9]*$ ]] || { echo "GLM5_FLASH_VERIFY_CHAIN_MIN_SEQUENCES must be a positive whole number" >&2; exit 2; }
    family_args+=(--verify-chain-min-sequences "$chain_from")
  fi
  spec_tau="$(get GLM5_FLASH_SPEC_TAU)"
  if [[ -n "$spec_tau" ]]; then
    [[ "$spec_tau" =~ ^(0?[.][0-9]*[1-9][0-9]*|1([.]0*)?)$ ]] || { echo "GLM5_FLASH_SPEC_TAU must be in (0, 1]" >&2; exit 2; }
    family_args+=(--spec-tau "$spec_tau")
  fi
  # GLM5_FLASH_DECODE_ROWS: the most rows one decode or verify step takes, 64 (default: the decode
  # programs' rows) or 128: steps past 64 rows run the wide _m128 programs (16 sequences verify 7
  # drafts each instead of 3; fewer rows keep the 64-row programs). Their replay records take
  # 321 MB more. One GPU only.
  decode_rows="$(get GLM5_FLASH_DECODE_ROWS 64)"
  case "$decode_rows" in
    ""|64) ;;
    128)
      if [[ $head_split != 0 ]]; then
        echo "GLM5_FLASH_DECODE_ROWS=128 runs the wide decode programs on one GPU; serve it without a head split" >&2
        exit 2
      fi
      family_args+=(--decode-rows 128) ;;
    *) echo "GLM5_FLASH_DECODE_ROWS must be 64 or 128" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_EXL3_SCHEDULE: the Spark EXL3 decode schedule, default or gb10. gb10 runs the
  # m1-gb10/m80-gb10 TP4 exports: the same products and sums (the same bits), with the weight
  # words staged evict-first in L2, and at m80 64x128 tiles at two CTAs per SM.
  exl3_schedule="$(get GLM5_FLASH_EXL3_SCHEDULE default)"
  case "$exl3_schedule" in
    default) ;;
    gb10)
      [[ "$ranks" != 0 ]] || { echo "GLM5_FLASH_EXL3_SCHEDULE=gb10 is a Spark expert schedule; SPARK_COUNT=0 runs none" >&2; exit 2; }
      spark_worker_args+=" --exl3-schedule gb10" ;;
    *) echo "GLM5_FLASH_EXL3_SCHEDULE must be default or gb10" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_EXL3_WORKER_PATH: how a Spark EXL3 worker uploads a call's inputs and waits for its
  # GPU work. async (the default): pinned staging, one batched asynchronous copy, the hidden rows
  # decoded in place and a polled stream; blocking: the earlier copies and a blocking synchronize.
  # Same bits.
  exl3_worker_path="$(get GLM5_FLASH_EXL3_WORKER_PATH async)"
  case "$exl3_worker_path" in
    async) ;;
    blocking) spark_worker_env+=" -e CUTEAFD_EXL3_WORKER_PATH=blocking" ;;
    *) echo "GLM5_FLASH_EXL3_WORKER_PATH must be async or blocking" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_EXL3_ROUTE_DUMP: an absolute directory on every Spark. Each worker appends its
  # calls' routes to DIR/routes.<executor>.bin, which SparkInfer's GLM Flash decode benchmark
  # replays (--routes file:PATH); GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS caps the calls (default 200000).
  exl3_route_dump="$(get GLM5_FLASH_EXL3_ROUTE_DUMP "")"
  exl3_route_dump_calls="$(get GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS 200000)"
  if [[ -n "$exl3_route_dump" ]]; then
    [[ "$exl3_route_dump" =~ ^/[A-Za-z0-9._/-]+$ ]] ||
      { echo "GLM5_FLASH_EXL3_ROUTE_DUMP must be an absolute directory" >&2; exit 2; }
    [[ "$exl3_route_dump_calls" =~ ^[1-9][0-9]{0,8}$ ]] ||
      { echo "GLM5_FLASH_EXL3_ROUTE_DUMP_CALLS must be a positive call count" >&2; exit 2; }
    [[ "$ranks" != 0 ]] || { echo "GLM5_FLASH_EXL3_ROUTE_DUMP records Spark expert calls; SPARK_COUNT=0 runs none" >&2; exit 2; }
    spark_worker_env+=" -v $exl3_route_dump:$exl3_route_dump -e CUTEAFD_EXL3_ROUTE_DUMP=$exl3_route_dump/routes"
    spark_worker_env+=" -e CUTEAFD_EXL3_ROUTE_DUMP_CALLS=$exl3_route_dump_calls"
  fi
  # MAX_CONTEXT_TOKENS (the longest request, prompt plus output; default 8192) goes up to
  # 1,048,576. Past 131,072 the image needs GLM 5.3 Flash's own extent (built with
  # CUTEAFD_GLMF_MAX_CONTEXT, e.g. 1048576: its index top-k exported there too); start-up names a
  # shorter one before loading weights. A 1M request holds about 6.5 GB of pool (compact index).
  glmf_context="$(get MAX_CONTEXT_TOKENS 8192)"
  if ! [[ "$glmf_context" =~ ^[1-9][0-9]{0,6}$ ]] || ((glmf_context > 1048576)); then
    echo "MAX_CONTEXT_TOKENS must be 1 to 1048576 for GLM 5.3 Flash" >&2
    exit 2
  fi
  host_bytes="$(get HOST_CACHE_BYTES)"
  if ((glmf_context > 131072)) && [[ "$host_bytes" == auto || ($prefix_marks == pool && -z "$host_bytes") ]]; then
    echo "note: HOST_CACHE_BYTES=auto sizes the RAM tier for $(get PREFIX_CACHE_ENTRIES 20) snapshots a bank of" \
      "MAX_CONTEXT_TOKENS=$glmf_context tokens, up to all but 10% of free RAM; set HOST_CACHE_BYTES (e.g. 64GiB) to bound it" >&2
  fi
  # GLM5_FLASH_REPLAY_RECORDS: where the KDA speculative replay records live, own (default: their
  # own 321 MB at 64 decode rows, 643 MB at 128) or shared (the prefill lanes' scratch, which no
  # decode step reads). One GPU with an automatic pool and Spark experts.
  replay_records="$(get GLM5_FLASH_REPLAY_RECORDS own)"
  case "$replay_records" in
    ""|own) ;;
    shared)
      if [[ $head_split != 0 ]]; then
        echo "GLM5_FLASH_REPLAY_RECORDS=shared keeps the records in one GPU's prefill scratch; serve it without a head split" >&2
        exit 2
      fi
      # The engine sizes such a pool after the step workspaces: an automatic pool beside Spark
      # experts, or any pool under a coordinator GPU budget.
      if [[ -z "$coordinator_budget" && ( "$glmf_pool" != 0 || "$ranks" == 0 ) ]]; then
        echo "GLM5_FLASH_REPLAY_RECORDS=shared needs the step workspaces allocated before the pool: an automatic pool" \
          "(POOL_TOKENS=auto) with Spark experts, or COORDINATOR_GPU_BUDGET_GIB" >&2
        exit 2
      fi
      family_args+=(--replay-records shared) ;;
    *) echo "GLM5_FLASH_REPLAY_RECORDS must be own or shared" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_DECODE_ROW_BUCKETS: on pads every speculative step past 16 rows to a row bucket (4-row
  # steps to 64, 8-row steps to 128), so the decode graphs hold a few row shapes. One GPU; off by default.
  row_buckets="$(get GLM5_FLASH_DECODE_ROW_BUCKETS off)"
  case "$row_buckets" in
    ""|off) ;;
    on)
      if [[ $head_split != 0 ]]; then
        echo "GLM5_FLASH_DECODE_ROW_BUCKETS=on pads one GPU's decode steps; serve it without a head split" >&2
        exit 2
      fi
      family_args+=(--decode-row-buckets) ;;
    *) echo "GLM5_FLASH_DECODE_ROW_BUCKETS must be on or off" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_DRAFT_HEAD: the drafter's vocabulary head over the BF16 head, exact (default: as the
  # target's own head, FP32 products and sums on CUDA cores) or tensor (from two draft blocks of 8
  # rows, a BF16 tensor-core GEMM with FP32 accumulation that reads the head once). Drafts only:
  # the target verifies every proposal.
  draft_head="$(get GLM5_FLASH_DRAFT_HEAD exact)"
  case "$draft_head" in
    ""|exact) ;;
    tensor) family_args+=(--draft-head tensor) ;;
    *) echo "GLM5_FLASH_DRAFT_HEAD must be exact or tensor" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_DRAFT_LINEAR: the FP8 drafter's GEMMs, w8a16 (default: the W8A16 GEMV in passes of
  # 64 rows), wide (the same bits in passes of 128 rows) or w8a8 (past one draft block, E4M3
  # activations per row and 128-wide K block on FP8 tensor cores). Drafts only.
  draft_linear="$(get GLM5_FLASH_DRAFT_LINEAR w8a16)"
  case "$draft_linear" in
    ""|w8a16) ;;
    wide|w8a8) family_args+=(--draft-linear "$draft_linear") ;;
    *) echo "GLM5_FLASH_DRAFT_LINEAR must be w8a16, wide or w8a8" >&2; exit 2 ;;
  esac
  # GLM5_FLASH_TARGET_HEAD: the target's logits through the BF16 head, exact (default: the pedantic
  # FP32 GEMM at every row count) or tensor (steps past 8 rows on BF16 tensor cores with FP32
  # accumulation: a change to the target's numerics, KL-gated). Needs GLM5_FLASH_FP8_HEAD=off.
  target_head="$(get GLM5_FLASH_TARGET_HEAD exact)"
  case "$target_head" in
    ""|exact) ;;
    tensor)
      if [[ $fp8_head == on ]]; then
        echo "GLM5_FLASH_TARGET_HEAD=tensor runs the BF16 head; set GLM5_FLASH_FP8_HEAD=off" >&2
        exit 2
      fi
      family_args+=(--target-head tensor) ;;
    *) echo "GLM5_FLASH_TARGET_HEAD must be exact or tensor" >&2; exit 2 ;;
  esac
fi
# GLM5_FLASH_MEMORY=auto: lay out what this launch would serve with the standard settings (the flags
# resolved above), on the free memory of the GPU it serves from, before any container changes. When
# the pool falls short, the launch runs again with the compact profile.
if [[ "$glmf_memory" == auto ]]; then
  glmf_sequences="$(get CONCURRENCY 8)"
  [[ "$glmf_sequences" =~ ^[1-9][0-9]*$ ]] || { echo "CONCURRENCY must be a positive whole number" >&2; exit 2; }
  glmf_need=$((glmf_context + 65536 * (glmf_sequences - 1)))
  if [[ "$head_split" == 1 ]]; then
    echo "note: GLM5_FLASH_MEMORY=auto keeps standard: compact serves from one GPU, and this launch splits heads over two" >&2
  elif [[ "$ranks" == 0 ]]; then
    echo "note: GLM5_FLASH_MEMORY=auto keeps standard: compact runs the routed experts on Sparks, and this launch runs" \
      "them on the GPU" >&2
  else
    glmf_free_gib="$(selected_gpu_free_gib "$gpu")"
    [[ -n "$glmf_free_gib" && ! "$glmf_free_gib" =~ ^0(\.0+)?$ ]] ||
      { echo "GLM5_FLASH_MEMORY=auto needs GPU $gpu's free memory from nvidia-smi (read: ${glmf_free_gib:-nothing} GiB);" \
        "set GLM5_FLASH_MEMORY=compact or standard" >&2; exit 2; }
    # The flags `cuteafd plan --layout` takes under serve-glmf's names, as this launch passes them.
    glmf_plan=(--vision "$vision" --audio "$audio" --json --layout --rtx 1 --rtx-gib "$glmf_free_gib"
      --coordinator-budget-gib "$glmf_free_gib" --spark-ranks "$ranks"
      --spark-budget-gib "$(python3 -c 'import sys; print(int(sys.argv[1]) / 2**30)' "$budget")"
      --concurrency "$glmf_sequences" --context-tokens "$glmf_context")
    for ((arg = 0; arg < ${#family_args[@]}; arg++)); do
      case "${family_args[arg]}" in
        --embedding-placement|--kda-fp8|--fp8-head|--index-cache|--kda-state|--prefix-marks|--replay-records|\
        --decode-rows|--prefill-lanes|--prefill-lane-rows|--headroom-gib|--graph-budget-mib|--pool-tokens|--draft-fp8|\
        --draft-linear|--draft-context-slots|--draft-sequences)
          glmf_plan+=("${family_args[arg]}" "${family_args[arg + 1]}"); arg=$((arg + 1)) ;;
        --decode-row-buckets) glmf_plan+=(--decode-row-buckets) ;;
      esac
    done
    [[ "${draft_args[0]:-}" != --draft ]] || glmf_plan+=(--draft "${draft_args[1]}")
    # Lazily captured graphs keep the graph allowance (a budget is forwarded above).
    [[ "$(get GLM5_FLASH_STARTUP_GRAPHS)" != off ]] || glmf_plan+=(--startup-graphs off)
    # A WIP slot plans with its own binary and program manifest, copied apart from the layout a
    # running launch may hold.
    glmf_plan_run=(cuteafd plan "$snapshot") glmf_plan_mounts=() glmf_plan_dir=""
    if [[ -n "$wip_slot" ]]; then
      glmf_plan_dir="$(mktemp -d)"
      for glmf_file in cuteafd PROGRAMS.json; do
        docker cp "cuteafd-coordinator-wip:/wip/slots/$wip_slot/coordinator/workspace/.cuteafd-wip/$glmf_file" \
          "$glmf_plan_dir/$glmf_file" >/dev/null ||
          { rm -rf "$glmf_plan_dir"; echo "GLM5_FLASH_MEMORY=auto: WIP slot $wip_slot has no coordinator $glmf_file" \
            "(build it with ./wip.sh --slot $wip_slot)" >&2; exit 1; }
      done
      glmf_plan_mounts=(-v "$glmf_plan_dir:/opt/cuteafd-plan:ro" --entrypoint /opt/cuteafd-plan/cuteafd)
      glmf_plan_run=(plan "$snapshot" --workspace-manifest /opt/cuteafd-plan/PROGRAMS.json)
    fi
    glmf_plan_json="$(docker run --rm --network none -v "$hub:/root/.cache/huggingface/hub:ro" "${glmf_plan_mounts[@]}" \
      "$coordinator_image" "${glmf_plan_run[@]}" "${glmf_plan[@]}")" || glmf_plan_json=""
    [[ -z "$glmf_plan_dir" ]] || rm -rf "$glmf_plan_dir"
    glmf_planned="$(python3 -c 'import json,sys; d=json.load(sys.stdin); m=d.get("memory_layout") or {}
print(int(m.get("pool_tokens") or 0) if d.get("fits") else 0)' <<<"$glmf_plan_json" 2>/dev/null || true)"
    [[ "$glmf_planned" =~ ^[0-9]+$ ]] ||
      { echo "GLM5_FLASH_MEMORY=auto could not lay out the standard settings with $coordinator_image (cuteafd plan" \
        "--layout); set GLM5_FLASH_MEMORY=compact or standard" >&2; exit 2; }
    glmf_why="the standard settings admit $glmf_planned KV tokens on GPU $gpu ($glmf_free_gib GiB free) for"
    glmf_why+=" $glmf_sequences sequences; one $glmf_context-token request and 65,536 tokens for each other sequence"
    glmf_why+=" need $glmf_need"
    if ((glmf_planned >= glmf_need)); then
      echo "note: GLM5_FLASH_MEMORY=auto keeps standard: $glmf_why" >&2
    else
      echo "note: GLM5_FLASH_MEMORY=auto takes compact: $glmf_why" >&2
      exec env CUTEAFD_GLMF_MEMORY_CHOSEN=compact bash "${BASH_SOURCE[0]}" "${launch_args[@]}"
    fi
  fi
fi
# INSTANCE names a launch that runs beside others on disjoint hardware
# (`cuteafd bench smoke` sets it): its coordinator container is
# cuteafd-coordinator-INSTANCE; empty keeps the one cuteafd-coordinator.
# SPARK_COUNT=0: the routed experts run on the coordinator GPU (--local-experts;
# GLM 5.3 Flash, MiMo V2 and Qwen 3.8), the natural minimum for checkpoints
# that fit one RTX.
if [[ "$ranks" == 0 ]]; then
  case "$family" in
    glm5_flash|mimo_v2|qwen4) family_args+=(--local-experts) ;;
    *) echo "SPARK_COUNT=0 (local experts) serves GLM 5.3 Flash, MiMo V2 and Qwen 3.8, not $family" >&2; exit 2 ;;
  esac
fi
# Check every selected image before --restart or checkpoint reads. The worker's
# resident admission validates the sibling for its full 4096-row workspace even
# when only decode requests use BF16, so this preflight requires the same coverage.
if [[ -n "$expert_input" && "$expert_input" != fp8 ]]; then
  [[ "$ranks" != 0 ]] || { echo "EXPERT_INPUT=$expert_input requires Spark experts" >&2; exit 2; }
  store_dtype="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("quantization_config", {}).get("store_dtype", "fp8"))' "$root/snapshots/$revision/config.json")"
  case "$store_dtype" in
    fp8) expert_geometry=mimo ;;
    mxfp4)
      hidden="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("text_config", c).get("hidden_size", 0))' "$root/snapshots/$revision/config.json")"
      case "$hidden" in
        4096) expert_geometry=mimof ;;
        6144) expert_geometry=mimop ;;
        *) echo "EXPERT_INPUT=$expert_input has no MXFP4 MiMo package for hidden_size=$hidden" >&2; exit 2 ;;
      esac ;;
    *) echo "EXPERT_INPUT=$expert_input has no package for store_dtype=$store_dtype" >&2; exit 2 ;;
  esac
  printf -v preflight_command '%q ' docker run --rm -i --entrypoint python3 "$spark_image" - "$expert_geometry" "tp$ranks" 4096
  for ((rank = 0; rank < ranks; rank++)); do
    host="$(get "SPARK_${rank}_HOST")"
    if ! ssh "$host" "$preflight_command" < "$repo_root/scripts/launch/preflight-fp8-bf16.py"; then
      echo "$host cannot serve EXPERT_INPUT=$expert_input; build $expert_geometry:fp8 with CUTEAFD_RELEASE_FP8_MOE_BF16_FAMILIES=$expert_geometry (or its WIP equivalent)" >&2
      exit 2
    fi
  done
fi
# Resolve cold placement before starting or removing containers. Off skips the
# planner and tower startup entirely; text-only families keep their old path.
vision_peers=()
encoder_ranks=()
encoder_hash=""
encoder_max_tokens=4096
encoder_port=$((port + 1))
if [[ ( "$family" == mimo_v2 || "$family" == qwen4 || "$family" == glm5_flash ) && "$vision" != off ]] &&
   python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("vision_config") else 1)' "$root/snapshots/$revision/config.json"; then
  plan_rtx=1; ((head_split == 0)) || plan_rtx=2
  plan_pool="$(get POOL_TOKENS auto)"; [[ "$plan_pool" != auto ]] || plan_pool=0
  plan_gib="${coordinator_budget:-95.5}"
  plan_json="$(docker run --rm --network none -v "$hub:/root/.cache/huggingface/hub:ro" \
    "$coordinator_image" cuteafd plan "$snapshot" --vision "$vision" --audio "$audio" --json --layout \
    --spark-ranks "$ranks" --spark-budget-gib "$(python3 -c 'import sys;print(int(sys.argv[1])/2**30)' "$budget")" \
    --rtx "$plan_rtx" --rtx-gib "$plan_gib" --pool-tokens "$plan_pool" --vision-replicas "$vision_replicas")"
  selected="$(python3 -c '
import json,sys
p=json.load(sys.stdin); e=p.get("encoder"); assert e is not None, "image checkpoint lacks encoder plan"
assert p["placement_supported"] and p["fits"], "encoder deployment cannot fit: "+str(p.get("hints"))
k=e["kind"]; kind=k["kind"]; h=p["encoder_plan_hash"]
cap=p.get("max_image_tokens") or (1024 if sys.argv[1]=="qwen4" else 4096)
assert type(cap) is int and 1<=cap<=4096, "invalid encoder image cap"
assert len(h)==64 and all(c in "0123456789abcdef" for c in h), "invalid encoder plan hash"
if kind=="spark":
    ranks=[k["rank"]]+e["replicas"]
    assert len(ranks)==len(set(ranks)) and all(0<=r<p["spark_ranks"] for r in ranks)
    print("spark:"+str(k["rank"]),h,",".join(map(str,ranks)),cap)
elif kind=="rtx": print("rtx:"+str(k["gpu"]),h,"-",cap)
elif kind=="off": print("off",h,"-",cap)
else: raise ValueError("idle-host launch needs an explicit inventory")
' "$family" <<<"$plan_json")"
  read -r vision encoder_hash rank_csv encoder_max_tokens <<<"$selected"
  if [[ "$vision" == spark:* ]]; then
    IFS=, read -r -a encoder_ranks <<<"$rank_csv"
    for encoder_rank in "${encoder_ranks[@]}"; do
      vision_peers+=("$(get "SPARK_${encoder_rank}_LANE_A"):$encoder_port")
    done
    family_args+=(--vision-peers "$(IFS=,; printf '%s' "${vision_peers[*]}")" --encoder-plan-hash "$encoder_hash" --encoder-revision "$revision")
  fi
elif [[ "$vision" == spark* || "$vision" == rtx* ]]; then
  release_die "explicit encoder placement requires a supported MiMo, Qwen or GLM Flash vision checkpoint"
fi
# Replace the original policy with the selected placement, without duplicated flags.
for ((arg = 0; arg < ${#family_args[@]}; arg++)); do
  [[ "${family_args[arg]}" != --vision ]] || family_args[arg+1]="$vision"
done
peers=()
# --restart removes this launcher's containers; stop.sh accepts the same keys.
# One model is served at a time: every expert worker on these hosts goes, whatever
# its port (a leftover worker of another model holds Spark memory and OOMs the next).
if [[ "$restart" == 1 ]]; then
  previous_csv=""
  if [[ "$ranks" == 0 && "$configured_ranks" != 0 ]]; then
    # Switching to local experts must release this coordinator's old workers.
    # A new local launch must not stop another instance's Spark workers.
    previous_csv="$(docker inspect --format '{{json .Config.Cmd}}' "$coordinator_name" 2>/dev/null \
      | python3 -c 'import json,sys; c=json.load(sys.stdin); print(c[c.index("--peers")+1])' 2>/dev/null || true)"
  fi
  docker rm -f "$coordinator_name" >/dev/null 2>&1 || true
  if [[ -n "$previous_csv" ]]; then
    IFS=, read -r -a previous_peers <<<"$previous_csv"
    for ((rank = 0; rank < configured_ranks; rank++)); do
      host="$(get "SPARK_${rank}_HOST")"; lane="$(get "SPARK_${rank}_LANE_A")"
      for peer in "${previous_peers[@]}"; do
        old_port="${peer##*:}"
        if [[ "${peer%:*}" == "$lane" && "$old_port" =~ ^[0-9]+$ ]]; then
          ssh "$host" "docker rm -f cuteafd-spark-expert-$host-$old_port >/dev/null 2>&1 || true"
        fi
      done
    done
  fi
  for ((rank = 0; rank < ranks; rank++)); do
    host="$(get "SPARK_${rank}_HOST")"
    # Workers are cuteafd-spark-expert-HOST-PORT; the persistent ./wip.sh container
    # (cuteafd-spark-expert-wip) and its build cache stay.
    ssh "$host" 'ids=$(docker ps -aq --filter "name=^cuteafd-spark-expert-.+-[0-9]+$"); [ -z "$ids" ] || docker rm -f $ids >/dev/null 2>&1 || true'
  done
fi
# FP8_EXPERT_PREFILL: how FP8 expert packages run prefill row counts: auto
# (default: wire rows W8A8 with E4M3 x E4M3 gate/up, BF16 rows W8A16), w8a16
# (the former programs) or w8a8 (also quantizes the BF16 rows of experts on the
# coordinator GPU). Spark workers and the coordinator both read it.
fp8_prefill="$(get FP8_EXPERT_PREFILL auto)"
case "$fp8_prefill" in auto|w8a8|w8a16) ;; *) echo "FP8_EXPERT_PREFILL must be auto, w8a8 or w8a16" >&2; exit 2 ;; esac
# CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP (local-ip=device,...; each process picks the entry
# for its own fabric address) reaches the coordinator and every worker when it is set. Without
# it each opens its first RDMA device, which need not carry the fabric address (GB10 exposes
# several RDMA functions per port).
device_map="${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-}"
device_map_env="" device_map_args=()
if [[ -n "$device_map" ]]; then
  [[ "$device_map" =~ ^[A-Za-z0-9.:=,_-]+$ ]] ||
    { echo "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP must be local-ip=device[,local-ip=device...]" >&2; exit 2; }
  device_map_env="-e CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=$device_map"
  device_map_args=(-e "CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=$device_map")
fi
# GB10 CUDA allocations cannot reclaim page cache: drop it on the expert hosts, through
# SparkNest's nest when it is installed, otherwise over ssh (passwordless sudo on each host).
spark_hosts=()
spark_host_names=()
for ((rank = 0; rank < ranks; rank++)); do
  spark_hosts+=(--host "$(get "SPARK_${rank}_HOST")")
  spark_host_names+=("$(get "SPARK_${rank}_HOST")")
done
drop_spark_caches() {
  if command -v nest >/dev/null; then nest drop-caches "${spark_hosts[@]}" >/dev/null; return; fi
  local host failed=0
  for host in "${spark_host_names[@]}"; do
    ssh -n "$host" 'sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null' || failed=1
  done
  return "$failed"
}
if [[ -n "$wip_slot" ]]; then
  # The slot must exist everywhere before anything starts, and the development images must
  # carry the SparkInfer revision this checkout pins.
  pinned_sparkinfer="$(python3 "$repo_root/scripts/build/verify-sparkinfer-source.py" --source "$repo_root/third_party/sparkinfer" \
    --lock "$repo_root/third_party/sparkinfer.lock.json" --print-revision)"
  release_require_dev_image_sparkinfer "$(hostname)" "$coordinator_image" \
    "$(docker image inspect -f '{{index .Config.Labels "io.cuteafd.sparkinfer.revision"}}' "$coordinator_image")" "$pinned_sparkinfer"
  release_stage_wip_layout cuteafd-coordinator-wip "$wip_slot" coordinator "$wip_layout"
  for ((rank = 0; rank < ranks; rank++)); do
    host="$(get "SPARK_${rank}_HOST")"
    ssh "$host" bash -s -- "$wip_slot" "$spark_image" "$pinned_sparkinfer" <<'STAGE' ||
set -euo pipefail
slot="$1" image="$2" pinned="$3"
label="$(docker image inspect -f '{{index .Config.Labels "io.cuteafd.sparkinfer.revision"}}' "$image" 2>/dev/null || true)"
[[ "$label" == "$pinned" ]] || { echo "$(hostname): $image carries SparkInfer ${label:-<none>}, this checkout pins $pinned" >&2; exit 1; }
layout="$HOME/.cache/cuteafd/wip-run/$slot" raw="$HOME/.cache/cuteafd/wip-run/$slot.tmp/raw"
rm -rf "$layout.tmp" && mkdir -p "$raw" "$layout.tmp/bin" "$layout.tmp/lib" "$layout.tmp/share"
docker cp "cuteafd-spark-expert-wip:/wip/slots/$slot/spark-expert/workspace/.cuteafd-wip/." "$raw/"
docker cp "cuteafd-spark-expert-wip:/wip/slots/$slot/spark-expert/workspace/docker/release-entrypoint.sh" "$raw/"
mv "$raw/cuteafd" "$layout.tmp/bin/cuteafd"
mv "$raw/libcuteafd_native.so" "$layout.tmp/lib/"
[[ ! -d "$raw/exl3" ]] || mv "$raw/exl3" "$layout.tmp/lib/exl3"
[[ ! -d "$raw/fp8" ]] || mv "$raw/fp8" "$layout.tmp/lib/fp8"
mv "$raw/"* "$layout.tmp/share/"
rm -rf "$raw" "$layout" && mv "$layout.tmp" "$layout"
STAGE
      { echo "$host: WIP slot $wip_slot is not staged (build it with ./wip.sh --slot $wip_slot)" >&2; exit 1; }
  done
fi
((ranks == 0)) || drop_spark_caches || echo "warning: could not drop Spark page caches" >&2
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  lane="$(get "SPARK_${rank}_LANE_A")"
  peers+=("$lane:$port")
  encoder_args=""
  for encoder_rank in "${encoder_ranks[@]}"; do
    if [[ "$rank" == "$encoder_rank" ]]; then
      encoder_args="--encoder --encoder-listen 0.0.0.0:$encoder_port --encoder-plan-hash $encoder_hash --encoder-revision $revision --encoder-max-tokens $encoder_max_tokens"
    fi
  done
  ssh "$host" "docker run -d --name cuteafd-spark-expert-$host-$port --restart no --gpus all --network host \
    --ipc host --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -e CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill$spark_worker_env $wip_worker_args $device_map_env \
    -v \$(readlink -f \$HOME/.cache/huggingface/hub):/root/.cache/huggingface/hub:ro '$spark_image' \
    cuteafd expertd-native --snapshot '$snapshot' --native-lib /opt/cuteafd/lib/libcuteafd_native.so \
    --rank $rank --world $ranks --capacity 4096 --device-budget-bytes $budget $layer_args$spark_worker_args \
    --listen 0.0.0.0:$port $encoder_args >/dev/null" &
done
wait
for ((rank = 0; rank < ranks; rank++)); do
  host="$(get "SPARK_${rank}_HOST")"
  # Expert readiness follows synchronous encoder startup in the same process.
  ready_deadline=$((SECONDS + 900))
  until ssh "$host" "docker logs cuteafd-spark-expert-$host-$port 2>&1 | grep -q 'worker ready'"; do
    ((SECONDS < ready_deadline)) || { echo "$host readiness timed out" >&2; exit 1; }
    ssh "$host" "docker ps -q -f name=cuteafd-spark-expert-$host-$port | grep -q ." ||
      { echo "$host expert worker exited:" >&2; ssh "$host" "docker logs --tail 20 cuteafd-spark-expert-$host-$port" >&2; exit 1; }
    sleep 2
  done
done
# Loading leaves ~10 GiB of checkpoint pages cached per Spark (sparknest passthrough: the
# workers' own fadvise cannot reach them), and GB10 CUDA allocations do not reclaim page
# cache: drop it once every rank is resident. CUTEAFD_SPARK_DROP_PAGE_CACHE=0 keeps it.
if ((ranks > 0 && ${#spark_hosts[@]} > 0)) && [[ "${CUTEAFD_SPARK_DROP_PAGE_CACHE:-1}" != 0 ]]; then
  drop_spark_caches || echo "warning: could not drop Spark page caches after loading" >&2
fi
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
  --security-opt "seccomp=$repo_root/docker/seccomp-code-bench.json" \
  --ulimit memlock=-1:-1 --device=/dev/infiniband -e RUST_LOG=info -e "CUTEAFD_SPARK_INTAKE=$intake" \
  -e "CUTEAFD_CONSOLE_TEXT=$([[ $console_text == on ]] && echo true || echo false)" "${bond_args[@]}" \
  -e "CUTEAFD_FP8_EXPERT_PREFILL=$fp8_prefill" -e "CUTEAFD_IMAGE=$coordinator_image" "${wip_mount_args[@]}" "${device_map_args[@]}" \
  -v "$hub:/root/.cache/huggingface/hub:ro" -v "$bench_dir:/root/.cache/cuteafd/bench" \
  "${chat_template_mounts[@]}" "${trace_args[@]}" "${probe_args[@]}" "$coordinator_image" cuteafd "${coordinator_budget_args[@]}" $serve --snapshot "$snapshot" \
  --native-lib /opt/cuteafd/lib/libcuteafd_native.so "${peer_args[@]}" --listen "$addr" \
  --max-sequences "$(get CONCURRENCY 8)" --max-context "$(get MAX_CONTEXT_TOKENS 8192)" \
  --max-output "$(get MAX_OUTPUT_TOKENS 4096)" "${dspark_args[@]}" \
  "${family_args[@]}" "${draft_args[@]}" "${served_args[@]}" >/dev/null
url="http://127.0.0.1:${addr##*:}"
ready_deadline=$((SECONDS + 900))
until curl --max-time 5 -sf "$url/health" >/dev/null; do
  if (( SECONDS >= ready_deadline )); then
    echo "coordinator readiness timed out: $(curl --max-time 5 -s "$url/health" || true)" >&2
    docker logs --tail 30 "$coordinator_name" >&2
    exit 1
  fi
  docker ps -q -f "name=^$coordinator_name\$" | grep -q . ||
    { echo "coordinator exited:" >&2; docker logs --tail 30 "$coordinator_name" >&2; exit 1; }
  sleep 2
done
echo "API ready at $url/v1/ ($(curl -s "$url/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])'))"
