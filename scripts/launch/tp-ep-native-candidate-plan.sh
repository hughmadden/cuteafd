#!/usr/bin/env bash
# Plan/command rendering for scripts/launch/run-tp-ep-native-candidate.sh.
#
# Sourced after scripts/lib/release-common.sh by the candidate launcher. The
# functions read the launcher's resolved variables and the argv builders, so
# they must be called only after configuration resolution.
#
# candidate_print_plan emits a deterministic, parseable plan with no Docker,
# SSH, GPU or network action.

candidate_print_plan() {
  echo "mode=plan"
  echo "run_id=$run_id"
  echo "topology tp=$spark_tp ep=$spark_ep world=$spark_world"
  echo "rtx_gpus=$rtx_gpus"
  echo "ownership=coordinator-selected exact layer IDs"
  echo "budget_bytes=$SPARK_DEVICE_BUDGET_BYTES"
  echo "admission=$admission"
  echo "worker_geometry=coordinator-selected typed model admission"
  echo "coordinator_container=$coordinator_container"
  echo "spark_container=$spark_container"
  echo "api_port=$api_port expert_port=$expert_port"
  echo "rail=A-only (candidate CLI has no secondary-rail endpoints; single rail, no fake dual)"
  echo "daemon_bin=$daemon_bin"
  echo "snapshot=$snapshot"
  echo "coordinator_native_lib=$coordinator_native_lib"
  echo "spark_native_lib=$spark_native_lib"
  echo "spark_lib_source=$spark_lib_source"
  echo "native_lib_env=coordinator:CUTEAFD_NATIVE_LIB=$coordinator_native_lib spark:CUTEAFD_NATIVE_LIB=$spark_native_lib"
  echo "artifact_contract=per-role daemon=$daemon_bin coordinator_lib=$coordinator_native_lib spark_lib=$spark_native_lib (per-host existence+arch verified on start; uniform Spark staging step below)"
  echo "wip_runtime_root=$wip_runtime_root"
  echo "coordinator_cuda_visible_devices=${coordinator_cuda_visible_devices:-unset} (coordinator container only)"
  echo "expect_spark_lib_sha256=${expect_spark_lib_sha256:-unset}"
  echo "role_manifest=${role_manifest:-unset}"
  echo "qualified=$qualified require_arch=$require_arch"
  echo "speculator=$SPECULATOR dspark_draft_limit=$dspark_draft_limit (coordinator only)"
  echo "host_device_map=CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP=${CUTEAFD_PROTOCOL_V2_VERBS_HOST_DEVICE_MAP:-unset} (required on both roles)"
  echo "tcp_timing=CUTEAFD_PROTOCOL_V2_TCP_TIMING=${CUTEAFD_PROTOCOL_V2_TCP_TIMING:-unset} (startup diagnostic; unset means the RC GID line is not emitted)"
  echo "sparkinfer_source=${CUTEAFD_SPARKINFER_SOURCE_DIR:-unset} pythonpath=${PYTHONPATH:-unset}"
  echo "ssh_bind=${ssh_bind_ip:-unset} ssh_targets=$(for i in "${!hosts[@]}"; do printf '%s=%s ' "$i" "${hosts[$i]}"; done)"
  echo "STEP prestage-prerequisite (NOT performed by this launcher; run before start)"
  for rank in "${!hosts[@]}"; do
    echo "PREREQ stage-$rank: host=${hosts[$rank]} install -D -m644 $spark_lib_source_host $spark_lib_stage_host && verify aarch64+sha; container reads $spark_native_lib (built source $spark_lib_source)"
  done
  for rank in "${!hosts[@]}"; do
    echo "STEP expert-start rank=$rank"
    echo "CMD expert-$rank: $(expert_command_string "$rank" "${hosts[$rank]}")"
  done
  echo "STEP worker-bootstrap-ready"
  for rank in "${!hosts[@]}"; do
    echo "CMD worker-bootstrap-ready-$rank: host=${hosts[$rank]} log=$wip_runtime_root/$expert_process.log offset=<pre-start byte size> match='native worker bootstrap ready ... rank=$rank world=$spark_world'"
  done
  echo "STEP coordinator-start"
  echo "CMD coordinator: $(coordinator_command_string)"
  echo "STEP worker-admission-ready"
  for rank in "${!hosts[@]}"; do
    echo "CMD worker-admission-ready-$rank: host=${hosts[$rank]} log=$wip_runtime_root/$expert_process.log offset=<pre-start byte size> match='coordinator-selected worker weights ready ... rank=$rank world=$spark_world layer_ids=<actual installed owners>' (or native empty expert worker ready)"
  done
  echo "STEP readiness-gate"
  local probe_host="$listen_host"; [[ "$probe_host" != "0.0.0.0" ]] || probe_host=127.0.0.1
  echo "CMD readiness-api: curl -fsS http://$probe_host:$api_port/health && curl -fsS http://$probe_host:$api_port/v1/models"
  echo "CMD readiness-workers: docker exec -e CUTEAFD_WIP_RUNTIME_ROOT=$wip_runtime_root $coordinator_container $wip_process status $coordinator_process; per Spark rank: -e CUTEAFD_WIP_RUNTIME_ROOT=$wip_runtime_root $spark_wip_process status $expert_process"
  echo "STEP first-request"
  echo "CMD first-request: curl -fsS -m 120 -H 'Content-Type: application/json' -d '<bounded max_tokens=1 payload>' http://$probe_host:$api_port/v1/chat/completions (initializes the lazy per-connection RDMA endpoints)"
  echo "STEP gid-capture"
  echo "CMD gid-capture: AFTER the first request, per rank grep the current process log slice for gid_index=/client_gid=/server_gid=; fail closed if absent; write $repo_root/runs/tp-ep-six/gid-binding-<run_id>.txt"
}
