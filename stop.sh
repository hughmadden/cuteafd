#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_root/scripts/lib/release-common.sh"

usage() {
  cat <<'EOF'
Usage: ./stop.sh [--config FILE] [--all]

Gracefully stops this instance's coordinator and EXPERT_PORT workers on every
configured Spark rank, regardless of SPARK_COUNT. Host scope is a superset of the
active ranks, but worker cleanup remains port-scoped. Release containers are removed;
only the port-tracked WIP processes are stopped. Shared persistent WIP containers,
other worker ports, slots, build caches, images and model caches remain untouched.

--all also removes every release expert worker on the configured hosts and stops
(but retains) their persistent WIP containers. Use only when those hosts are not
shared with another deployment.

Every configured host and every cleanup phase is attempted even if one host or
phase fails; the script exits nonzero if anything could not be stopped.

--config FILE selects an entire alternate configuration file. Its Spark host
keys define the cleanup scope. Only syntax, known keys, value domains and host
tokens are validated: a file that is incomplete or invalid for launching (for
example six hosts without SPARK_TP/SPARK_EP) still stops every host it names.
INSTANCE in that file selects the coordinator container to stop: with INSTANCE
set it is cuteafd-coordinator-INSTANCE, otherwise the shared cuteafd-coordinator.
EOF
}

config="$repo_root/cuteafd.config"
stop_all=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 && -n "$2" ]] || release_die "--config requires a configuration file"
      config="$2"
      shift 2
      ;;
    --all) stop_all=1; shift ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown stop argument: $1"
      ;;
  esac
done

release_load_config "$config" stop
# Widen cleanup past the active ranks before any stop helper runs. The load
# above validates the configuration first (syntax, known keys, value domains),
# so a missing, malformed or unsafe file fails before any container or process
# is touched. Stop mode skips launch-only topology/readiness rules, so a file
# that is incomplete for launching (for example six hosts without SPARK_TP/
# SPARK_EP) still cleans every host it names.
release_select_stop_hosts
release_need docker
release_need ssh
release_need ss
release_need ps

docker info >/dev/null 2>&1 ||
  release_die "local Docker daemon is unavailable"

echo "== stopping CUTEAFD release services =="
echo "  Spark cleanup hosts: $(release_stop_hosts | paste -sd, -)"
failed=0
release_stop_wip_services || failed=1
if ((stop_all)); then
  release_stop_wip_containers || failed=1
  release_stop_all_worker_containers || failed=1
fi
release_stop_services \
  "$RELEASE_COORDINATOR_CONTAINER_NAME" \
  "$RELEASE_SPARK_CONTAINER_PREFIX" || failed=1
((failed == 0)) ||
  release_die "one or more CUTEAFD services or containers could not be stopped"
echo "CUTEAFD release and WIP services are stopped."
