#!/usr/bin/env bash
# Source alongside release-common.sh. Locks always live on the coordinator,
# including for an export running over SSH on the seed Spark.
release_with_export_locks() (
  set -euo pipefail
  local host="$1" container="$2"
  shift 2
  local lock_dir="$HOME/.cache/cuteafd"
  local wait_seconds="${CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS:-1200}"
  [[ "$wait_seconds" =~ ^[1-9][0-9]*$ ]] || release_die "CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS must be positive seconds"
  mkdir -p "$lock_dir"
  exec 9>"$lock_dir/sparks.lock" 8>"$lock_dir/gpu1.lock"
  echo "== waiting for release export hardware locks (sparks then gpu1, ${wait_seconds}s each) =="
  flock -w "$wait_seconds" 9 || release_die "timed out waiting for sparks.lock"
  flock -w "$wait_seconds" 8 || release_die "timed out waiting for gpu1.lock"
  # The Docker client being terminated does not stop a daemon-owned container.
  # Remove our uniquely named export on every exit, while both locks are held.
  cleanup_export() {
    local output
    if [[ -n "$host" ]]; then
      output="$(timeout 60 ssh "${release_ssh_opts[@]}" "$host" docker rm -f "$container" 2>&1)" && return 0
    else
      output="$(timeout 60 docker rm -f "$container" 2>&1)" && return 0
    fi
    # --rm has already removed a successful export; absence is also clean.
    [[ "$output" == *"No such container:"* ]] && return 0
    echo "$output" >&2
    return 1
  }
  trap 'rc=$?; cleanup_export || { echo "failed to remove export container $container" >&2; rc=1; }; exit "$rc"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  "$@"
)
