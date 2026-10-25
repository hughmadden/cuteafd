#!/usr/bin/env bash
# Source alongside release-common.sh. Locks always live on the coordinator,
# including for an export running over SSH on the seed Spark.
#
# release_with_export_locks HOST CONTAINER LOCKS PREFLIGHT CMD...: run CMD (a
# command or shell function) holding LOCKS (comma list of scripts/hw/hwlock.py
# names) after a fair hwlock turn. PREFLIGHT, when non-empty, is a shell check
# hwlock runs before enqueue and again at grant; failure stops before CMD. The
# named CONTAINER (on HOST over SSH, or local when HOST is empty) is removed on
# every exit while the locks are still held, after the optional caller hook
# release_export_locked_cleanup (e.g. a GPU watchdog stop).
release_with_export_locks() (
  set -euo pipefail
  local host="$1" container="$2" locks="$3" preflight="$4"
  shift 4
  local lock_dir="${CUTEAFD_LOCK_DIR:-$HOME/.cache/cuteafd}"
  local wait_seconds="${CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS:-1200}"
  [[ "$wait_seconds" =~ ^[1-9][0-9]*$ ]] || release_die "CUTEAFD_RELEASE_LOCK_TIMEOUT_SECONDS must be positive seconds"
  [[ "$locks" =~ ^[a-z0-9-]+(,[a-z0-9-]+)*$ ]] || release_die "export locks must be a comma list of hardware lock names: $locks"
  mkdir -p "$lock_dir"
  # Queue fairly (scripts/hw/hwlock.py) so the waiter isn't overtaken by later
  # single-lock jobs; hwlock releases its ticket once the turn is ours, and the
  # flocks below then hold the locks for the export's lifetime.
  local hwlock name fd
  local -a lock_names preflight_args=()
  hwlock="$(dirname "${BASH_SOURCE[0]}")/../hw/hwlock.py"
  [[ -z "$preflight" ]] || preflight_args=(--preflight "$preflight")
  # hwlock prints the canonical order, so the flocks below follow it too.
  local lock_text
  lock_text="$(python3 -I -c \
    'import sys; sys.path.insert(0, sys.argv[1]); import hwlock; print("\n".join(hwlock.canonical(sys.argv[2].split(","))))' \
    "$(dirname "$hwlock")" "$locks" 2>/dev/null)" || release_die "unknown hardware locks: $locks"
  mapfile -t lock_names <<<"$lock_text"
  local lock_list="${lock_names[*]}"
  echo "== waiting for a fair turn on release export hardware locks (${lock_list// /, }; ${wait_seconds}s) =="
  local -a lock_fds=()
  for name in "${lock_names[@]}"; do
    exec {fd}>"$lock_dir/$name.lock"
    lock_fds+=("$fd")
  done
  local hwlock_status=0
  CUTEAFD_LOCK_DIR="$lock_dir" python3 "$hwlock" --locks "$locks" --wait "$wait_seconds" \
    "${preflight_args[@]}" -- true || hwlock_status=$?
  case "$hwlock_status" in
    0) ;;
    78) release_die "export preflight failed for ${lock_list// /.lock and }.lock" ;;
    *) release_die "timed out waiting for a fair turn on ${lock_list// /.lock and }.lock" ;;
  esac
  local index
  for index in "${!lock_names[@]}"; do
    flock -w "$wait_seconds" "${lock_fds[$index]}" || release_die "timed out waiting for ${lock_names[$index]}.lock"
  done
  echo "== export hardware locks held: ${lock_list// /, } ($(date -Is)) =="
  # The Docker client being terminated does not stop a daemon-owned container.
  # Remove our uniquely named export on every exit, while every lock is held.
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
  trap 'rc=$?; ! declare -F release_export_locked_cleanup >/dev/null || release_export_locked_cleanup || true; cleanup_export || { echo "failed to remove export container $container" >&2; rc=1; }; exit "$rc"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  "$@"
)
