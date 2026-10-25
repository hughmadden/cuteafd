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
  local hwlock lock_text lock_list
  local -a preflight_args=()
  hwlock="$(dirname "${BASH_SOURCE[0]}")/../hw/hwlock.py"
  [[ -z "$preflight" ]] || preflight_args=(--preflight "$preflight")
  lock_text="$(python3 -I -c \
    'import sys; sys.path.insert(0, sys.argv[1]); import hwlock; print(" ".join(hwlock.canonical(sys.argv[2].split(","))))' \
    "$(dirname "$hwlock")" "$locks" 2>/dev/null)" || release_die "unknown hardware locks: $locks"
  lock_list="$lock_text"
  echo "== waiting for a fair turn on release export hardware locks (${lock_list// /, }; ${wait_seconds}s) =="
  # One live hwlock guard owns the fair ticket and then the flocks, from grant
  # until we write done: no other waiter can interpose and the grant preflight
  # still describes the hardware CMD runs on. The guard ignores TERM/INT/HUP so
  # a leg's group signal cannot drop the locks before cleanup; the EXIT trap
  # below always ends it (done, then a bounded wait, then SIGKILL).
  local ipc ready_fd done_fd guard_pid line guard_status
  ipc="$(mktemp -d "${TMPDIR:-/tmp}/cuteafd-export-lease.XXXXXXXX")"
  mkfifo "$ipc/ready" "$ipc/done"
  exec {ready_fd}<>"$ipc/ready" {done_fd}<>"$ipc/done"
  (
    trap '' TERM INT HUP
    exec {ready_fd}>&- {done_fd}>&-
    CUTEAFD_LOCK_DIR="$lock_dir" exec python3 "$hwlock" --locks "$locks" --wait "$wait_seconds" \
      "${preflight_args[@]}" -- bash -c 'printf "granted\n" >"$1" && read -r _ <"$2"' _ "$ipc/ready" "$ipc/done"
  ) &
  guard_pid=$!
  release_export_guard_end() {
    printf 'done\n' >&"$done_fd" 2>/dev/null || true
    local i
    for ((i = 0; i < 100; i++)); do
      kill -0 "$guard_pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "$guard_pid" 2>/dev/null || true
    wait "$guard_pid" 2>/dev/null || true
    rm -rf "$ipc"
  }
  trap 'rc=$?; release_export_guard_end; exit "$rc"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  while :; do
    if read -r -t 1 line <&"$ready_fd" && [[ "$line" == granted ]]; then
      break
    fi
    if ! kill -0 "$guard_pid" 2>/dev/null; then
      guard_status=0
      wait "$guard_pid" || guard_status=$?
      case "$guard_status" in
        78) release_die "export preflight failed for ${lock_list// /.lock and }.lock" ;;
        *) release_die "timed out waiting for a fair turn on ${lock_list// /.lock and }.lock (hwlock exit $guard_status)" ;;
      esac
    fi
  done
  echo "== export hardware locks held: ${lock_list// /, } ($(date -Is)) =="
  # The Docker client being terminated does not stop a daemon-owned container.
  # Remove our uniquely named export on every exit while the guard still holds
  # every lock; only then is the guard released.
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
  trap 'rc=$?; ! declare -F release_export_locked_cleanup >/dev/null || release_export_locked_cleanup || true; cleanup_export || { echo "failed to remove export container $container" >&2; rc=1; }; release_export_guard_end; exit "$rc"' EXIT
  "$@"
)
