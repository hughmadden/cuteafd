#!/usr/bin/env bash
# Source and call cuteafd_compiler_cache_setup BUILD_DIR, or use as a compiler wrapper.

cuteafd_compiler_cache_warn() {
  if [[ "${cuteafd_cache_warned:-0}" == 0 ]]; then
    printf 'warning: cuteafd compiler cache disabled: %s; using plain compilers\n' "$*" >&2
    cuteafd_cache_warned=1
  fi
}

cuteafd_compiler_cache_setup() {
  [[ -n "${CUTEAFD_KACHE:-}" ]] || return 0
  export CUTEAFD_KACHE_MODE=disabled
  local build_dir="$1" wrapper cache config remote="${CUTEAFD_KACHE_REMOTE:-}"
  wrapper="$(command -v -- "$CUTEAFD_KACHE" 2>/dev/null)" || {
    cuteafd_compiler_cache_warn "kache not found ($CUTEAFD_KACHE)"; return 0;
  }
  [[ -x "$wrapper" ]] && timeout 5 "$wrapper" --version >/dev/null 2>&1 || {
    cuteafd_compiler_cache_warn "kache is not executable on $(uname -m)"; return 0;
  }
  # Do not silently replace an agent's existing wrapper or native toolchain.
  if [[ -n "${RUSTC_WRAPPER:-}${RUSTC_WORKSPACE_WRAPPER:-}${CMAKE_C_COMPILER_LAUNCHER:-}${CMAKE_CXX_COMPILER_LAUNCHER:-}${CMAKE_CUDA_COMPILER_LAUNCHER:-}" ||
        "${CC:-cc}" == *[[:space:]]* || "${CXX:-c++}" == *[[:space:]]* ]]; then
    cuteafd_compiler_cache_warn 'an existing compiler wrapper is configured'; return 0
  fi
  cache="${CUTEAFD_KACHE_CACHE_DIR:-$HOME/.cache/cuteafd/kache}/$(uname -m)"
  # Local index/runtime must stay on a build-safe local filesystem, never the remote.
  if ! cuteafd_compiler_cache_local_fs "$cache" >/dev/null 2>&1 ||
     ! python3 "$(dirname "${BASH_SOURCE[0]}")/assert-build-filesystem.py" "$cache" > /dev/null 2>&1 ||
     ! mkdir -p "$cache" "$build_dir/compiler-cache"; then
    cuteafd_compiler_cache_warn "local cache unavailable ($cache)"; return 0
  fi
  if [[ -n "$remote" ]]; then
    # Run the probe out of process: an unavailable hard NFS mount must not hang a build.
    if ! timeout -k 1 3 python3 - "$remote" <<'PY'
import os, sys, tempfile
from pathlib import Path
path = Path(sys.argv[1])
path.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix=".cuteafd-kache-probe-", dir=path) as directory:
    old = Path(directory) / "stage"
    old.write_bytes(b"cache probe")
    new = Path(directory) / "committed"
    os.replace(old, new)
    assert new.read_bytes() == b"cache probe"
PY
    then
      cuteafd_compiler_cache_warn "remote unavailable ($remote)"; return 0
    fi
  fi
  config="$build_dir/compiler-cache/config.toml"
  if ! python3 - "$config" "$remote" "$(uname -m)" <<'PY'
import json, sys
from pathlib import Path
text = '''[cache]
local_max_size = "30GiB"
auto_clean_orphaned_targets = false
auto_share_target_files = false
auto_clean_unused_units_days = 0
seed_new_targets = false
adaptive_incremental = false
build_script_hermetic = false
'''
if sys.argv[2]:
    text += '\n[cache.remote]\ntype = "filesystem"\npath = ' + json.dumps(sys.argv[2])
    text += '\nprefix = ' + json.dumps('artifacts/' + sys.argv[3]) + '\n'
Path(sys.argv[1]).write_text(text)
PY
  then
    cuteafd_compiler_cache_warn 'cannot write cache configuration'; return 0
  fi
  # cc recognizes this basename; cmake-rs otherwise drops the cc wrapper.
  local launcher
  launcher="$(realpath "${BASH_SOURCE[0]}")"
  if [[ "$launcher" == *[[:space:]]* || "$wrapper" == *[[:space:]]* ]]; then
    cuteafd_compiler_cache_warn 'wrapper paths contain whitespace'; return 0
  fi
  export CUTEAFD_KACHE_ACTIVE="$wrapper" KACHE_CONFIG="$config" KACHE_HOST_CONFIG=
  export KACHE_CACHE_DIR="$cache" KACHE_BUILD_SCRIPT_CACHE=0 KACHE_OUT_DIR_ALIAS=0
  export KACHE_VERIFY_RESTORES=always
  export CUTEAFD_KACHE_WARNING_DIR="$build_dir/compiler-cache/warned"
  rmdir "$CUTEAFD_KACHE_WARNING_DIR" 2>/dev/null || true
  export RUSTC_WRAPPER="$launcher" CC_KNOWN_WRAPPER_CUSTOM=compiler-cache
  export CC="$launcher ${CC:-cc}" CXX="$launcher ${CXX:-c++}"
  export CMAKE_C_COMPILER_LAUNCHER="$launcher" CMAKE_CXX_COMPILER_LAUNCHER="$launcher"
  # Native CUDA remains opt-in with the rest of the build; not part of the CPU pilot.
  export CMAKE_CUDA_COMPILER_LAUNCHER="$launcher"
  export CUTEAFD_KACHE_MODE=enabled
}

cuteafd_compiler_cache_local_fs() {
  timeout -k 1 5 python3 - "$1" <<'PY'
import json, subprocess, sys
from pathlib import Path
path = Path(sys.argv[1]).resolve()
while not path.exists():
    path = path.parent
mounts = json.loads(subprocess.check_output(['findmnt', '--json', '--target', str(path),
                                           '--output', 'FSTYPE'], text=True))['filesystems']
raise SystemExit(0 if len(mounts) == 1 and mounts[0]['fstype'] in
                 {'ext4', 'xfs', 'btrfs', 'zfs', 'tmpfs', 'overlay'} else 1)
PY
}

cuteafd_compiler_cache_cmake_args() {
  local native_dir="$1" language launcher=
  if [[ "${CUTEAFD_KACHE_MODE:-disabled}" == enabled ]]; then
    launcher="$RUSTC_WRAPPER"
  elif ! [[ -f "$native_dir/CMakeCache.txt" ]] ||
       ! grep -qE 'CMAKE_(C|CXX|CUDA)_COMPILER_LAUNCHER.*compiler-cache.sh' "$native_dir/CMakeCache.txt"; then
    return 0
  fi
  # Explicit values override old CMake caches; opt-out removes our old launcher.
  for language in C CXX CUDA; do
    printf '%s\n' "-DCMAKE_${language}_COMPILER_LAUNCHER=$launcher"
  done
}

# Docker argv rendering is deliberately a no-op unless explicitly configured.
# Call on the Docker host; ARM machines need their own native kache executable.
cuteafd_compiler_cache_docker_args() {
  [[ -n "${CUTEAFD_KACHE:-}" ]] || return 0
  # Even an unavailable opt-in records that this build actually ran plain.
  printf '%s\n' -e CUTEAFD_KACHE_REQUESTED=1
  local wrapper cache
  wrapper="$(command -v -- "$CUTEAFD_KACHE" 2>/dev/null)" || {
    cuteafd_compiler_cache_warn "kache not found ($CUTEAFD_KACHE)"; return 0;
  }
  if ! [[ -f "$wrapper" && -x "$wrapper" ]]; then
    cuteafd_compiler_cache_warn "kache is not an executable file ($wrapper)"; return 0
  fi
  wrapper="$(realpath "$wrapper")"
  cache="${CUTEAFD_KACHE_CACHE_DIR:-$HOME/.cache/cuteafd/kache}"
  cache="$(realpath -m "$cache")"
  if [[ "$wrapper$cache${CUTEAFD_KACHE_REMOTE:-}" == *','* ]]; then
    cuteafd_compiler_cache_warn 'Docker mount paths contain a comma'; return 0
  fi
  if ! cuteafd_compiler_cache_local_fs "$cache" >/dev/null 2>&1 ||
     ! python3 "$(dirname "${BASH_SOURCE[0]}")/assert-build-filesystem.py" "$cache" >/dev/null 2>&1 || ! mkdir -p "$cache"; then
    cuteafd_compiler_cache_warn "local cache unavailable ($cache)"; return 0
  fi
  if [[ -n "${CUTEAFD_KACHE_REMOTE:-}" ]] && ! timeout -k 1 3 test -d "$CUTEAFD_KACHE_REMOTE"; then
    cuteafd_compiler_cache_warn "remote unavailable ($CUTEAFD_KACHE_REMOTE)"; return 0
  fi
  printf '%s\n' --mount "type=bind,src=$wrapper,dst=/opt/cuteafd-kache,readonly" \
    --mount "type=bind,src=$cache,dst=/opt/cuteafd-kache-cache" \
    -e CUTEAFD_KACHE=/opt/cuteafd-kache -e CUTEAFD_KACHE_CACHE_DIR=/opt/cuteafd-kache-cache
  if [[ -n "${CUTEAFD_KACHE_REMOTE:-}" ]]; then
    printf '%s\n' --mount "type=bind,src=$CUTEAFD_KACHE_REMOTE,dst=/opt/cuteafd-kache-remote" \
      -e CUTEAFD_KACHE_REMOTE=/opt/cuteafd-kache-remote
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  # One failing cache invocation disables it for the remainder of this build.
  if [[ -d "${CUTEAFD_KACHE_WARNING_DIR:-/nonexistent}" ]]; then
    exec "$@"
  fi
  # Preserve genuine compiler errors, but never let an optional cache break a build.
  if [[ -n "${CUTEAFD_KACHE_ACTIVE:-}" ]] &&
     timeout -k 5 "${CUTEAFD_KACHE_TIMEOUT_SECONDS:-300}" "$CUTEAFD_KACHE_ACTIVE" "$@"; then
    exit 0
  fi
  if [[ -n "${CUTEAFD_KACHE_ACTIVE:-}" ]] && mkdir "${CUTEAFD_KACHE_WARNING_DIR:-/nonexistent}" 2>/dev/null; then
    printf 'warning: cuteafd kache invocation failed or timed out; retrying with plain compiler\n' >&2
  fi
  exec "$@"
fi
