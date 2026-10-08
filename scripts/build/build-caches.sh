#!/usr/bin/env bash
# Host-side cache admission and Docker argv shared by release, WIP and dev.

cuteafd_build_cache_defaults() {
  if [[ "${CUTEAFD_BUILD_CACHES:-on}" == off ]]; then
    export CUTEAFD_KACHE= CUTEAFD_KACHE_SPARK= CUTEAFD_SCCACHE_CUDA=0
  else
    export CUTEAFD_KACHE="${CUTEAFD_KACHE-1}"
    export CUTEAFD_KACHE_SPARK="${CUTEAFD_KACHE_SPARK-1}"
    export CUTEAFD_SCCACHE_CUDA="${CUTEAFD_SCCACHE_CUDA-1}"
  fi
}

# stdout is argv (one argument per line); diagnostics always go to stderr.
# A dry plan checks existing parents but never creates or probes cache contents.
cuteafd_build_cache_docker_args() {
  local build_root="$1" container_home="$2" tc_hash="$3" mode="${4:-prepare}"
  python3 "$(dirname "${BASH_SOURCE[0]}")/build-cache-plan.py" \
    --build-root "$build_root" --container-home "$container_home" \
    --toolchain "$tc_hash" --mode "$mode" --arch "${5:-$(uname -m)}"
}

# Cargo's offline resolution is probed without changing the caller's --locked policy.
# A first build may populate missing crates; a complete cache never refreshes the index.
cuteafd_build_cache_cargo_offline() {
  [[ "${CUTEAFD_BUILD_CACHES:-on}" != off && -z "${CARGO_NET_OFFLINE:-}" ]] || return 0
  if cargo metadata --locked --offline --format-version 1 --manifest-path "$1" >/dev/null 2>&1; then
    export CARGO_NET_OFFLINE=true
    printf 'Cargo cache: complete locked dependencies; offline\n'
  else
    printf 'Cargo cache: incomplete locked dependencies; online population allowed\n'
  fi
}
