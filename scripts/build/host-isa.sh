# shellcheck shell=bash
# Host CPU targets for the daemon (Rust) and libcuteafd_native.so (C/C++),
# sourced by build-release-artifacts.sh and build-wip-artifacts.sh.
#
# Every Spark is a GB10 (Cortex-X925 + Cortex-A725, SVE2/BF16/I8MM/crypto), so
# the arm64 build has one tuned target. Coordinators differ (AVX-512 PRO 6000
# hosts, AVX2-only 5090 desktops), so x86_64 images carry a portable
# x86-64-v2 base plus an x86-64-v4 variant that the daemon selects at startup
# (rust/crates/cuteafd-ffi/src/host_isa.rs). The base is v2, not v3: LLVM's AVX2
# code for the CPU sampler's masked exp reduction ran 1.6x slower than v2 on
# Zen 5 (min_p row 425 vs 200 us; v4 231), and v2 runs on every x86-64 host
# in scope. Explicit targets make the build host irrelevant: the same flags come
# out of raptor, oliver, a LAN-only build seed or a Spark.
#
# Artifact layout: `cuteafd` and `libcuteafd_native.so` are the base (first
# ISA); each further ISA adds bin-variants/<isa>/cuteafd and
# lib-variants/<isa>/libcuteafd_native.so, which images and WIP layouts install
# as /opt/cuteafd/bin/<isa>/cuteafd and /opt/cuteafd/lib/<isa>/libcuteafd_native.so.

# Rust: LLVM's cortex-x925 leaves the optional crypto extensions off and turns
# on MTE; GB10 has crypto (aes, sha2, sha3) and no MTE. neoverse-v3 would add
# RNG, which GB10 also lacks.
cuteafd_host_isa_rustflags() {
  case "$1" in
    x86-64-v2) printf '%s' '-C target-cpu=x86-64-v2' ;;
    x86-64-v4) printf '%s' '-C target-cpu=x86-64-v4' ;;
    grace) printf '%s' '-C target-cpu=cortex-x925 -C target-feature=+aes,+sha2,+sha3,-mte' ;;
    *) echo "unknown host ISA: $1" >&2; return 2 ;;
  esac
}

cuteafd_host_isa_triple() {
  case "$1" in
    x86-64-v2|x86-64-v4) printf 'x86_64-unknown-linux-gnu' ;;
    grace) printf 'aarch64-unknown-linux-gnu' ;;
    *) echo "unknown host ISA: $1" >&2; return 2 ;;
  esac
}

# x86-64-v4 when this CPU can run it (the same flags the daemon's selector checks).
cuteafd_host_isa_native_x86() {
  local flags
  flags="$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)"
  for flag in avx512f avx512bw avx512cd avx512dq avx512vl; do
    [[ " $flags " == *" $flag "* ]] || { printf 'x86-64-v2'; return; }
  done
  printf 'x86-64-v4'
}

# cuteafd_host_isa_list MODE [OVERRIDE]: the ISAs to build, base first, one per line.
# MODE release builds every coordinator variant; MODE wip builds only the one
# this host runs. OVERRIDE (CUTEAFD_RELEASE_HOST_ISA / CUTEAFD_WIP_HOST_ISA) is a
# space- or comma-separated list, base first; its architecture must match.
cuteafd_host_isa_list() {
  local mode="$1" override="${2:-}" arch isa
  arch="$(uname -m)"
  local -a list=()
  if [[ -n "$override" ]]; then
    read -ra list <<<"${override//,/ }"
  elif [[ "$arch" == aarch64 ]]; then
    list=(grace)
  elif [[ "$arch" == x86_64 && "$mode" == release ]]; then
    list=(x86-64-v2 x86-64-v4)
  elif [[ "$arch" == x86_64 ]]; then
    list=("$(cuteafd_host_isa_native_x86)")
  else
    echo "no host ISA targets for $arch" >&2; return 2
  fi
  ((${#list[@]})) || { echo "empty host ISA list" >&2; return 2; }
  for isa in "${list[@]}"; do
    case "$arch:$isa" in
      x86_64:x86-64-v2|x86_64:x86-64-v4|aarch64:grace) ;;
      *) echo "host ISA $isa cannot be built on $arch" >&2; return 2 ;;
    esac
  done
  printf '%s\n' "${list[@]}"
}

# cuteafd_host_isa_cargo_build TARGET_DIR_BASE ISA MANIFEST [cargo args...]: one
# variant's daemon into TARGET_DIR_BASE-ISA (each variant its own target dir, so
# cache keys and fingerprints never mix). --target keeps build scripts and
# proc-macros generic: only the shipped binary gets the ISA flags.
cuteafd_host_isa_cargo_build() {
  local base="$1" isa="$2" manifest="$3"; shift 3
  local flags triple
  flags="$(cuteafd_host_isa_rustflags "$isa")" || return
  triple="$(cuteafd_host_isa_triple "$isa")" || return
  CARGO_TARGET_DIR="$base-$isa" RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$flags" \
    cargo build --locked --manifest-path "$manifest" -p cuteafd-daemon --release --target "$triple" "$@"
}

# The daemon a cuteafd_host_isa_cargo_build call produced.
cuteafd_host_isa_daemon() {
  printf '%s-%s/%s/release/cuteafd' "$1" "$2" "$(cuteafd_host_isa_triple "$2")"
}

# cuteafd_host_isa_install OUTPUT NATIVE_BUILD TARGET_DIR_BASE ISA...: install the
# base daemon and library at OUTPUT and every further ISA under
# OUTPUT/{bin,lib}-variants/<isa>/. Both directories always exist, so an image
# can COPY them unconditionally.
cuteafd_host_isa_install() {
  local output="$1" native="$2" target="$3"; shift 3
  local base="$1" isa
  rm -rf "$output/bin-variants" "$output/lib-variants"
  mkdir -p "$output/bin-variants" "$output/lib-variants"
  install -m 0755 "$(cuteafd_host_isa_daemon "$target" "$base")" "$output/cuteafd"
  install -m 0755 "$native/libcuteafd_native.so" "$output/libcuteafd_native.so"
  shift
  for isa in "$@"; do
    install -D -m 0755 "$(cuteafd_host_isa_daemon "$target" "$isa")" "$output/bin-variants/$isa/cuteafd"
    install -D -m 0755 "$native/$isa/libcuteafd_native.so" "$output/lib-variants/$isa/libcuteafd_native.so"
  done
}

# cuteafd_native_build_args: extra Ninja arguments for `cmake --build DIR --`.
# GB10 (every aarch64 builder) has 10 Cortex-X925 + 10 Cortex-A725 cores, about
# 15 X925-equivalents; -j stays CMAKE_BUILD_PARALLEL_LEVEL (or Ninja's default)
# and -l 14 stops starting compiles and exports while the host is already
# loaded, a brake on the overload behind GB10 driver NV_ERR_NO_MEMORY failures.
cuteafd_native_build_args() {
  [[ "$(uname -m)" != aarch64 ]] || printf '%s\n' -l 14
}
