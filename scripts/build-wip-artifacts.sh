#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: build-wip-artifacts.sh SOURCE_DIR ROLE CUDA_ARCH BUILD_DIR OUTPUT_DIR" >&2
  exit 2
}

[[ $# -eq 5 ]] || usage
source_dir="$(realpath "$1")"
role="$2"
cuda_arch="$3"
build_dir="$(realpath -m "$4")"
output_dir="$(realpath -m "$5")"
# Check source and destinations before creating files or invoking Cargo. This
# also rejects NTFS exposed under a container alias such as /scratch.
python3 "$(dirname "$0")/assert-build-filesystem.py" "$source_dir" "$build_dir" "$output_dir" "${CARGO_HOME:-$HOME/.cargo}" "${TMPDIR:-/tmp}"

case "$role" in
  coordinator)
    coordinator_aot=ON
    xgrammar=ON
    ;;
  expert)
    coordinator_aot=OFF
    xgrammar=OFF
    ;;
  *)
    echo "ROLE must be coordinator or expert" >&2
    exit 2
    ;;
esac
spark_tp_roles="${CUTEAFD_WIP_SPARK_TP_ROLES:-}"
if [[ -n "$spark_tp_roles" ]]; then
  IFS=';' read -ra spark_tp_role_list <<<"$spark_tp_roles"
  for spark_tp_role in "${spark_tp_role_list[@]}"; do
    case "$spark_tp_role" in
      tp2|tp3|tp6) ;;
      *) echo "CUTEAFD_WIP_SPARK_TP_ROLES accepts only tp2, tp3 and tp6, got: $spark_tp_role" >&2; exit 2 ;;
    esac
  done
  [[ "$role" == expert ]] ||
    { echo "CUTEAFD_WIP_SPARK_TP_ROLES is only valid for the expert role" >&2; exit 2; }
fi

# Extra routed-expert kernel families (FAMILY:ROLE list, e.g. dsv4f:spark),
# validated by native/cmake/expert_families.cmake. Empty keeps the V4.1 image.
# Both builds take the same list; CMake keeps the entries for its architecture.
expert_families="${CUTEAFD_WIP_EXPERT_FAMILIES:-}"
# Official-only WIP builds may skip the EXL3 quantization AOT entirely. The
# default stays ON so every existing slot and script is byte-compatible; the
# native expert path does not require the EXL3 package.
exl3_aot="${CUTEAFD_WIP_EXL3_AOT:-ON}"
nvfp4_aot="${CUTEAFD_WIP_NVFP4_AOT:-ON}"
case "$exl3_aot" in ON|OFF) ;; *) echo "CUTEAFD_WIP_EXL3_AOT must be ON or OFF, got: $exl3_aot" >&2; exit 2 ;; esac
case "$nvfp4_aot" in ON|OFF) ;; *) echo "CUTEAFD_WIP_NVFP4_AOT must be ON or OFF, got: $nvfp4_aot" >&2; exit 2 ;; esac
[[ "$cuda_arch" =~ ^[0-9]+$ ]] || {
  echo "CUDA_ARCH must be numeric" >&2
  exit 2
}
[[ -f "$source_dir/rust/Cargo.toml" && -f "$source_dir/native/CMakeLists.txt" ]] || {
  echo "SOURCE_DIR is not a CUTEAFD source tree: $source_dir" >&2
  exit 2
}

python3 "$source_dir/scripts/verify-sparkinfer-source.py" \
  --source "$source_dir/third_party/sparkinfer" \
  --lock "$source_dir/third_party/sparkinfer.lock.json"
if [[ "$xgrammar" == ON ]]; then
  python3 "$source_dir/scripts/verify-xgrammar-source.py" \
    --source "$source_dir/third_party/xgrammar" \
    --lock "$source_dir/third_party/xgrammar.lock.json"
fi

mkdir -p "$build_dir" "$output_dir"
export PYO3_PYTHON=python3
export PYTHONPATH="$source_dir/third_party/sparkinfer:$source_dir/python/reference/cuteafd_reference:$source_dir/python/reference${PYTHONPATH:+:$PYTHONPATH}"
export CARGO_TARGET_DIR="$build_dir/cargo-target"

# The WIP sync chain (rsync -a + docker cp) can leave source mtimes older
# than the previous build's fingerprints; cargo/ninja then silently reuse
# stale objects and the slot ships binaries that do not match the frozen
# source. Fingerprint each build tree's content and, only where it changed,
# refresh mtimes so the dependency trackers see the new content. Python
# tools drive the AOT exports, so a python change also refreshes native/.
wip_tree_fingerprint() {
  find "$@" -type f \( -name '*.rs' -o -name '*.toml' -o -name '*.lock' -o -name '*.cu' -o -name '*.cc' -o -name '*.h' -o -name '*.cmake' -o -name 'CMakeLists.txt' -o -name '*.py' \) \
    -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
}
wip_rust_fingerprint="$(wip_tree_fingerprint "$source_dir/rust")"
wip_native_fingerprint="$(wip_tree_fingerprint "$source_dir/native" "$source_dir/python")"
wip_fingerprint_marker="$build_dir/.source-content-fingerprint"
wip_previous_fingerprint="$(cat "$wip_fingerprint_marker" 2>/dev/null || true)"
wip_previous_rust="$(cut -d' ' -f1 <<<"$wip_previous_fingerprint")"
wip_previous_native="$(cut -d' ' -f2 <<<"$wip_previous_fingerprint")"
if [[ -z "$wip_previous_fingerprint" || "$wip_previous_rust" != "$wip_rust_fingerprint" ]]; then
  find "$source_dir/rust" -type f -exec touch {} +
fi
if [[ -z "$wip_previous_fingerprint" || "$wip_previous_native" != "$wip_native_fingerprint" ]]; then
  find "$source_dir/native" "$source_dir/python" -type f -exec touch {} +
fi
wip_current_fingerprint="$wip_rust_fingerprint $wip_native_fingerprint"

cargo build \
  --quiet \
  --manifest-path "$source_dir/rust/Cargo.toml" \
  -p cuteafd-daemon \
  --release

cmake \
  -S "$source_dir/native" \
  -B "$build_dir/native" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTEAFD_ENABLE_CUDA=ON \
  -DCUTEAFD_ENABLE_V41_EXPERT_AOT=ON \
  -DCUTEAFD_V41_SPARK_TP_ROLES="$spark_tp_roles" \
  -DCUTEAFD_EXPERT_FAMILIES="$expert_families" \
  -DCUTEAFD_ENABLE_V41_NVFP4_AOT="$nvfp4_aot" \
  -DCUTEAFD_ENABLE_V41_EXL3_AOT="$exl3_aot" \
  -DCUTEAFD_V41_EXL3_BITS="${CUTEAFD_WIP_EXL3_BITS:-2;3}" \
  -DCUTEAFD_ENABLE_V41_LOCAL_EXPERT_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_TP2_EXPERT_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_FP8_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_DSV4_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_DSV4_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_GLM_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_GLM_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_MIMO_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_MIMO_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_MIMO_GEOMETRIES="$(g="${CUTEAFD_WIP_MIMO_GEOMETRIES:-mimo}"; echo "${g//,/;}")" \
  -DCUTEAFD_ENABLE_GLMF_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_GLMF_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_QWEN4_AOT="$( [[ "$role" == coordinator ]] && echo "${CUTEAFD_WIP_QWEN4_AOT:-OFF}" || echo OFF)" \
  -DCUTEAFD_ENABLE_V41_ATTENTION_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_HC_LAGGED_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_V41_NARROW_AOT="$coordinator_aot" \
  -DCUTEAFD_ENABLE_RDMA=ON \
  -DCUTEAFD_ENABLE_SPARKINFER_AOT=OFF \
  -DCUTEAFD_ENABLE_SPARKINFER_COORDINATOR_AOT=OFF \
  -DCUTEAFD_ENABLE_DS4_FLASH_AOT=OFF \
  -DCUTEAFD_ENABLE_W8A16_AOT=OFF \
  -DCUTEAFD_SPARKINFER_SOURCE_DIR="$source_dir/third_party/sparkinfer" \
  -DCUTEAFD_SPARKINFER_LOCK_FILE="$source_dir/third_party/sparkinfer.lock.json" \
  -DCUTEAFD_ENABLE_NCCL=OFF \
  -DCUTEAFD_ENABLE_XGRAMMAR="$xgrammar" \
  -DCUTEAFD_XGRAMMAR_SOURCE_DIR="$source_dir/third_party/xgrammar" \
  -DCUTEAFD_XGRAMMAR_LOCK_FILE="$source_dir/third_party/xgrammar.lock.json" \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -DCUTEAFD_CUDA_ARCHITECTURES="$cuda_arch"
cmake --build "$build_dir/native"
printf '%s' "$wip_current_fingerprint" >"$wip_fingerprint_marker"

install -m 0755 "$CARGO_TARGET_DIR/release/cuteafd" "$output_dir/cuteafd"
install -m 0755 "$build_dir/native/libcuteafd_native.so" "$output_dir/libcuteafd_native.so"
# The EXL3 package is only built and installed when the opt-in is ON. An
# official-only WIP build (CUTEAFD_WIP_EXL3_AOT=OFF) has no exl3/ directory and
# the native launch path never references one.
if [[ "$exl3_aot" == ON ]]; then
  wip_exl3_bits="${CUTEAFD_WIP_EXL3_BITS:-2;3}"
  wip_exl3_tag="k${wip_exl3_bits//[;]/}"
  wip_exl3_tag="${wip_exl3_tag//,/}"
  python3 "$source_dir/python/tools/package_v41_exl3_aot.py" install \
    --package "$build_dir/native/exl3-$wip_exl3_tag" --output "$output_dir/exl3/exl3-$wip_exl3_tag"
  python3 "$source_dir/python/tools/package_v41_exl3_aot.py" verify \
    --package "$output_dir/exl3/exl3-$wip_exl3_tag" --role "$role"
  # Other expert geometries (FAMILY:exl3-kTIERS entries) ship as exl3-FAMILY-kTIERS.
  IFS=';' read -ra wip_family_list <<<"$expert_families"
  for wip_family in "${wip_family_list[@]}"; do
    [[ "$wip_family" == *:exl3-k* ]] || continue
    wip_package="exl3-${wip_family%%:*}-${wip_family#*:exl3-}"
    python3 "$source_dir/python/tools/package_v41_exl3_aot.py" install \
      --package "$build_dir/native/$wip_package" --output "$output_dir/exl3/$wip_package"
    python3 "$source_dir/python/tools/package_v41_exl3_aot.py" verify \
      --package "$output_dir/exl3/$wip_package" --role "$role"
  done
fi
# Exact FP8 expert packages (FAMILY:fp8 entries) ship as fp8/fp8-FAMILY.
IFS=';' read -ra wip_fp8_list <<<"$expert_families"
for wip_family in "${wip_fp8_list[@]}"; do
  [[ "$wip_family" == *:fp8 ]] || continue
  wip_package="fp8-${wip_family%%:*}"
  mkdir -p "$output_dir/fp8"
  rm -rf "$output_dir/fp8/$wip_package"
  cp -a "$build_dir/native/fp8/$wip_package" "$output_dir/fp8/$wip_package"
  python3 "$source_dir/python/tools/package_fp8_moe_aot.py" verify --package "$output_dir/fp8/$wip_package"
done
install -m 0644 "$build_dir/native/v41_experts/v41_experts.json" "$output_dir/V41_EXPERT_AOT.json"
# Always emit the built-role manifest (empty for the legacy default). Roles are
# derived from the AOT export manifests CMake actually produced and bound to the
# built library hash, so a stale/partial export cannot advertise a role.
python3 "$source_dir/scripts/write-v41-expert-tp-manifest.py" \
  --role "$role" \
  --requested "$spark_tp_roles" \
  --native-build-dir "$build_dir/native" \
  --native-library "$build_dir/native/libcuteafd_native.so" \
  --output "$output_dir/V41_EXPERT_TP_AOT.json"
if [[ "$coordinator_aot" == ON ]]; then
  install -m 0644 "$build_dir/native/v41_fp8/v41_fp8.json" "$output_dir/V41_FP8_AOT.json"
else
  printf '%s\n' '{"schema":1,"role":"expert","enabled":false}' >"$output_dir/V41_FP8_AOT.json"
fi
(
  cd "$output_dir"
  sha256sum cuteafd libcuteafd_native.so V41_EXPERT_AOT.json V41_EXPERT_TP_AOT.json V41_FP8_AOT.json >ARTIFACT_SHA256SUMS
  sha256sum -c ARTIFACT_SHA256SUMS
)
