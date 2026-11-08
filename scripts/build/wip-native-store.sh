#!/usr/bin/env bash
# Shared by the build and wip.sh's host-side no-GPU-turn probe.
wip_native_required_items() {
  local item family package
  for item in libcuteafd_native.so PROGRAMS.json V41_EXPERT_AOT.json V41_EXPERT_TP_AOT.json V41_FP8_AOT.json; do
    printf 'file %s\n' "$item"
  done
  printf 'dir %s\n' lib-variants V41_EXPERT_WORKSPACES
  for item in "${host_isas[@]:1}"; do
    printf 'file lib-variants/%s/libcuteafd_native.so\n' "$item"
  done
  if [[ "$v41_coordinator_aot" == ON ]]; then
    for item in rtx_backbone rtx_tp2; do
      printf 'file V41_EXPERT_WORKSPACES/mxfp4_%s.json\n' "$item"
      [[ "$v41_nvfp4_aot" != ON ]] || printf 'file V41_EXPERT_WORKSPACES/nvfp4_%s.json\n' "$item"
    done
  fi
  local -a families
  IFS=';' read -ra families <<<"$expert_families"
  for family in "${families[@]}"; do
    case "$family" in
      *:exl3-k*)
        [[ "$exl3_aot" == ON ]] || continue
        printf 'dir exl3\nfile exl3/exl3-%s-%s/manifest.json\n' "${family%%:*}" "${family#*:exl3-}"
        continue ;;
      *:fp8) package="fp8-${family%%:*}" ;;
      step5:bf16) package=fp8-step5-bf16-weights ;;
      step5:nvfp4) package=fp8-step5-nvfp4a4 ;;
      *:nvfp4|*:nvfp4a4) package="fp8-${family%%:*}-${family#*:}" ;;
      *) continue ;;
    esac
    printf 'dir fp8\nfile fp8/%s/manifest.json\n' "$package"
    if [[ "$role" == expert && "$family" == *:fp8 && ";$bf16_families;" == *";${family%%:*};"* ]]; then
      printf 'file fp8/%s-bf16/manifest.json\n' "$package"
    fi
    if [[ "$family" == *:nvfp4 && "$family" != step5:nvfp4 ]]; then
      printf 'file fp8/%sa4/manifest.json\n' "$package"
    fi
    if [[ "$role" == coordinator && "$family" == qwen4:nvfp4* ]]; then
      printf 'file fp8/fp8-qwen4/manifest.json\n'
    fi
  done
}

wip_native_store_valid() {
  local entry="$1" required="$2" kind item
  [[ -f "$entry/COMPLETE" ]] || return 1
  if [[ ! -s "$required" ]]; then
    echo "WIP native cache: refusing $entry: missing required inventory $required" >&2
    return 1
  fi
  while read -r kind item; do
    if [[ ( "$kind" == file && -f "$entry/output/$item" ) || ( "$kind" == dir && -d "$entry/output/$item" ) ]]; then
      continue
    fi
    echo "WIP native cache: refusing $entry: missing required $kind $item" >&2
    return 1
  done <"$required"
  if ! (cd "$entry/output" && sha256sum --quiet --strict -c ../NATIVE_SHA256SUMS); then
    echo "WIP native cache: refusing $entry: native checksums failed" >&2
    return 1
  fi
}
