#!/usr/bin/env bash
# GPUs reach containers through CDI (--device nvidia.com/gpu=NAME), never
# --gpus: the legacy hook adds device rules behind systemd's back, so a
# `systemctl daemon-reload` strips them from running containers (NVML dies).
# A CDI device is passed by Docker itself and survives the reload.

cuteafd_cdi_require() { # cuteafd_cdi_require NAME...   (fail unless the CDI spec lists each)
  local name spec found
  [[ "${CUTEAFD_CDI_CHECK:-on}" != off ]] || return 0   # script tests only: no real spec on the machine
  local -a specs=(${CUTEAFD_CDI_SPECS:-/var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml})
  for name in "$@"; do
    found=0
    for spec in "${specs[@]}"; do
      [[ -r "$spec" ]] && grep -Eq "^[[:space:]]*-?[[:space:]]*name:[[:space:]]*[\"']?${name}[\"']?[[:space:]]*\$" "$spec" && found=1 && break
    done
    ((found)) || { echo "cuteafd: CDI device nvidia.com/gpu=$name is not in /var/run/cdi/nvidia.yaml on $(hostname); run: sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml" >&2; return 1; }
  done
}

cuteafd_cdi_only_all() { # true when every readable spec names only `all`
  local spec any=0
  local -a specs=(${CUTEAFD_CDI_SPECS:-/var/run/cdi/nvidia.yaml /etc/cdi/nvidia.yaml})
  for spec in "${specs[@]}"; do
    [[ -r "$spec" ]] || continue
    any=1
    grep -E "^[[:space:]]*-?[[:space:]]*name:" "$spec" | grep -Eqv "name:[[:space:]]*[\"']?all[\"']?[[:space:]]*\$" && return 1
  done
  ((any))
}

cuteafd_cdi_single_gpu_host() {
  [[ -z "${CUTEAFD_CDI_GPU_COUNT:-}" ]] || { [[ "$CUTEAFD_CDI_GPU_COUNT" == 1 ]]; return; }  # script tests only
  local smi
  smi="$(command -v nvidia-smi || { [[ -x /usr/lib/wsl/lib/nvidia-smi ]] && echo /usr/lib/wsl/lib/nvidia-smi; })" || return 1
  [[ -n "$smi" && "$("$smi" -L 2>/dev/null | wc -l)" == 1 ]]
}

cuteafd_cdi_gpu_args() { # cuteafd_cdi_gpu_args all|INDEX_OR_UUID[,INDEX_OR_UUID...]  -> CUTEAFD_CDI_ARGS
  local -a names
  IFS=, read -ra names <<<"$1"
  ((${#names[@]})) && [[ -n "${names[0]}" ]] || { echo "cuteafd: empty GPU request" >&2; return 1; }
  # WSL2's spec lists only `all`; on a single-GPU host that is the one GPU asked for.
  if ((${#names[@]} == 1)) && [[ "${names[0]}" != all ]] && cuteafd_cdi_only_all && cuteafd_cdi_single_gpu_host; then
    names=(all)
  fi
  cuteafd_cdi_require "${names[@]}" || return 1
  CUTEAFD_CDI_ARGS=()
  local name
  for name in "${names[@]}"; do CUTEAFD_CDI_ARGS+=(--device "nvidia.com/gpu=$name"); done
}
