#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${CUTEAFD_NVFP4_MODEL_OPT_IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}"
fixture="${CUTEAFD_NVFP4_MODEL_OPT_FIXTURE:-tests/fixtures/nvfp4/real_tensor_decode.json}"
output="${CUTEAFD_NVFP4_MODEL_OPT_OUTPUT:-tests/fixtures/nvfp4/modelopt_reference.json}"
gpus="${CUTEAFD_NVFP4_MODEL_OPT_DOCKER_GPUS:-all}"
device="${CUTEAFD_NVFP4_MODEL_OPT_DEVICE:-cuda}"
# The verifier writes its --output reference into the bind-mounted checkout, so
# it runs as the invoking user: the fixture stays deletable without sudo. `--user`
# bypasses the image's passwd lookup, so the identity and the caches are passed
# explicitly (the image's own CARGO_HOME/HOME belong to root).
container_home="${CUTEAFD_NVFP4_MODEL_OPT_CONTAINER_HOME:-/tmp/cuteafd-dev-home}"

docker_args=(
  run --rm -i
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v "$repo_root:/workspace/cuteafd"
  -w /workspace/cuteafd
  -e "CUTEAFD_NVFP4_MODEL_OPT_IMAGE=$image"
  -e "CUTEAFD_NVFP4_MODEL_OPT_DEVICE=$device"
  --user "$(id -u):$(id -g)"
  -e "HOME=$container_home"
  -e "USER=$(id -un)"
  -e "LOGNAME=$(id -un)"
  -e "TORCHINDUCTOR_CACHE_DIR=$container_home/torchinductor"
  -e "CARGO_HOME=$container_home/cargo"
  --entrypoint python
)
if [ -n "$gpus" ] && [ "$gpus" != "none" ]; then
  docker_args+=(--gpus "$gpus")
fi

docker "${docker_args[@]}" "$image" \
  python/tools/hf/verify_nvfp4_modelopt_real_tensor_decode_fixture.py \
    --fixture "$fixture" \
    --output "$output" \
    --device "$device"
