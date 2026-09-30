set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

model_id := env_var_or_default("CUTEAFD_MODEL_ID", "deepseek-ai/DeepSeek-V4.1-Flash")
spark_hosts := env_var_or_default("CUTEAFD_SPARK_HOSTS", "ostrich,dodo,emu,kiwi")
base_image := env_var_or_default("CUTEAFD_CONTAINER_BASE", "nvcr.io/nvidia/pytorch:26.05-py3")

default:
    @just --list

doctor-host:
    scripts/launch/doctor.sh --role coordinator --model-id "{{ model_id }}"

doctor:
    scripts/launch/cuteafd doctor --role coordinator --model-id "{{ model_id }}"

doctor-container-coordinator:
    scripts/build/cuteafd-dev.sh coordinator cuteafd doctor --role coordinator --model-id "{{ model_id }}"

doctor-hosts HOSTS=spark_hosts:
    scripts/launch/run-on-hosts.sh "{{ HOSTS }}" 'cd {{ justfile_directory() }} && scripts/launch/doctor.sh --role expert --model-id "{{ model_id }}"'

build-rust:
    cargo build --manifest-path rust/Cargo.toml --workspace

test-rust:
    scripts/lib/run-with-python-env.sh \
      cargo test --manifest-path rust/Cargo.toml --workspace

test-rust-fast:
    RUSTFLAGS="${RUSTFLAGS:--Awarnings}" \
      CUTEAFD_DISABLE_NATIVE_AUTO_DISCOVERY=1 \
      scripts/lib/run-with-python-env.sh \
      cargo test --manifest-path rust/Cargo.toml --workspace --exclude cuteafd-daemon
    RUSTFLAGS="${RUSTFLAGS:--Awarnings}" \
      env -u CUTEAFD_NATIVE_LIB -u CUTEAFD_REAL_FULL_CUDA_REFERENCE_KERNELS -u CUTEAFD_B12X \
      CUTEAFD_DISABLE_NATIVE_AUTO_DISCOVERY=1 \
      scripts/lib/run-with-python-env.sh \
      cargo test --manifest-path rust/Cargo.toml -p cuteafd-daemon -- \
        --skip real_checkpoint \
        --skip when_available \
        --skip when_cuda_available \
        --skip when_cuda_enabled \
        --skip native_available \
        --skip real_full_preflight \
        --skip real_full_runtime \
        --skip real_full_info_from_report \
        --skip uses_coord_dense_graph_slot \
        --skip uses_coord_sparse_a_graph_slot \
        --skip replays_same_bucket_when_rows_change \
        --skip cuda_graph \
        --skip b12x \
        --skip triton

test-python:
    cd python && ../scripts/lib/run-with-python-env.sh \
      uv run --frozen pytest reference/tests ../scripts/tests

test-native:
    cmake -S native -B native/build -G Ninja -DCUTEAFD_ENABLE_CUDA=OFF -DCUTEAFD_ENABLE_RDMA=OFF
    cmake --build native/build
    ctest --test-dir native/build --output-on-failure
    native_lib="{{ justfile_directory() }}/native/build/libcuteafd_native.so"; \
      for test_name in \
        tests::native_version_call \
        tests::error_propagation \
        tests::allocate_copy_free_roundtrip \
        tests::rdma_device_info_and_host_buffer_plan; do \
        CUTEAFD_NATIVE_LIB="$native_lib" cargo test \
          --manifest-path rust/Cargo.toml -p cuteafd-ffi \
          "$test_name" -- --exact || exit; \
      done

test-native-rdma OUT="reports/phase0_artifacts/native_rdma_enabled_build_status.json":
    python python/tools/aot/check_native_rdma_build.py --clean --output "{{ OUT }}"

test-smoke: doctor-host build-rust test-rust

docker-build-coordinator:
    docker build \
      --platform linux/amd64 \
      --build-arg BASE_IMAGE="{{ base_image }}" \
      --build-arg CUTEAFD_ROLE=coordinator \
      --build-arg CUDA_ARCH=120 \
      --build-arg TARGET_PLATFORM=linux/amd64 \
      -f docker/Dockerfile.dev \
      -t cuteafd-coordinator-dev .

docker-gpu-check IMAGE="cuteafd-coordinator-dev":
    CUTEAFD_DOCKER_GPU_VERIFY_IMAGE="{{ IMAGE }}" scripts/build/configure-docker-nvidia-runtime.sh --verify-only

docker-configure-nvidia-runtime:
    sudo scripts/build/configure-docker-nvidia-runtime.sh

docker-shell-coordinator *ARGS:
    scripts/build/cuteafd-dev.sh coordinator {{ ARGS }}

docker-shell-spark *ARGS:
    scripts/build/cuteafd-dev.sh expert {{ ARGS }}

api-smoke MODEL=model_id URL="http://127.0.0.1:8000":
    scripts/qualify/api-smoke.sh "{{ URL }}" "{{ MODEL }}"

transport-capabilities BENCHMARK_JSONL="reports/phase0_artifacts/benchmarks/phase0_results.jsonl" OUT="reports/phase0_artifacts/transport_capabilities.json":
    scripts/launch/cuteafd transport-capabilities --benchmark-jsonl "{{ BENCHMARK_JSONL }}" --out "{{ OUT }}"

bench-rdma HOST_A HOST_B:
    scripts/bench/bench-rdma-pair.sh "{{ HOST_A }}" "{{ HOST_B }}"

bench-verbs-app HOST_A HOST_B:
    scripts/bench/bench-verbs-app-pair.sh "{{ HOST_A }}" "{{ HOST_B }}"

bench-verbs-app-coordinator HOSTS=spark_hosts:
    scripts/bench/bench-verbs-app-coordinator-links.sh "{{ HOSTS }}"
