"""CPU checks for nvcc registration forwarding and workspace-free TP2 preload."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/shared/src/tp2_kernel_inventory.cc"
ABI = """#pragma once
#include <cstdint>
#define CUDARTAPI
struct uint3 { unsigned x, y, z; };
struct dim3 { unsigned x, y, z; };
struct cudaFuncAttributes {};
enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidDeviceFunction = 98 };
extern "C" void __cudaRegisterFunction(void**, const char*, char*, const char*,
    int, uint3*, uint3*, dim3*, dim3*, int*);
extern "C" cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, const void*);
"""
CALLER = r'''
#include <cuteafd_cuda_registration_abi.h>
extern "C" void register_kernel(const char* host, const char* name) {
  __cudaRegisterFunction(nullptr, host, nullptr, name, -1, nullptr, nullptr,
      nullptr, nullptr, nullptr);
}
'''
MOCK = r'''
#include <cuteafd_cuda_registration_abi.h>
#include <cassert>
#include <vector>
extern "C" void register_kernel(const char*, const char*);
extern "C" int32_t cuteafd_tp2_native_initialize();
static int device = 0, calls = 0, fail_at = -1;
static std::vector<const void*> registered;
static std::vector<const void*> visited[2];
extern "C" void __cudaRegisterFunction(void** fat, const char* host, char* fun,
    const char* name, int limit, uint3* tid, uint3* bid, dim3* block,
    dim3* grid, int* warp) {
  assert(!fat && host && !fun && name && limit == -1);
  assert(!tid && !bid && !block && !grid && !warp);
  registered.push_back(host);
}
extern "C" cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, const void* host) {
  if (calls++ == fail_at) return static_cast<cudaError_t>(719);
  visited[device].push_back(host);
  return cudaSuccess;
}
int main(int argc, char**) {
  assert(cuteafd_tp2_native_initialize() == 98); // wrap absent/empty fails closed
  const char* names[] = {"select_fast_kernel", "softmax_select_kernel",
      "rtx_tp2_sum", "wait_flag_abortable", "brand_new_helper"};
  static char identities[5];
  for (int i = 0; i < 5; ++i) {
    register_kernel(&identities[i], names[i]);
    if (i < 3) assert(cuteafd_tp2_native_initialize() == 98);
  }
  if (argc > 1) {
    assert(cuteafd_tp2_native_initialize() == 98); // no --wrap is never accepted
    return 0;
  }
  register_kernel(&identities[0], names[0]);
  for (device = 0; device < 2; ++device) {
    visited[device].clear();
    assert(cuteafd_tp2_native_initialize() == 0);
    assert(visited[device].size() == 5); // duplicate forwarding, unique inventory
    for (int i = 0; i < 5; ++i) assert(visited[device][i] == &identities[i]);
  }
  device = 0; fail_at = calls + 2;
  assert(cuteafd_tp2_native_initialize() == 719);
  assert(registered.size() == 6);
}
'''


def test_compiled_registration_inventory_forwards_and_preloads_every_new_kernel(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "cuteafd_cuda_registration_abi.h").write_text(ABI)
    (tmp_path / "caller.cc").write_text(CALLER)
    (tmp_path / "mock.cc").write_text(MOCK)
    command = [compiler, "-std=c++17", "-pthread", "-I", str(tmp_path), str(SOURCE),
               str(tmp_path / "caller.cc"), str(tmp_path / "mock.cc")]
    for wrapped in [True, False]:
        executable = tmp_path / ("wrapped" if wrapped else "unwrapped")
        subprocess.run(command + (["-Wl,--wrap=__cudaRegisterFunction"] if wrapped else
                       ["-Wl,--defsym=__real___cudaRegisterFunction=__cudaRegisterFunction"])
                       + ["-o", str(executable)], check=True, capture_output=True)
        subprocess.run([str(executable)] + ([] if wrapped else ["no-wrap"]), check=True,
                       capture_output=True)
    (tmp_path / "cuteafd_cuda_registration_abi.h").write_text(
        ABI.replace("int, uint3*", "long, uint3*"))
    result = subprocess.run([compiler, "-std=c++17", "-I", str(tmp_path),
                             "-fsyntax-only", str(SOURCE)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "CUDA registration ABI changed" in result.stderr


def test_cuda_build_attests_real_header_and_always_wraps_registration():
    cmake = (ROOT / "native/CMakeLists.txt").read_text()
    assert 'crt/host_runtime.h" CUTEAFD_CUDA_HOST_RUNTIME' in cmake
    assert 'message(FATAL_ERROR "Cannot attest the CUDA registration ABI' in cmake
    assert 'list(APPEND CUTEAFD_NATIVE_SOURCES shared/src/tp2_kernel_inventory.cc)' in cmake
    assert 'target_link_options(cuteafd_native PRIVATE "-Wl,--wrap=__cudaRegisterFunction")' in cmake
    constructor = (ROOT / "rust/crates/cuteafd-daemon/src/shared/peer_split.rs").read_text()
    constructor = constructor.split("pub fn new(", 1)[1].split("pub fn new_abortable", 1)[0]
    assert constructor.index("library.tp2_native_initialize()?") < constructor.index("let zeroed")
    assert 'elapsed_ms = started.elapsed().as_secs_f64() * 1000.0' in constructor


def test_inventory_signature_matches_installed_toolkit_without_cuda_initialization(tmp_path):
    header = Path("/usr/local/cuda/include/crt/host_runtime.h")
    compiler = shutil.which("g++")
    if not header.exists() or not compiler:
        pytest.skip("CUDA headers or compiler unavailable")
    declaration = re.search(r"extern void CUDARTAPI __cudaRegisterFunction\([^;]*;", header.read_text())
    assert declaration, "toolkit registration ABI missing"
    (tmp_path / "cuteafd_cuda_registration_abi.h").write_text(
        '#include <cuda_runtime.h>\nextern "C" {\n' + declaration[0] + '\n}\n')
    subprocess.run([compiler, "-std=c++17", "-I", str(tmp_path),
                    "-I", str(header.parents[1]), "-fsyntax-only", str(SOURCE)],
                   check=True, capture_output=True)
