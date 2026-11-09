"""CPU proof of native allocation-boundary capture before constructor cleanup."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_native_failure_wrappers_capture_live_state_and_preserve_raw_codes(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "memory_failure.cc"
    source.write_text(r'''
#include "memory_failure.h"
#include <cassert>
#include <cstring>
static int live = 0, captured_live = 0, calls = 0, code = 0;
static bool graph_alive = true;
static const char* api = nullptr;
void event(int32_t, uintptr_t, size_t, int32_t released, const char* file, int32_t line,
    uint32_t kind, const char* owner) {
  // An allocation names its tag; a release names none.
  if (!released) assert(file && line > 0 && kind == CUTEAFD_MEMORY_KIND_KV && owner && std::strcmp(owner, "test/kv") == 0);
  else assert(kind == 0 && owner == nullptr);
  live += released ? -1 : 1;
}
void failure(const cuteafd_memory_failure_t* trigger) {
  captured_live = live; code = trigger->code; api = trigger->api;
  if (std::strcmp(api, "cudaGraphInstantiate") == 0) {
    assert(graph_alive && trigger->handle == 0x1234 && !trigger->bytes_known);
  }
}
int allocate(void** output) {
  ++calls;
  if (calls == 2) return 2;
  *output = reinterpret_cast<void*>(0x1000); return 0;
}
int graph_failure() {
  struct Graph { ~Graph() { graph_alive = false; } } graph;
  return CUTEAFD_CHECK_MEMORY("cudaGraphInstantiate", "cuda", 2, 0, 0, reinterpret_cast<void*>(0x1234));
}
int main() {
  cuteafd_set_memory_event_callback_v3(event);
  cuteafd_set_memory_failure_callback(failure);
  void* first = nullptr;
  void* second = nullptr;
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &first, 4096, 0, 0, "test/kv", allocate(&first)) == 1);
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &first, 4096, 0, CUTEAFD_MEMORY_KIND_KV, "", allocate(&first)) == 1);
  assert(calls == 0 && first == nullptr && live == 0);
  assert(CUTEAFD_TRACK_MALLOC(64, 0, "test/kv") == nullptr);
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &first, 4096, 0, CUTEAFD_MEMORY_KIND_KV, "test/kv", allocate(&first)) == 0);
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &second, 8192, 0, CUTEAFD_MEMORY_KIND_KV, "test/kv", allocate(&second)) == 2);
  uint32_t kind = 0; char owner[32] = {}; size_t required = 0;
  assert(cuteafd_last_allocation_tag(&kind, owner, sizeof(owner), &required) == 0);
  assert(kind == CUTEAFD_MEMORY_KIND_KV && std::strcmp(owner, "test/kv") == 0);
  assert(calls == 2 && captured_live == 1 && code == 2);
  assert(CUTEAFD_TRACK_FREE(first, 0) == 0 && live == 0);
  assert(graph_failure() == 2);
  assert(!graph_alive);
  assert(CUTEAFD_CHECK_MEMORY("cublasCreate", "cublas", 3, 0, 0, nullptr) == 3);
  assert(code == 3 && std::strcmp(api, "cublasCreate") == 0);
  assert(!graph_alive);
  assert(CUTEAFD_CHECK_MEMORY("cudaProbe", "cuda", 801, 0, 0, nullptr) == 801);
  assert(code == 3);
  size_t bytes = 1;
  void* host = CUTEAFD_MALLOC(bytes++);
  assert(bytes == 2);
  free(host);
}
''')
    executable = tmp_path / "memory_failure"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/shared/include"), str(source), str(ROOT / "native/shared/src/memory_failure.cc"),
                            "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr


def test_allocation_failure_format_preserves_original_error(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "allocation_failure.cc"
    source.write_text(r'''
#include "memory_failure.h"
#include <cassert>
#include <limits>
int main() {
  cuteafd_allocation_diagnostics diagnostics;
  diagnostics.device = 1;
  diagnostics.free_bytes = 4096;
  diagnostics.total_bytes = 98304;
  const std::string original = "cudaErrorMemoryAllocation (out of memory)";
  const auto format = [&](const char* owner) {
    return cuteafd_format_allocation_failure("cudaMalloc failed", 8192, owner, original, diagnostics);
  };
  assert(format("kv/pages") == "cudaMalloc failed: cudaErrorMemoryAllocation (out of memory); "
      "requested_bytes=8192; owner=kv/pages; device=1; free_bytes=4096; total_bytes=98304");
  diagnostics.memory_error = "cudaErrorUnknown (unknown error)";
  assert(format("weights/core") == "cudaMalloc failed: cudaErrorMemoryAllocation (out of memory); "
      "requested_bytes=8192; owner=weights/core; device=1; free/total unavailable: cudaErrorUnknown (unknown error)");
  diagnostics.device_error = "cudaErrorNoDevice (no CUDA-capable device is detected)";
  const auto unavailable = format(nullptr);
  assert(unavailable == "cudaMalloc failed: cudaErrorMemoryAllocation (out of memory); "
      "requested_bytes=8192; owner=unavailable; device unavailable: cudaErrorNoDevice (no CUDA-capable device is detected); "
      "free/total unavailable: cudaErrorUnknown (unknown error)");
  assert(format("") == unavailable);
  diagnostics.memory_error.clear();
  assert(format("host/staging").find("free_bytes=4096; total_bytes=98304") != std::string::npos);
  const size_t large = std::numeric_limits<size_t>::max();
  const auto wide = cuteafd_format_allocation_failure("cudaHostAlloc failed", large, "transport/rings", original, diagnostics);
  assert(wide.find("requested_bytes=" + std::to_string(large)) != std::string::npos);
  assert(wide.find("cudaHostAlloc failed: " + original) == 0);
}
''')
    executable = tmp_path / "allocation_failure"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/shared/include"), str(source),
                            str(ROOT / "native/shared/src/memory_failure.cc"), "-o", str(executable)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr


def test_cmake_rejects_wrapped_source_without_real_callbacks(tmp_path):
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake unavailable")
    audit = (ROOT / "native/CMakeLists.txt").read_text().split(
        "# Follow local implementation includes too (the vision test includes a .cu).", 1)[1]
    (tmp_path / "wrapped.cc").write_text('#include "memory_failure.h"\nint main() {}\n')
    (tmp_path / "indirect.cc").write_text('#include "wrapped.cc"\n')
    project = "cmake_minimum_required(VERSION 3.24)\nproject(fixture CXX)\n"
    project += "add_executable(missing indirect.cc)\n" + audit
    (tmp_path / "CMakeLists.txt").write_text(project)
    result = subprocess.run([cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "missing uses wrapped allocators but does not link" in result.stderr
    project = project.replace("add_executable(missing indirect.cc)",
        'add_library(cuteafd_memory_callbacks OBJECT "' + str(ROOT / "native/shared/src/memory_failure.cc") + '")\n'
        'target_include_directories(cuteafd_memory_callbacks PRIVATE "' + str(ROOT / "native/shared/include") + '")\n'
        'add_executable(missing indirect.cc)\n'
        'target_link_libraries(missing PRIVATE cuteafd_memory_callbacks)')
    (tmp_path / "CMakeLists.txt").write_text(project)
    result = subprocess.run([cmake, "-S", str(tmp_path), "-B", str(tmp_path / "linked")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
