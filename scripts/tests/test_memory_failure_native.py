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


def test_graph_capture_failure_clears_last_error_before_peer_eager_retry(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    native = (ROOT / "native/shared/src/cuteafd_native.cc").read_text()
    # Compile the production entry points against a CPU runtime which retains
    # recoverable errors exactly until cudaGetLastError consumes them.
    entry_points = native.split('extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(', 1)[1]
    entry_points = 'extern "C" cuteafd_status_t cuteafd_cuda_graph_begin_capture(' + entry_points.split(
        'extern "C" cuteafd_status_t cuteafd_cuda_graph_launch(', 1)[0]
    diagnostics = (ROOT / "native/shared/include/cuteafd_graph_diagnostics.h").read_text()
    clear_error = diagnostics.split("inline cudaError_t diagnostic_cuda", 1)[1].split("using GraphEdgesQuery", 1)[0]
    source = tmp_path / "graph_capture_failure.cc"
    source.write_text(r'''
#include "cuteafd_native.h"
#include <cassert>
#include <string>
#define CUTEAFD_NATIVE_ENABLE_CUDA 1
#define CUTEAFD_CHECK_MEMORY(api, domain, call, bytes, known, handle) (call)
using cudaError_t = int;
using cudaStream_t = void*;
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;
constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorStreamCaptureInvalidated = 901;
constexpr int cudaErrorIllegalAddress = 700, cudaStreamCaptureModeThreadLocal = 1;
static int last_error = 0, sticky_error = 0, injected = 0, point = 0, live_graphs = 0, live_execs = 0;
static bool active = false, partial_exec = false, census = false;
static std::string message;
int cudaGetLastError() { int status = sticky_error ? sticky_error : last_error; last_error = 0; return status; }
int outcome(int at) {
  const int status = sticky_error ? sticky_error : (point == at ? injected : 0);
  if (status) last_error = status; // Successful CUDA calls do not consume a previous error.
  return status;
}
int cudaStreamBeginCapture(void*, int) { int status = outcome(1); if (!status) active = true; return status; }
int cudaStreamEndCapture(void*, void** graph) {
  assert(active); active = false;
  const int status = outcome(2);
  if (!status) { *graph = reinterpret_cast<void*>(0x1000); ++live_graphs; }
  return status;
}
int cudaGraphInstantiate(void** exec, void*, int) {
  const int status = outcome(3);
  if (!status || partial_exec) { *exec = reinterpret_cast<void*>(0x2000); ++live_execs; }
  return status;
}
int cudaGraphDestroy(void*) { assert(live_graphs > 0); --live_graphs; return outcome(4); }
int cudaGraphExecDestroy(void*) { assert(live_execs > 0); --live_execs; return outcome(5); }
int cudaGraphGetNodes(void*, void**, size_t* nodes) { *nodes = 3; return 0; }
static cuteafd_status_t end_capture_retained(void*, cuteafd_cuda_graph_capture_info_t*, bool);
int cudaMemGetInfo(size_t* free, size_t* total) { *free = 1 << 20; *total = 2 << 20; return 0; }
bool graph_census_enabled() { return census; }
void record_graph_census(void*, void*, int, size_t, int, size_t) {}
cuteafd_status_t ok() { message.clear(); return CUTEAFD_STATUS_OK; }
cuteafd_status_t fail(cuteafd_status_t status, const char* action) { message = action; return status; }
cuteafd_status_t fail_cuda(cuteafd_status_t status, const char* action, int error) {
  message = std::string(action) + ": " + std::to_string(error); return status;
}
cuteafd_status_t fill_cuda_graph_capture_info(void* graph, void* exec, cuteafd_cuda_graph_capture_info_t* out) {
  out->graph = graph; out->graph_exec = exec; out->node_count = 3; return ok();
}
namespace cuteafd {
''' + "inline cudaError_t diagnostic_cuda" + clear_error + r'''
int inspect_graph_topology(void*, size_t* count, size_t* width, int* spans) {
  *count = 3; *width = 1; *spans = 1; return 0;
}
}
''' + entry_points + r'''
int peer_wait_abortable() { return cudaGetLastError(); }
int main() {
  void* stream = reinterpret_cast<void*>(0x3000);
  for (bool with_census : {false, true}) {
    census = with_census;
    // Both ranks share this host thread's last-error slot. A valid eager wait
    // after either rank's failed instantiate must not poison the TP2 submission.
    for (int failing_rank : {0, 1}) {
      int captures[2] = {}, eager[2] = {}, cancelled[2] = {};
      for (int rank : {0, 1}) {
        point = rank == failing_rank ? 3 : 0; injected = cudaErrorMemoryAllocation;
        assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
        void* exec = nullptr; size_t nodes = 0, width = 0; int spans = 0;
        const auto status = cuteafd_cuda_graph_end_capture_topology(stream, &exec, &nodes, &width, &spans);
        assert(!active && live_graphs == 0);
        if (status != CUTEAFD_STATUS_OK) {
          assert(message == "cudaGraphInstantiate failed; lower GPU_MEMORY_CAP or increase GPU_RESERVE to leave graph headroom: 2" && exec == nullptr);
          ++cancelled[rank]; ++eager[rank];
        } else {
          assert(nodes == 3 && width == 1 && spans == 1);
          ++captures[rank]; assert(cudaGraphExecDestroy(exec) == 0);
        }
        assert(peer_wait_abortable() == 0 && live_execs == 0);
      }
      assert(cancelled[failing_rank] == 1 && eager[failing_rank] == 1);
      assert(captures[1 - failing_rank] == 1); // Never rerun the peer's queued work.
      assert(peer_wait_abortable() == 0); // A subsequent submission remains usable.
    }
  }
  census = false;
  // All three end-capture APIs share recovery, including partially returned execs.
  for (int wrapper : {0, 1, 2}) for (int failure : {2, 3, 4}) {
    point = failure; injected = failure == 2 ? cudaErrorStreamCaptureInvalidated : cudaErrorMemoryAllocation;
    partial_exec = true;
    assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
    void* exec = nullptr; size_t nodes = 0, width = 0; int spans = 0;
    cuteafd_cuda_graph_capture_info_t info{};
    const auto status = wrapper == 0 ? cuteafd_cuda_graph_end_capture(stream, &exec)
      : wrapper == 1 ? cuteafd_cuda_graph_end_capture_topology(stream, &exec, &nodes, &width, &spans)
      : cuteafd_cuda_graph_end_capture_retained(stream, &info);
    // The retained API transfers its source graph, so it has no destroy boundary.
    if (wrapper == 2 && failure == 4) {
      assert(status == CUTEAFD_STATUS_OK);
      cudaGraphDestroy(info.graph); cudaGraphExecDestroy(info.graph_exec); cudaGetLastError();
    } else {
      const auto expected = failure == 3 ? CUTEAFD_STATUS_ALLOCATION_FAILED : CUTEAFD_STATUS_INTERNAL_ERROR;
      assert(status == expected && exec == nullptr && info.graph == nullptr);
    }
    assert(!active && live_graphs == 0 && live_execs == 0 && peer_wait_abortable() == 0);
  }
  point = 1; injected = cudaErrorMemoryAllocation;
  assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_INTERNAL_ERROR);
  assert(!active && message == "cudaStreamBeginCapture failed: 2" && peer_wait_abortable() == 0);
  // Clearing a recoverable error slot cannot erase a genuine sticky GPU fault.
  point = 0;
  assert(cuteafd_cuda_graph_begin_capture(stream) == CUTEAFD_STATUS_OK);
  sticky_error = cudaErrorIllegalAddress;
  cuteafd_cuda_graph_capture_info_t info{};
  assert(cuteafd_cuda_graph_end_capture_retained(stream, &info) == CUTEAFD_STATUS_INTERNAL_ERROR);
  assert(!active && peer_wait_abortable() == cudaErrorIllegalAddress);
}
''')
    executable = tmp_path / "graph_capture_failure"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/shared/include"), str(source), "-o", str(executable)],
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
