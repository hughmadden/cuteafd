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
extern "C" void cuteafd_memory_event(int32_t, uintptr_t, size_t, int32_t released) {
  live += released ? -1 : 1;
}
extern "C" void cuteafd_memory_failure(const cuteafd_memory_failure_t* trigger) {
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
int main() {
  void* first = nullptr;
  void* second = nullptr;
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &first, 4096, 0, allocate(&first)) == 0);
  assert(CUTEAFD_TRACK_ALLOC("cudaMalloc", &second, 8192, 0, allocate(&second)) == 2);
  assert(calls == 2 && captured_live == 1 && code == 2);
  assert(CUTEAFD_TRACK_FREE(first, 0) == 0 && live == 0);
  void* graph = reinterpret_cast<void*>(0x1234);
  assert(CUTEAFD_CHECK_MEMORY("cudaGraphInstantiate", "cuda", 2, 0, 0, graph) == 2);
  graph_alive = false;
  assert(CUTEAFD_CHECK_MEMORY("cublasCreate", "cublas", 3, 0, 0, nullptr) == 3);
  assert(code == 3 && std::strcmp(api, "cublasCreate") == 0);
  assert(!graph_alive);
}
''')
    executable = tmp_path / "memory_failure"
    built = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "native/shared/include"), str(source),
                            "-o", str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
