#include <cuteafd_cuda_registration_abi.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

// This undocumented ABI must match the installed toolkit, not a guessed shim.
using RegisterFunction = void CUDARTAPI(void**, const char*, char*, const char*,
    int, uint3*, uint3*, dim3*, dim3*, int*);
static_assert(std::is_same_v<decltype(__cudaRegisterFunction), RegisterFunction>,
    "CUDA registration ABI changed: review the TP2 kernel inventory wrapper");
extern "C" RegisterFunction __real___cudaRegisterFunction;

namespace {
struct Kernel { const void* host; std::string name; };
struct Inventory { std::mutex mutex; std::vector<Kernel> kernels; };
Inventory& inventory() {
  // Registrations run during DSO initialization, before ordinary static owners.
  static Inventory* state = new Inventory;
  return *state;
}
}

extern "C" void CUDARTAPI __wrap___cudaRegisterFunction(void** fat_binary,
    const char* host, char* device, const char* name, int thread_limit,
    uint3* tid, uint3* bid, dim3* block, dim3* grid, int* warp_size) {
  __real___cudaRegisterFunction(fat_binary, host, device, name, thread_limit,
      tid, bid, block, grid, warp_size);
  auto& state = inventory();
  std::lock_guard<std::mutex> guard(state.mutex);
  if (std::none_of(state.kernels.begin(), state.kernels.end(),
      [host](const Kernel& kernel) { return kernel.host == host; })) {
    state.kernels.push_back({host, name ? name : ""});
  }
}

// Only native runtime-registered kernels are inventoried here. CuTe AOT
// cudaLibrary programs retain their explicit programs.load_matching startup.
extern "C" int32_t cuteafd_tp2_native_initialize() {
  auto& state = inventory();
  std::lock_guard<std::mutex> guard(state.mutex);
  // Missing wrap linkage or changed nvcc registration must never silently pass.
  for (const char* required : {"select_fast_kernel", "softmax_select_kernel",
      "rtx_tp2_sum", "wait_flag_abortable"}) {
    if (std::none_of(state.kernels.begin(), state.kernels.end(),
        [required](const Kernel& kernel) { return kernel.name.find(required) != std::string::npos; })) {
      return cudaErrorInvalidDeviceFunction;
    }
  }
  for (const auto& kernel : state.kernels) {
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, kernel.host);
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}
