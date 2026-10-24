// cuteafd_steer_rows: the SparkInfer steer_rows_one (k_max == 1) and
// steer_rows_general (k_max <= 32) exports behind the contract ABI.
#include "cuteafd_steer_rows.h"
#include "steer_rows_general.h"
#include "steer_rows_one.h"
#include <atomic>
#include <mutex>

namespace {
constexpr int kDevices = 32;
cuteafd_steer_rows_one_Kernel_Module_t module_one{};
cuteafd_steer_rows_general_Kernel_Module_t module_general{};
std::mutex mutex;
std::atomic<bool> loaded[kDevices]{};

int load_device(int device) {
  std::lock_guard<std::mutex> guard(mutex);
  if (loaded[device].load(std::memory_order_acquire)) return 0;
  int result = 0;
  auto* one = &module_one.module;
  auto* general = &module_general.module;
  if (!*one) {
    void* args[] = {&one, &result};
    _mlir_cuteafd_steer_rows_one_cuda_init(args);
    if (result) return result;
  }
  if (!*general) {
    void* args[] = {&general, &result};
    _mlir_cuteafd_steer_rows_general_cuda_init(args);
    if (result) return result;
  }
  {
    void* args[] = {&one, &device, &result};
    _mlir_cuteafd_steer_rows_one_cuda_load_to_device(args);
    if (result) return result;
  }
  {
    void* args[] = {&general, &device, &result};
    _mlir_cuteafd_steer_rows_general_cuda_load_to_device(args);
    if (result) return result;
  }
  loaded[device].store(true, std::memory_order_release);
  return 0;
}

int current_device(int* device) {
  auto status = cudaGetDevice(device);
  if (status) return status;
  return *device < 0 || *device >= kDevices ? cudaErrorInvalidDevice : 0;
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }
}  // namespace

extern "C" int cuteafd_steer_rows_load(void) {
  int device = -1;
  if (auto status = current_device(&device)) return status;
  return load_device(device);
}

extern "C" int cuteafd_steer_rows(__nv_bfloat16* x, int64_t ld, int32_t rows, int32_t h,
    const int32_t* vidx, const float* a, const float* b, const int32_t* k, int32_t k_max,
    cudaStream_t stream) {
  if (rows < 0 || h < 128 || h > 8192 || h % 128 || ld < h || ld % 8 || k_max < 1 || k_max > 32)
    return cudaErrorInvalidValue;
  if (rows == 0) return 0;
  if (!x || !vidx || !a || !b || !k || !aligned16(x) || !aligned16(a) || !aligned16(b))
    return cudaErrorInvalidValue;
  int device = -1;
  if (auto status = current_device(&device)) return status;
  if (!loaded[device].load(std::memory_order_acquire)) {
    // Module loading is not a stream operation: never attempt it mid-capture.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (auto status = cudaStreamIsCapturing(stream, &capture)) return status;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    if (auto status = load_device(device)) return status;
  }
  // The generated wrappers take non-const pointers; the kernels only read these.
  auto* v = const_cast<int32_t*>(vidx);
  auto* ta = const_cast<float*>(a);
  auto* tb = const_cast<float*>(b);
  auto* tk = const_cast<int32_t*>(k);
  if (k_max == 1)
    return cute_dsl_cuteafd_steer_rows_one_wrapper(&module_one, x, v, ta, tb, tk, ld, rows, h, k_max, stream);
  return cute_dsl_cuteafd_steer_rows_general_wrapper(&module_general, x, v, ta, tb, tk, ld, rows, h, k_max,
      stream);
}
