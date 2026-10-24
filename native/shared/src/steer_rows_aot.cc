// cuteafd_steer_rows and cuteafd_add_rows: the SparkInfer steer_rows_one
// (k_max == 1), steer_rows_general (k_max <= 32) and add_rows_f32 exports
// behind the weight-variant ABI (cuteafd_steer_rows.h).
#include "cuteafd_steer_rows.h"
#include "add_rows_f32.h"
#include "steer_rows_general.h"
#include "steer_rows_one.h"
#include <atomic>
#include <mutex>

namespace {
constexpr int kDevices = 32;
cuteafd_steer_rows_one_Kernel_Module_t module_one{};
cuteafd_steer_rows_general_Kernel_Module_t module_general{};
cuteafd_add_rows_f32_Kernel_Module_t module_add{};
std::mutex mutex;
std::atomic<bool> loaded[kDevices]{};

using Entry = void (*)(void**);
struct Program { cudaLibrary_t* library; Entry init, load; };

int load_device(int device) {
  std::lock_guard<std::mutex> guard(mutex);
  if (loaded[device].load(std::memory_order_acquire)) return 0;
  const Program programs[] = {
    {&module_one.module, _mlir_cuteafd_steer_rows_one_cuda_init, _mlir_cuteafd_steer_rows_one_cuda_load_to_device},
    {&module_general.module, _mlir_cuteafd_steer_rows_general_cuda_init,
     _mlir_cuteafd_steer_rows_general_cuda_load_to_device},
    {&module_add.module, _mlir_cuteafd_add_rows_f32_cuda_init, _mlir_cuteafd_add_rows_f32_cuda_load_to_device},
  };
  for (const auto& program : programs) {
    int result = 0;
    auto* library = program.library;
    if (!*library) {
      void* args[] = {&library, &result};
      program.init(args);
      if (result) return result;
    }
    void* args[] = {&library, &device, &result};
    program.load(args);
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

// Module loading is not a stream operation: never attempt it mid-capture.
int ensure_loaded(cudaStream_t stream) {
  int device = -1;
  if (auto status = current_device(&device)) return status;
  if (loaded[device].load(std::memory_order_acquire)) return 0;
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (auto status = cudaStreamIsCapturing(stream, &capture)) return status;
  if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  return load_device(device);
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
  if (auto status = ensure_loaded(stream)) return status;
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

extern "C" int cuteafd_add_rows(__nv_bfloat16* y, int64_t ld_y, const float* d, int64_t ld_d, int32_t rows,
    int32_t h, const int32_t* vidx, const int32_t* on, cudaStream_t stream) {
  if (rows < 0 || h < 8 || h > 65536 || h % 8 || ld_y < h || ld_y % 8 || ld_d < h || ld_d % 4)
    return cudaErrorInvalidValue;
  if (rows == 0) return 0;
  if (!y || !d || !vidx || !on || !aligned16(y) || !aligned16(d)) return cudaErrorInvalidValue;
  if (auto status = ensure_loaded(stream)) return status;
  return cute_dsl_cuteafd_add_rows_f32_wrapper(&module_add, y, const_cast<float*>(d),
      const_cast<int32_t*>(vidx), const_cast<int32_t*>(on), ld_y, ld_d, rows, h, stream);
}
