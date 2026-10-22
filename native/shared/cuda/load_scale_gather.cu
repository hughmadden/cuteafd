// Load-time per-row scale gathers on the GPU (see load_scale_gather.h).
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>

#include "load_scale_gather.h"

namespace {
__global__ void gather_kernel(const float* grid, const uint32_t* index, float* row_major, float* k_major,
                              int64_t rows, int64_t kb, int64_t grid_cols, int64_t col0) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * kb;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = i / kb, b = i % kb;
    const float v = cuteafd_scale_gather::gathered(grid, index, grid_cols, col0, r, b);
    if (row_major) row_major[r * kb + b] = v;
    if (k_major) k_major[b * rows + r] = v;
  }
}
}  // namespace

// `index` [rows] (device), `grid` [*, grid_cols], outputs row major [rows, kb]
// and/or K-block major [kb, rows] (either may be null, not both).
extern "C" int32_t cuteafd_load_scale_gather(const void* grid, const void* index, void* row_major, void* k_major,
                                             int64_t rows, int64_t kb, int64_t grid_cols, int64_t col0, void* stream) {
  if (!grid || !index || rows < 1 || kb < 1 || col0 < 0 || col0 + kb > grid_cols || (!row_major && !k_major))
    return cudaErrorInvalidValue;
  int device = 0, sms = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  const int64_t blocks = std::min<int64_t>((rows * kb + 255) / 256, int64_t(sms > 0 ? sms : 1) * 8);
  gather_kernel<<<unsigned(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(grid), static_cast<const uint32_t*>(index), static_cast<float*>(row_major),
      static_cast<float*>(k_major), rows, kb, grid_cols, col0);
  return cudaGetLastError();
}
