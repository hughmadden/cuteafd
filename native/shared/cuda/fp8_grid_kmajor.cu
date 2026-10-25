// Load-time widening of a 128x128 FP8 block-scale grid [ceil(rows/128), kb]
// to per-row scales stored K-block major [kb, rows]:
// kmajor[b * rows + r] = grid[(r / 128) * kb + b] (the block-FP8 prefill
// GEMM's operand). A pure gather, so the result is exact.
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>

namespace {
__global__ void grid_kmajor_kernel(const float* grid, float* kmajor, int64_t rows, int64_t kb) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * kb;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t b = i / rows, r = i % rows;
    kmajor[i] = grid[(r / 128) * kb + b];
  }
}
}  // namespace

extern "C" int32_t cuteafd_fp8_grid_kmajor(const void* grid, void* kmajor, int64_t rows, int64_t kb, void* stream) {
  if (rows < 1 || kb < 1) return cudaErrorInvalidValue;
  int device = 0, sms = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  const int64_t blocks = std::min<int64_t>((rows * kb + 255) / 256, int64_t(sms > 0 ? sms : 1) * 8);
  grid_kmajor_kernel<<<unsigned(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(grid), static_cast<float*>(kmajor), rows, kb);
  return cudaGetLastError();
}
