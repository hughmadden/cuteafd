// FP8 E4M3 weights with FP32 128x128 block scales (GLM's checkpoint format)
// dequantized to BF16 at load, exactly as the reference does:
// bf16(float(w) * scale[row / 128, col / 128]).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
__global__ void fp8_block_dequant_kernel(const __nv_fp8_e4m3* w, const float* scale, __nv_bfloat16* out,
                                         int rows, int cols) {
  const int scale_cols = (cols + 127) / 128;
  const uint64_t total = uint64_t(rows) * cols;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += uint64_t(gridDim.x) * blockDim.x) {
    const int row = int(i / cols), col = int(i % cols);
    out[i] = __float2bfloat16(float(w[i]) * scale[(row / 128) * scale_cols + col / 128]);
  }
}
}  // namespace

extern "C" int32_t cuteafd_fp8_block_dequant(const void* w, const void* scale, void* out, int32_t rows,
                                             int32_t cols, void* stream) {
  if (rows < 1 || cols < 1) return cudaErrorInvalidValue;
  const uint64_t total = uint64_t(rows) * cols;
  const int blocks = int(total / 256 + 1 < 65535 * 8 ? total / 256 + 1 : 65535 * 8);
  fp8_block_dequant_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_fp8_e4m3*>(w), static_cast<const float*>(scale), static_cast<__nv_bfloat16*>(out),
      rows, cols);
  return cudaGetLastError();
}
