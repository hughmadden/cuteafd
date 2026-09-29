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
// One CTA per 128x128 block: amax over the block, scale = amax / 448 (1 for
// an all-zero block), E4M3 = rn_satfinite(w / scale). Dequantizing gives
// the weight MmaFp8Gemv applies: bf16(float(q) * scale).
__global__ void fp8_block_quant_kernel(const __nv_bfloat16* w, __nv_fp8_e4m3* q, float* scale, int rows, int cols) {
  const int scale_cols = (cols + 127) / 128;
  const int r0 = blockIdx.y * 128, c0 = blockIdx.x * 128;
  float amax = 0.0f;
  for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
    const int r = r0 + i / 128, c = c0 + i % 128;
    if (r < rows && c < cols) amax = fmaxf(amax, fabsf(__bfloat162float(w[uint64_t(r) * cols + c])));
  }
  __shared__ float warp_max[32];
  for (int offset = 16; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    amax = threadIdx.x < blockDim.x / 32 ? warp_max[threadIdx.x] : 0.0f;
    for (int offset = 16; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    if (threadIdx.x == 0) warp_max[0] = amax > 0.0f ? amax / 448.0f : 1.0f;
  }
  __syncthreads();
  const float s = warp_max[0];
  if (threadIdx.x == 0) scale[blockIdx.y * scale_cols + blockIdx.x] = s;
  for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
    const int r = r0 + i / 128, c = c0 + i % 128;
    if (r < rows && c < cols) {
      const uint64_t at = uint64_t(r) * cols + c;
      q[at] = __nv_fp8_e4m3(__bfloat162float(w[at]) / s);
    }
  }
}
}  // namespace

extern "C" int32_t cuteafd_fp8_block_quant(const void* w, void* q, void* scale, int32_t rows, int32_t cols,
                                           void* stream) {
  if (rows < 1 || cols < 1) return cudaErrorInvalidValue;
  const dim3 grid((cols + 127) / 128, (rows + 127) / 128);
  fp8_block_quant_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(w), static_cast<__nv_fp8_e4m3*>(q), static_cast<float*>(scale), rows, cols);
  return cudaGetLastError();
}

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
