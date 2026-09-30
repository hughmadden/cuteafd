// FP8 E4M3 weights with FP32 128x128 block scales (GLM's checkpoint format)
// dequantized to BF16 at load, exactly as the reference does:
// bf16(float(w) * scale[row / 128, col / 128]).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
// FP8 scale rules for copies of BF16 weights (per block of values with
// absolute maximum amax): 0 amax / 448; 1 the smallest power of two >=
// amax / 448 (Hugh Madden's glm53f-afd pow2_scale: values with at most E4M3's
// 3 mantissa bits then quantize exactly); 2 whichever of the two leaves the
// smaller squared error over the block. 1 for an all-zero block.
__device__ __forceinline__ float pow2_scale(float amax) {
  const uint32_t b = __float_as_uint(amax);
  int x = int(b >> 23) - 135 + int((b & 0x7FFFFFu) > 0x600000u);
  x = x < -126 ? -126 : (x > 127 ? 127 : x);
  return __uint_as_float(uint32_t(x + 127) << 23);
}
__device__ __forceinline__ float quant_error(float x, float s) {
  const float d = float(__nv_fp8_e4m3(x / s)) * s - x;
  return d * d;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = 16; offset; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

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
__global__ void fp8_block_quant_kernel(const __nv_bfloat16* w, __nv_fp8_e4m3* q, float* scale, int rows, int cols,
                                       int rule) {
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
    if (threadIdx.x == 0) {
      warp_max[0] = amax > 0.0f ? amax / 448.0f : 1.0f;
      warp_max[1] = amax > 0.0f ? pow2_scale(amax) : 1.0f;
    }
  }
  __syncthreads();
  float s = warp_max[rule == 1 ? 1 : 0];
  if (rule == 2) {
    // Squared error of the block under both scales.
    const float sa = warp_max[0], sp = warp_max[1];
    float ea = 0.0f, ep = 0.0f;
    for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
      const int r = r0 + i / 128, c = c0 + i % 128;
      if (r < rows && c < cols) {
        const float x = __bfloat162float(w[uint64_t(r) * cols + c]);
        ea += quant_error(x, sa);
        ep += quant_error(x, sp);
      }
    }
    ea = warp_sum(ea);
    ep = warp_sum(ep);
    __shared__ float errs[2][32];
    __syncthreads();
    if ((threadIdx.x & 31) == 0) {
      errs[0][threadIdx.x >> 5] = ea;
      errs[1][threadIdx.x >> 5] = ep;
    }
    __syncthreads();
    ea = ep = 0.0f;
    for (int i = 0; i < int(blockDim.x / 32); ++i) {
      ea += errs[0][i];
      ep += errs[1][i];
    }
    s = ep < ea ? sp : sa;
  }
  if (threadIdx.x == 0) scale[blockIdx.y * scale_cols + blockIdx.x] = s;
  for (int i = threadIdx.x; i < 128 * 128; i += blockDim.x) {
    const int r = r0 + i / 128, c = c0 + i % 128;
    if (r < rows && c < cols) {
      const uint64_t at = uint64_t(r) * cols + c;
      q[at] = __nv_fp8_e4m3(__bfloat162float(w[at]) / s);
    }
  }
}
// One warp per (row, 128-wide K block): scale = amax / 448 (1 for an
// all-zero block), E4M3 = rn_satfinite(w / scale). The per-row x 128-K layout
// (`[rows, cols / 128]` scales) of the MmaFp8Gemv row_scales programs.
__global__ void fp8_row_quant_kernel(const __nv_bfloat16* w, __nv_fp8_e4m3* q, float* scale, int rows, int cols,
                                     int rule) {
  const int blocks = cols / 128;
  const uint64_t unit = uint64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (unit >= uint64_t(rows) * blocks) return;
  const int lane = threadIdx.x & 31;
  const uint64_t base = (unit / blocks) * uint64_t(cols) + (unit % blocks) * 128 + lane * 4;
  float x[4];
  float amax = 0.0f;
  for (int j = 0; j < 4; ++j) {
    x[j] = __bfloat162float(w[base + j]);
    amax = fmaxf(amax, fabsf(x[j]));
  }
  for (int offset = 16; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  const float sa = amax > 0.0f ? amax / 448.0f : 1.0f, sp = amax > 0.0f ? pow2_scale(amax) : 1.0f;
  float s = rule == 1 ? sp : sa;
  if (rule == 2) {
    float ea = 0.0f, ep = 0.0f;
    for (int j = 0; j < 4; ++j) {
      ea += quant_error(x[j], sa);
      ep += quant_error(x[j], sp);
    }
    s = warp_sum(ep) < warp_sum(ea) ? sp : sa;
  }
  if (lane == 0) scale[unit] = s;
  for (int j = 0; j < 4; ++j) q[base + j] = __nv_fp8_e4m3(x[j] / s);
}
}  // namespace

// `rule`: 0 amax / 448, 1 power of two, 2 the better of the two per block (see pow2_scale).
extern "C" int32_t cuteafd_fp8_row_quant_rule(const void* w, void* q, void* scale, int32_t rows, int32_t cols,
                                              int32_t rule, void* stream) {
  if (rows < 1 || cols < 128 || cols % 128 || rule < 0 || rule > 2) return cudaErrorInvalidValue;
  const uint64_t units = uint64_t(rows) * (cols / 128);
  fp8_row_quant_kernel<<<unsigned((units + 7) / 8), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(w), static_cast<__nv_fp8_e4m3*>(q), static_cast<float*>(scale), rows, cols,
      rule);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_fp8_row_quant(const void* w, void* q, void* scale, int32_t rows, int32_t cols,
                                         void* stream) {
  return cuteafd_fp8_row_quant_rule(w, q, scale, rows, cols, 0, stream);
}

extern "C" int32_t cuteafd_fp8_block_quant_rule(const void* w, void* q, void* scale, int32_t rows, int32_t cols,
                                                int32_t rule, void* stream) {
  if (rows < 1 || cols < 1 || rule < 0 || rule > 2) return cudaErrorInvalidValue;
  const dim3 grid((cols + 127) / 128, (rows + 127) / 128);
  fp8_block_quant_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(w), static_cast<__nv_fp8_e4m3*>(q), static_cast<float*>(scale), rows, cols,
      rule);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_fp8_block_quant(const void* w, void* q, void* scale, int32_t rows, int32_t cols,
                                           void* stream) {
  return cuteafd_fp8_block_quant_rule(w, q, scale, rows, cols, 0, stream);
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
