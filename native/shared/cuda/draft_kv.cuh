#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace cuteafd::draft_kv {

// Qwen's token/head E4M3 record, planar as in Step-5: codes, FP32
// head descales, then zero padding to a 16-byte row boundary.
__host__ __device__ constexpr uint64_t plane_bytes(int heads, int dim) {
  return (uint64_t(heads) * (dim + 4) + 15) / 16 * 16;
}

template <int dim>
__device__ __forceinline__ void store(void* plane, uint64_t row, int head,
                                    int heads, const float (&values)[dim / 32]) {
  const int lane = threadIdx.x % 32;
  float amax = 0;
#pragma unroll
  for (int j = 0; j < dim / 32; ++j) amax = fmaxf(amax, fabsf(values[j]));
#pragma unroll
  for (int offset = 16; offset; offset >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  const float scale = amax > 0 ? __fdiv_rn(amax, 448.0f) : 1.0f;
  const float inverse = __fdiv_rn(1.0f, scale);
  auto* record = static_cast<uint8_t*>(plane) + row * plane_bytes(heads, dim);
  if (lane == 0) reinterpret_cast<float*>(record + heads * dim)[head] = scale;
#pragma unroll
  for (int j = 0; j < dim / 32; ++j) {
    const float x = values[j] * inverse;
    const float x1 = __shfl_xor_sync(0xffffffffu, x, 1);
    const float x2 = __shfl_xor_sync(0xffffffffu, x, 2);
    const float x3 = __shfl_xor_sync(0xffffffffu, x1, 2);
    if (lane % 4 == 0) {
      const auto lo = __nv_cvt_float2_to_fp8x2(make_float2(x, x1), __NV_SATFINITE, __NV_E4M3);
      const auto hi = __nv_cvt_float2_to_fp8x2(make_float2(x2, x3), __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint32_t*>(record + head * dim + lane + j * 32) = uint32_t(lo) | (uint32_t(hi) << 16);
    }
  }
  if (head == 0 && lane < plane_bytes(heads, dim) - uint64_t(heads) * (dim + 4))
    record[heads * (dim + 4) + lane] = 0;
}

template <int dim>
__device__ __forceinline__ uint4 load8(const void* plane, uint64_t row, int head,
                                     int heads, int d) {
  const auto* record = static_cast<const uint8_t*>(plane) + row * plane_bytes(heads, dim);
  const float scale = reinterpret_cast<const float*>(record + heads * dim)[head];
  const uint64_t codes = *reinterpret_cast<const uint64_t*>(record + head * dim + d);
  uint4 result;
  auto* pairs = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __nv_fp8x2_e4m3 packed;
    packed.__x = uint16_t(codes >> (16 * j));
    const float2 values = static_cast<float2>(packed);
    pairs[j] = __floats2bfloat162_rn(values.x * scale, values.y * scale);
  }
  return result;
}
}  // namespace cuteafd::draft_kv
