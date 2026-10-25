#include "../../../shared/cuda/draft_attention.cuh"
#include "../../../shared/cuda/draft_kv.cuh"
// DFlash block drafter of MiMo V2.6 Pro (the snapshot's dflash/: five
// Qwen3-style layers, 128 query / 8 KV heads of 128, attention sinks, value
// scale, NeoX RoPE on the first 64 dims, non-causal block attention over a
// 1024-position sliding window of target context). The non-GEMM pieces the
// GLM DFlash2 kernels (glm_dflash.cu) do not cover; GEMMs run through cuBLAS
// and RMSNorm, SwiGLU, taps and the top-16 through the glm_dflash exports.
//
// Statement order and BF16 boundaries follow python/reference/families/mimo_v2/mimo_dflash/
// reference.py: per-head RMSNorm rounds the unit-RMS value before the weight;
// RoPE rounds cos/sin, each product and the sum; values are bf16(v * scale);
// attention keeps FP32 scores and BF16 probabilities.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float r(float v) { return __bfloat162float(__float2bfloat16(v)); }

constexpr int kHeadDim = 128;
constexpr int kNormThreads = 512;

// One warp per (row, head) of [q heads | k heads]: per-head RMSNorm, then NeoX
// RoPE on the first `rope_dim` (64 or 128) dims at the row's position; k and
// bf16(v * v_scale) go to row `slots[row]` (row itself when slots is null,
// skipped when negative) of the [slot, kv_heads, 128] caches.
template <bool fp8>
__global__ void qk_rope_kernel(const bf16* qkv, const bf16* q_norm, const bf16* k_norm, const int64_t* positions,
                               const int32_t* slots, bf16* q_out, bf16* k_out, bf16* v_out, int rows, int heads,
                               int kv_heads, int rope_dim, float theta, float eps, float v_scale) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  const int per_row = heads + kv_heads;
  if (warp >= rows * per_row) return;
  const int row = warp / per_row, head = warp % per_row;
  const bool is_q = head < heads;
  const int slot = slots != nullptr ? slots[row] : row;
  if (!is_q && slot < 0) return;
  const uint64_t width = uint64_t(heads + 2 * kv_heads) * kHeadDim;
  const bf16* in = qkv + row * width + uint64_t(head) * kHeadDim;
  const bf16* norm = is_q ? q_norm : k_norm;
  // Lane holds elements lane + 32 * j.
  float x[4], sum = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] = f(in[lane + 32 * j]);
    sum += x[j] * x[j];
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float inv = rsqrtf(sum / kHeadDim + eps);
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = r(r(x[j] * inv) * f(norm[lane + 32 * j]));
  const float position = float(positions[row]);
  const int quarters = rope_dim / 32;  // 2: pairs (j, j ^ 1) in j < 2; 4: pairs (j, j ^ 2)
  float y[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (j >= quarters) {
      y[j] = x[j];
      continue;
    }
    const int e = lane + 32 * j, i = e % (rope_dim / 2);
    const float inv_freq = 1.0f / powf(theta, float(2 * i) / rope_dim);
    const float angle = position * inv_freq;
    const float c = r(cosf(angle)), s = r(sinf(angle));
    const int partner = quarters == 2 ? (j ^ 1) : (j ^ 2);
    const float pair = x[partner];
    y[j] = j < quarters / 2 ? r(r(x[j] * c) - r(pair * s)) : r(r(x[j] * c) + r(pair * s));
  }
  if (is_q) {
    bf16* out = q_out + (uint64_t(row) * heads + head) * kHeadDim;
#pragma unroll
    for (int j = 0; j < 4; ++j) out[lane + 32 * j] = __float2bfloat16(y[j]);
    return;
  }
  const int kv = head - heads;
  bf16* k = k_out + (uint64_t(slot) * kv_heads + kv) * kHeadDim;
  bf16* v = v_out + (uint64_t(slot) * kv_heads + kv) * kHeadDim;
  const bf16* v_in = qkv + row * width + uint64_t(heads + kv_heads + kv) * kHeadDim;
  if constexpr (fp8) {
    float values[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) values[j] = r(f(v_in[lane + 32 * j]) * v_scale);
    cuteafd::draft_kv::store<128>(k_out, slot, kv, kv_heads, y);
    cuteafd::draft_kv::store<128>(v_out, slot, kv, kv_heads, values);
  } else {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    k[lane + 32 * j] = __float2bfloat16(y[j]);
    v[lane + 32 * j] = __float2bfloat16(f(v_in[lane + 32 * j]) * v_scale);
  }
  }
}

// h = bf16(h + delta); n = RMSNorm(h) * w (Qwen3 rounding), one block per row.
__global__ void add_norm_kernel(bf16* h, const bf16* delta, const bf16* w, bf16* n, int width, float eps) {
  __shared__ float partial[kNormThreads / 32];
  const uint64_t row = blockIdx.x;
  float sum = 0;
  for (int i = threadIdx.x; i < width; i += kNormThreads) {
    const float v = r(f(h[row * width + i]) + f(delta[row * width + i]));
    h[row * width + i] = __float2bfloat16(v);
    sum += v * v;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
  __syncthreads();
  float total = 0;
  for (int i = 0; i < kNormThreads / 32; ++i) total += partial[i];
  const float inv = rsqrtf(total / width + eps);
  for (int i = threadIdx.x; i < width; i += kNormThreads)
    n[row * width + i] = __float2bfloat16(r(f(h[row * width + i]) * inv) * f(w[i]));
}

}  // namespace

extern "C" int32_t cuteafd_mimo_dflash_qk_rope(const void* qkv, const void* q_norm, const void* k_norm,
                                               const void* positions, const void* slots, void* q_out, void* k_out,
                                               void* v_out, int32_t rows, int32_t heads, int32_t kv_heads,
                                               int32_t rope_dim, float theta, float eps, float v_scale, void* stream) {
  if (rows < 1 || kv_heads < 1 || heads < 0 || (rope_dim != 64 && rope_dim != 128)) return cudaErrorInvalidValue;
  const int warps = rows * (heads + kv_heads);
  qk_rope_kernel<false><<<(warps * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(qkv), static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm),
      static_cast<const int64_t*>(positions), static_cast<const int32_t*>(slots), static_cast<bf16*>(q_out),
      static_cast<bf16*>(k_out), static_cast<bf16*>(v_out), rows, heads, kv_heads, rope_dim, theta, eps, v_scale);
  return cudaGetLastError();
}

// Workspace bytes cuteafd_mimo_dflash_attention needs.
extern "C" uint64_t cuteafd_mimo_dflash_attention_workspace(int32_t sequences, int32_t heads, int32_t kv_heads,
                                                             int32_t block_rows, int32_t max_keys) {
  if (kv_heads < 1 || heads < 1 || heads % kv_heads || block_rows < 1) return 0;
  const int slices = (heads / kv_heads * block_rows + cuteafd::draft_attention::queries_per_slice - 1)
      / cuteafd::draft_attention::queries_per_slice;
  return cuteafd::draft_attention::workspace(sequences, kv_heads, slices, max_keys, kHeadDim);
}

// out [sequences * block_rows, heads * 128]: softmax(q k^T * scale [; sink]) v over
// each sequence's ring context (positions ctx_ends[s] - ctx_lengths[s] ..
// ctx_ends[s] - 1 at seq_slots[s] * ring + position % ring; with window > 0
// only those within `window` of the block row's position) and its block rows
// (k_block/v_block [rows, kv_heads, 128]). sinks: BF16 [heads] or null.
extern "C" int32_t cuteafd_mimo_dflash_attention(const void* q, const void* k_block, const void* v_block,
                                                 const void* k_ring, const void* v_ring, const void* seq_slots,
                                                 const void* ctx_lengths, const void* ctx_ends, const void* sinks,
                                                 void* out, void* workspace, int32_t sequences, int32_t block_rows,
                                                 int32_t heads, int32_t kv_heads, int32_t ring, int32_t max_keys,
                                                 int32_t window, float scale, void* stream) {
  if (kv_heads < 1 || heads < 1 || heads % kv_heads || block_rows < 1) return cudaErrorInvalidValue;
  const int slices = (heads / kv_heads * block_rows + cuteafd::draft_attention::queries_per_slice - 1)
      / cuteafd::draft_attention::queries_per_slice;
  return cuteafd::draft_attention::launch(q, k_block, v_block, k_ring, v_ring, seq_slots,
      ctx_lengths, ctx_ends, sinks, out, workspace, sequences, block_rows, heads, kv_heads,
      ring, max_keys, window, false, kHeadDim, slices, false, scale, static_cast<cudaStream_t>(stream));
}

// h = h + delta (BF16), n = RMSNorm(h) * w for `rows` rows of `width` (<= 8192 * 4).
extern "C" int32_t cuteafd_mimo_dflash_add_norm(void* h, const void* delta, const void* w, void* n, int32_t rows,
                                                int32_t width, float eps, void* stream) {
  if (rows < 1 || width < 1) return cudaErrorInvalidValue;
  add_norm_kernel<<<rows, kNormThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<bf16*>(h), static_cast<const bf16*>(delta), static_cast<const bf16*>(w), static_cast<bf16*>(n),
      width, eps);
  return cudaGetLastError();
}


extern "C" int32_t cuteafd_mimo_dflash_qk_rope_fp8(const void* qkv, const void* q_norm, const void* k_norm,
                                               const void* positions, const void* slots, void* q_out, void* k_out,
                                               void* v_out, int32_t rows, int32_t heads, int32_t kv_heads,
                                               int32_t rope_dim, float theta, float eps, float v_scale, void* stream) {
  if (rows < 1 || kv_heads < 1 || heads < 0 || (rope_dim != 64 && rope_dim != 128)) return cudaErrorInvalidValue;
  const int warps = rows * (heads + kv_heads);
  qk_rope_kernel<true><<<(warps * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(qkv), static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm),
      static_cast<const int64_t*>(positions), static_cast<const int32_t*>(slots), static_cast<bf16*>(q_out),
      static_cast<bf16*>(k_out), static_cast<bf16*>(v_out), rows, heads, kv_heads, rope_dim, theta, eps, v_scale);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_mimo_dflash_attention_fp8(const void* q, const void* k_block, const void* v_block,
                                                 const void* k_ring, const void* v_ring, const void* seq_slots,
                                                 const void* ctx_lengths, const void* ctx_ends, const void* sinks,
                                                 void* out, void* workspace, int32_t sequences, int32_t block_rows,
                                                 int32_t heads, int32_t kv_heads, int32_t ring, int32_t max_keys,
                                                 int32_t window, float scale, void* stream) {
  if (kv_heads < 1 || heads < 1 || heads % kv_heads || block_rows < 1) return cudaErrorInvalidValue;
  const int slices = (heads / kv_heads * block_rows + cuteafd::draft_attention::queries_per_slice - 1)
      / cuteafd::draft_attention::queries_per_slice;
  return cuteafd::draft_attention::launch(q, k_block, v_block, k_ring, v_ring, seq_slots,
      ctx_lengths, ctx_ends, sinks, out, workspace, sequences, block_rows, heads, kv_heads,
      ring, max_keys, window, false, kHeadDim, slices, true, scale, static_cast<cudaStream_t>(stream));
}
