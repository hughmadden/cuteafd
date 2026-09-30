// DFlash block drafter of MiMo V2.6 Pro (the snapshot's dflash/: five
// Qwen3-style layers, 128 query / 8 KV heads of 128, attention sinks, value
// scale, NeoX RoPE on the first 64 dims, non-causal block attention over a
// 1024-position sliding window of target context). The non-GEMM pieces the
// GLM DFlash2 kernels (glm_dflash.cu) do not cover; GEMMs run through cuBLAS
// and RMSNorm, SwiGLU, taps and the top-16 through the glm_dflash exports.
//
// Statement order and BF16 boundaries follow python/reference/mimo_dflash/
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
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    k[lane + 32 * j] = __float2bfloat16(y[j]);
    v[lane + 32 * j] = __float2bfloat16(f(v_in[lane + 32 * j]) * v_scale);
  }
}

// Block attention with an optional per-head sink logit: every block row sees
// the whole block (non-causal) and the sequence's context entries within the
// window (row r at position end + r sees context positions q with end + r -
// q < window; window 0 = all `ctx` entries). One CTA per (key chunk, kv head x
// 64-query slice, sequence); a second pass merges chunks and adds the sink.
constexpr int kAttnThreads = 256;
constexpr int kChunk = 128;
constexpr int kTile = 32;
constexpr int kQueries = 64;

__global__ void attention_partial_kernel(const bf16* q, const bf16* k_block, const bf16* v_block, const bf16* k_ring,
                                         const bf16* v_ring, const int32_t* seq_slots, const int32_t* ctx_lengths,
                                         const int32_t* ctx_ends, float* partial_o, float* partial_ml, int ring,
                                         int heads, int kv_heads, int block_rows, int chunks, int slices, float scale,
                                         int window) {
  const int chunk = blockIdx.x, kvh = blockIdx.y / slices, slice = blockIdx.y % slices, s = blockIdx.z;
  const int group = heads / kv_heads, queries = group * block_rows;
  const int ctx = ctx_lengths[s], end = ctx_ends[s], total = ctx + block_rows;
  const int first = chunk * kChunk, last = min(total, first + kChunk);
  const int t = threadIdx.x, local = t / 4, sub = t % 4, query = slice * kQueries + local;
  const uint64_t out_base = (((uint64_t(s) * kv_heads + kvh) * slices + slice) * chunks + chunk) * kQueries + local;
  float* o_out = partial_o + out_base * kHeadDim + sub * 32;
  if (first >= last) {
    if (sub == 0) {
      partial_ml[out_base * 2] = -CUDART_INF_F;
      partial_ml[out_base * 2 + 1] = 0;
    }
    for (int d = 0; d < 32; ++d) o_out[d] = 0;
    return;
  }
  __shared__ bf16 qs[kQueries][kHeadDim];
  __shared__ bf16 ks[kTile][kHeadDim + 8];
  __shared__ bf16 vs[kTile][kHeadDim];
  __shared__ float ps[kQueries][kTile + 1];
  // Query `query` is block row query / group, head kvh * group + query % group.
  for (int i = t; i < kQueries * kHeadDim; i += kAttnThreads) {
    const int qi = slice * kQueries + i / kHeadDim, d = i % kHeadDim;
    const int row = qi / group, head = kvh * group + qi % group;
    qs[i / kHeadDim][d] = qi < queries ? q[((uint64_t(s) * block_rows + row) * heads + head) * kHeadDim + d]
                                       : bf16(0.0f);
  }
  const int lowest = window > 0 ? end + min(query / group, block_rows - 1) - window + 1 : INT32_MIN;
  float m = -CUDART_INF_F, l = 0, acc[32];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d] = 0;
  const int slot = seq_slots[s];
  for (int tile = first; tile < last; tile += kTile) {
    const int n = min(kTile, last - tile);
    __syncthreads();
    for (int i = t; i < kTile * (kHeadDim / 8); i += kAttnThreads) {
      const int key = i / (kHeadDim / 8), d = (i % (kHeadDim / 8)) * 8;
      uint4 kv4 = make_uint4(0, 0, 0, 0), vv4 = make_uint4(0, 0, 0, 0);
      if (key < n) {
        const int j = tile + key;
        const bf16 *kp, *vp;
        if (j < ctx) {
          const int position = end - ctx + j;
          const uint64_t at = ((uint64_t(slot) * ring + position % ring) * kv_heads + kvh) * kHeadDim + d;
          kp = k_ring + at;
          vp = v_ring + at;
        } else {
          const uint64_t at = ((uint64_t(s) * block_rows + (j - ctx)) * kv_heads + kvh) * kHeadDim + d;
          kp = k_block + at;
          vp = v_block + at;
        }
        kv4 = *reinterpret_cast<const uint4*>(kp);
        vv4 = *reinterpret_cast<const uint4*>(vp);
      }
      *reinterpret_cast<uint4*>(&ks[key][d]) = kv4;
      *reinterpret_cast<uint4*>(&vs[key][d]) = vv4;
    }
    __syncthreads();
    float score[8];
    float tile_max = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int key = sub * 8 + j;
      float dot = 0;
      for (int d = 0; d < kHeadDim; d += 2) {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qs[local][d]));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ks[key][d]));
        dot = fmaf(a.x, b.x, dot);
        dot = fmaf(a.y, b.y, dot);
      }
      const int j_key = tile + key;
      const bool visible = key < n && (j_key >= ctx || end - ctx + j_key >= lowest);
      score[j] = visible ? dot * scale : -CUDART_INF_F;
      tile_max = fmaxf(tile_max, score[j]);
    }
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 2));
    const float m_new = fmaxf(m, tile_max);
    const float factor = m_new == -CUDART_INF_F ? 1.0f : expf(m - m_new);
    float tile_sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float p = m_new == -CUDART_INF_F ? 0.0f : expf(score[j] - m_new);
      tile_sum += p;
      ps[local][sub * 8 + j] = r(p);
    }
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 2);
    l = l * factor + tile_sum;
    m = m_new;
    __syncwarp();
#pragma unroll
    for (int d = 0; d < 32; ++d) acc[d] *= factor;
    for (int key = 0; key < n; ++key) {
      const float p = ps[local][key];
#pragma unroll
      for (int d = 0; d < 32; d += 2) {
        const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vs[key][sub * 32 + d]));
        acc[d] = fmaf(p, v.x, acc[d]);
        acc[d + 1] = fmaf(p, v.y, acc[d + 1]);
      }
    }
  }
  if (sub == 0) {
    partial_ml[out_base * 2] = m;
    partial_ml[out_base * 2 + 1] = l;
  }
#pragma unroll
  for (int d = 0; d < 32; ++d) o_out[d] = acc[d];
}

// out [rows, heads * 128] BF16 from the chunk partials; the sink logit (when
// given) joins the denominator with no value.
__global__ void attention_merge_kernel(const float* partial_o, const float* partial_ml, const bf16* sinks, bf16* out,
                                       int heads, int kv_heads, int block_rows, int chunks, int slices) {
  const int query = blockIdx.x, kvh = blockIdx.y, s = blockIdx.z, d = threadIdx.x;
  const int group = heads / kv_heads;
  const int slice = query / kQueries, local = query % kQueries;
  const int row = query / group, head = kvh * group + query % group;
  const uint64_t base = ((uint64_t(s) * kv_heads + kvh) * slices + slice) * chunks;
  float m = -CUDART_INF_F;
  for (int c = 0; c < chunks; ++c) m = fmaxf(m, partial_ml[((base + c) * kQueries + local) * 2]);
  const float sink = sinks != nullptr ? f(sinks[head]) : -CUDART_INF_F;
  m = fmaxf(m, sink);
  float l = sinks != nullptr ? expf(sink - m) : 0.0f, o = 0;
  for (int c = 0; c < chunks; ++c) {
    const uint64_t at = (base + c) * kQueries + local;
    const float cm = partial_ml[at * 2];
    if (cm == -CUDART_INF_F) continue;
    const float w = expf(cm - m);
    l += partial_ml[at * 2 + 1] * w;
    o += partial_o[at * kHeadDim + d] * w;
  }
  out[((uint64_t(s) * block_rows + row) * heads + head) * kHeadDim + d] = __float2bfloat16(o / l);
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
  qk_rope_kernel<<<(warps * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(qkv), static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm),
      static_cast<const int64_t*>(positions), static_cast<const int32_t*>(slots), static_cast<bf16*>(q_out),
      static_cast<bf16*>(k_out), static_cast<bf16*>(v_out), rows, heads, kv_heads, rope_dim, theta, eps, v_scale);
  return cudaGetLastError();
}

// Workspace bytes cuteafd_mimo_dflash_attention needs.
extern "C" uint64_t cuteafd_mimo_dflash_attention_workspace(int32_t sequences, int32_t heads, int32_t kv_heads,
                                                             int32_t block_rows, int32_t max_keys) {
  const uint64_t chunks = (uint64_t(max_keys) + kChunk - 1) / kChunk;
  const uint64_t slices = (uint64_t(heads / kv_heads) * block_rows + kQueries - 1) / kQueries;
  return uint64_t(sequences) * kv_heads * slices * chunks * kQueries * (kHeadDim + 2) * sizeof(float);
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
  if (sequences < 1 || kv_heads < 1 || heads % kv_heads || window < 0 || block_rows < 1) return cudaErrorInvalidValue;
  const int queries = heads / kv_heads * block_rows;
  const int slices = (queries + kQueries - 1) / kQueries;
  const int chunks = (max_keys + kChunk - 1) / kChunk;
  auto* o = static_cast<float*>(workspace);
  auto* ml = o + uint64_t(sequences) * kv_heads * slices * chunks * kQueries * kHeadDim;
  auto s = static_cast<cudaStream_t>(stream);
  attention_partial_kernel<<<dim3(chunks, kv_heads * slices, sequences), kAttnThreads, 0, s>>>(
      static_cast<const bf16*>(q), static_cast<const bf16*>(k_block), static_cast<const bf16*>(v_block),
      static_cast<const bf16*>(k_ring), static_cast<const bf16*>(v_ring), static_cast<const int32_t*>(seq_slots),
      static_cast<const int32_t*>(ctx_lengths), static_cast<const int32_t*>(ctx_ends), o, ml, ring, heads, kv_heads,
      block_rows, chunks, slices, scale, window);
  attention_merge_kernel<<<dim3(queries, kv_heads, sequences), kHeadDim, 0, s>>>(
      o, ml, static_cast<const bf16*>(sinks), static_cast<bf16*>(out), heads, kv_heads, block_rows, chunks, slices);
  return cudaGetLastError();
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
