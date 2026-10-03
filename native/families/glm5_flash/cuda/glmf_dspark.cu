// dSpark drafter for GLM 5.3 Flash (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview):
// the non-GEMM pieces of the context update, the Qwen3 block body (64-wide
// heads, causal inside the block) and the Markov / confidence heads.
// GEMMs run through cuBLAS or the FP8 GEMV; RMSNorm, SiLU-mul and the mHC tap
// mean are the DFlash2 kernels (glm_dflash.cu); the vocabulary head is the
// engine's. python/reference/families/glm5_flash/dspark/reference.py is the oracle.
//
// BF16 boundaries follow the Speculators model in transformers: Qwen3RMSNorm
// rounds the unit-RMS value before the weight, RoPE rounds each product and the
// sum, residual adds round once. Attention keeps FP32 probabilities; the head
// logits and the Markov bias are FP32.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float r(float v) { return __bfloat162float(__float2bfloat16(v)); }

constexpr int kHeadDim = 64;

// One warp per (row, head) of [q heads | kv heads]: per-head RMSNorm, then
// rotate_half RoPE at the row's position (lane holds elements lane and
// lane + 32, a rotation pair); k and v rows go to `slots[row]` (row when
// null, skipped when negative) of the [slot, kv_heads, 64] caches.
__global__ void qk_rope_kernel(const bf16* qkv, const bf16* q_norm, const bf16* k_norm, const int64_t* positions,
                               const int32_t* slots, bf16* q_out, bf16* k_out, bf16* v_out, int rows, int heads,
                               int kv_heads, float theta, float eps) {
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
  float x0 = f(in[lane]), x1 = f(in[lane + 32]);
  float sum = x0 * x0 + x1 * x1;
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  const float inv = rsqrtf(sum / kHeadDim + eps);
  x0 = r(r(x0 * inv) * f(norm[lane]));
  x1 = r(r(x1 * inv) * f(norm[lane + 32]));
  const float inv_freq = 1.0f / powf(theta, float(2 * lane) / kHeadDim);
  const float angle = float(positions[row]) * inv_freq;
  const float c = r(cosf(angle)), s = r(sinf(angle));
  const float y0 = r(r(x0 * c) - r(x1 * s));
  const float y1 = r(r(x1 * c) + r(x0 * s));
  if (is_q) {
    bf16* out = q_out + (uint64_t(row) * heads + head) * kHeadDim;
    out[lane] = __float2bfloat16(y0);
    out[lane + 32] = __float2bfloat16(y1);
    return;
  }
  const int kv = head - heads;
  bf16* k = k_out + (uint64_t(slot) * kv_heads + kv) * kHeadDim;
  bf16* v = v_out + (uint64_t(slot) * kv_heads + kv) * kHeadDim;
  const bf16* v_in = qkv + row * width + uint64_t(heads + kv_heads + kv) * kHeadDim;
  k[lane] = __float2bfloat16(y0);
  k[lane + 32] = __float2bfloat16(y1);
  v[lane] = v_in[lane];
  v[lane + 32] = v_in[lane + 32];
}

// Block attention: every block row sees the sequence's last `ctx[s]` context
// entries (ring slots) and the block rows up to itself (all of them when not
// causal). One CTA per (key chunk, kv head, sequence); one warp per query
// (block row x query head of the kv head's group); a second pass merges the
// chunks. Scores and probabilities stay FP32.
constexpr int kChunk = 128;
constexpr int kTile = 32;
constexpr int kMaxQueries = 32;

__global__ void attention_partial_kernel(const bf16* q, const bf16* k_block, const bf16* v_block, const bf16* k_ring,
                                         const bf16* v_ring, const int32_t* seq_slots, const int32_t* ctx_lengths,
                                         const int32_t* ctx_ends, float* partial_o, float* partial_ml, int ring,
                                         int heads, int kv_heads, int block_rows, int chunks, float scale,
                                         int causal) {
  const int chunk = blockIdx.x, kvh = blockIdx.y, s = blockIdx.z;
  const int group = heads / kv_heads;
  const int query = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = query / group, head = kvh * group + query % group;
  const int ctx = ctx_lengths[s], end = ctx_ends[s], total = ctx + block_rows;
  const int first = chunk * kChunk, last = min(total, first + kChunk);
  const uint64_t out_base = ((uint64_t(s) * kv_heads + kvh) * chunks + chunk) * kMaxQueries + query;
  __shared__ float qs[kMaxQueries][kHeadDim];
  __shared__ __nv_bfloat162 ks[kTile][kHeadDim / 2 + 1];
  __shared__ __nv_bfloat162 vs[kTile][kHeadDim / 2];
  const bf16* q_row = q + ((uint64_t(s) * block_rows + row) * heads + head) * kHeadDim;
  qs[query][lane] = f(q_row[lane]);
  qs[query][lane + 32] = f(q_row[lane + 32]);
  float m = -CUDART_INF_F, l = 0, acc0 = 0, acc1 = 0;
  const int slot = seq_slots[s];
  for (int tile = first; tile < last; tile += kTile) {
    const int n = min(kTile, last - tile);
    __syncthreads();
    for (int i = threadIdx.x; i < kTile * (kHeadDim / 8); i += blockDim.x) {
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
      const __nv_bfloat162* kv2 = reinterpret_cast<const __nv_bfloat162*>(&kv4);
#pragma unroll
      for (int e = 0; e < 4; ++e) ks[key][d / 2 + e] = kv2[e];
      *reinterpret_cast<uint4*>(&vs[key][d / 2]) = vv4;
    }
    __syncthreads();
    // Lane `lane` scores key tile + lane.
    float dot = 0;
#pragma unroll 8
    for (int d = 0; d < kHeadDim / 2; ++d) {
      const float2 b = __bfloat1622float2(ks[lane][d]);
      dot = fmaf(qs[query][2 * d], b.x, dot);
      dot = fmaf(qs[query][2 * d + 1], b.y, dot);
    }
    const int j = tile + lane;
    const bool visible = lane < n && (j < ctx || !causal || j - ctx <= row);
    const float score = visible ? dot * scale : -CUDART_INF_F;
    float tile_max = score;
    for (int o = 16; o; o >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, o));
    const float m_new = fmaxf(m, tile_max);
    const float factor = m_new == -CUDART_INF_F ? 1.0f : expf(m - m_new);
    const float p = m_new == -CUDART_INF_F ? 0.0f : expf(score - m_new);
    float tile_sum = p;
    for (int o = 16; o; o >>= 1) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, o);
    l = l * factor + tile_sum;
    m = m_new;
    acc0 *= factor;
    acc1 *= factor;
    for (int key = 0; key < n; ++key) {
      const float pk = __shfl_sync(0xffffffffu, p, key);
      const float2 v = __bfloat1622float2(vs[key][lane]);
      acc0 = fmaf(pk, v.x, acc0);
      acc1 = fmaf(pk, v.y, acc1);
    }
  }
  if (lane == 0) {
    partial_ml[out_base * 2] = m;
    partial_ml[out_base * 2 + 1] = l;
  }
  partial_o[out_base * kHeadDim + 2 * lane] = acc0;
  partial_o[out_base * kHeadDim + 2 * lane + 1] = acc1;
}

// out [rows, heads * 64] BF16 from the chunk partials.
__global__ void attention_merge_kernel(const float* partial_o, const float* partial_ml, bf16* out, int heads,
                                       int kv_heads, int block_rows, int chunks) {
  const int query = blockIdx.x, kvh = blockIdx.y, s = blockIdx.z, d = threadIdx.x;
  const int group = heads / kv_heads;
  const uint64_t base = (uint64_t(s) * kv_heads + kvh) * chunks;
  float m = -CUDART_INF_F;
  for (int c = 0; c < chunks; ++c) m = fmaxf(m, partial_ml[((base + c) * kMaxQueries + query) * 2]);
  float l = 0, o = 0;
  for (int c = 0; c < chunks; ++c) {
    const uint64_t at = (base + c) * kMaxQueries + query;
    const float cm = partial_ml[at * 2];
    if (cm == -CUDART_INF_F) continue;
    const float w = expf(cm - m);
    l += partial_ml[at * 2 + 1] * w;
    o += partial_o[at * kHeadDim + d] * w;
  }
  const int row = query / group, head = kvh * group + query % group;
  out[((uint64_t(s) * block_rows + row) * heads + head) * kHeadDim + d] = __float2bfloat16(o / l);
}

constexpr int kNormThreads = 512;

template <int THREADS>
__device__ float block_sum(float v, float* shared) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = v;
  __syncthreads();
  float total = 0;
  for (int i = 0; i < THREADS / 32; ++i) total += shared[i];
  __syncthreads();
  return total;
}

// residual_out = bf16(residual + delta); normalized = bf16(bf16(h * rsqrt(mean(h^2) + eps)) * w).
__global__ void add_rmsnorm_kernel(const bf16* residual, const bf16* delta, const bf16* w, bf16* residual_out,
                                   bf16* normalized, int width, float eps) {
  __shared__ float partial[kNormThreads / 32];
  const uint64_t row = blockIdx.x;
  float sum = 0;
  for (int i = threadIdx.x; i < width; i += kNormThreads) {
    const float h = r(f(residual[row * width + i]) + f(delta[row * width + i]));
    residual_out[row * width + i] = __float2bfloat16(h);
    sum += h * h;
  }
  const float inv = rsqrtf(block_sum<kNormThreads>(sum, partial) / width + eps);
  // Each thread rereads the sums it wrote (residual_out may alias residual).
  for (int i = threadIdx.x; i < width; i += kNormThreads)
    normalized[row * width + i] = __float2bfloat16(r(f(residual_out[row * width + i]) * inv) * f(w[i]));
}

// Markov chain step `step` of every sequence: value[v] = logits[s, step, v] +
// w2[v] . w1[prev_s] with prev the anchor at step 0, else the previous draft.
// Exact pruning: |w2[v] . e| <= |w2[v]| |e| (Cauchy-Schwarz), so a row whose
// logit plus that bound (with slack for FP32 rounding) stays below the value
// of the row's base argmax `top` cannot win and its W2 row is never read; the
// rows that can win are scored exactly as an unpruned pass would (same dot,
// same order). A warp reads each surviving W2 row once for every sequence.
// Block-local best per sequence (largest value, lowest index on ties).
constexpr int kMarkovBlocks = 296;
constexpr int kMarkovThreads = 256;
constexpr int kMarkovRank = 256;
constexpr int kMaxSequences = 32;

// Lane's share of w2[v] . e (elements 8 * lane .. 8 * lane + 7), warp-summed.
__device__ __forceinline__ float markov_dot(const bf16* w2, uint32_t v, const float* e, int lane) {
  const uint4 raw = *reinterpret_cast<const uint4*>(w2 + uint64_t(v) * kMarkovRank + 8 * lane);
  const __nv_bfloat162* pair = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float dot = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 x = __bfloat1622float2(pair[k]);
    dot = fmaf(x.x, e[8 * lane + 2 * k], dot);
    dot = fmaf(x.y, e[8 * lane + 2 * k + 1], dot);
  }
  for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
  return dot;
}

__global__ void markov_partial_kernel(const float* logits, const bf16* w1, const bf16* w2, const float* norms,
                                      const uint32_t* top, const uint32_t* anchors, const uint32_t* drafts,
                                      float* partial_value, uint32_t* partial_index, int sequences, int vocab,
                                      int block, int step) {
  __shared__ float embed[kMaxSequences][kMarkovRank];
  __shared__ float floor_value[kMaxSequences], reach[kMaxSequences];
  for (int i = threadIdx.x; i < sequences * kMarkovRank; i += blockDim.x) {
    const int s = i / kMarkovRank, c = i % kMarkovRank;
    const uint32_t prev = step == 0 ? anchors[s] : drafts[s * block + step - 1];
    embed[s][c] = f(w1[uint64_t(prev) * kMarkovRank + c]);
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  // Per sequence: |e| and the value of the base argmax row (scored as below).
  for (int s = warp; s < sequences; s += warps) {
    float sum = 0;
    for (int c = lane; c < kMarkovRank; c += 32) sum += embed[s][c] * embed[s][c];
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const uint64_t row = uint64_t(s) * block + step;
    const uint32_t t = top[row];
    const float value = logits[row * vocab + t] + markov_dot(w2, t, embed[s], lane);
    if (lane == 0) {
      floor_value[s] = value;
      reach[s] = sqrtf(sum) * 1.0001f + 1e-6f;
    }
  }
  __syncthreads();
  float best[kMaxSequences];
  uint32_t best_index[kMaxSequences];
#pragma unroll
  for (int s = 0; s < kMaxSequences; ++s) {
    best[s] = -CUDART_INF_F;
    best_index[s] = UINT32_MAX;
  }
  for (int v = blockIdx.x * warps + warp; v < vocab; v += gridDim.x * warps) {
    const float n = norms[v] * 1.0001f + 1e-6f;
    bool any = false;
#pragma unroll
    for (int s = 0; s < kMaxSequences; ++s) {
      if (s >= sequences) break;
      any |= logits[(uint64_t(s) * block + step) * vocab + v] + n * reach[s] >= floor_value[s];
    }
    if (!any) continue;
#pragma unroll
    for (int s = 0; s < kMaxSequences; ++s) {
      if (s >= sequences) break;
      const float value = logits[(uint64_t(s) * block + step) * vocab + v] + markov_dot(w2, v, embed[s], lane);
      if (value > best[s] || (value == best[s] && uint32_t(v) < best_index[s])) {
        best[s] = value;
        best_index[s] = v;
      }
    }
  }
  __shared__ float values[kMarkovThreads / 32][kMaxSequences];
  __shared__ uint32_t indices[kMarkovThreads / 32][kMaxSequences];
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < kMaxSequences; ++s) {
      values[warp][s] = best[s];
      indices[warp][s] = best_index[s];
    }
  }
  __syncthreads();
  if (threadIdx.x < sequences) {
    const int s = threadIdx.x;
    float b = values[0][s];
    uint32_t bi = indices[0][s];
    for (int i = 1; i < warps; ++i)
      if (values[i][s] > b || (values[i][s] == b && indices[i][s] < bi)) {
        b = values[i][s];
        bi = indices[i][s];
      }
    partial_value[s * gridDim.x + blockIdx.x] = b;
    partial_index[s * gridDim.x + blockIdx.x] = bi;
  }
}

// top[row] = argmax of logits[row] (FP32 [rows, vocab]; lowest index on ties).
__global__ void row_argmax_kernel(const float* logits, uint32_t* top, int vocab) {
  const uint64_t row = blockIdx.x;
  float b = -CUDART_INF_F;
  uint32_t bi = UINT32_MAX;
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    const float x = logits[row * vocab + v];
    if (x > b) {
      b = x;
      bi = v;
    }
  }
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, b, o);
    const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > b || (ov == b && oi < bi)) {
      b = ov;
      bi = oi;
    }
  }
  __shared__ float wv[32];
  __shared__ uint32_t wi[32];
  if (threadIdx.x % 32 == 0) {
    wv[threadIdx.x / 32] = b;
    wi[threadIdx.x / 32] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < int(blockDim.x / 32); ++i)
      if (wv[i] > b || (wv[i] == b && wi[i] < bi)) {
        b = wv[i];
        bi = wi[i];
      }
    top[row] = bi;
  }
}

// norms[v] = |w2[v]| (FP32) of the rank-256 rows.
__global__ void row_norms_kernel(const bf16* w2, float* norms, int vocab) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= vocab) return;
  float sum = 0;
  for (int c = lane; c < kMarkovRank; c += 32) {
    const float x = f(w2[uint64_t(warp) * kMarkovRank + c]);
    sum += x * x;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) norms[warp] = sqrtf(sum);
}

__global__ void markov_final_kernel(const float* partial_value, const uint32_t* partial_index, uint32_t* drafts,
                                    int partials, int block, int step) {
  const int s = blockIdx.x, lane = threadIdx.x;
  float b = -CUDART_INF_F;
  uint32_t bi = UINT32_MAX;
  for (int i = lane; i < partials; i += 32) {
    const float v = partial_value[s * partials + i];
    const uint32_t index = partial_index[s * partials + i];
    if (v > b || (v == b && index < bi)) {
      b = v;
      bi = index;
    }
  }
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, b, o);
    const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > b || (ov == b && oi < bi)) {
      b = ov;
      bi = oi;
    }
  }
  if (lane == 0) drafts[s * block + step] = bi;
}

// confidence[s, k] = sigmoid(w[:hidden] . h[s, k] + w[hidden:] . w1[prev] + bias), prev
// the anchor at k = 0, else draft k - 1; one warp per (sequence, row).
__global__ void confidence_kernel(const bf16* hidden, const bf16* w1, const bf16* weight, const bf16* bias,
                                  const uint32_t* anchors, const uint32_t* drafts, float* out, int rows, int block,
                                  int width, int rank) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= rows) return;
  const int s = warp / block, k = warp % block;
  const uint32_t prev = k == 0 ? anchors[s] : drafts[s * block + k - 1];
  float sum = 0;
  for (int i = lane; i < width; i += 32) sum = fmaf(f(hidden[uint64_t(warp) * width + i]), f(weight[i]), sum);
  for (int i = lane; i < rank; i += 32) sum = fmaf(f(w1[uint64_t(prev) * rank + i]), f(weight[width + i]), sum);
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) out[warp] = 1.0f / (1.0f + expf(-(sum + f(bias[0]))));
}

}  // namespace

// qkv [rows, (heads + 2 kv_heads) * 64]: q heads -> q_out [rows, heads, 64]
// (normed, roped); k (normed, roped) and v -> row slots[row] (or row when
// slots is null; negative slots are skipped) of k_out/v_out [*, kv_heads, 64].
extern "C" int32_t cuteafd_glmf_dspark_qk_rope(const void* qkv, const void* q_norm, const void* k_norm,
                                               const void* positions, const void* slots, void* q_out, void* k_out,
                                               void* v_out, int32_t rows, int32_t heads, int32_t kv_heads,
                                               float theta, float eps, void* stream) {
  if (rows < 1 || kv_heads < 1 || heads < 0) return cudaErrorInvalidValue;
  const int warps = rows * (heads + kv_heads);
  qk_rope_kernel<<<(warps * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(qkv), static_cast<const bf16*>(q_norm), static_cast<const bf16*>(k_norm),
      static_cast<const int64_t*>(positions), static_cast<const int32_t*>(slots), static_cast<bf16*>(q_out),
      static_cast<bf16*>(k_out), static_cast<bf16*>(v_out), rows, heads, kv_heads, theta, eps);
  return cudaGetLastError();
}

// Workspace bytes cuteafd_glmf_dspark_attention needs.
extern "C" uint64_t cuteafd_glmf_dspark_attention_workspace(int32_t sequences, int32_t kv_heads, int32_t max_keys) {
  const uint64_t chunks = (uint64_t(max_keys) + kChunk - 1) / kChunk;
  return uint64_t(sequences) * kv_heads * chunks * kMaxQueries * (kHeadDim + 2) * sizeof(float);
}

// out [sequences * block_rows, heads * 64] = softmax(q k^T * scale) v over each
// sequence's context entries (ring slots seq_slots[s] * ring + position % ring
// for positions ctx_ends[s] - ctx_lengths[s] .. ctx_ends[s] - 1) and its block
// rows (k_block/v_block [rows, kv_heads, 64]; with `causal`, row r sees block
// rows 0..r). heads / kv_heads * block_rows must be at most 32; max_keys
// bounds ctx_lengths + block_rows.
extern "C" int32_t cuteafd_glmf_dspark_attention(const void* q, const void* k_block, const void* v_block,
                                                 const void* k_ring, const void* v_ring, const void* seq_slots,
                                                 const void* ctx_lengths, const void* ctx_ends, void* out,
                                                 void* workspace, int32_t sequences, int32_t block_rows,
                                                 int32_t heads, int32_t kv_heads, int32_t ring, int32_t max_keys,
                                                 int32_t causal, float scale, void* stream) {
  if (sequences < 1 || kv_heads < 1 || heads % kv_heads || heads / kv_heads * block_rows > kMaxQueries)
    return cudaErrorInvalidValue;
  const int queries = heads / kv_heads * block_rows;
  const int chunks = (max_keys + kChunk - 1) / kChunk;
  auto* o = static_cast<float*>(workspace);
  auto* ml = o + uint64_t(sequences) * kv_heads * chunks * kMaxQueries * kHeadDim;
  auto s = static_cast<cudaStream_t>(stream);
  attention_partial_kernel<<<dim3(chunks, kv_heads, sequences), queries * 32, 0, s>>>(
      static_cast<const bf16*>(q), static_cast<const bf16*>(k_block), static_cast<const bf16*>(v_block),
      static_cast<const bf16*>(k_ring), static_cast<const bf16*>(v_ring), static_cast<const int32_t*>(seq_slots),
      static_cast<const int32_t*>(ctx_lengths), static_cast<const int32_t*>(ctx_ends), o, ml, ring, heads, kv_heads,
      block_rows, chunks, scale, causal);
  attention_merge_kernel<<<dim3(queries, kv_heads, sequences), kHeadDim, 0, s>>>(
      o, ml, static_cast<bf16*>(out), heads, kv_heads, block_rows, chunks);
  return cudaGetLastError();
}

// residual_out = residual + delta (may alias residual), normalized = RMSNorm(residual_out) * w.
extern "C" int32_t cuteafd_glmf_dspark_add_rmsnorm(const void* residual, const void* delta, const void* w,
                                                   void* residual_out, void* normalized, int32_t rows, int32_t width,
                                                   float eps, void* stream) {
  if (rows < 1 || width < 1) return cudaErrorInvalidValue;
  add_rmsnorm_kernel<<<rows, kNormThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(residual), static_cast<const bf16*>(delta), static_cast<const bf16*>(w),
      static_cast<bf16*>(residual_out), static_cast<bf16*>(normalized), width, eps);
  return cudaGetLastError();
}

// Workspace bytes cuteafd_glmf_dspark_markov needs for `sequences` x `block` rows.
extern "C" uint64_t cuteafd_glmf_dspark_markov_workspace(int32_t sequences, int32_t block) {
  return uint64_t(sequences) * kMarkovBlocks * 8 + uint64_t(sequences) * block * 4;
}

// norms [vocab] F32 = row norms of the Markov projection w2 (BF16 [vocab, 256]), once at load.
extern "C" int32_t cuteafd_glmf_dspark_markov_norms(const void* w2, void* norms, int32_t vocab, int32_t rank,
                                                    void* stream) {
  if (vocab < 1 || rank != kMarkovRank) return cudaErrorInvalidValue;
  row_norms_kernel<<<(vocab * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(w2), static_cast<float*>(norms), vocab);
  return cudaGetLastError();
}

// drafts [sequences, block] U32: for step k, argmax over the vocabulary of
// logits[s, k] (FP32 [sequences * block, vocab]) + w2 @ w1[prev] (BF16 [vocab, 256])
// with prev = anchors[s] at k = 0, else draft k - 1 (the vanilla Markov head, greedy);
// `norms` from cuteafd_glmf_dspark_markov_norms.
extern "C" int32_t cuteafd_glmf_dspark_markov(const void* logits, const void* w1, const void* w2, const void* norms,
                                              const void* anchors, void* drafts, void* workspace, int32_t sequences,
                                              int32_t block, int32_t vocab, int32_t rank, void* stream) {
  if (sequences < 1 || sequences > kMaxSequences || block < 1 || rank != kMarkovRank) return cudaErrorInvalidValue;
  auto s = static_cast<cudaStream_t>(stream);
  auto* values = static_cast<float*>(workspace);
  auto* indices = reinterpret_cast<uint32_t*>(values + uint64_t(sequences) * kMarkovBlocks);
  auto* top = indices + uint64_t(sequences) * kMarkovBlocks;
  row_argmax_kernel<<<sequences * block, 1024, 0, s>>>(static_cast<const float*>(logits), top, vocab);
  for (int step = 0; step < block; ++step) {
    markov_partial_kernel<<<kMarkovBlocks, kMarkovThreads, 0, s>>>(
        static_cast<const float*>(logits), static_cast<const bf16*>(w1), static_cast<const bf16*>(w2),
        static_cast<const float*>(norms), top, static_cast<const uint32_t*>(anchors),
        static_cast<const uint32_t*>(drafts), values, indices, sequences, vocab, block, step);
    markov_final_kernel<<<sequences, 32, 0, s>>>(values, indices, static_cast<uint32_t*>(drafts), kMarkovBlocks,
                                                 block, step);
  }
  return cudaGetLastError();
}

// confidence [sequences * block] F32 from the final-norm rows `hidden` [rows, width],
// the Markov embedding of each row's previous token and the head (weight
// [width + rank] BF16, bias [1] BF16).
extern "C" int32_t cuteafd_glmf_dspark_confidence(const void* hidden, const void* w1, const void* weight,
                                                  const void* bias, const void* anchors, const void* drafts, void* out,
                                                  int32_t sequences, int32_t block, int32_t width, int32_t rank,
                                                  void* stream) {
  if (sequences < 1 || block < 1 || width < 1 || rank < 1) return cudaErrorInvalidValue;
  const int rows = sequences * block;
  confidence_kernel<<<(rows * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(hidden), static_cast<const bf16*>(w1), static_cast<const bf16*>(weight),
      static_cast<const bf16*>(bias), static_cast<const uint32_t*>(anchors), static_cast<const uint32_t*>(drafts),
      static_cast<float*>(out), rows, block, width, rank);
  return cudaGetLastError();
}
