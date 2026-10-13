#include "draft_attention.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>
#include <mma.h>
#include <algorithm>
#include <limits.h>

namespace cuteafd::draft_attention {
namespace {
using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;
constexpr int tile_keys = 32;

int sm_count() {
  // Cache per calling thread and device, not one card's SM count globally.
  static thread_local int cached_device = -1, cached_count = 0;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  if (device != cached_device) {
    int count = 0;
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
      return 0;
    cached_device = device;
    cached_count = count;
  }
  return cached_count;
}

int partition_capacity(int kv_heads, int slices, int max_keys, int sms) {
  const int tiles = (max_keys + tile_keys - 1) / tile_keys;
  return std::min(tiles, (2 * sms + kv_heads * slices - 1) / (kv_heads * slices));
}

// Query geometry selects two warps for m16 or four for m32. Each query tile
// shares a 32-key tile; PV warps cover disjoint halves of the value dimension.
template <int dim, int query_rows, bool direct>
__global__ void partial_kernel(const bf16* q, const bf16* k_block, const bf16* v_block,
    const bf16* k_ring, const bf16* v_ring, const int32_t* slots, const int32_t* lengths,
    const int32_t* ends, float* partial_o, float* partial_ml, int heads, int kv_heads,
    int block_rows, int ring, int slices, int partitions, int tiles_per_partition,
    int max_keys, int window, bool causal, float scale, const bf16* sinks, bf16* out) {
  constexpr int block_threads = query_rows * 4;
  constexpr int stride = dim + 8;
  constexpr int score_stride = 40;
  const int partition = blockIdx.x, kvh = blockIdx.y / slices, slice = blockIdx.y % slices;
  const int sequence = blockIdx.z, t = threadIdx.x, warp = t / 32;
  const int local = t / 4, sub = t % 4, query = slice * query_rows + local;
  const int group = heads / kv_heads, queries = group * block_rows;
  const int ctx = lengths[sequence], end = ends[sequence], slot = slots[sequence];
  const int first = partition * tiles_per_partition * tile_keys;
  const int last = min(min(ctx + block_rows, max_keys), first + tiles_per_partition * tile_keys);
  const uint64_t base = (((uint64_t(sequence) * kv_heads + kvh) * slices + slice)
      * partitions + partition) * query_rows;
  if (first >= last) {
    if constexpr (!direct) {
      if (sub == 0) {
        partial_ml[(base + local) * 2] = -CUDART_INF_F;
        partial_ml[(base + local) * 2 + 1] = 0;
      }
      for (int i = t; i < query_rows * dim; i += block_threads) partial_o[base * dim + i] = 0;
    }
    return;
  }
  __shared__ __align__(32) bf16 qs[query_rows][stride];
  __shared__ __align__(32) bf16 ks[tile_keys][stride];
  __shared__ __align__(32) bf16 vs[tile_keys][stride];
  __shared__ __align__(32) bf16 probability[query_rows][score_stride];
  // Scores and logical-row accumulator rescaling occupy this arena in turn.
  __shared__ __align__(32) float output[query_rows][dim];
  __shared__ float factors[query_rows];
  for (int i = t; i < query_rows * dim; i += block_threads) {
    const int qi = slice * query_rows + i / dim, d = i % dim;
    const int row = qi / group, head = kvh * group + qi % group;
    qs[i / dim][d] = qi < queries
        ? q[((uint64_t(sequence) * block_rows + row) * heads + head) * dim + d]
        : __float2bfloat16(0);
  }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[dim / 32];
#pragma unroll
  for (int d = 0; d < dim / 32; ++d) wmma::fill_fragment(accum[d], 0.0f);
  float m = -CUDART_INF_F, l = 0;
  const int lowest = window > 0 ? end + min(query / group, block_rows - 1) - window + 1 : INT32_MIN;
  for (int tile = first; tile < last; tile += tile_keys) {
    __syncthreads();
    for (int i = t; i < tile_keys * (dim / 8); i += block_threads) {
      const int key = i / (dim / 8), d = i % (dim / 8) * 8, j = tile + key;
      uint4 k4 = make_uint4(0, 0, 0, 0), v4 = make_uint4(0, 0, 0, 0);
      if (j < last) {
        const bf16 *kp, *vp;
        if (j < ctx) {
          const int position = end - ctx + j;
          const uint64_t at = ((uint64_t(slot) * ring + position % ring) * kv_heads + kvh) * dim + d;
          kp = k_ring + at; vp = v_ring + at;
        } else {
          const uint64_t at = ((uint64_t(sequence) * block_rows + j - ctx) * kv_heads + kvh) * dim + d;
          kp = k_block + at; vp = v_block + at;
        }
        k4 = *reinterpret_cast<const uint4*>(kp);
        v4 = *reinterpret_cast<const uint4*>(vp);
      }
      *reinterpret_cast<uint4*>(&ks[key][d]) = k4;
      *reinterpret_cast<uint4*>(&vs[key][d]) = v4;
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> score;
    wmma::fill_fragment(score, 0.0f);
#pragma unroll
    for (int k = 0; k < dim; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
      wmma::load_matrix_sync(a, &qs[(warp / 2) * 16][k], stride);
      wmma::load_matrix_sync(b, &ks[(warp % 2) * 16][k], stride);
      wmma::mma_sync(score, a, b, score);
    }
    wmma::store_matrix_sync(&output[(warp / 2) * 16][(warp % 2) * 16], score,
        dim, wmma::mem_row_major);
    __syncthreads();
    float logits[8], tile_max = -CUDART_INF_F;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int key = sub * 8 + j, position = tile + key;
      const bool visible = query < queries && position < last &&
          (position < ctx ? end - ctx + position >= lowest : !causal || position - ctx <= query / group);
      logits[j] = visible ? output[local][key] * scale : -CUDART_INF_F;
      tile_max = fmaxf(tile_max, logits[j]);
    }
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));
    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 2));
    const float next_m = fmaxf(m, tile_max);
    const float factor = next_m == -CUDART_INF_F ? 1.0f : expf(m - next_m);
    float tile_sum = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float p = next_m == -CUDART_INF_F ? 0.0f : expf(logits[j] - next_m);
      probability[local][sub * 8 + j] = __float2bfloat16(p);
      tile_sum += p;
    }
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 1);
    tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, 2);
    l = l * factor + tile_sum;
    m = next_m;
    if (sub == 0) factors[local] = factor;
    __syncthreads();
    // Fragment lane/row mapping is opaque. Rescale by logical row in shared
    // storage, then reload, as in the existing V4.1 WMMA attention path.
#pragma unroll
    for (int d = 0; d < dim / 32; ++d)
      wmma::store_matrix_sync(&output[(warp / 2) * 16][(warp % 2) * (dim / 2) + d * 16],
          accum[d], dim, wmma::mem_row_major);
    __syncthreads();
    for (int i = t; i < query_rows * dim; i += block_threads) output[i / dim][i % dim] *= factors[i / dim];
    __syncthreads();
#pragma unroll
    for (int d = 0; d < dim / 32; ++d)
      wmma::load_matrix_sync(accum[d], &output[(warp / 2) * 16][(warp % 2) * (dim / 2) + d * 16],
          dim, wmma::mem_row_major);
#pragma unroll
    for (int k = 0; k < tile_keys; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> p;
      wmma::load_matrix_sync(p, &probability[(warp / 2) * 16][k], score_stride);
#pragma unroll
      for (int d = 0; d < dim / 32; ++d) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> v;
        wmma::load_matrix_sync(v, &vs[k][(warp % 2) * (dim / 2) + d * 16], stride);
        wmma::mma_sync(accum[d], p, v, accum[d]);
      }
    }
  }
#pragma unroll
  for (int d = 0; d < dim / 32; ++d)
    wmma::store_matrix_sync(&output[(warp / 2) * 16][(warp % 2) * (dim / 2) + d * 16],
        accum[d], dim, wmma::mem_row_major);
  __syncthreads();
  if constexpr (direct) {
    if (sub == 0) {
      const int head = kvh * group + query % group;
      const float sink = sinks && query < queries ? __bfloat162float(sinks[head]) : -CUDART_INF_F;
      const float maximum = fmaxf(m, sink), factor = expf(m - maximum);
      factors[local] = factor / (l * factor + (sinks ? expf(sink - maximum) : 0.0f));
    }
    __syncthreads();
    for (int i = t; i < query_rows * dim; i += block_threads) {
      const int qi = slice * query_rows + i / dim, d = i % dim;
      if (qi < queries) {
        const int row = qi / group, head = kvh * group + qi % group;
        out[((uint64_t(sequence) * block_rows + row) * heads + head) * dim + d]
            = __float2bfloat16(output[i / dim][d] * factors[i / dim]);
      }
    }
  } else {
    for (int i = t; i < query_rows * dim; i += block_threads) partial_o[base * dim + i] = output[i / dim][i % dim];
    if (sub == 0) {
      partial_ml[(base + local) * 2] = m;
      partial_ml[(base + local) * 2 + 1] = l;
    }
  }
}

template <int dim, int query_rows>
__global__ void merge_kernel(const float* partial_o, const float* partial_ml, const bf16* sinks,
    bf16* out, int heads, int kv_heads, int block_rows, int slices, int partitions) {
  const int query = blockIdx.x, kvh = blockIdx.y, sequence = blockIdx.z, d = threadIdx.x;
  const int group = heads / kv_heads, row = query / group, head = kvh * group + query % group;
  const int slice = query / query_rows, local = query % query_rows;
  const uint64_t base = ((uint64_t(sequence) * kv_heads + kvh) * slices + slice) * partitions;
  const float sink = sinks ? __bfloat162float(sinks[head]) : -CUDART_INF_F;
  float m = sink;
  for (int p = 0; p < partitions; ++p)
    m = fmaxf(m, partial_ml[((base + p) * query_rows + local) * 2]);
  float l = sinks ? expf(sink - m) : 0, value = 0;
  for (int p = 0; p < partitions; ++p) {
    const uint64_t at = (base + p) * query_rows + local;
    const float pm = partial_ml[at * 2];
    if (pm == -CUDART_INF_F) continue;
    const float factor = expf(pm - m);
    l += partial_ml[at * 2 + 1] * factor;
    value += partial_o[at * dim + d] * factor;
  }
  out[((uint64_t(sequence) * block_rows + row) * heads + head) * dim + d] = __float2bfloat16(value / l);
}

template <int dim, int query_rows, bool direct>
int dispatch(const void* q, const void* k_block, const void* v_block, const void* k_ring,
    const void* v_ring, const void* slots, const void* lengths, const void* ends, const void* sinks,
    void* out, void* scratch, int sequences, int block_rows, int heads, int kv_heads, int ring,
    int max_keys, int window, bool causal, int admitted_slices, int sms, float scale, cudaStream_t stream) {
  const int queries = heads / kv_heads * block_rows;
  const int slices = query_rows == 16 ? 1 : admitted_slices;
  const int tiles = (max_keys + tile_keys - 1) / tile_keys;
  // Compact m16 partials may use more key splits without growing the m32 arena.
  const int capacity = std::min(tiles, partition_capacity(kv_heads, admitted_slices, max_keys, sms)
      * admitted_slices * queries_per_slice / (slices * query_rows));
  const int wave = (2 * sms + sequences * kv_heads * slices - 1) / (sequences * kv_heads * slices);
  const int wanted = direct ? 1 : std::min(capacity, std::max(wave,
      query_rows == 16 ? (tiles + 3) / 4 : 1));
  const int tiles_per_partition = (tiles + wanted - 1) / wanted;
  const int partitions = (tiles + tiles_per_partition - 1) / tiles_per_partition;
  auto* partial_o = static_cast<float*>(scratch);
  auto* partial_ml = partial_o + uint64_t(sequences) * kv_heads * slices * partitions * query_rows * dim;
  partial_kernel<dim, query_rows, direct><<<dim3(partitions, kv_heads * slices, sequences), query_rows * 4, 0, stream>>>(
      static_cast<const bf16*>(q), static_cast<const bf16*>(k_block), static_cast<const bf16*>(v_block),
      static_cast<const bf16*>(k_ring), static_cast<const bf16*>(v_ring), static_cast<const int32_t*>(slots),
      static_cast<const int32_t*>(lengths), static_cast<const int32_t*>(ends), partial_o, partial_ml, heads,
      kv_heads, block_rows, ring, slices, partitions, tiles_per_partition, max_keys, window, causal, scale,
      static_cast<const bf16*>(sinks), static_cast<bf16*>(out));
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  if constexpr (!direct) {
    merge_kernel<dim, query_rows><<<dim3(queries, kv_heads, sequences), dim, 0, stream>>>(partial_o, partial_ml,
        static_cast<const bf16*>(sinks), static_cast<bf16*>(out), heads, kv_heads, block_rows, slices, partitions);
  }
  return cudaGetLastError();
}

template <int dim>
int route(const void* q, const void* k_block, const void* v_block, const void* k_ring,
    const void* v_ring, const void* slots, const void* lengths, const void* ends, const void* sinks,
    void* out, void* scratch, int sequences, int block_rows, int heads, int kv_heads, int ring,
    int max_keys, int window, bool causal, int slices, int sms, float scale, cudaStream_t stream) {
  const bool skinny = heads / kv_heads * block_rows <= 16;
  const bool direct = max_keys <= tile_keys;
#define ROUTE(Q, D) dispatch<dim, Q, D>(q, k_block, v_block, k_ring, v_ring, slots, lengths, ends, sinks, \
    out, scratch, sequences, block_rows, heads, kv_heads, ring, max_keys, window, causal, slices, sms, scale, stream)
  if (skinny) return direct ? ROUTE(16, true) : ROUTE(16, false);
  return direct ? ROUTE(32, true) : ROUTE(32, false);
#undef ROUTE
}
}  // namespace

uint64_t workspace(int sequences, int kv_heads, int slices, int max_keys, int head_dim) {
  if (sequences < 1 || kv_heads < 1 || slices < 1 || max_keys < 1 ||
      (head_dim != 64 && head_dim != 128)) return 0;
  const int sms = sm_count();
  if (sms < 1) return 0;
  const int capacity = partition_capacity(kv_heads, slices, max_keys, sms);
  return uint64_t(sequences) * kv_heads * slices * capacity * queries_per_slice * (head_dim + 2) * sizeof(float);
}

int launch(const void* q, const void* k_block, const void* v_block, const void* k_ring,
    const void* v_ring, const void* slots, const void* lengths, const void* ends, const void* sinks,
    void* out, void* scratch, int sequences, int block_rows, int heads, int kv_heads, int ring,
    int max_keys, int window, bool causal, int head_dim, int slices, float scale, cudaStream_t stream) {
  if (sequences < 1 || block_rows < 1 || heads < 1 || kv_heads < 1 || heads % kv_heads ||
      ring < 1 || max_keys < block_rows || window < 0 || slices < 1 ||
      (heads / kv_heads * block_rows + queries_per_slice - 1) / queries_per_slice > slices ||
      !q || !k_block || !v_block || !k_ring || !v_ring || !slots || !lengths || !ends || !out || !scratch)
    return cudaErrorInvalidValue;
  const int sms = sm_count();
  if (sms < 1) return cudaErrorInvalidDevice;
  if (head_dim == 128) return route<128>(q, k_block, v_block, k_ring, v_ring, slots, lengths, ends,
      sinks, out, scratch, sequences, block_rows, heads, kv_heads, ring, max_keys, window, causal, slices, sms, scale, stream);
  if (head_dim == 64) return route<64>(q, k_block, v_block, k_ring, v_ring, slots, lengths, ends,
      sinks, out, scratch, sequences, block_rows, heads, kv_heads, ring, max_keys, window, causal, slices, sms, scale, stream);
  return cudaErrorInvalidValue;
}
}  // namespace cuteafd::draft_attention
