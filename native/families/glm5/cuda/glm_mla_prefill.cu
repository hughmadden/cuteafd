// Sparse MLA prefill over FP8 latent records on F16 or E4M3 tensor cores (SM120):
// GLM 5.x (656-byte records: 512 E4M3 codes, 4 FP32 group scales, 64 BF16
// RoPE dims; 576-wide absorbed query) and GLM 5.3 Flash (528-byte records,
// no RoPE; 512-wide query).
//
//   out[r, h, :] = softmax_t(scale * q[r, h] . k_t) @ v_t   over the selected
//   records t = indices[r, 0 .. lengths[r]) (negative entries masked), with
//   k_t = (codes_t * group scales_t | rope_t) and v_t = codes_t * scales_t.
//
// One CTA per (row, 32 heads): eight warps, two 16-head groups of four.
// Tiles of 32 records: each thread prefetches its share of tile i + 2's E4M3
// codes and group scales into registers while tile i is computed, and turns
// tile i + 1's (codes times their record's FP32 group scale, rounded once to
// F16's 11-bit significand) into the other half of a double-buffered F16 tile
// that both head groups read with ldmatrix; RoPE dims (GLM 5.x) arrive by
// cp.async beside it. Warp w of a group owns latent channels [128 w, 128 w +
// 128): its QK partial covers those channels (plus 16 of the 64 RoPE dims on
// BF16 tensor cores); the four partials meet in shared memory, where warp w
// sums records [8 w, 8 w + 8), and after one exchange of slice maxima every
// warp applies the same online-softmax step to its slice and stores P (F16)
// once for the group. Each warp then multiplies P by the tile
// (ldmatrix.trans) into its 128 output channels. The query enters the F16
// products scaled per head by a power of two (exact for BF16 values), undone
// in FP32. Accumulation is FP32 throughout; one CTA barrier per tile.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kT = 32;             // records per tile
constexpr int kD = 512;            // latent channels
constexpr int kRope = 64;          // GLM 5.x RoPE dims
constexpr int kHeads = 32;         // heads per CTA
constexpr int kThreads = 256;
constexpr int kTileRowBytes = kD * 2;  // F16 tile row: 64 16-byte units
constexpr int kPartStride = kT + 8;    // FP32 partial scores row (16 heads x 32 records)
constexpr int kPStride = 40;           // F16 P row (32 records + pad: conflict-free ldmatrix)

__device__ __forceinline__ int swz(int token, int unit) { return unit ^ (token & 7); }

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float2 e4m3x2_to_float2(uint32_t two) {
  return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two & 0xFFFFu), __NV_E4M3)));
}

__device__ __forceinline__ float bf16_bits(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }

__device__ __forceinline__ void group_barrier(int group) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(1 + group), "r"(128));
}

struct Smem {
  int32_t index[4][kT];  // selected slots of tiles i .. i + 3 (ring)
  float slice_max[2][4][16];
  float slice_sum[2][4][16];
  __align__(16) __half p[2][16][kPStride];
};

// Dynamic shared memory: the double-buffered F16 scaled-code tile [2][kT][512]
// (swizzled 16-byte units), RoPE rows [2][kT][64] BF16 (GLM 5.x), the small
// exchange buffers, and the FP32 partial scores [2 groups][4 warps][16][kPartStride].
constexpr int kRopeBytes = 2 * kT * kRope * 2;
template <bool kHasRope>
__host__ __device__ constexpr int smem_bytes() {
  return 2 * kT * kTileRowBytes + (kHasRope ? kRopeBytes : 0) + int(sizeof(Smem)) + 2 * 4 * 16 * kPartStride * 4;
}

template <bool kHasRope>
__global__ void __launch_bounds__(kThreads, 1)
mla_prefill_kernel(const bf16* __restrict__ q, const uint8_t* __restrict__ kv, const int32_t* __restrict__ indices,
                   const int32_t* __restrict__ lengths, bf16* __restrict__ out, int heads, int topk, int rec,
                   float scale_log2) {
  constexpr int kQk = kHasRope ? kD + kRope : kD;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* tile = smem;                                        // [2][kT][1024]
  uint8_t* rope = tile + 2 * kT * kTileRowBytes;               // [2][kT][128]
  Smem& s = *reinterpret_cast<Smem*>(rope + (kHasRope ? kRopeBytes : 0));
  float* part = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(&s) + sizeof(Smem));

  const int row = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int group = warp / 4, cw = warp % 4;
  const int g = lane / 4, t4 = lane % 4;
  const int head0 = blockIdx.y * kHeads + group * 16;
  const int length = min(max(lengths[row], 0), topk);
  const int tiles = (length + kT - 1) / kT;
  const int32_t* sel = indices + int64_t(row) * topk;

  // Per-head power-of-two query shift: the head's largest |q| (latent part) lands
  // just under 2^14, so its BF16 values convert to F16 exactly.
  __shared__ float shift[kHeads];
  for (int h = warp; h < kHeads; h += kThreads / 32) {
    const bf16* qh = q + (int64_t(row) * heads + blockIdx.y * kHeads + h) * kQk;
    float m = 0;
    for (int i = lane; i < kD; i += 32) m = fmaxf(m, fabsf(__bfloat162float(qh[i])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) shift[h] = m > 0 ? exp2f(float(13 - ilogbf(m))) : 1.0f;
  }
  __syncthreads();
  const float up_lo = shift[group * 16 + g], up_hi = shift[group * 16 + g + 8];
  // A fragments: 16 heads x this warp's 128 latent channels (8 k16 steps), F16 x shift.
  uint32_t qa[8][4];
  {
    const bf16* q_lo = q + (int64_t(row) * heads + head0 + g) * kQk;
    const bf16* q_hi = q_lo + 8 * kQk;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int c = cw * 128 + ks * 16 + 2 * t4;
      const uint32_t lo0 = *reinterpret_cast<const uint32_t*>(q_lo + c);
      const uint32_t hi0 = *reinterpret_cast<const uint32_t*>(q_hi + c);
      const uint32_t lo8 = *reinterpret_cast<const uint32_t*>(q_lo + c + 8);
      const uint32_t hi8 = *reinterpret_cast<const uint32_t*>(q_hi + c + 8);
      auto conv = [](uint32_t v, float up) {
        return pack_f16(bf16_bits(uint16_t(v & 0xFFFF)) * up, bf16_bits(uint16_t(v >> 16)) * up);
      };
      qa[ks][0] = conv(lo0, up_lo);
      qa[ks][1] = conv(hi0, up_hi);
      qa[ks][2] = conv(lo8, up_lo);
      qa[ks][3] = conv(hi8, up_hi);
    }
  }
  // RoPE A fragment: 16 heads x dims [16 cw, 16 cw + 16), BF16 as stored.
  uint32_t ra[4] = {0, 0, 0, 0};
  if (kHasRope) {
    const bf16* q_lo = q + (int64_t(row) * heads + head0 + g) * kQk + kD + cw * 16 + 2 * t4;
    const bf16* q_hi = q_lo + 8 * kQk;
    ra[0] = *reinterpret_cast<const uint32_t*>(q_lo);
    ra[1] = *reinterpret_cast<const uint32_t*>(q_hi);
    ra[2] = *reinterpret_cast<const uint32_t*>(q_lo + 8);
    ra[3] = *reinterpret_cast<const uint32_t*>(q_hi + 8);
  }
  const float down_lo = 1.0f / up_lo, down_hi = 1.0f / up_hi;

  // Slots of tile i into the index ring (written three tiles ahead of the mask).
  auto index = [&](int i) {
    if (tid < kT) {
      const int e = i * kT + tid;
      s.index[i % 4][tid] = e < length ? sel[e] : -1;
    }
  };
  // This thread's share of a tile: record tid / 8, 16-code units part8 + 8 j
  // (channel group j), and the four group scales.
  const int my_token = tid / 8, part8 = tid % 8;
  uint4 codes[4];
  float scales[4];
  auto load = [&](int i) {
    const int32_t slot = s.index[i % 4][my_token];
    const uint8_t* r = kv + int64_t(slot >= 0 ? slot : 0) * rec;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      codes[j] = slot >= 0 ? __ldg(reinterpret_cast<const uint4*>(r + (part8 + 8 * j) * 16)) : make_uint4(0, 0, 0, 0);
    }
    const float4 sc = slot >= 0 ? __ldg(reinterpret_cast<const float4*>(r + kD)) : make_float4(0, 0, 0, 0);
    scales[0] = sc.x;
    scales[1] = sc.y;
    scales[2] = sc.z;
    scales[3] = sc.w;
  };
  // Tile i's RoPE rows into rope buffer i & 1 (one 16-byte unit per thread).
  auto load_rope = [&](int i) {
    if (kHasRope) {
      const int32_t slot = s.index[i % 4][my_token];
      const uint8_t* r = kv + int64_t(slot >= 0 ? slot : 0) * rec;
      cp_async16(smem_addr(rope + ((i & 1) * kT + my_token) * kRope * 2 + part8 * 16), r + kD + 16 + part8 * 16,
                 slot >= 0);
      cp_async_commit();
    }
  };
  // Codes x group scale -> F16 tile buffer `buffer`.
  auto convert = [&](int buffer) {
    uint8_t* rows = tile + (buffer * kT + my_token) * kTileRowBytes;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int unit = part8 + 8 * j;
      const uint32_t words[4] = {codes[j].x, codes[j].y, codes[j].z, codes[j].w};
      uint32_t h[8];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 a = e4m3x2_to_float2(words[k]), b = e4m3x2_to_float2(words[k] >> 16);
        h[2 * k] = pack_f16(a.x * scales[j], a.y * scales[j]);
        h[2 * k + 1] = pack_f16(b.x * scales[j], b.y * scales[j]);
      }
      *reinterpret_cast<uint4*>(rows + swz(my_token, 2 * unit) * 16) = make_uint4(h[0], h[1], h[2], h[3]);
      *reinterpret_cast<uint4*>(rows + swz(my_token, 2 * unit + 1) * 16) = make_uint4(h[4], h[5], h[6], h[7]);
    }
  };

  float acc[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0;
  float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F, l_lo = 0, l_hi = 0;
  float* my_part = part + ((group * 4 + cw) * 16) * kPartStride;
  const float* group_part = part + (group * 4 * 16) * kPartStride;

  // Prologue: tile 0 converted with its RoPE rows, tile 1 in registers.
  index(0);
  index(1);
  index(2);
  __syncthreads();
  if (tiles > 0) {
    load(0);
    load_rope(0);
    convert(0);
    load(1);
    cp_async_wait<0>();
  }
  __syncthreads();
  for (int i = 0; i < tiles; ++i) {
    const int stage = i & 1;
    // Buffers of tile i - 1 and ring slot (i + 3) % 4 are free past the barrier.
    if (i + 1 < tiles) load_rope(i + 1);
    index(i + 3);
    const uint32_t tile_base = smem_addr(tile + stage * kT * kTileRowBytes);
    // Scores: this warp's 128 channels (times their group scale) + 16 RoPE dims.
    float sn[4][4], sr[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) sn[n][0] = sn[n][1] = sn[n][2] = sn[n][3] = sr[n][0] = sr[n][1] = sr[n][2] = sr[n][3] = 0;
    {
      const int mi = lane / 8, r = lane % 8;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
          const int token = 8 * (j + mi / 2) + r;
          const int unit = cw * 16 + 2 * ks + (mi & 1);
          uint32_t b[4];
          ldsm_x4(b, tile_base + token * kTileRowBytes + swz(token, unit) * 16);
          mma_f16(sn[j], qa[ks], b[0], b[1]);
          mma_f16(sn[j + 1], qa[ks], b[2], b[3]);
        }
      }
      if (kHasRope) {
        const uint32_t rope_base = smem_addr(rope + stage * kT * kRope * 2);
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
          const int token = 8 * (j + mi / 2) + r;
          uint32_t b[4];
          ldsm_x4(b, rope_base + token * kRope * 2 + cw * 32 + (mi & 1) * 16);
          mma_bf16(sr[j], ra, b[0], b[1]);
          mma_bf16(sr[j + 1], ra, b[2], b[3]);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int t0 = 8 * n + 2 * t4;
      *reinterpret_cast<float2*>(my_part + g * kPartStride + t0) =
          make_float2(sn[n][0] * down_lo + sr[n][0], sn[n][1] * down_lo + sr[n][1]);
      *reinterpret_cast<float2*>(my_part + (g + 8) * kPartStride + t0) =
          make_float2(sn[n][2] * down_hi + sr[n][2], sn[n][3] * down_hi + sr[n][3]);
    }
    group_barrier(group);
    // Warp cw's slice: records [8 cw, 8 cw + 8), heads g and g + 8, masked, log2 units.
    const int t0 = 8 * cw + 2 * t4;
    float v[4] = {0, 0, 0, 0};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float2 lo = *reinterpret_cast<const float2*>(group_part + (w * 16 + g) * kPartStride + t0);
      const float2 hi = *reinterpret_cast<const float2*>(group_part + (w * 16 + g + 8) * kPartStride + t0);
      v[0] += lo.x;
      v[1] += lo.y;
      v[2] += hi.x;
      v[3] += hi.y;
    }
    const bool ok0 = s.index[i % 4][t0] >= 0, ok1 = s.index[i % 4][t0 + 1] >= 0;
    v[0] = ok0 ? v[0] * scale_log2 : -CUDART_INF_F;
    v[1] = ok1 ? v[1] * scale_log2 : -CUDART_INF_F;
    v[2] = ok0 ? v[2] * scale_log2 : -CUDART_INF_F;
    v[3] = ok1 ? v[3] * scale_log2 : -CUDART_INF_F;
    float smax_lo = fmaxf(v[0], v[1]), smax_hi = fmaxf(v[2], v[3]);
    for (int o = 1; o <= 2; o <<= 1) {
      smax_lo = fmaxf(smax_lo, __shfl_xor_sync(0xffffffffu, smax_lo, o));
      smax_hi = fmaxf(smax_hi, __shfl_xor_sync(0xffffffffu, smax_hi, o));
    }
    if (t4 == 0) {
      s.slice_max[group][cw][g] = smax_lo;
      s.slice_max[group][cw][g + 8] = smax_hi;
    }
    group_barrier(group);
    float tmax_lo = s.slice_max[group][0][g], tmax_hi = s.slice_max[group][0][g + 8];
#pragma unroll
    for (int w = 1; w < 4; ++w) {
      tmax_lo = fmaxf(tmax_lo, s.slice_max[group][w][g]);
      tmax_hi = fmaxf(tmax_hi, s.slice_max[group][w][g + 8]);
    }
    const float mn_lo = fmaxf(m_lo, tmax_lo), mn_hi = fmaxf(m_hi, tmax_hi);
    const float corr_lo = mn_lo == -CUDART_INF_F ? 1.0f : exp2f(m_lo - mn_lo);
    const float corr_hi = mn_hi == -CUDART_INF_F ? 1.0f : exp2f(m_hi - mn_hi);
    m_lo = mn_lo;
    m_hi = mn_hi;
    const float e0 = mn_lo == -CUDART_INF_F ? 0.0f : exp2f(v[0] - mn_lo);
    const float e1 = mn_lo == -CUDART_INF_F ? 0.0f : exp2f(v[1] - mn_lo);
    const float e2 = mn_hi == -CUDART_INF_F ? 0.0f : exp2f(v[2] - mn_hi);
    const float e3 = mn_hi == -CUDART_INF_F ? 0.0f : exp2f(v[3] - mn_hi);
    // This warp's slice sums (its records only); the four slices meet at the end.
    l_lo = l_lo * corr_lo + e0 + e1;
    l_hi = l_hi * corr_hi + e2 + e3;
    *reinterpret_cast<uint32_t*>(&s.p[group][g][t0]) = pack_f16(e0, e1);
    *reinterpret_cast<uint32_t*>(&s.p[group][g + 8][t0]) = pack_f16(e2, e3);
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      acc[n][0] *= corr_lo;
      acc[n][1] *= corr_lo;
      acc[n][2] *= corr_hi;
      acc[n][3] *= corr_hi;
    }
    group_barrier(group);
    {
      const int mi = lane / 8, r = lane % 8;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        uint32_t pa[4];
        ldsm_x4(pa, smem_addr(&s.p[group][r + 8 * (mi & 1)][16 * kk + 8 * (mi >> 1)]));
        const int token = 16 * kk + 8 * (mi & 1) + r;
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
          const int unit = cw * 16 + 2 * pair + (mi >> 1);
          uint32_t b[4];
          ldsm_x4_trans(b, tile_base + token * kTileRowBytes + swz(token, unit) * 16);
          mma_f16(acc[2 * pair], pa, b[0], b[1]);
          mma_f16(acc[2 * pair + 1], pa, b[2], b[3]);
        }
      }
    }
    // Tile i + 1 (in registers) into the other buffer, whose last readers (tile
    // i - 1) passed the previous barrier; then fetch tile i + 2.
    if (i + 1 < tiles) {
      convert(stage ^ 1);
      if (i + 2 < tiles) load(i + 2);
      cp_async_wait<0>();
    }
    __syncthreads();
  }
  for (int o = 1; o <= 2; o <<= 1) {
    l_lo += __shfl_xor_sync(0xffffffffu, l_lo, o);
    l_hi += __shfl_xor_sync(0xffffffffu, l_hi, o);
  }
  if (t4 == 0) {
    s.slice_sum[group][cw][g] = l_lo;
    s.slice_sum[group][cw][g + 8] = l_hi;
  }
  __syncthreads();
  l_lo = l_hi = 0;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    l_lo += s.slice_sum[group][w][g];
    l_hi += s.slice_sum[group][w][g + 8];
  }
  const float inv_lo = l_lo > 0 ? 1.0f / l_lo : 0.0f, inv_hi = l_hi > 0 ? 1.0f / l_hi : 0.0f;
  bf16* o_lo = out + (int64_t(row) * heads + head0 + g) * kD + cw * 128 + 2 * t4;
  bf16* o_hi = o_lo + 8 * kD;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    *reinterpret_cast<__nv_bfloat162*>(o_lo + 8 * n) = __floats2bfloat162_rn(acc[n][0] * inv_lo, acc[n][1] * inv_lo);
    *reinterpret_cast<__nv_bfloat162*>(o_hi + 8 * n) = __floats2bfloat162_rn(acc[n][2] * inv_hi, acc[n][3] * inv_hi);
  }
}


// ---------------------------------------------------------------------------
// E4M3 tensor-core kernel (kernel 2, two-term P, is the engine default): QK and
// PV on mma.m16n8k32.e4m3 (twice the F16 rate on SM120), the records' codes
// copied as stored (cp.async, no dequantization), RoPE on BF16 tensor cores.
//
// One CTA per (row, 16 * kGroups heads): kGroups head groups of four warps;
// warp w of a group owns latent channels [128 w, 128 w + 128), exactly one
// FP32 scale group of every record. Tiles of 32 records in a ring of three
// stages released per warp through mbarriers (no CTA barrier per tile: the
// head groups drift, one's softmax overlapping another's products): codes
// [32][512] (16-byte units swizzled by token & 7), group scales [32][4], RoPE
// rows [32][64] BF16 (swizzled likewise), and the next tile's slots.
//   QK: the query enters as E4M3 scaled per head by a power of two (its
//   largest latent |q| lands in (224, 448]); kQTerms = 2 adds a second E4M3
//   term for the remainder (q ~ hi + lo, about 7 significant bits). Warp w's
//   partial is scaled in FP32 by each record's group scale w, then takes its
//   16 RoPE dims (BF16 query times the same power of two, exact) on BF16
//   tensor cores; the four partials meet in shared memory and the online
//   softmax runs in FP32, each thread on four records of one head.
//   PV: out[h, c] = sum_t p[h, t] codes[t, c] scale[t, group(c)]: the softmax
//   threads write P times each group's scales as four E4M3 copies (kPTerms =
//   2: hi + lo, about 7 significant bits), each copy scaled by a power of two
//   that only decreases along the row (the largest group scale seen times it
//   stays at most 448); warp w multiplies its copy by the codes, read with
//   ldmatrix.trans and byte-permuted into k32 fragments (the tile's record
//   order inside k is permuted the same way in P), and folds a change of the
//   power of two into its online-softmax correction.
// Accumulation is FP32 throughout. The F16 kernel above stays as kernel 0.
namespace e4m3 {

constexpr int kStages = 3;
constexpr int kStageCodes = kT * kD;        // 16384
constexpr int kStageRope = kT * kRope * 2;  // 4096
constexpr int kStageScales = kT * 16;       // 512
constexpr int kPartBytes = 16 * kT * 4;     // one warp's FP32 partial scores [16][32]

template <bool kHasRope>
__host__ __device__ constexpr int stage_bytes() {
  return kStageCodes + (kHasRope ? kStageRope : 0) + kStageScales;
}

template <int kGroups>
struct Smem {
  uint64_t full[kStages];   // the stage's tile landed (every thread's copies)
  uint64_t empty[kStages];  // every warp is done with the stage's tile
  int32_t slots[4][kT];     // the selected slots of tiles i .. i + 3 (ring; copied with tile i - 1)
  float slice_max[kGroups][2][16];  // per head: the tile maxima of its two record halves
  float slice_sum[kGroups][2][16];
  int32_t qexp[kGroups * 16];
};

template <bool kHasRope, int kGroups>
__host__ __device__ constexpr int smem_bytes() {
  return kStages * stage_bytes<kHasRope>() + kGroups * 4 * kPartBytes + int(sizeof(Smem<kGroups>));
}

// Float offset of partial-score pair (row, pair) in a warp's [16][32] block: pairs XOR-swizzled by
// row so that both the fragment-order stores and the slice reads are conflict-free.
__device__ __forceinline__ int part_at(int row, int pair) { return row * kT + 2 * (pair ^ ((row & 3) << 2)); }

__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_addr(bar)) : "memory");
}
// Arrives on `bar` once this thread's earlier cp.async copies have landed.
__device__ __forceinline__ void cp_async_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_addr(bar)) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait(uint64_t* bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(done)
      : "r"(smem_addr(bar)), "r"(parity)
      : "memory");
  return done;
}
// Backs off between polls so that a waiting warp leaves the issue slots to the warps it waits for.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  while (!mbar_try_wait(bar, parity)) __nanosleep(64);
}

__device__ __forceinline__ void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t to_e4m3x2(float a, float b) {
  return __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}

// E4M3 pair (hi) of (a, b), and with kTerms = 2 the pair of the remainders (lo).
template <int kTerms>
__device__ __forceinline__ void quant2(float a, float b, uint32_t& hi, uint32_t& lo) {
  hi = to_e4m3x2(a, b);
  if (kTerms == 2) {
    // The remainder in F16 (exact for hi; x rounded to 11 bits, well past lo's 4).
    const __half2 x = __floats2half2_rn(a, b);
    const __half2 back(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(hi), __NV_E4M3));
    lo = __nv_cvt_halfraw2_to_fp8x2(__hsub2(x, back), __NV_SATFINITE, __NV_E4M3);
  } else {
    lo = 0;
  }
}

__device__ __forceinline__ float pow2(int e) { return __int_as_float((127 + e) << 23); }

// floor(log2(448 / m)) for finite m > 0 (448 = 1.75 x 2^8), clamped to normal powers of two.
__device__ __forceinline__ int e4m3_exponent(float m) {
  const uint32_t bits = __float_as_uint(m);
  const int e = int(bits >> 23) - 127 + ((bits & 0x7FFFFF) > 0x600000);
  return min(max(8 - e, -120), 120);
}

template <bool kHasRope, int kGroups, int kQTerms, int kPTerms>
__global__ void __launch_bounds__(128 * kGroups, 1)
mla_prefill_kernel(const bf16* __restrict__ q, const uint8_t* __restrict__ kv, const int32_t* __restrict__ indices,
                   const int32_t* __restrict__ lengths, bf16* __restrict__ out, int heads, int topk, int rec,
                   float scale_log2) {
  constexpr int kThreads = 128 * kGroups;
  constexpr int kHeadsCta = 16 * kGroups;
  constexpr int kQk = kHasRope ? kD + kRope : kD;
  constexpr int kStage = stage_bytes<kHasRope>();
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* part_base = smem + kStages * kStage;  // [kGroups][4 warps][16][32] FP32; P copies alias it
  Smem<kGroups>& s = *reinterpret_cast<Smem<kGroups>*>(part_base + kGroups * 4 * kPartBytes);

  const int row = blockIdx.x, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int group = warp / 4, cw = warp % 4;
  const int g = lane / 4, t4 = lane % 4;
  const int mi = lane / 8, r8 = lane % 8;
  const int head0 = blockIdx.y * kHeadsCta + group * 16;
  const int length = min(max(lengths[row], 0), topk);
  const int tiles = (length + kT - 1) / kT;
  const int32_t* sel = indices + int64_t(row) * topk;

  if (tid == 0) {
    for (int k = 0; k < kStages; ++k) {
      mbar_init(&s.full[k], kThreads);
      mbar_init(&s.empty[k], kThreads / 32);
    }
  }
  // Tile i's selected slots (8 units of `indices`) into ring entry i % 4 (positions past the
  // row's length are masked where read).
  auto load_slots = [&](int i) {
    if (tid < kT / 4 && i * kT < topk) {
      cp_async16(smem_addr(&s.slots[i % 4][4 * tid]), sel + i * kT + 4 * tid, true);
    }
  };
  load_slots(0);
  cp_async_commit();
  // Per-head power of two: the head's largest latent |q| lands in (224, 448].
  for (int h = warp; h < kHeadsCta; h += kThreads / 32) {
    const bf16* qh = q + (int64_t(row) * heads + blockIdx.y * kHeadsCta + h) * kQk;
    float m = 0;
    for (int i = lane; i < kD; i += 32) m = fmaxf(m, fabsf(__bfloat162float(qh[i])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) s.qexp[h] = m > 0 ? e4m3_exponent(m) : 0;
  }
  cp_async_wait<0>();
  __syncthreads();

  // Tile i's records into stage i % 3, after its slots landed (with tile i - 1): warp w copies
  // the 32 code units of records w + 16 j (w + 8 j with two head groups), one per lane,
  // swizzled; then the 8 RoPE units (GLM 5.x) and the scale unit of every record, one per
  // thread; then tile i + 1's slots. Masked slots zero-fill. Every thread arrives on the
  // stage's full barrier once its copies land.
  constexpr int kWarps = kThreads / 32, kCodeRounds = kT / kWarps;
  constexpr int kExtra = (kHasRope ? kT * 8 : 0) + kT, kExtraRounds = (kExtra + kThreads - 1) / kThreads;
  const int code_dst = warp * kD + swz(warp, lane) * 16;
  auto slot_at = [&](int i, int t) { return i * kT + t < length ? s.slots[i % 4][t] : -1; };
  auto load = [&](int i, uint8_t* stage) {
#pragma unroll
    for (int j = 0; j < kCodeRounds; ++j) {
      const int32_t slot = slot_at(i, warp + kWarps * j);
      cp_async16(smem_addr(stage + code_dst + j * kWarps * kD), kv + int64_t(max(slot, 0)) * rec + lane * 16,
                 slot >= 0);
    }
#pragma unroll
    for (int j = 0; j < kExtraRounds; ++j) {
      const int u = j * kThreads + tid;
      if (u < kExtra) {
        const bool rope_unit = kHasRope && u < kT * 8;
        const int token = rope_unit ? u / 8 : u - kExtra + kT, unit = u % 8;
        const int32_t slot = slot_at(i, token);
        const int src = rope_unit ? kD + 16 + unit * 16 : kD;
        const int dst = rope_unit ? kStageCodes + token * kRope * 2 + swz(token, unit) * 16
                                  : kStageCodes + (kHasRope ? kStageRope : 0) + token * 16;
        cp_async16(smem_addr(stage + dst), kv + int64_t(max(slot, 0)) * rec + src, slot >= 0);
      }
    }
    load_slots(i + 1);
  };
  // Softmax ownership: warp w's thread (g, t4) takes head hs = g + 8 (w & 1) and the tile's records
  // t0 + {0, 1, 8, 9} (t0 = 16 (w >> 1) + 2 t4): exactly the four E4M3 P bytes that thread (g, t4)
  // of every warp holds as A register w (a0..a3) for the PV product.
  const int hs = g + 8 * (cw & 1), t0 = 16 * (cw >> 1) + 2 * t4;
  if (tiles > 0) {
    load(0, smem);
    cp_async_arrive(&s.full[0]);
  }

  const float qd_lo = pow2(-s.qexp[group * 16 + g]), qd_hi = pow2(-s.qexp[group * 16 + g + 8]);
  // A fragments: 16 heads x this warp's 128 latent channels (4 k32 steps), E4M3 x 2^qexp.
  uint32_t qa[kQTerms][4][4];
  {
    const float up_lo = 1.0f / qd_lo, up_hi = 1.0f / qd_hi;
    const bf16* q_lo = q + (int64_t(row) * heads + head0 + g) * kQk;
    const bf16* q_hi = q_lo + 8 * kQk;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const int c = cw * 128 + ks * 32 + 4 * t4 + (f / 2) * 16;
        const uint2 v = *reinterpret_cast<const uint2*>((f & 1 ? q_hi : q_lo) + c);
        const float up = f & 1 ? up_hi : up_lo;
        uint32_t h0, l0, h1, l1;
        quant2<kQTerms>(bf16_bits(uint16_t(v.x & 0xFFFF)) * up, bf16_bits(uint16_t(v.x >> 16)) * up, h0, l0);
        quant2<kQTerms>(bf16_bits(uint16_t(v.y & 0xFFFF)) * up, bf16_bits(uint16_t(v.y >> 16)) * up, h1, l1);
        qa[0][ks][f] = h0 | (h1 << 16);
        if (kQTerms == 2) qa[kQTerms - 1][ks][f] = l0 | (l1 << 16);
      }
    }
  }
  // RoPE A fragment: 16 heads x dims [16 cw, 16 cw + 16), BF16 x 2^qexp (exact), so that the
  // partial scores stay in the head's units until the softmax.
  uint32_t ra[4] = {0, 0, 0, 0};
  if (kHasRope) {
    const bf16* q_lo = q + (int64_t(row) * heads + head0 + g) * kQk + kD + cw * 16 + 2 * t4;
    const bf16* q_hi = q_lo + 8 * kQk;
    const __nv_bfloat162 up_lo = __float2bfloat162_rn(1.0f / qd_lo), up_hi = __float2bfloat162_rn(1.0f / qd_hi);
    auto up = [](uint32_t v, __nv_bfloat162 f) {
      const __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&v), f);
      return *reinterpret_cast<const uint32_t*>(&r);
    };
    ra[0] = up(*reinterpret_cast<const uint32_t*>(q_lo), up_lo);
    ra[1] = up(*reinterpret_cast<const uint32_t*>(q_hi), up_hi);
    ra[2] = up(*reinterpret_cast<const uint32_t*>(q_lo + 8), up_lo);
    ra[3] = up(*reinterpret_cast<const uint32_t*>(q_hi + 8), up_hi);
  }
  const float sl_lo = qd_lo * scale_log2, sl_hi = qd_hi * scale_log2;

  float acc[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0;
  float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F, l_own = 0;
  // Powers of two of the four groups' P copies: each only ever decreases (to the largest that keeps
  // the tile's largest scale times it at most 448), so warp w's acc, held in units of 2^pexp[w],
  // rarely needs rescaling for it. Identical in every warp.
  int pexp[4] = {120, 120, 120, 120};
  uint8_t* group_bytes = part_base + group * 4 * kPartBytes;
  float* my_part = reinterpret_cast<float*>(group_bytes + cw * kPartBytes);
  const float* group_part = reinterpret_cast<const float*>(group_bytes);
  // No CTA barrier per tile: the four head groups drift apart (one's softmax overlaps another's
  // MMAs), bounded by the stage ring: tile i + 1 is copied into the stage of tile i - 2.
  // P copies (alias the group's partials once they are summed): [4 scale groups][kPTerms][8 g][4
  // t4][4 registers a0..a3 x 4 bytes]: one 16-byte A fragment per PV thread.
  uint8_t* pbuf = group_bytes;

  int stage_index = 0, phase = 0;  // tile i's stage (i % 3) and its full-barrier parity
  for (int i = 0; i < tiles; ++i) {
    mbar_wait(&s.full[stage_index], phase);
    const int next_index = stage_index == kStages - 1 ? 0 : stage_index + 1;
    if (i + 1 < tiles) {
      // Tile i + 1 into the stage of tile i - 2 (its slots landed with tile i).
      if (i + 1 >= kStages) mbar_wait(&s.empty[next_index], phase ^ (next_index != 0));
      load(i + 1, smem + next_index * kStage);
      cp_async_arrive(&s.full[next_index]);
    }
    int valid = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) valid |= int(slot_at(i, t0 + (k & 1) + 8 * (k >> 1)) >= 0) << k;
    const uint8_t* stage = smem + stage_index * kStage;
    const uint32_t codes = smem_addr(stage);
    const float* scales = reinterpret_cast<const float*>(stage + kStageCodes + (kHasRope ? kStageRope : 0));

    // Scores: this warp's 128 channels times their group scale, plus 16 RoPE dims.
    float sn[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n) sn[n][0] = sn[n][1] = sn[n][2] = sn[n][3] = 0;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        const int token = 8 * (j + mi / 2) + r8;
        const int unit = cw * 8 + 2 * ks + (mi & 1);
        uint32_t b[4];
        ldsm_x4(b, codes + token * kD + swz(token, unit) * 16);
#pragma unroll
        for (int qt = 0; qt < kQTerms; ++qt) {
          mma_e4m3(sn[j], qa[qt][ks], b[0], b[1]);
          mma_e4m3(sn[j + 1], qa[qt][ks], b[2], b[3]);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int t0 = 8 * n + 2 * t4;
      const float s0 = scales[t0 * 4 + cw], s1 = scales[(t0 + 1) * 4 + cw];
      sn[n][0] *= s0;
      sn[n][1] *= s1;
      sn[n][2] *= s0;
      sn[n][3] *= s1;
    }
    if (kHasRope) {
      const uint32_t rope = codes + kStageCodes;
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        const int token = 8 * (j + mi / 2) + r8;
        uint32_t b[4];
        ldsm_x4(b, rope + token * kRope * 2 + swz(token, 2 * cw + (mi & 1)) * 16);
        mma_bf16(sn[j], ra, b[0], b[1]);
        mma_bf16(sn[j + 1], ra, b[2], b[3]);
      }
    }
    // The group's P copies of tile i - 1 (aliasing the partials) are read.
    group_barrier(group);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      *reinterpret_cast<float2*>(my_part + part_at(g, 4 * n + t4)) = make_float2(sn[n][0], sn[n][1]);
      *reinterpret_cast<float2*>(my_part + part_at(g + 8, 4 * n + t4)) = make_float2(sn[n][2], sn[n][3]);
    }
    // The tile's largest scale per group -> the P copies' powers of two.
    // (Scales are non-negative: their bits order as unsigned integers.)
    int acc_shift = 0;  // pexp[cw] - its value for the previous tiles
    {
      const uint4 sc = *reinterpret_cast<const uint4*>(scales + lane * 4);
      const uint32_t mx[4] = {__reduce_max_sync(0xffffffffu, sc.x), __reduce_max_sync(0xffffffffu, sc.y),
                              __reduce_max_sync(0xffffffffu, sc.z), __reduce_max_sync(0xffffffffu, sc.w)};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int e = mx[k] ? min(pexp[k], e4m3_exponent(__uint_as_float(mx[k]))) : pexp[k];
        acc_shift = k == cw ? e - pexp[k] : acc_shift;
        pexp[k] = e;
      }
    }
    group_barrier(group);
    // This thread's four scores (head hs, records t0 + {0, 1, 8, 9}), masked, log2 units.
    float v[4] = {0, 0, 0, 0};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float2 x = *reinterpret_cast<const float2*>(group_part + w * 16 * kT + part_at(hs, t0 / 2));
      const float2 y = *reinterpret_cast<const float2*>(group_part + w * 16 * kT + part_at(hs, t0 / 2 + 4));
      v[0] += x.x;
      v[1] += x.y;
      v[2] += y.x;
      v[3] += y.y;
    }
    const float sl = cw & 1 ? sl_hi : sl_lo;
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = valid >> k & 1 ? v[k] * sl : -CUDART_INF_F;
    float smax = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
    for (int o = 1; o <= 2; o <<= 1) smax = fmaxf(smax, __shfl_xor_sync(0xffffffffu, smax, o));
    if (t4 == 0) s.slice_max[group][cw >> 1][hs] = smax;
    group_barrier(group);
    const float tmax_lo = fmaxf(s.slice_max[group][0][g], s.slice_max[group][1][g]);
    const float tmax_hi = fmaxf(s.slice_max[group][0][g + 8], s.slice_max[group][1][g + 8]);
    const float mn_lo = fmaxf(m_lo, tmax_lo), mn_hi = fmaxf(m_hi, tmax_hi);
    float corr_lo = mn_lo == -CUDART_INF_F ? 1.0f : exp2f(m_lo - mn_lo);
    float corr_hi = mn_hi == -CUDART_INF_F ? 1.0f : exp2f(m_hi - mn_hi);
    m_lo = mn_lo;
    m_hi = mn_hi;
    const float mn = cw & 1 ? mn_hi : mn_lo;
    float e[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) e[k] = mn == -CUDART_INF_F ? 0.0f : exp2f(v[k] - mn);
    l_own = l_own * (cw & 1 ? corr_hi : corr_lo) + (e[0] + e[1]) + (e[2] + e[3]);
    // P times each group's scales (and its power of two) as E4M3 copies: register cw of the
    // A fragments.
    {
      const float4 sa = *reinterpret_cast<const float4*>(scales + t0 * 4);
      const float4 sb = *reinterpret_cast<const float4*>(scales + (t0 + 1) * 4);
      const float4 sc = *reinterpret_cast<const float4*>(scales + (t0 + 8) * 4);
      const float4 sd = *reinterpret_cast<const float4*>(scales + (t0 + 9) * 4);
      const float s0[4] = {sa.x, sa.y, sa.z, sa.w}, s1[4] = {sb.x, sb.y, sb.z, sb.w};
      const float s2[4] = {sc.x, sc.y, sc.z, sc.w}, s3[4] = {sd.x, sd.y, sd.z, sd.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float f = pow2(pexp[k]);
        uint32_t h01, l01, h89, l89;
        quant2<kPTerms>(e[0] * (s0[k] * f), e[1] * (s1[k] * f), h01, l01);
        quant2<kPTerms>(e[2] * (s2[k] * f), e[3] * (s3[k] * f), h89, l89);
        uint8_t* p = pbuf + k * kPTerms * 512 + (g * 4 + t4) * 16 + 4 * cw;
        *reinterpret_cast<uint32_t*>(p) = h01 | h89 << 16;
        if (kPTerms == 2) *reinterpret_cast<uint32_t*>(p + 512) = l01 | l89 << 16;
      }
    }
    if (acc_shift) {
      const float f = pow2(acc_shift);
      corr_lo *= f;
      corr_hi *= f;
    }
    if (!__all_sync(0xffffffffu, corr_lo == 1.0f && corr_hi == 1.0f)) {
#pragma unroll
      for (int n = 0; n < 16; ++n) {
        acc[n][0] *= corr_lo;
        acc[n][1] *= corr_lo;
        acc[n][2] *= corr_hi;
        acc[n][3] *= corr_hi;
      }
    }
    group_barrier(group);
    {
      uint32_t pa[kPTerms][4];
#pragma unroll
      for (int pt = 0; pt < kPTerms; ++pt) {
        const uint4 a = *reinterpret_cast<const uint4*>(pbuf + (cw * kPTerms + pt) * 512 + (g * 4 + t4) * 16);
        pa[pt][0] = a.x;
        pa[pt][1] = a.y;
        pa[pt][2] = a.z;
        pa[pt][3] = a.w;
      }
      const int token = 8 * mi + r8;
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int unit = cw * 8 + u;
        uint32_t b[4];
        ldsm_x4_trans(b, codes + token * kD + swz(token, unit) * 16);
        // b[m]: records 8 m + 2 t4, + 1 x channels 2 g, 2 g + 1 of the unit.
        const uint32_t even0 = prmt(b[0], b[1], 0x6420), even1 = prmt(b[2], b[3], 0x6420);
        const uint32_t odd0 = prmt(b[0], b[1], 0x7531), odd1 = prmt(b[2], b[3], 0x7531);
#pragma unroll
        for (int pt = 0; pt < kPTerms; ++pt) {
          mma_e4m3(acc[2 * u], pa[pt], even0, even1);
          mma_e4m3(acc[2 * u + 1], pa[pt], odd0, odd1);
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&s.empty[stage_index]);
    phase ^= next_index == 0;
    stage_index = next_index;
  }
  for (int o = 1; o <= 2; o <<= 1) l_own += __shfl_xor_sync(0xffffffffu, l_own, o);
  if (t4 == 0) s.slice_sum[group][cw >> 1][hs] = l_own;
  group_barrier(group);
  const float l_lo = s.slice_sum[group][0][g] + s.slice_sum[group][1][g];
  const float l_hi = s.slice_sum[group][0][g + 8] + s.slice_sum[group][1][g + 8];
  const int acc_exp = cw == 0 ? pexp[0] : cw == 1 ? pexp[1] : cw == 2 ? pexp[2] : pexp[3];
  const float down = pow2(-acc_exp);
  const float inv_lo = l_lo > 0 ? down / l_lo : 0.0f, inv_hi = l_hi > 0 ? down / l_hi : 0.0f;
  // acc[2 u] holds channels 16 u' + 4 t4 + {0, 2}, acc[2 u + 1] channels + {1, 3} (u' = 8 cw + u).
  bf16* o_lo = out + (int64_t(row) * heads + head0 + g) * kD + cw * 128 + 4 * t4;
  bf16* o_hi = o_lo + 8 * kD;
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(acc[2 * u][0] * inv_lo, acc[2 * u + 1][0] * inv_lo);
    const __nv_bfloat162 b = __floats2bfloat162_rn(acc[2 * u][1] * inv_lo, acc[2 * u + 1][1] * inv_lo);
    const __nv_bfloat162 c = __floats2bfloat162_rn(acc[2 * u][2] * inv_hi, acc[2 * u + 1][2] * inv_hi);
    const __nv_bfloat162 d = __floats2bfloat162_rn(acc[2 * u][3] * inv_hi, acc[2 * u + 1][3] * inv_hi);
    *reinterpret_cast<uint2*>(o_lo + 16 * u) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
    *reinterpret_cast<uint2*>(o_hi + 16 * u) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&c), *reinterpret_cast<const uint32_t*>(&d));
  }
}

}  // namespace e4m3

template <bool kHasRope, int kGroups, int kQTerms, int kPTerms>
int launch_e4m3(const void* q, const void* kv, const void* indices, const void* lengths, void* out, int rows,
                int heads, int topk, int record_bytes, float scale_log2, cudaStream_t s) {
  auto kernel = e4m3::mla_prefill_kernel<kHasRope, kGroups, kQTerms, kPTerms>;
  constexpr int bytes = e4m3::smem_bytes<kHasRope, kGroups>();
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  kernel<<<dim3(rows, heads / (16 * kGroups)), 128 * kGroups, bytes, s>>>(
      static_cast<const bf16*>(q), static_cast<const uint8_t*>(kv), static_cast<const int32_t*>(indices),
      static_cast<const int32_t*>(lengths), static_cast<bf16*>(out), heads, topk, record_bytes, scale_log2);
  return cudaGetLastError();
}

template <bool kHasRope, int kQTerms, int kPTerms>
int launch_e4m3_heads(const void* q, const void* kv, const void* indices, const void* lengths, void* out, int rows,
                      int heads, int topk, int record_bytes, float scale_log2, cudaStream_t s) {
  return heads % 64 == 0
             ? launch_e4m3<kHasRope, 4, kQTerms, kPTerms>(q, kv, indices, lengths, out, rows, heads, topk,
                                                          record_bytes, scale_log2, s)
             : launch_e4m3<kHasRope, 2, kQTerms, kPTerms>(q, kv, indices, lengths, out, rows, heads, topk,
                                                          record_bytes, scale_log2, s);
}

template <int kQTerms, int kPTerms>
int launch_e4m3_records(const void* q, const void* kv, const void* indices, const void* lengths, void* out, int rows,
                        int heads, int topk, int record_bytes, float scale_log2, cudaStream_t s) {
  return record_bytes == 656
             ? launch_e4m3_heads<true, kQTerms, kPTerms>(q, kv, indices, lengths, out, rows, heads, topk, record_bytes,
                                                          scale_log2, s)
             : launch_e4m3_heads<false, kQTerms, kPTerms>(q, kv, indices, lengths, out, rows, heads, topk,
                                                           record_bytes, scale_log2, s);
}

}  // namespace

// out [rows, heads, 512] BF16 = sparse MLA of q [rows, heads, 576 or 512] BF16
// over the records kv (slot * record_bytes; 656 with RoPE, 528 without) at
// indices [rows, topk] (the first lengths[r] entries; negative ones masked).
// scale_log2 = softmax scale * log2(e). heads must be a multiple of 32.
// kernel: 0 the F16 kernel; E4M3 kernels 1 (one-term query and P), 2 (two-term
// P), 3 (two-term query), 4 (two-term query and P).
extern "C" int32_t cuteafd_glm_mla_prefill(const void* q, const void* kv, const void* indices, const void* lengths,
                                           void* out, int32_t rows, int32_t heads, int32_t topk,
                                           int32_t record_bytes, float scale_log2, int32_t kernel, void* stream) {
  if (rows < 1 || heads % kHeads || topk < 1 || (record_bytes != 656 && record_bytes != 528) || kernel < 0 ||
      kernel > 4) {
    return cudaErrorInvalidValue;
  }
  auto s = static_cast<cudaStream_t>(stream);
  switch (kernel) {
    case 1:
      return launch_e4m3_records<1, 1>(q, kv, indices, lengths, out, rows, heads, topk, record_bytes, scale_log2, s);
    case 2:
      return launch_e4m3_records<1, 2>(q, kv, indices, lengths, out, rows, heads, topk, record_bytes, scale_log2, s);
    case 3:
      return launch_e4m3_records<2, 1>(q, kv, indices, lengths, out, rows, heads, topk, record_bytes, scale_log2, s);
    case 4:
      return launch_e4m3_records<2, 2>(q, kv, indices, lengths, out, rows, heads, topk, record_bytes, scale_log2, s);
    default:
      break;
  }
  const bool rope = record_bytes == 656;
  const int bytes = rope ? smem_bytes<true>() : smem_bytes<false>();
  const dim3 grid(rows, heads / kHeads);
  if (rope) {
    cudaFuncSetAttribute(mla_prefill_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    mla_prefill_kernel<true><<<grid, kThreads, bytes, s>>>(
        static_cast<const bf16*>(q), static_cast<const uint8_t*>(kv), static_cast<const int32_t*>(indices),
        static_cast<const int32_t*>(lengths), static_cast<bf16*>(out), heads, topk, record_bytes, scale_log2);
  } else {
    cudaFuncSetAttribute(mla_prefill_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    mla_prefill_kernel<false><<<grid, kThreads, bytes, s>>>(
        static_cast<const bf16*>(q), static_cast<const uint8_t*>(kv), static_cast<const int32_t*>(indices),
        static_cast<const int32_t*>(lengths), static_cast<bf16*>(out), heads, topk, record_bytes, scale_log2);
  }
  return cudaGetLastError();
}

namespace {

__global__ void diff_stats(const bf16* __restrict__ a, const bf16* __restrict__ ref, int64_t n, double* stats) {
  double d2 = 0, r2 = 0, dmax = 0;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    const double x = __bfloat162float(a[i]), y = __bfloat162float(ref[i]);
    d2 += (x - y) * (x - y);
    r2 += y * y;
    dmax = fmax(dmax, fabs(x - y));
  }
  for (int o = 16; o; o >>= 1) {
    d2 += __shfl_xor_sync(0xffffffffu, d2, o);
    r2 += __shfl_xor_sync(0xffffffffu, r2, o);
    dmax = fmax(dmax, __shfl_xor_sync(0xffffffffu, dmax, o));
  }
  if (threadIdx.x % 32 == 0) {
    atomicAdd(&stats[0], d2);
    atomicAdd(&stats[1], r2);
    // Non-negative doubles order as their bit patterns.
    atomicMax(reinterpret_cast<unsigned long long*>(&stats[2]), __double_as_longlong(dmax));
  }
}

}  // namespace

// Diagnostics (CUTEAFD_MLA_PREFILL_CHECK): runs every kernel on the same inputs and writes, per
// kernel k = 0..4, stats[3 k ..] = {relative L2, max abs difference, relative L2 of the output
// itself (1)} against kernel 4 (two-term E4M3: the closest to exact). Synchronizes the stream;
// allocates its own buffers.
extern "C" int32_t cuteafd_glm_mla_prefill_check(const void* q, const void* kv, const void* indices,
                                                 const void* lengths, int32_t rows, int32_t heads, int32_t topk,
                                                 int32_t record_bytes, float scale_log2, double* stats,
                                                 void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  const int64_t n = int64_t(rows) * heads * kD;
  void *ref = nullptr, *out = nullptr;
  double* dev = nullptr;
  cudaError_t e = cudaMalloc(&ref, n * 2);
  if (e == cudaSuccess) e = cudaMalloc(&out, n * 2);
  if (e == cudaSuccess) e = cudaMalloc(&dev, 3 * sizeof(double));
  if (e == cudaSuccess) {
    e = cudaError_t(cuteafd_glm_mla_prefill(q, kv, indices, lengths, ref, rows, heads, topk, record_bytes, scale_log2,
                                            4, stream));
  }
  for (int k = 0; k < 5 && e == cudaSuccess; ++k) {
    e = cudaError_t(cuteafd_glm_mla_prefill(q, kv, indices, lengths, out, rows, heads, topk, record_bytes, scale_log2,
                                            k, stream));
    double h[3] = {0, 0, 0};
    if (e == cudaSuccess) e = cudaMemsetAsync(dev, 0, 3 * sizeof(double), s);
    if (e == cudaSuccess) {
      diff_stats<<<1024, 256, 0, s>>>(static_cast<const bf16*>(out), static_cast<const bf16*>(ref), n, dev);
      e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemcpyAsync(h, dev, sizeof(h), cudaMemcpyDeviceToHost, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    stats[3 * k] = h[1] > 0 ? sqrt(h[0] / h[1]) : 0;
    stats[3 * k + 1] = h[2];
    stats[3 * k + 2] = sqrt(h[1] / double(n));
  }
  cudaFree(ref);
  cudaFree(out);
  cudaFree(dev);
  return e;
}
