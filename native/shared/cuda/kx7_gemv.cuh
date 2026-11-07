// The KX7 decode-row GEMV and its BF16 twin (kx7.cu dispatches rows <= 64 here).
//
// Structure (both kernels): a CTA owns 8 * G * RG weight rows (output columns) and up to MT
// 16-row activation tiles (rows <= 16 * MT); its KS * RG warps split K into KS contiguous slices
// (the pack's K split) for each of RG row-groups. Lane (g = lane / 4, j = lane % 4) handles weight
// row 8 * gi + g of each group gi and, per 64-wide K chunk, the 16 elements 16 j .. 16 j + 15 of
// that chunk: MMA step t takes k-slots 2j, 2j+1 <-> elements 16j+4t+{0,1} and 2j+8, 2j+9 <->
// 16j+4t+{2,3}, so the lane's 16 elements are the 8 bf16x2 words of four m16n8k16 B fragments and
// the matching A fragments come from two contiguous 32-byte activation runs (rows g and g + 8 of a
// tile). Each chunk's weight fragments are decoded (KX7) or loaded (BF16) once and multiplied
// into every activation tile; activation fragments are reloaded per tile. FP32 accumulation; the
// KS warps of a row-group fold their partials in ascending K-slice order. Every output element
// therefore has one accumulator, one K order and one fold whatever MT, G, RG and the live row
// count are: a row's bits do not depend on the batch width, and the KX7 kernel equals the BF16
// twin bit for bit (they differ only in where the weight words come from).
//
// Output: row r, column c lands at out[r * ldo + c], so a caller can write one matrix's columns
// into a wider row (ldo >= the columns). One KX7 launch can cover up to kMaxGroup packs of one
// shape and code (Kx7GemvGroup): its CTAs are the packs' CTAs in order, each pack's columns at its
// col0, and every output element still has its pack's single K order and fold.
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#include "kx7_decode.cuh"

namespace kx7gemv {

// The most packs one grouped KX7 launch covers (GLM Flash's q, k and v).
constexpr int kMaxGroup = 3;

// One pack of a launch: its device buffers, its code's scalars, its rows and where they land.
struct Kx7GemvPack {
  const uint8_t* streams;
  const uint32_t* offsets;
  const uint16_t* lengths;
  const uint32_t* lut;
  int lut_words;
  uint32_t esc_shift;
  uint32_t esc_mask;
  int n;     // weight rows (output columns) of this pack, a multiple of the CTA's columns
  int col0;  // first output column of this pack
};

// The packs of one launch: one shape, K split, layout and code (the kernel's TWOCLASS).
struct Kx7GemvGroup {
  Kx7GemvPack pack[kMaxGroup];
  int packs;
  int streams_per_row;
  uint32_t stride;
  int ragged;
};

__device__ __forceinline__ void mma_bf16_16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Activations: read-only, shared by every CTA (L1-cached).
__device__ __forceinline__ uint4 ld_nc_16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// BF16 twin weights: streamed once (no L1 allocation).
__device__ __forceinline__ uint4 ld_stream_16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

template <int KS, int RG, int G, int MT>
struct Geometry {
  static constexpr int WARPS = KS * RG;
  static constexpr int THREADS = WARPS * 32;
  static constexpr int COLS = 8 * G * RG;
  static constexpr int ROWS = 16 * MT;
  static constexpr int FRAGS = MT * G * 4;
  static constexpr int FOLD_BYTES = WARPS * FRAGS * 32 * 4;
  static constexpr int RING_BYTES = WARPS * G * KX7_RING_BYTES;
  static constexpr int kx7_smem(int lut_words) {
    return lut_words * 4 + (RING_BYTES > FOLD_BYTES ? RING_BYTES : FOLD_BYTES);
  }
  static constexpr int bf16_smem() { return FOLD_BYTES; }
};

// Activation fragments of one tile and chunk: rows g and g + 8 of the tile, elements 16j..16j+15.
// A lane whose row is past the live count reads the last live row (no branch); those MMA results
// land in rows that are never stored.
__device__ __forceinline__ void load_x(uint32_t xa[16], const uint16_t* xlo, const uint16_t* xhi) {
  const uint4 a = ld_nc_16(xlo), b = ld_nc_16(xlo + 8), c = ld_nc_16(xhi), d = ld_nc_16(xhi + 8);
  xa[0] = a.x; xa[1] = a.y; xa[2] = a.z; xa[3] = a.w; xa[4] = b.x; xa[5] = b.y; xa[6] = b.z; xa[7] = b.w;
  xa[8] = c.x; xa[9] = c.y; xa[10] = c.z; xa[11] = c.w; xa[12] = d.x; xa[13] = d.y; xa[14] = d.z; xa[15] = d.w;
}

__device__ __forceinline__ void mma_chunk(float acc[4], const uint32_t xa[16], const uint32_t bw[8]) {
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const uint32_t a[4] = {xa[2 * t], xa[8 + 2 * t], xa[2 * t + 1], xa[8 + 2 * t + 1]};
    const uint32_t b[2] = {bw[2 * t], bw[2 * t + 1]};
    mma_bf16_16816(acc, a, b);
  }
}

// One chunk's MMAs over every activation tile, reloading each tile's fragments.
template <int G, int MT>
__device__ __forceinline__ void mma_tiles(float (&acc)[MT][G][4], const uint32_t (&bw)[G][8], const uint16_t* X,
                                          int rows, int K, int col, int g) {
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    uint32_t xa[16];
    const int lo = min(16 * mt + g, rows - 1), hi = min(16 * mt + g + 8, rows - 1);
    load_x(xa, X + size_t(lo) * K + col, X + size_t(hi) * K + col);
#pragma unroll
    for (int gi = 0; gi < G; ++gi) mma_chunk(acc[mt][gi], xa, bw[gi]);
  }
}

// The KS warps of a row-group fold in ascending K-slice order; warp (ks, rg) stores the fragments
// i with i % KS == ks, rounded once to BF16, at out[row * ldo + c0 + column within the CTA].
template <int KS, int RG, int G, int MT>
__device__ __forceinline__ void fold_and_store(float (&acc)[MT][G][4], float* partial, __nv_bfloat16* __restrict__ out,
                                               int rows, int ldo, int c0, int ks, int rg, int lane, int g, int j) {
  constexpr int FRAGS = MT * G * 4;
  const int w = rg * KS + ks;
#pragma unroll
  for (int i = 0; i < FRAGS; ++i) partial[(w * FRAGS + i) * 32 + lane] = acc[i / (4 * G)][(i / 4) % G][i % 4];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < FRAGS; ++i) {
    if (i % KS == ks) {
      float total = 0.0f;
#pragma unroll
      for (int src = 0; src < KS; ++src) total += partial[((rg * KS + src) * FRAGS + i) * 32 + lane];
      const int mt = i / (4 * G), gi = (i / 4) % G, r = i % 4;
      const int row = 16 * mt + g + 8 * (r / 2);
      const int col = c0 + 8 * (rg * G + gi) + 2 * j + (r % 2);
      if (row < rows) out[size_t(row) * ldo + col] = __float2bfloat16_rn(total);
    }
  }
}

// Pack `i` of a group by value, through constant indices (no local copy of the parameter block).
__device__ __forceinline__ Kx7GemvPack pick(const Kx7GemvGroup& group, int i) {
  static_assert(kMaxGroup == 3, "pick names every pack slot");
  return i == 0 ? group.pack[0] : (i == 1 ? group.pack[1] : group.pack[2]);
}

// The KX7 GEMV: for each pack P of `group`, out[:, P.col0 .. P.col0 + P.n) (row stride ldo) BF16 =
// X [rows, K] BF16 @ W_P^T, W_P a v1.1-layout pack under the two-class code (TWOCLASS) or its
// Huffman fallback. The grid is the packs' CTAs in order (P.n / COLS each).
template <int KS, int RG, int G, int MT, bool TWOCLASS>
__global__ void __launch_bounds__(KS * RG * 32)
    kx7_gemv_kernel(const uint16_t* __restrict__ X, int rows, int K, int ldo, const Kx7GemvGroup group,
                    __nv_bfloat16* __restrict__ out) {
  using Geo = Geometry<KS, RG, G, MT>;
  extern __shared__ __align__(16) uint32_t smem[];
  int block = blockIdx.x, at = 0;
#pragma unroll
  for (int i = 0; i + 1 < kMaxGroup; ++i) {
    const int ctas = pick(group, at).n / Geo::COLS;
    if (at + 1 < group.packs && block >= ctas) {
      block -= ctas;
      ++at;
    }
  }
  const Kx7GemvPack pack = pick(group, at);
  uint32_t* lut = smem;
  uint32_t* ring = smem + pack.lut_words;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, j = lane & 3;
  const int ks = warp % KS, rg = warp / KS;
  const int n0 = block * Geo::COLS;
  const int kpw = K / KS, chunks = kpw / 64, kb = ks * kpw;
  kx7_load_tables(lut, pack.lut, pack.lut_words, tid, Geo::THREADS);
  __syncthreads();
  float acc[MT][G][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int gi = 0; gi < G; ++gi) acc[mt][gi][0] = acc[mt][gi][1] = acc[mt][gi][2] = acc[mt][gi][3] = 0.0f;
  Kx7Stream s[G];
  uint32_t* ring_lane[G];
#pragma unroll
  for (int gi = 0; gi < G; ++gi) {
    const int n = n0 + 8 * (rg * G + gi) + g;
    const size_t sidx = size_t(n) * group.streams_per_row + ks * 4 + j;
    ring_lane[gi] = ring + (warp * G + gi) * (KX7_RING_BYTES / 4) + lane;
    kx7_init(s[gi], pack.streams, pack.offsets[sidx], pack.lengths[sidx], group.stride, group.ragged != 0,
             ring_lane[gi]);
  }
  for (int c = 0; c < chunks; ++c) {
    uint32_t bw[G][8];
    kx7_decode16_multi<G, TWOCLASS>(s, ring_lane, lut, pack.esc_shift, pack.esc_mask, group.stride, bw);
    mma_tiles<G, MT>(acc, bw, X, rows, K, kb + 16 * j + 64 * c, g);
  }
  __syncthreads();  // the ring is dead: reuse it for the fold
  fold_and_store<KS, RG, G, MT>(acc, reinterpret_cast<float*>(ring), out, rows, ldo, pack.col0 + n0, ks, rg, lane, g,
                                j);
}

// The BF16 twin (tests, and BF16 rows beside KX7 ones): the same kernel reading row-major BF16
// weights W [N, K] into out[:, 0 .. N) with row stride ldo.
template <int KS, int RG, int G, int MT>
__global__ void __launch_bounds__(KS * RG * 32)
    bf16_gemv_kernel(const uint16_t* __restrict__ X, int rows, int N, int K, const uint16_t* __restrict__ W, int ldo,
                     __nv_bfloat16* __restrict__ out) {
  using Geo = Geometry<KS, RG, G, MT>;
  extern __shared__ __align__(16) uint32_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, j = lane & 3;
  const int ks = warp % KS, rg = warp / KS;
  const int n0 = blockIdx.x * Geo::COLS;
  const int kpw = K / KS, chunks = kpw / 64, kb = ks * kpw;
  float acc[MT][G][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int gi = 0; gi < G; ++gi) acc[mt][gi][0] = acc[mt][gi][1] = acc[mt][gi][2] = acc[mt][gi][3] = 0.0f;
  const uint16_t* wrow[G];
#pragma unroll
  for (int gi = 0; gi < G; ++gi) wrow[gi] = W + size_t(n0 + 8 * (rg * G + gi) + g) * K + kb + 16 * j;
  for (int c = 0; c < chunks; ++c) {
    uint32_t bw[G][8];
#pragma unroll
    for (int gi = 0; gi < G; ++gi) {
      const uint4 a = ld_stream_16(wrow[gi] + 64 * c), b = ld_stream_16(wrow[gi] + 64 * c + 8);
      bw[gi][0] = a.x; bw[gi][1] = a.y; bw[gi][2] = a.z; bw[gi][3] = a.w;
      bw[gi][4] = b.x; bw[gi][5] = b.y; bw[gi][6] = b.z; bw[gi][7] = b.w;
    }
    mma_tiles<G, MT>(acc, bw, X, rows, K, kb + 16 * j + 64 * c, g);
  }
  (void)N;
  fold_and_store<KS, RG, G, MT>(acc, reinterpret_cast<float*>(smem), out, rows, ldo, n0, ks, rg, lane, g, j);
}

}  // namespace kx7gemv
