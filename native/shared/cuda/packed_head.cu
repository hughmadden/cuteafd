// Exact 12-bit BF16 vocabulary heads for skinny row counts: every weight of a
// BF16 [n, k] head (n % 16 == 0, k % 128 == 0) in 12 bits, plus a per-row table
// of the few values 12 bits cannot hold, applied by a tensor-core GEMV built
// like fp8_gemv.cu's W8A16 path (BF16 activations, FP32 sums).
//
// Format. Row r keeps a base exponent e0[r]: of the windows [e0, e0 + 15] of
// BF16 biased exponents (e0 <= 240), the one holding the most of the row's
// values, ties to the lowest e0. A value whose exponent e is in the window is
// its sign, code = e - e0 (4 bits) and its 7 mantissa bits, in two planes so a
// reader can take one:
//   coarse byte = sign << 7 | code << 3 | mantissa bits 6..4
//   fine nibble = mantissa bits 3..0
// Exact decode: bf16 = sign << 15 | (e0 + code) << 7 | m6..4 << 4 | m3..0.
// Coarse decode (drafters): the same with m3..0 = 0b1000, the midpoint of the
// dropped bits. A value outside the window is an exception: its planes hold
// it clamped into the window with its sign (below: exponent e0, mantissa 0;
// above: e0 + 15, mantissa 127), and a CSR table (row_ptr [n + 1], columns,
// values: u32, u32, BF16) records its exact value, in column order. The exact
// GEMV adds (exact - coded) x[m][column] per exception into the row's sum
// after the K reduction, in table order; the coarse GEMV ignores the table.
//
// Layout. Both planes follow fp8_gemv.cu's A-fragment order: tiles of 16 rows
// x 32 k, 32 lanes each. Lane l (g = l / 4, c = 2 (l % 4)) of a tile holds,
// per k16 step s, mma.m16n8k16's A registers a0..a3: (row g, k c c+1),
// (row g + 8, k c c+1), (row g, k c+8 c+9), (row g + 8, k c+8 c+9). The coarse
// plane keeps 16 bytes per lane per tile (byte 8 s + 2 j + h: register j's
// value h), the fine plane 8 bytes (32-bit word s, nibble 4 j + 16 h), so a lane
// reads one coalesced 16-byte and one 8-byte word per tile and turns each
// register into bf16x2 with a byte permute, a shift, a mask and an add.
//
// Pack, from BF16 weights already resident: per row a 256-bin exponent
// histogram picks e0 and counts the exceptions; one scan gives row_ptr; one
// warp per (tile, K block) writes the planes and one warp per row the table.
// Then the pack verifies itself: an independent decode of every value (planes,
// base and table) against the BF16 source, bit for bit, counting failing rows.
//
// Every output element's MMAs, split-K sums and exception fix-ups are its own
// and in a fixed order, so a row's logits do not depend on how many activation
// rows the call has; the split plan depends only on the shape and SM count.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "gemv_tiles.cuh"

namespace {

using namespace cuteafd::gemv_tiles;

// Largest base exponent: the window [e0, e0 + 15] stays in the 8-bit field.
constexpr uint32_t kMaxBase = 240;
// The `planes` arrays the entry points take.
enum Plane { kCoarse, kFine, kBase, kRowPtr, kColumns, kValues, kPlanes };
// Status words (u64): exceptions, rows failing verification, the first such row
// (all ones if none), and the most exceptions in any row.
enum Status { kExceptions, kBadRows, kFirstBadRow, kMaxRowExceptions, kStatusWords };

constexpr int kBaseThreads = 128;
constexpr int kScanThreads = 1024;
constexpr int kRowWarps = 4;  // rows per CTA of the table and verify kernels

__host__ __device__ __forceinline__ uint32_t exponent_of(uint32_t bits) { return (bits >> 7) & 0xffu; }

// The value row base `e0` codes for BF16 `bits`: itself inside the window,
// else clamped into it with its sign (the window's smallest or largest magnitude).
__device__ __forceinline__ uint32_t clamp_to_window(uint32_t bits, uint32_t e0) {
  const uint32_t e = exponent_of(bits), sign = bits & 0x8000u;
  if (e < e0) return sign | (e0 << 7);
  if (e > e0 + 15) return sign | ((e0 + 15) << 7) | 0x7fu;
  return bits;
}

__device__ __forceinline__ float bf16_value(uint32_t bits) { return __uint_as_float(bits << 16); }

bool valid_shape(int32_t n, int32_t k) {
  return n >= 16 && n % 16 == 0 && k >= 128 && k % 128 == 0 && uint64_t(n) * uint64_t(k) < (uint64_t(1) << 32);
}

// --- pack -------------------------------------------------------------------

// One CTA per row: the exponent histogram, the window holding the most values
// (ties: the lowest base) and the row's exception count, into row_ptr[r + 1].
__global__ void __launch_bounds__(kBaseThreads) base_kernel(const uint16_t* __restrict__ w, uint8_t* __restrict__ base,
                                                            uint32_t* __restrict__ row_ptr,
                                                            unsigned long long* __restrict__ status, int k) {
  __shared__ uint32_t histogram[256];
  __shared__ unsigned long long best[kBaseThreads / 32];
  const int r = blockIdx.x, t = threadIdx.x;
  for (int i = t; i < 256; i += kBaseThreads) histogram[i] = 0;
  __syncthreads();
  const uint16_t* row = w + size_t(r) * k;
  for (int col = t; col < k; col += kBaseThreads) atomicAdd(&histogram[exponent_of(row[col])], 1u);
  __syncthreads();
  // The count above, 255 - e0 below: the largest key is the fullest window with the lowest base.
  unsigned long long key = 0;
  for (uint32_t e0 = t; e0 <= kMaxBase; e0 += kBaseThreads) {
    uint32_t count = 0;
    for (int i = 0; i < 16; ++i) count += histogram[e0 + i];
    const unsigned long long candidate = (static_cast<unsigned long long>(count) << 8) | (255u - e0);
    key = candidate > key ? candidate : key;
  }
  for (int offset = 16; offset; offset >>= 1) {
    const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, offset);
    key = other > key ? other : key;
  }
  if (t % 32 == 0) best[t / 32] = key;
  __syncthreads();
  if (t == 0) {
    for (int i = 1; i < kBaseThreads / 32; ++i) key = best[i] > key ? best[i] : key;
    const uint32_t exceptions = uint32_t(k) - uint32_t(key >> 8);
    base[r] = uint8_t(255u - uint32_t(key & 0xffu));
    row_ptr[r + 1] = exceptions;
    if (r == 0) row_ptr[0] = 0;
    atomicMax(&status[kMaxRowExceptions], static_cast<unsigned long long>(exceptions));
  }
}

// row_ptr[1..n]: per-row counts to running totals, in one CTA. Each thread sums
// a contiguous slice, one thread scans the slice sums, the slices rewrite
// themselves. The total goes to status[kExceptions] (n * k < 2^32: no overflow).
__global__ void __launch_bounds__(kScanThreads) scan_kernel(uint32_t* __restrict__ row_ptr,
                                                            unsigned long long* __restrict__ status, int n) {
  __shared__ unsigned long long sums[kScanThreads];
  const int t = threadIdx.x;
  const int per = (n + kScanThreads - 1) / kScanThreads;
  const int begin = min(n, t * per), end = min(n, begin + per);
  unsigned long long sum = 0;
  for (int i = begin; i < end; ++i) sum += row_ptr[i + 1];
  sums[t] = sum;
  __syncthreads();
  if (t == 0) {
    unsigned long long running = 0;
    for (int i = 0; i < kScanThreads; ++i) {
      const unsigned long long slice = sums[i];
      sums[i] = running;
      running += slice;
    }
    status[kExceptions] = running;
  }
  __syncthreads();
  unsigned long long running = sums[t];
  for (int i = begin; i < end; ++i) {
    running += row_ptr[i + 1];
    row_ptr[i + 1] = uint32_t(running);
  }
}

// One warp per (16-row tile, 128-wide K block): its four 16 x 32 tiles of both planes.
__global__ void planes_kernel(const uint16_t* __restrict__ w, const uint8_t* __restrict__ base,
                              uint4* __restrict__ coarse, uint2* __restrict__ fine, int k) {
  const int rt = blockIdx.x, kb = blockIdx.y, lane = threadIdx.x;
  const int g = lane / 4, c = (lane % 4) * 2;
  const uint16_t* rows[2] = {w + size_t(rt * 16 + g) * k, w + size_t(rt * 16 + g + 8) * k};
  const uint32_t bases[2] = {base[rt * 16 + g], base[rt * 16 + g + 8]};
  for (int kt = 0; kt < 4; ++kt) {
    uint8_t bytes[16];
    uint32_t words[2] = {0, 0};
    for (int step = 0; step < 2; ++step) {
      const int k0 = kb * 128 + kt * 32 + step * 16 + c;
      for (int j = 0; j < 4; ++j) {  // register j: row g + 8 (j & 1), k0 + 8 (j >> 1)
        const uint32_t e0 = bases[j & 1];
        for (int h = 0; h < 2; ++h) {
          const uint32_t coded = clamp_to_window(rows[j & 1][k0 + (j >> 1) * 8 + h], e0);
          bytes[step * 8 + j * 2 + h] =
              uint8_t(((coded >> 8) & 0x80u) | ((exponent_of(coded) - e0) << 3) | ((coded >> 4) & 7u));
          words[step] |= (coded & 0xfu) << (4 * j + 16 * h);
        }
      }
    }
    const size_t index = (size_t(rt) * (k / 32) + kb * 4 + kt) * 32 + lane;
    uint4 v;
    memcpy(&v, bytes, 16);
    coarse[index] = v;
    fine[index] = make_uint2(words[0], words[1]);
  }
}

// One warp per row: (column, exact BF16) of every value outside the row's
// window, in column order, from row_ptr[r]. Nothing when the table overflows.
__global__ void __launch_bounds__(kRowWarps * 32) table_kernel(const uint16_t* __restrict__ w,
                                                               const uint8_t* __restrict__ base,
                                                               const uint32_t* __restrict__ row_ptr,
                                                               uint32_t* __restrict__ columns,
                                                               uint16_t* __restrict__ values, long long capacity,
                                                               int n, int k) {
  const int lane = threadIdx.x & 31, r = blockIdx.x * kRowWarps + threadIdx.x / 32;
  if (r >= n || static_cast<long long>(row_ptr[n]) > capacity) return;
  const uint16_t* row = w + size_t(r) * k;
  const uint32_t e0 = base[r];
  uint32_t at = row_ptr[r];
  for (int col0 = 0; col0 < k; col0 += 32) {
    const uint32_t bits = row[col0 + lane], e = exponent_of(bits);
    const bool outside = e < e0 || e > e0 + 15;
    const uint32_t ballot = __ballot_sync(0xffffffffu, outside);
    if (outside) {
      const uint32_t i = at + __popc(ballot & ((1u << lane) - 1));
      columns[i] = uint32_t(col0 + lane);
      values[i] = uint16_t(bits);
    }
    at += __popc(ballot);
  }
}

// The BF16 the planes and base `e0` give row r, column col, decoded by position
// (independently of the GEMV's register decode).
__device__ __forceinline__ uint32_t decode_at(const uint8_t* __restrict__ coarse, const uint32_t* __restrict__ fine,
                                              uint32_t e0, int r, int col, int k) {
  const int g = r & 7, upper = (r >> 3) & 1, kk = col & 31;
  const int step = kk >> 4, half = (kk >> 3) & 1, within = kk & 7;
  const int lane = g * 4 + (within >> 1), h = within & 1, j = half * 2 + upper;
  const size_t index = (size_t(r >> 4) * (k / 32) + (col >> 5)) * 32 + lane;
  const uint32_t byte = coarse[index * 16 + step * 8 + j * 2 + h];
  const uint32_t nibble = (fine[index * 2 + step] >> (4 * j + 16 * h)) & 0xfu;
  return ((byte & 0x80u) << 8) | (((byte & 0x7fu) << 4) + (e0 << 7)) | nibble;
}

// One warp per row. A row passes when its base and row_ptr range are sane, and
// every value decodes to its source, except exactly the table's entries, in
// column order, whose planes hold the clamped value and whose table value is
// the source. Failing rows are counted, and the first one kept.
__global__ void __launch_bounds__(kRowWarps * 32) verify_kernel(
    const uint16_t* __restrict__ w, const uint8_t* __restrict__ coarse, const uint32_t* __restrict__ fine,
    const uint8_t* __restrict__ base, const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ columns,
    const uint16_t* __restrict__ values, long long capacity, unsigned long long* __restrict__ status, int n, int k) {
  const int lane = threadIdx.x & 31, r = blockIdx.x * kRowWarps + threadIdx.x / 32;
  if (r >= n) return;
  const uint32_t total = row_ptr[n], begin = row_ptr[r], end = row_ptr[r + 1], e0 = base[r];
  bool ok = static_cast<long long>(total) <= capacity && e0 <= kMaxBase && begin <= end && end <= total;
  uint32_t found = 0;
  for (int col0 = 0; ok && col0 < k; col0 += 32) {
    const int col = col0 + lane;
    const uint32_t source = w[size_t(r) * k + col];
    const uint32_t coded = decode_at(coarse, fine, e0, r, col, k);
    const bool exception = coded != source;
    const uint32_t ballot = __ballot_sync(0xffffffffu, exception);
    bool good = true;
    if (exception) {
      const uint32_t i = begin + found + __popc(ballot & ((1u << lane) - 1));
      good = i < end && columns[i] == uint32_t(col) && values[i] == source && coded == clamp_to_window(source, e0);
    }
    found += __popc(ballot);
    ok = __all_sync(0xffffffffu, good);
  }
  if (lane == 0 && !(ok && begin + found == end)) {
    atomicAdd(&status[kBadRows], 1ull);
    atomicMin(&status[kFirstBadRow], static_cast<unsigned long long>(r));
  }
}

// --- GEMV -------------------------------------------------------------------

// One packed head's device arrays.
struct Head {
  const uint4* coarse;
  const uint2* fine;
  const uint8_t* base;
  const uint32_t* row_ptr;
  const uint32_t* columns;
  const uint16_t* values;
};

// Activations [rows, k] BF16 -> B fragments, unconverted: per 32-wide k tile
// and 8-row group, 32 lanes x 16 bytes; lane l (row group * 8 + l / 4,
// c = 2 (l % 4)) holds the k pairs c c+1 and c+8 c+9 of both k16 steps.
__global__ void pack_x_kernel(const uint32_t* __restrict__ x, uint4* __restrict__ xp, int rows, int k, int groups) {
  const int kt = blockIdx.x, group = blockIdx.y, lane = threadIdx.x;
  const int m = group * 8 + lane / 4, c = (lane % 4) * 2;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (m < rows) {
    const uint32_t* pairs = x + size_t(m) * (k / 2) + kt * 16 + c / 2;
    v = make_uint4(pairs[0], pairs[4], pairs[8], pairs[12]);
  }
  xp[(size_t(kt) * groups + group) * 32 + lane] = v;
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t selector) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(0u), "r"(selector));
  return d;
}

// Selectors for two coarse bytes of a word, each followed by its sign replicated.
constexpr uint32_t kBytes01 = 0x9180u;
constexpr uint32_t kBytes23 = 0xb3a2u;

// Two values' coarse bytes (picked by SELECTOR from `word`) and fine nibbles
// (bits 0..3 and 16..19 of `nibbles`) to the bf16x2 they code at `bias` (e0 << 7
// in both halves, plus the coarse midpoint when there are no nibbles). With
// each byte's sign replicated above it, one shift puts the sign on bit 15 and
// code and mantissa on bits 4..10; the mask clears the rest, and the sum of the
// disjoint fields carries code + e0 into the exponent.
template <uint32_t SELECTOR>
__device__ __forceinline__ uint32_t decode2(uint32_t word, uint32_t nibbles, uint32_t bias) {
  return ((prmt(word, SELECTOR) << 4) & 0x87f087f0u) + bias + nibbles;
}

// One k16 step's A fragment from its coarse words (registers 0 1, 2 3) and fine word.
template <bool COARSE>
__device__ __forceinline__ void fragment(uint32_t (&a)[4], uint32_t w01, uint32_t w23, uint32_t fine,
                                         uint32_t bias_lo, uint32_t bias_hi) {
  constexpr uint32_t kNibbles = 0x000f000fu;
  a[0] = decode2<kBytes01>(w01, COARSE ? 0u : fine & kNibbles, bias_lo);
  a[1] = decode2<kBytes23>(w01, COARSE ? 0u : (fine >> 4) & kNibbles, bias_hi);
  a[2] = decode2<kBytes01>(w23, COARSE ? 0u : (fine >> 8) & kNibbles, bias_lo);
  a[3] = decode2<kBytes23>(w23, COARSE ? 0u : (fine >> 12) & kNibbles, bias_hi);
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// One row's exceptions: its table range and base.
struct RowTable {
  uint32_t begin, end, e0;
};

__device__ __forceinline__ RowTable row_table(const Head& head, int row) {
  return {head.row_ptr[row], head.row_ptr[row + 1], head.base[row]};
}

// What the planes miss of an exception: exact - coded, in FP32.
__device__ __forceinline__ float exception_delta(uint32_t exact, uint32_t e0) {
  return __fsub_rn(bf16_value(exact), bf16_value(clamp_to_window(exact, e0)));
}

// A row's exact sum for the activation row `x`: `sum` (the coded weights'
// reduction), then delta * x per exception in table order, one FMA each.
__device__ __forceinline__ float fix_exceptions(float sum, const Head& head, const RowTable& table,
                                                const uint16_t* __restrict__ x) {
  for (uint32_t i = table.begin; i < table.end; ++i) {
    sum = __fmaf_rn(exception_delta(head.values[i], table.e0), bf16_value(x[head.columns[i]]), sum);
  }
  return sum;
}

__device__ __forceinline__ void store(void* out, int out_f32, size_t at, float v) {
  if (out_f32) {
    static_cast<float*>(out)[at] = v;
  } else {
    static_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16(v);
  }
}

// One warp per (16-row tile, K split), GROUPS 8-row activation groups. Each
// 128-wide K block's MMAs sum into their own registers before joining the
// FP32 total, as fp8_gemv.cu's gemv_kernel.
template <int GROUPS, bool COARSE>
__global__ void __launch_bounds__(kWarps * 32)
    gemv_kernel(Head head, const uint4* __restrict__ xp, const uint16_t* __restrict__ x, void* __restrict__ out,
                float* __restrict__ partial, int out_f32, int rows, int n, int k, int kb_per_split, int splits) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * kWarps + threadIdx.x / 32;
  const int tiles = n / 16;
  if (warp >= tiles * splits) return;
  const int rt = warp % tiles, split = warp / tiles;
  const int kbs = k / 128;
  const int kb0 = split * kb_per_split, kb1 = min(kbs, kb0 + kb_per_split);
  const int g = lane / 4, c = (lane % 4) * 2;
  const int row_lo = rt * 16 + g, row_hi = row_lo + 8;
  const uint32_t midpoint = COARSE ? 0x00080008u : 0u;
  const uint32_t bias_lo = (uint32_t(head.base[row_lo]) << 7) * 0x00010001u + midpoint;
  const uint32_t bias_hi = (uint32_t(head.base[row_hi]) << 7) * 0x00010001u + midpoint;
  float acc[GROUPS][4];
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
  const uint4* crow = head.coarse + size_t(rt) * (k / 32) * 32 + lane;
  const uint2* frow = head.fine + size_t(rt) * (k / 32) * 32 + lane;
  uint4 a[4];
  uint2 f[4];
  if (kb0 < kb1) {
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      a[kt] = load_stream(crow + size_t(kb0 * 4 + kt) * 32);
      if (!COARSE) f[kt] = load_stream(frow + size_t(kb0 * 4 + kt) * 32);
    }
  }
  for (int kb = kb0; kb < kb1; ++kb) {
    uint4 next[4];
    uint2 fnext[4];
    if (kb + 1 < kb1) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
        next[kt] = load_stream(crow + size_t((kb + 1) * 4 + kt) * 32);
        if (!COARSE) fnext[kt] = load_stream(frow + size_t((kb + 1) * 4 + kt) * 32);
      }
    }
    float blk[GROUPS][4];
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) blk[i][0] = blk[i][1] = blk[i][2] = blk[i][3] = 0.0f;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      uint32_t a0[4], a1[4];
      fragment<COARSE>(a0, a[kt].x, a[kt].y, COARSE ? 0u : f[kt].x, bias_lo, bias_hi);
      fragment<COARSE>(a1, a[kt].z, a[kt].w, COARSE ? 0u : f[kt].y, bias_lo, bias_hi);
      const uint4* b = xp + (size_t(kb * 4 + kt) * GROUPS) * 32 + lane;
#pragma unroll
      for (int i = 0; i < GROUPS; ++i) {
        const uint4 bv = __ldg(b + i * 32);
        mma_bf16(blk[i], a0, bv.x, bv.y);
        mma_bf16(blk[i], a1, bv.z, bv.w);
      }
    }
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) {
      acc[i][0] += blk[i][0];
      acc[i][1] += blk[i][1];
      acc[i][2] += blk[i][2];
      acc[i][3] += blk[i][3];
    }
    if (kb + 1 < kb1) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) {
        a[kt] = next[kt];
        if (!COARSE) f[kt] = fnext[kt];
      }
    }
  }
  // C fragment: c0, c1 weight row g, activation rows c, c + 1; c2, c3 weight
  // row g + 8. Exact single-split outputs take their rows' exceptions here: each
  // output gets fix_exceptions' FMA sequence, with the table read once per row.
  if (!COARSE && splits == 1) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const RowTable table = row_table(head, half ? row_hi : row_lo);
      for (uint32_t e = table.begin; e < table.end; ++e) {
        const float delta = exception_delta(head.values[e], table.e0);
        const uint32_t column = head.columns[e];
#pragma unroll
        for (int i = 0; i < GROUPS; ++i) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int m = i * 8 + c + h;
            if (m < rows) {
              acc[i][half * 2 + h] = __fmaf_rn(delta, bf16_value(x[size_t(m) * k + column]), acc[i][half * 2 + h]);
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int m = i * 8 + c + (j & 1);
      const int row = j < 2 ? row_lo : row_hi;
      if (m >= rows) continue;
      if (splits > 1) {
        partial[(size_t(split) * rows + m) * n + row] = acc[i][j];
      } else {
        store(out, out_f32, size_t(m) * n + row, acc[i][j]);
      }
    }
  }
}

// The K splits' sum in split order, then (exact) the exceptions, as the
// single-split epilogue does.
__global__ void reduce_kernel(Head head, const float* __restrict__ partial, const uint16_t* __restrict__ x,
                              void* __restrict__ out, int out_f32, int exact, int rows, int n, int k, int splits) {
  const size_t total = size_t(rows) * n;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    float v = 0.0f;
    for (int s = 0; s < splits; ++s) v += partial[size_t(s) * total + i];
    if (exact) v = fix_exceptions(v, head, row_table(head, int(i % n)), x + (i / n) * k);
    store(out, out_f32, i, v);
  }
}

template <bool COARSE>
void launch_gemv(int groups, dim3 grid, cudaStream_t s, const Head& head, const uint4* xp, const uint16_t* x, void* out,
                 float* partial, int out_f32, int rows, int n, int k, int kb_per_split, int splits) {
  const dim3 block(kWarps * 32);
  switch (groups) {
    case 1: gemv_kernel<1, COARSE><<<grid, block, 0, s>>>(head, xp, x, out, partial, out_f32, rows, n, k, kb_per_split, splits); break;
    case 2: gemv_kernel<2, COARSE><<<grid, block, 0, s>>>(head, xp, x, out, partial, out_f32, rows, n, k, kb_per_split, splits); break;
    case 4: gemv_kernel<4, COARSE><<<grid, block, 0, s>>>(head, xp, x, out, partial, out_f32, rows, n, k, kb_per_split, splits); break;
    default: gemv_kernel<8, COARSE><<<grid, block, 0, s>>>(head, xp, x, out, partial, out_f32, rows, n, k, kb_per_split, splits); break;
  }
}

}  // namespace

// Verifies a packed head against its BF16 source `w` [n, k] (see the file
// comment): status words 1 (failing rows) and 2 (the first, all ones if none)
// are reset and written, the others left as the pack wrote them; the table's
// size is row_ptr[n]. `planes` are coarse, fine, base, row_ptr, columns,
// values; `capacity` the table's entries. Asynchronous on `stream`; read the
// status after it drains.
extern "C" int32_t cuteafd_packed_head_verify(const void* w, const void* const* planes, int64_t capacity, int32_t n,
                                              int32_t k, void* status, void* stream) {
  if (!w || !planes || !status || !valid_shape(n, k) || capacity < 0 || capacity > INT64_C(0xffffffff)) {
    return cudaErrorInvalidValue;
  }
  for (int i = 0; i < kPlanes; ++i) {
    if (!planes[i] && !(capacity == 0 && (i == kColumns || i == kValues))) return cudaErrorInvalidValue;
  }
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  unsigned long long* words = static_cast<unsigned long long*>(status);
  cudaError_t error = cudaMemsetAsync(words + kBadRows, 0, sizeof(*words), s);
  if (error == cudaSuccess) error = cudaMemsetAsync(words + kFirstBadRow, 0xff, sizeof(*words), s);
  if (error != cudaSuccess) return error;
  verify_kernel<<<(n + kRowWarps - 1) / kRowWarps, kRowWarps * 32, 0, s>>>(
      static_cast<const uint16_t*>(w), static_cast<const uint8_t*>(planes[kCoarse]),
      static_cast<const uint32_t*>(planes[kFine]), static_cast<const uint8_t*>(planes[kBase]),
      static_cast<const uint32_t*>(planes[kRowPtr]), static_cast<const uint32_t*>(planes[kColumns]),
      static_cast<const uint16_t*>(planes[kValues]), capacity, words, n, k);
  return cudaGetLastError();
}

// Packs the BF16 head `w` [n, k] (n % 16 == 0, k % 128 == 0, n * k < 2^32)
// into `planes` (coarse n * k bytes, fine n * k / 2 bytes, base n bytes,
// row_ptr n + 1 u32, columns `capacity` u32, values `capacity` BF16; 16-byte
// aligned planes) and verifies it. `status` holds four u64 words: exceptions
// (above `capacity`, the table is left unwritten and every row fails), failing
// rows, the first failing row (all ones if none), the most exceptions in a row.
// Asynchronous on `stream`; the caller drains it before reading the status.
extern "C" int32_t cuteafd_packed_head_pack(const void* w, void* const* planes, int64_t capacity, int32_t n,
                                            int32_t k, void* status, void* stream) {
  if (!w || !planes || !status || !valid_shape(n, k) || capacity < 0 || capacity > INT64_C(0xffffffff)) {
    return cudaErrorInvalidValue;
  }
  for (int i = 0; i < kPlanes; ++i) {
    if (!planes[i] && !(capacity == 0 && (i == kColumns || i == kValues))) return cudaErrorInvalidValue;
  }
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  unsigned long long* words = static_cast<unsigned long long*>(status);
  const uint16_t* source = static_cast<const uint16_t*>(w);
  uint8_t* base = static_cast<uint8_t*>(planes[kBase]);
  uint32_t* row_ptr = static_cast<uint32_t*>(planes[kRowPtr]);
  const cudaError_t error = cudaMemsetAsync(words, 0, kStatusWords * sizeof(*words), s);
  if (error != cudaSuccess) return error;
  base_kernel<<<n, kBaseThreads, 0, s>>>(source, base, row_ptr, words, k);
  scan_kernel<<<1, kScanThreads, 0, s>>>(row_ptr, words, n);
  planes_kernel<<<dim3(n / 16, k / 128), 32, 0, s>>>(source, base, static_cast<uint4*>(planes[kCoarse]),
                                                       static_cast<uint2*>(planes[kFine]), k);
  table_kernel<<<(n + kRowWarps - 1) / kRowWarps, kRowWarps * 32, 0, s>>>(
      source, base, row_ptr, static_cast<uint32_t*>(planes[kColumns]), static_cast<uint16_t*>(planes[kValues]),
      capacity, n, k);
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) return launched;
  return cuteafd_packed_head_verify(w, planes, capacity, n, k, status, stream);
}

// Scratch bytes cuteafd_packed_head_linear needs for these shapes (0 for an
// invalid one): the B fragments of 64 rows and, when the head's split plan
// splits K on this device, the partial sums of a pass.
extern "C" size_t cuteafd_packed_head_workspace(int32_t rows, int32_t k, int32_t n) {
  if (rows < 1 || !valid_shape(n, k)) return 0;
  const int chunk = rows < kMaxRows ? rows : kMaxRows;
  int kb_per_split, splits;
  split_plan(n, k, &kb_per_split, &splits);
  return align256(size_t(k) * kMaxRows * 2) +
         (splits > 1 ? align256(size_t(splits) * chunk * n * sizeof(float)) : 0);
}

// out [rows, n] (BF16, or FP32 with out_f32) = x [rows, k] BF16 @ W^T for a
// head packed by cuteafd_packed_head_pack: exact (both planes and the table),
// or with `coarse` the coarse plane alone (`planes` fine, row_ptr, columns and
// values unused). `workspace` holds cuteafd_packed_head_workspace bytes.
extern "C" int32_t cuteafd_packed_head_linear(const void* x, const void* const* planes, void* out, int32_t out_f32,
                                              int32_t rows, int32_t k, int32_t n, int32_t coarse, void* workspace,
                                              size_t workspace_bytes, void* stream) {
  if (!x || !planes || !out || !workspace || rows < 1 || !valid_shape(n, k)) return cudaErrorInvalidValue;
  if (!planes[kCoarse] || !planes[kBase] || (!coarse && (!planes[kFine] || !planes[kRowPtr]))) {
    return cudaErrorInvalidValue;
  }
  if (workspace_bytes < cuteafd_packed_head_workspace(rows, k, n)) return cudaErrorInvalidValue;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  const Head head{static_cast<const uint4*>(planes[kCoarse]),     static_cast<const uint2*>(planes[kFine]),
                  static_cast<const uint8_t*>(planes[kBase]),     static_cast<const uint32_t*>(planes[kRowPtr]),
                  static_cast<const uint32_t*>(planes[kColumns]), static_cast<const uint16_t*>(planes[kValues])};
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  uint4* xp = reinterpret_cast<uint4*>(ws);
  float* partial = reinterpret_cast<float*>(ws + align256(size_t(k) * kMaxRows * 2));
  int kb_per_split, splits;
  split_plan(n, k, &kb_per_split, &splits);
  const size_t out_row = size_t(n) * (out_f32 ? 4 : 2);
  const int warps = (n / 16) * splits;
  const dim3 grid((warps + kWarps - 1) / kWarps);
  for (int first = 0; first < rows; first += kMaxRows) {
    const int m = rows - first < kMaxRows ? rows - first : kMaxRows;
    const int groups = m <= 8 ? 1 : m <= 16 ? 2 : m <= 32 ? 4 : 8;
    const uint16_t* xm = static_cast<const uint16_t*>(x) + size_t(first) * k;
    void* om = static_cast<uint8_t*>(out) + size_t(first) * out_row;
    pack_x_kernel<<<dim3(k / 32, groups), 32, 0, s>>>(reinterpret_cast<const uint32_t*>(xm), xp, m, k, groups);
    if (coarse) {
      launch_gemv<true>(groups, grid, s, head, xp, xm, om, partial, out_f32, m, n, k, kb_per_split, splits);
    } else {
      launch_gemv<false>(groups, grid, s, head, xp, xm, om, partial, out_f32, m, n, k, kb_per_split, splits);
    }
    if (splits > 1) {
      const size_t total = size_t(m) * n;
      const int blocks = int((total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096);
      reduce_kernel<<<blocks, 256, 0, s>>>(head, partial, xm, om, out_f32, coarse ? 0 : 1, m, n, k, splits);
    }
  }
  return cudaGetLastError();
}
