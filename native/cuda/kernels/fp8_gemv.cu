// W8A16 linear for skinny row counts (drafters, FP8 LM heads): E4M3 weights
// with one FP32 scale per output row and 128-wide K block ([n, k / 128], the
// fp8_row_quant layout), BF16 activations, FP32 accumulation on tensor cores.
//
// Weights are packed at load into mma.m16n8k16 A-fragment order: tiles of
// 16 rows x 32 k, each 32 lanes x 16 bytes, so a warp streams its tile row
// with one coalesced 16-byte load per lane and converts E4M3 pairs straight
// to the f16x2 registers the MMA takes (exact: every E4M3 value is an f16).
// Activations take the MMA's B side (swap AB: out^T = W x^T): each row is
// scaled by a power of two into f16 range (exact within it) and packed in
// B-fragment order once per call. Rows beyond 64 run in chunks of 64.
// Split-K (a deterministic second pass) keeps narrow outputs busy.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kWarps = 4;
constexpr int kMaxRows = 64;

// Smallest power of two >= x (x > 0).
__device__ __forceinline__ float pow2_ceil(float x) {
  int e;
  const float m = frexpf(x, &e);  // x = m * 2^e, m in [0.5, 1)
  return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

// Scale of a 128-value block with absolute maximum `amax`: amax / 448, or
// with `pow2` the smallest power of two >= amax / 448 (BF16 weights with at
// most E4M3's 3 mantissa bits then quantize exactly); 1 for an all-zero block.
__device__ __forceinline__ float block_scale(float amax, int pow2) {
  if (!(amax > 0.0f)) return 1.0f;
  const float s = amax / 448.0f;
  return pow2 ? pow2_ceil(s) : s;
}

__device__ __forceinline__ uint8_t e4m3(float x) {
  return __nv_fp8_e4m3(x).__x;
}

// One warp per (16-row tile, 128-wide K block): per-row scales, then the
// block's four 16 x 32 tiles in A-fragment order.
__global__ void pack_kernel(const __nv_bfloat16* __restrict__ w, uint8_t* __restrict__ packed,
                            float* __restrict__ scale, int n, int k, int pow2) {
  const int kb = blockIdx.x, rt = blockIdx.y, lane = threadIdx.x;
  const int kbs = k / 128;
  __shared__ float s_scale[16];
  for (int r = 0; r < 16; ++r) {
    const __nv_bfloat16* row = w + size_t(rt * 16 + r) * k + kb * 128 + lane * 4;
    float amax = 0.0f;
    for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(__bfloat162float(row[j])));
    for (int offset = 16; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    if (lane == 0) {
      const float s = block_scale(amax, pow2);
      s_scale[r] = s;
      scale[size_t(rt * 16 + r) * kbs + kb] = s;
    }
  }
  __syncwarp();
  const int g = lane / 4, c = (lane % 4) * 2;
  const float s_lo = s_scale[g], s_hi = s_scale[g + 8];
  const __nv_bfloat16* lo = w + size_t(rt * 16 + g) * k;
  const __nv_bfloat16* hi = w + size_t(rt * 16 + g + 8) * k;
  auto q = [&](const __nv_bfloat16* row, float s, int col) { return e4m3(__bfloat162float(row[col]) / s); };
  for (int kt = 0; kt < 4; ++kt) {
    uint8_t out[16];
    for (int step = 0; step < 2; ++step) {
      const int k0 = kb * 128 + kt * 32 + step * 16 + c;
      uint8_t* o = out + step * 8;
      o[0] = q(lo, s_lo, k0);
      o[1] = q(lo, s_lo, k0 + 1);
      o[2] = q(hi, s_hi, k0);
      o[3] = q(hi, s_hi, k0 + 1);
      o[4] = q(lo, s_lo, k0 + 8);
      o[5] = q(lo, s_lo, k0 + 9);
      o[6] = q(hi, s_hi, k0 + 8);
      o[7] = q(hi, s_hi, k0 + 9);
    }
    uint4 v;
    memcpy(&v, out, 16);
    reinterpret_cast<uint4*>(packed)[(size_t(rt) * (k / 32) + kb * 4 + kt) * 32 + lane] = v;
  }
}

// Per activation row: a power of two that brings its absolute maximum to at
// most 2^14 (f16 holds it exactly), and its inverse for the output.
__global__ void row_scale_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ sx, int rows, int k) {
  const int m = blockIdx.x;
  float amax = 0.0f;
  if (m < rows) {
    for (int i = threadIdx.x; i < k; i += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(x[size_t(m) * k + i])));
  }
  __shared__ float warp_max[32];
  for (int offset = 16; offset; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < int(blockDim.x / 32); ++i) amax = fmaxf(amax, warp_max[i]);
    amax = fmaxf(amax, warp_max[0]);
    // amax * s <= 2^14; a zero (or padding) row keeps 1.
    const float s = amax > 0.0f ? 16384.0f / pow2_ceil(amax) : 1.0f;
    sx[m] = s;
    sx[kMaxRows + m] = 1.0f / s;
  }
}

// Activations [rows, k] -> f16 B fragments: per 32-wide k tile and 8-row
// group, 32 lanes x 16 bytes (two k16 steps of b0b1, b2b3).
__global__ void pack_x_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ sx,
                              uint4* __restrict__ xp, int rows, int k, int groups) {
  const int kt = blockIdx.x, group = blockIdx.y, lane = threadIdx.x;
  const int m = group * 8 + lane / 4, c = (lane % 4) * 2;
  const float s = sx[m];
  auto h = [&](int col) -> __half {
    return m < rows ? __float2half_rn(__bfloat162float(x[size_t(m) * k + col]) * s) : __float2half_rn(0.0f);
  };
  __half v[8];
  for (int step = 0; step < 2; ++step) {
    const int k0 = kt * 32 + step * 16 + c;
    v[step * 4 + 0] = h(k0);
    v[step * 4 + 1] = h(k0 + 1);
    v[step * 4 + 2] = h(k0 + 8);
    v[step * 4 + 3] = h(k0 + 9);
  }
  uint4 out;
  memcpy(&out, v, 16);
  xp[(size_t(kt) * groups + group) * 32 + lane] = out;
}

__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint32_t pair) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
  return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 load_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// One warp per (16-row tile, K split). GROUPS 8-row activation groups.
template <int GROUPS>
__global__ void __launch_bounds__(kWarps * 32) gemv_kernel(
    const uint4* __restrict__ wp, const float* __restrict__ scale, const uint4* __restrict__ xp,
    const float* __restrict__ sx, void* __restrict__ out, float* __restrict__ partial, int out_f32, int rows, int n,
    int k, int kb_per_split, int splits) {
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * kWarps + threadIdx.x / 32;
  const int tiles = n / 16;
  if (warp >= tiles * splits) return;
  const int rt = warp % tiles, split = warp / tiles;
  const int kbs = k / 128;
  const int kb0 = split * kb_per_split, kb1 = min(kbs, kb0 + kb_per_split);
  const int g = lane / 4, c = (lane % 4) * 2;
  float acc[GROUPS][4];
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
  const uint4* wrow = wp + size_t(rt) * (k / 32) * 32 + lane;
  const float* s_lo = scale + size_t(rt * 16 + g) * kbs;
  const float* s_hi = scale + size_t(rt * 16 + g + 8) * kbs;
  uint4 a[4];
  if (kb0 < kb1) {
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) a[kt] = load_stream(wrow + size_t(kb0 * 4 + kt) * 32);
  }
  for (int kb = kb0; kb < kb1; ++kb) {
    uint4 next[4];
    if (kb + 1 < kb1) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) next[kt] = load_stream(wrow + size_t((kb + 1) * 4 + kt) * 32);
    }
    float blk[GROUPS][4];
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) blk[i][0] = blk[i][1] = blk[i][2] = blk[i][3] = 0.0f;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const uint32_t a0[4] = {fp8x2_to_f16x2(a[kt].x & 0xffff), fp8x2_to_f16x2(a[kt].x >> 16),
                              fp8x2_to_f16x2(a[kt].y & 0xffff), fp8x2_to_f16x2(a[kt].y >> 16)};
      const uint32_t a1[4] = {fp8x2_to_f16x2(a[kt].z & 0xffff), fp8x2_to_f16x2(a[kt].z >> 16),
                              fp8x2_to_f16x2(a[kt].w & 0xffff), fp8x2_to_f16x2(a[kt].w >> 16)};
      const uint4* b = xp + (size_t(kb * 4 + kt) * GROUPS) * 32 + lane;
#pragma unroll
      for (int i = 0; i < GROUPS; ++i) {
        const uint4 bv = __ldg(b + i * 32);
        mma(blk[i], a0, bv.x, bv.y);
        mma(blk[i], a1, bv.z, bv.w);
      }
    }
    const float lo = __ldg(s_lo + kb), hi = __ldg(s_hi + kb);
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) {
      acc[i][0] += blk[i][0] * lo;
      acc[i][1] += blk[i][1] * lo;
      acc[i][2] += blk[i][2] * hi;
      acc[i][3] += blk[i][3] * hi;
    }
    if (kb + 1 < kb1) {
#pragma unroll
      for (int kt = 0; kt < 4; ++kt) a[kt] = next[kt];
    }
  }
  // C fragment: c0, c1 weight row g, activation rows c, c + 1; c2, c3 weight row g + 8.
  const int row_lo = rt * 16 + g, row_hi = row_lo + 8;
#pragma unroll
  for (int i = 0; i < GROUPS; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int m = i * 8 + c + (j & 1);
      const int col = j < 2 ? row_lo : row_hi;
      if (m >= rows) continue;
      if (splits > 1) {
        partial[(size_t(split) * rows + m) * n + col] = acc[i][j];
      } else {
        const float v = acc[i][j] * sx[kMaxRows + m];
        if (out_f32) {
          static_cast<float*>(out)[size_t(m) * n + col] = v;
        } else {
          static_cast<__nv_bfloat16*>(out)[size_t(m) * n + col] = __float2bfloat16(v);
        }
      }
    }
  }
}

__global__ void reduce_kernel(const float* __restrict__ partial, const float* __restrict__ sx, void* __restrict__ out,
                              int out_f32, int rows, int n, int splits) {
  const size_t total = size_t(rows) * n;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    float v = 0.0f;
    for (int s = 0; s < splits; ++s) v += partial[size_t(s) * total + i];
    v *= sx[kMaxRows + int(i / n)];
    if (out_f32) {
      static_cast<float*>(out)[i] = v;
    } else {
      static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16(v);
    }
  }
}

int sm_count() {
  static int count = 0;
  if (count == 0) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    if (count <= 0) count = 128;
  }
  return count;
}

void split_plan(int n, int k, int* kb_per_split, int* splits) {
  const int tiles = n / 16, kbs = k / 128;
  const int wanted = (sm_count() * 16 + tiles - 1) / tiles;
  int s = wanted < 1 ? 1 : (wanted > kbs ? kbs : wanted);
  *kb_per_split = (kbs + s - 1) / s;
  *splits = (kbs + *kb_per_split - 1) / *kb_per_split;
}

size_t align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

}  // namespace

// Packs a BF16 [n, k] weight (n % 16 == 0, k % 128 == 0) into `packed`
// (n * k bytes, A-fragment order) and `scale` ([n, k / 128] FP32).
extern "C" int32_t cuteafd_fp8_w8a16_pack(const void* w, void* packed, void* scale, int32_t n, int32_t k,
                                          int32_t pow2, void* stream) {
  if (n < 16 || n % 16 || k < 128 || k % 128) return cudaErrorInvalidValue;
  pack_kernel<<<dim3(k / 128, n / 16), 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(w), static_cast<uint8_t*>(packed), static_cast<float*>(scale), n, k, pow2);
  return cudaGetLastError();
}

// Scratch bytes cuteafd_fp8_w8a16_linear needs for these shapes.
extern "C" size_t cuteafd_fp8_w8a16_workspace(int32_t rows, int32_t k, int32_t n) {
  if (n < 16 || k < 128) return 0;
  const int chunk = rows < kMaxRows ? rows : kMaxRows;
  int kb_per_split, splits;
  split_plan(n, k, &kb_per_split, &splits);
  return align256(2 * kMaxRows * sizeof(float)) + align256(size_t(k) * kMaxRows * 2) +
         (splits > 1 ? align256(size_t(splits) * chunk * n * sizeof(float)) : 0);
}

// out [rows, n] (BF16, or FP32 with out_f32) = x [rows, k] BF16 @ W^T for a
// packed weight; `workspace` holds cuteafd_fp8_w8a16_workspace bytes.
extern "C" int32_t cuteafd_fp8_w8a16_linear(const void* x, const void* packed, const void* scale, void* out,
                                            int32_t out_f32, int32_t rows, int32_t k, int32_t n, void* workspace,
                                            size_t workspace_bytes, void* stream) {
  if (rows < 1 || n < 16 || n % 16 || k < 128 || k % 128) return cudaErrorInvalidValue;
  if (workspace_bytes < cuteafd_fp8_w8a16_workspace(rows, k, n)) return cudaErrorInvalidValue;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  float* sx = reinterpret_cast<float*>(ws);
  uint4* xp = reinterpret_cast<uint4*>(ws + align256(2 * kMaxRows * sizeof(float)));
  float* partial = reinterpret_cast<float*>(ws + align256(2 * kMaxRows * sizeof(float)) +
                                            align256(size_t(k) * kMaxRows * 2));
  int kb_per_split, splits;
  split_plan(n, k, &kb_per_split, &splits);
  const size_t out_row = size_t(n) * (out_f32 ? 4 : 2);
  for (int first = 0; first < rows; first += kMaxRows) {
    const int m = rows - first < kMaxRows ? rows - first : kMaxRows;
    const int groups = m <= 8 ? 1 : m <= 16 ? 2 : m <= 32 ? 4 : 8;
    const __nv_bfloat16* xm = static_cast<const __nv_bfloat16*>(x) + size_t(first) * k;
    void* om = static_cast<uint8_t*>(out) + size_t(first) * out_row;
    row_scale_kernel<<<groups * 8, 256, 0, s>>>(xm, sx, m, k);
    pack_x_kernel<<<dim3(k / 32, groups), 32, 0, s>>>(xm, sx, xp, m, k, groups);
    const int warps = (n / 16) * splits;
    const dim3 grid((warps + kWarps - 1) / kWarps), block(kWarps * 32);
    const uint4* wp = static_cast<const uint4*>(packed);
    const float* sc = static_cast<const float*>(scale);
    switch (groups) {
      case 1: gemv_kernel<1><<<grid, block, 0, s>>>(wp, sc, xp, sx, om, partial, out_f32, m, n, k, kb_per_split, splits); break;
      case 2: gemv_kernel<2><<<grid, block, 0, s>>>(wp, sc, xp, sx, om, partial, out_f32, m, n, k, kb_per_split, splits); break;
      case 4: gemv_kernel<4><<<grid, block, 0, s>>>(wp, sc, xp, sx, om, partial, out_f32, m, n, k, kb_per_split, splits); break;
      default: gemv_kernel<8><<<grid, block, 0, s>>>(wp, sc, xp, sx, om, partial, out_f32, m, n, k, kb_per_split, splits); break;
    }
    if (splits > 1) {
      const size_t total = size_t(m) * n;
      const int blocks = int((total + 255) / 256 < 4096 ? (total + 255) / 256 : 4096);
      reduce_kernel<<<blocks, 256, 0, s>>>(partial, sx, om, out_f32, m, n, splits);
    }
  }
  return cudaGetLastError();
}
