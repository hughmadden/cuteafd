// Component probe, not wired: an SM89 V4.1 mHC boundary that reproduces
// today's SM89 decode arithmetic bit for bit at every row count, for timing
// against the native begin (mixes + pre + RMSNorm) and SparkInfer's lagged
// begin exported for sm_89 (v41_hc_lagged_probe.cc).
//
// Begin (begin_kernel / begin_p1_kernel), one launch:
//  - projection CTAs: the 24 FP32 projection partials of the residual in the
//    split-8 arithmetic of the hc_project AOT (lane-strided fma.rn chain over
//    80 steps per 2560-wide K part, then the shfl_down tree), each fn slice
//    read once for all rows of the CTA's row tile;
//  - one norm CTA per row: the collapse with the incoming coefficients and its
//    RMSNorm in the arithmetic of pre_kernel + ds41_rmsnorm_cached_bf16_kernel<5120>.
// Finish: finish_mixes_kernel<8> (verbatim), on two warps (finish2_kernel), at
// the end of the begin (last-arriver mode), or folded into post
// (post_finish_kernel: post_vec4_kernel's arithmetic after the row's finish).
//
// Build: nvcc -O3 -DNDEBUG -arch=sm_89 -shared -Xcompiler -fPIC
//   native/tests/v41_hc_sm89_probe.cu -o libv41_hc_sm89_probe.so
// Driver: python/tools/qualify/deepseek_v41/bench_v41_hc_sm89.py.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int kHidden = 5120, kK = 20480, kParts = 8, kPartK = 2560, kSteps = 80;
constexpr int kNormThreads = 256;  // ds41_rmsnorm_cached_bf16_kernel's kBlock

__device__ __forceinline__ float bf16_to_f32(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}
__device__ __forceinline__ uint16_t f32_to_bf16(float value) {
  const uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}
__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset; offset >>= 1) value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// Rows NR of one warp's projections PW over this CTA's K part. Chains run in
// step order from zero exactly as the AOT lane loop; SQ adds the residual
// square chain, which is the same for every projection, stored to all 24.
template <int PW, int NR, bool SQ>
__device__ __forceinline__ void chains(const uint16_t* __restrict__ s_x, const float (&w)[PW][kSteps],
    int lane, float2* __restrict__ partials, int row, int p0, int part) {
  float acc[PW][NR], square[NR];
  #pragma unroll
  for (int n = 0; n < NR; ++n) {
    square[n] = 0.0f;
    #pragma unroll
    for (int q = 0; q < PW; ++q) acc[q][n] = 0.0f;
  }
  #pragma unroll
  for (int j = 0; j < kSteps; ++j) {
    float x[NR];
    #pragma unroll
    for (int n = 0; n < NR; ++n) x[n] = bf16_to_f32(s_x[n * kPartK + 32 * j + lane]);
    #pragma unroll
    for (int q = 0; q < PW; ++q)
      #pragma unroll
      for (int n = 0; n < NR; ++n) acc[q][n] = fmaf(x[n], w[q][j], acc[q][n]);
    if constexpr (SQ) {
      #pragma unroll
      for (int n = 0; n < NR; ++n) square[n] = fmaf(x[n], x[n], square[n]);
    }
  }
  #pragma unroll
  for (int q = 0; q < PW; ++q)
    #pragma unroll
    for (int n = 0; n < NR; ++n) {
      const float dot = warp_sum(acc[q][n]);
      if (lane == 0) partials[((row + n) * 24 + p0 + q) * kParts + part].x = dot;
    }
  if constexpr (SQ) {
    #pragma unroll
    for (int n = 0; n < NR; ++n) {
      const float total = __shfl_sync(0xffffffffu, warp_sum(square[n]), 0);
      if (lane < 24) partials[((row + n) * 24 + lane) * kParts + part].y = total;
    }
  }
}

// pre_kernel + ds41_rmsnorm_cached_bf16_kernel<5120> for one row; T threads
// emulate the norm's 256 (each owns columns t, t+256, ...).
template <int T>
__device__ __forceinline__ void norm_row(const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ incoming, const uint16_t* __restrict__ weight,
    uint16_t* __restrict__ out, int row, float eps, float* scratch) {
  constexpr int VT = kNormThreads / T, kPer = kHidden / kNormThreads;
  static_assert(VT * T == kNormThreads);
  const int tid = threadIdx.x;
  const float* pre = incoming + uint64_t(row) * 4;
  const float pre0 = pre[0], pre1 = pre[1], pre2 = pre[2], pre3 = pre[3];
  const __nv_bfloat16* base = residual + uint64_t(row) * 4 * kHidden;
  float values[VT][kPer];
  #pragma unroll
  for (int v = 0; v < VT; ++v) {
    const int vt = tid + v * T;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int col = vt + i * kNormThreads;
      float s = 0;
      s = __fadd_rn(s, __fmul_rn(pre0, __bfloat162float(base[0 * kHidden + col])));
      s = __fadd_rn(s, __fmul_rn(pre1, __bfloat162float(base[1 * kHidden + col])));
      s = __fadd_rn(s, __fmul_rn(pre2, __bfloat162float(base[2 * kHidden + col])));
      s = __fadd_rn(s, __fmul_rn(pre3, __bfloat162float(base[3 * kHidden + col])));
      values[v][i] = __bfloat162float(__float2bfloat16_rn(s));
      sum += values[v][i] * values[v][i];
    }
    scratch[vt] = sum;
  }
  __syncthreads();
  if (tid < 32) {
    float partial[kNormThreads / 32];
    #pragma unroll
    for (int i = 0; i < kNormThreads / 32; ++i) partial[i] = scratch[tid + i * 32];
    #pragma unroll
    for (int stride = kNormThreads / 64; stride > 0; stride >>= 1) {
      #pragma unroll
      for (int i = 0; i < stride; ++i) partial[i] += partial[i + stride];
    }
    float reduced = partial[0];
    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, reduced, stride);
      if (tid < stride) reduced += other;
    }
    if (tid == 0) scratch[0] = reduced;
  }
  __syncthreads();
  const float inv = rsqrtf(scratch[0] / static_cast<float>(kHidden) + eps);
  #pragma unroll
  for (int v = 0; v < VT; ++v)
    #pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int col = tid + v * T + i * kNormThreads;
      out[uint64_t(row) * kHidden + col] = f32_to_bf16(values[v][i] * inv * bf16_to_f32(weight[col]));
    }
}

// finish_mixes_kernel<8>'s arithmetic for one row on one warp (L2 reads of the
// partials other CTAs wrote; __syncwarp in place of the one-warp block barrier).
__device__ __forceinline__ void finish_row(const float* partials, const float* scale,
    const float* base, float* pre, float* post, float* comb, uint64_t row, int lane, float* projected) {
  if(lane<24) {
    float dot=0, square=0;
#pragma unroll
    for(int part=0;part<8;++part) {
      const uint64_t at=((row*24+lane)*8+part)*2;
      dot=__fadd_rn(dot,__ldcg(partials+at));
      square=__fadd_rn(square,__ldcg(partials+at+1));
    }
    projected[lane]=dot*rsqrtf(square/20480.0f+1e-20f);
  }
  __syncwarp();
  if(lane<4) {
    const int j=lane;
    const float a=fmaf(projected[j],scale[0],base[j]);
    const float b=fmaf(projected[j+4],scale[1],base[j+4]);
    pre[row*4+j]=1.0f/(1.0f+expf(-a))+1e-6f;
    post[row*4+j]=2.0f/(1.0f+expf(-b));
  }
  {
    float value=lane<16 ? fmaf(projected[8+lane],scale[2],base[8+lane]) : 0;
    float maximum=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,1));
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,2));
    value=expf(value-maximum);
    float sum=value+__shfl_xor_sync(0xffffffffu,value,1);
    sum+=__shfl_xor_sync(0xffffffffu,sum,2);
    value=value/sum+1e-6f;
    sum=value+__shfl_xor_sync(0xffffffffu,value,4);
    sum+=__shfl_xor_sync(0xffffffffu,sum,8);
    value/=sum+1e-6f;
    for(int iteration=1;iteration<20;++iteration) {
      sum=value+__shfl_xor_sync(0xffffffffu,value,1);
      sum+=__shfl_xor_sync(0xffffffffu,sum,2);
      value/=sum+1e-6f;
      sum=value+__shfl_xor_sync(0xffffffffu,value,4);
      sum+=__shfl_xor_sync(0xffffffffu,sum,8);
      value/=sum+1e-6f;
    }
    if(lane<16) comb[row*16+lane]=value;
  }
}

__device__ __forceinline__ void cp_async16(void* shared, const void* global) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address), "l"(global) : "memory");
}

struct Finish {
  const float* scale; const float* base; float* pre; float* post; float* comb; unsigned* counters;
};

// Grid: projection CTAs (part, projection tile, row tile), then one norm CTA per row.
template <int W, int PW, bool FS, bool FIN>
__global__ void __launch_bounds__(W * 32, 1)
begin_kernel(const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
             const float* __restrict__ incoming, const uint16_t* __restrict__ weight,
             float2* __restrict__ partials, uint16_t* __restrict__ normalized,
             int rows, int rows_per_cta, int projection_ctas, float eps, Finish finish) {
  extern __shared__ __align__(16) unsigned char smem[];
  if (int(blockIdx.x) >= projection_ctas) {
    norm_row<W * 32>(residual, incoming, weight, normalized, int(blockIdx.x) - projection_ctas, eps,
                     reinterpret_cast<float*>(smem));
    return;
  }
  constexpr int PT = W * PW, kTiles = 24 / PT;
  static_assert(kTiles * PT == 24);
  const int cta = blockIdx.x, part = cta % kParts, tile = (cta / kParts) % kTiles;
  const int row0 = (cta / (kParts * kTiles)) * rows_per_cta;
  const int count = min(rows_per_cta, rows - row0);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int p0 = tile * PT + warp * PW, k0 = part * kPartK;
  float w[PW][kSteps];
  uint16_t* s_x = reinterpret_cast<uint16_t*>(smem + (FS ? PT * kPartK * 4 : 0));
  const uint4* source = reinterpret_cast<const uint4*>(residual);
  if constexpr (FS) {
    // 16-byte copies of this CTA's fn slices, then lane-strided register fills.
    float* s_fn = reinterpret_cast<float*>(smem);
    for (int index = threadIdx.x; index < PT * (kPartK / 4); index += W * 32) {
      const int q = index / (kPartK / 4), c = index % (kPartK / 4);
      cp_async16(s_fn + index * 4, fn + uint64_t(tile * PT + q) * kK + k0 + c * 4);
    }
    for (int index = threadIdx.x; index < count * (kPartK / 8); index += W * 32) {
      const int r = index / (kPartK / 8), c = index % (kPartK / 8);
      cp_async16(reinterpret_cast<uint4*>(s_x) + index, source + (uint64_t(row0 + r) * kK + k0) / 8 + c);
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    #pragma unroll
    for (int q = 0; q < PW; ++q)
      #pragma unroll
      for (int j = 0; j < kSteps; ++j) w[q][j] = s_fn[(warp * PW + q) * kPartK + 32 * j + lane];
  } else {
    #pragma unroll
    for (int q = 0; q < PW; ++q)
      #pragma unroll
      for (int j = 0; j < kSteps; ++j) w[q][j] = __ldg(fn + uint64_t(p0 + q) * kK + k0 + 32 * j + lane);
    for (int index = threadIdx.x; index < count * (kPartK / 8); index += W * 32) {
      const int r = index / (kPartK / 8), c = index % (kPartK / 8);
      reinterpret_cast<uint4*>(s_x)[index] = __ldg(source + (uint64_t(row0 + r) * kK + k0) / 8 + c);
    }
    __syncthreads();
  }
  for (int r = 0; r < count; r += 2) {
    const bool square = tile == 0 && ((r >> 1) % W) == warp;
    const uint16_t* x = s_x + r * kPartK;
    if (r + 1 < count) {
      if (square) chains<PW, 2, true>(x, w, lane, partials, row0 + r, p0, part);
      else chains<PW, 2, false>(x, w, lane, partials, row0 + r, p0, part);
    } else {
      if (square) chains<PW, 1, true>(x, w, lane, partials, row0 + r, p0, part);
      else chains<PW, 1, false>(x, w, lane, partials, row0 + r, p0, part);
    }
  }
  if constexpr (FIN) {
    // The last of the 8 * kTiles projection CTAs of a row finishes it.
    __shared__ int last[18];
    __shared__ float projected[W][24];
    __threadfence();
    __syncthreads();
    if (int(threadIdx.x) < count) {
      __threadfence();
      last[threadIdx.x] = atomicAdd(finish.counters + row0 + threadIdx.x, 1u) == kParts * kTiles - 1;
    }
    __syncthreads();
    for (int r = warp; r < count; r += W) {
      if (!last[r]) continue;
      __threadfence();
      finish_row(reinterpret_cast<const float*>(partials), finish.scale, finish.base, finish.pre,
                 finish.post, finish.comb, uint64_t(row0 + r), lane, projected[warp]);
      if (lane == 0) finish.counters[row0 + r] = 0;
    }
  }
}

// One (projection, part) per CTA: 192 CTAs per row tile stream fn's 1.97 MB at
// once; the warps split the tile's rows in pairs; part-0..7 squares come from
// the projection-0 CTAs. Same chains and stores as begin_kernel.
template <int W>
__global__ void __launch_bounds__(W * 32, 2)
begin_p1_kernel(const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
                const float* __restrict__ incoming, const uint16_t* __restrict__ weight,
                float2* __restrict__ partials, uint16_t* __restrict__ normalized,
                int rows, int rows_per_cta, int projection_ctas, float eps) {
  extern __shared__ __align__(16) unsigned char smem[];
  if (int(blockIdx.x) >= projection_ctas) {
    norm_row<W * 32>(residual, incoming, weight, normalized, int(blockIdx.x) - projection_ctas, eps,
                     reinterpret_cast<float*>(smem));
    return;
  }
  const int cta = blockIdx.x, part = cta % kParts, p = (cta / kParts) % 24;
  const int row0 = (cta / (kParts * 24)) * rows_per_cta, count = min(rows_per_cta, rows - row0);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, k0 = part * kPartK;
  float* s_fn = reinterpret_cast<float*>(smem);
  uint16_t* s_x = reinterpret_cast<uint16_t*>(smem + kPartK * 4);
  for (int index = threadIdx.x; index < kPartK / 4; index += W * 32)
    cp_async16(s_fn + index * 4, fn + uint64_t(p) * kK + k0 + index * 4);
  const uint4* source = reinterpret_cast<const uint4*>(residual);
  for (int index = threadIdx.x; index < count * (kPartK / 8); index += W * 32) {
    const int r = index / (kPartK / 8), c = index % (kPartK / 8);
    cp_async16(reinterpret_cast<uint4*>(s_x) + index, source + (uint64_t(row0 + r) * kK + k0) / 8 + c);
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  float w[1][kSteps];
  #pragma unroll
  for (int j = 0; j < kSteps; ++j) w[0][j] = s_fn[32 * j + lane];
  for (int r = 2 * warp; r < count; r += 2 * W) {
    const uint16_t* x = s_x + r * kPartK;
    if (r + 1 < count) {
      if (p == 0) chains<1, 2, true>(x, w, lane, partials, row0 + r, p, part);
      else chains<1, 2, false>(x, w, lane, partials, row0 + r, p, part);
    } else {
      if (p == 0) chains<1, 1, true>(x, w, lane, partials, row0 + r, p, part);
      else chains<1, 1, false>(x, w, lane, partials, row0 + r, p, part);
    }
  }
}

// finish_mixes_kernel<8>'s arithmetic on two warps: warp 1 computes pre/post
// while warp 0 runs Sinkhorn, instead of one after the other.
__global__ void __launch_bounds__(64)
finish2_kernel(const float* partials, const float* scale, const float* base, float* pre,
               float* post, float* comb) {
  const uint64_t row = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __shared__ float projected[24];
  if (warp == 0 && lane < 24) {
    float dot=0, square=0;
#pragma unroll
    for(int part=0;part<8;++part) {
      const uint64_t at=((row*24+lane)*8+part)*2;
      dot=__fadd_rn(dot,partials[at]);
      square=__fadd_rn(square,partials[at+1]);
    }
    projected[lane]=dot*rsqrtf(square/20480.0f+1e-20f);
  }
  __syncthreads();
  if (warp == 1) {
    if(lane<4) {
      const int j=lane;
      const float a=fmaf(projected[j],scale[0],base[j]);
      const float b=fmaf(projected[j+4],scale[1],base[j+4]);
      pre[row*4+j]=1.0f/(1.0f+expf(-a))+1e-6f;
      post[row*4+j]=2.0f/(1.0f+expf(-b));
    }
    return;
  }
  float value=lane<16 ? fmaf(projected[8+lane],scale[2],base[8+lane]) : 0;
  float maximum=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,1));
  maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,2));
  value=expf(value-maximum);
  float sum=value+__shfl_xor_sync(0xffffffffu,value,1);
  sum+=__shfl_xor_sync(0xffffffffu,sum,2);
  value=value/sum+1e-6f;
  sum=value+__shfl_xor_sync(0xffffffffu,value,4);
  sum+=__shfl_xor_sync(0xffffffffu,sum,8);
  value/=sum+1e-6f;
  for(int iteration=1;iteration<20;++iteration) {
    sum=value+__shfl_xor_sync(0xffffffffu,value,1);
    sum+=__shfl_xor_sync(0xffffffffu,sum,2);
    value/=sum+1e-6f;
    sum=value+__shfl_xor_sync(0xffffffffu,value,4);
    sum+=__shfl_xor_sync(0xffffffffu,sum,8);
    value/=sum+1e-6f;
  }
  if(lane<16) comb[row*16+lane]=value;
}

__device__ __forceinline__ uint2 ld_u2(const void* p) {
  uint2 v;
  asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ void st_u2(void* p, uint2 v) {
  asm volatile("st.global.wb.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(v.x), "r"(v.y));
}
__device__ __forceinline__ float lo16(uint32_t u) { return __bfloat162float(__ushort_as_bfloat16(uint16_t(u))); }
__device__ __forceinline__ float hi16(uint32_t u) { return __bfloat162float(__ushort_as_bfloat16(uint16_t(u >> 16))); }
// compute_post_scalar from v41_hc.cu (= post_kernel's per-element arithmetic).
__device__ __forceinline__ __nv_bfloat16 post_scalar(float r0, float r1, float r2, float r3, float sub,
    float c0, float c1, float c2, float c3, float p) {
  float sum = 0.0f;
  sum = __fadd_rn(sum, __fmul_rn(c0, r0));
  sum = __fadd_rn(sum, __fmul_rn(c1, r1));
  sum = __fadd_rn(sum, __fmul_rn(c2, r2));
  sum = __fadd_rn(sum, __fmul_rn(c3, r3));
  const float expansion = __fmul_rn(p, sub);
  return __float2bfloat16_rn(__fadd_rn(expansion, sum));
}
__device__ __forceinline__ uint32_t post_pair(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t sub,
    float c0, float c1, float c2, float c3, float p) {
  const __nv_bfloat16 a = post_scalar(lo16(r0), lo16(r1), lo16(r2), lo16(r3), lo16(sub), c0, c1, c2, c3, p);
  const __nv_bfloat16 b = post_scalar(hi16(r0), hi16(r1), hi16(r2), hi16(r3), hi16(sub), c0, c1, c2, c3, p);
  return uint32_t(__bfloat16_as_ushort(a)) | (uint32_t(__bfloat16_as_ushort(b)) << 16);
}

// The boundary's finish moved after the sublayer: each of a row's 5 CTAs runs
// finish_mixes_kernel<8>'s arithmetic (warp 0 Sinkhorn, warp 1 pre/post) into
// shared memory while its residual loads are in flight, then post_vec4_kernel's
// arithmetic. CTA 0 of the row publishes next pre, post and comb.
__global__ void __launch_bounds__(256)
post_finish_kernel(const __nv_bfloat16* __restrict__ sublayer, const __nv_bfloat16* __restrict__ residual,
                   const float* __restrict__ partials, const float* __restrict__ scale,
                   const float* __restrict__ base, float* __restrict__ pre, float* __restrict__ post,
                   float* __restrict__ comb, __nv_bfloat16* __restrict__ output) {
  const uint64_t row = blockIdx.y;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const bool publish = blockIdx.x == 0;
  __shared__ float projected[24], s_post[4], s_comb[16];
  const uint32_t col = (blockIdx.x * 256 + threadIdx.x) * 4;
  const uint64_t r_idx = row * 20480 + col, s_idx = row * 5120 + col;
  const uint2 sub = ld_u2(sublayer + s_idx);
  const uint2 r0 = ld_u2(residual + r_idx), r1 = ld_u2(residual + r_idx + 5120);
  const uint2 r2 = ld_u2(residual + r_idx + 10240), r3 = ld_u2(residual + r_idx + 15360);
  if (warp == 0 && lane < 24) {
    float dot=0, square=0;
#pragma unroll
    for(int part=0;part<8;++part) {
      const uint64_t at=((row*24+lane)*8+part)*2;
      dot=__fadd_rn(dot,partials[at]);
      square=__fadd_rn(square,partials[at+1]);
    }
    projected[lane]=dot*rsqrtf(square/20480.0f+1e-20f);
  }
  __syncthreads();
  if (warp == 1 && lane < 4) {
    const int j=lane;
    const float a=fmaf(projected[j],scale[0],base[j]);
    const float b=fmaf(projected[j+4],scale[1],base[j+4]);
    const float p_=1.0f/(1.0f+expf(-a))+1e-6f;
    const float q_=2.0f/(1.0f+expf(-b));
    s_post[j] = q_;
    if (publish) { pre[row*4+j]=p_; post[row*4+j]=q_; }
  } else if (warp == 0) {
    float value=lane<16 ? fmaf(projected[8+lane],scale[2],base[8+lane]) : 0;
    float maximum=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,1));
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,2));
    value=expf(value-maximum);
    float sum=value+__shfl_xor_sync(0xffffffffu,value,1);
    sum+=__shfl_xor_sync(0xffffffffu,sum,2);
    value=value/sum+1e-6f;
    sum=value+__shfl_xor_sync(0xffffffffu,value,4);
    sum+=__shfl_xor_sync(0xffffffffu,sum,8);
    value/=sum+1e-6f;
    for(int iteration=1;iteration<20;++iteration) {
      sum=value+__shfl_xor_sync(0xffffffffu,value,1);
      sum+=__shfl_xor_sync(0xffffffffu,sum,2);
      value/=sum+1e-6f;
      sum=value+__shfl_xor_sync(0xffffffffu,value,4);
      sum+=__shfl_xor_sync(0xffffffffu,sum,8);
      value/=sum+1e-6f;
    }
    if(lane<16) { s_comb[lane]=value; if (publish) comb[row*16+lane]=value; }
  }
  __syncthreads();
  uint2* out = reinterpret_cast<uint2*>(output + r_idx);
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    const float c0 = s_comb[0 + d], c1 = s_comb[4 + d], c2 = s_comb[8 + d], c3 = s_comb[12 + d], p_ = s_post[d];
    uint2 o;
    o.x = post_pair(r0.x, r1.x, r2.x, r3.x, sub.x, c0, c1, c2, c3, p_);
    o.y = post_pair(r0.y, r1.y, r2.y, r3.y, sub.y, c0, c1, c2, c3, p_);
    st_u2(out + d * 1280, o);
  }
}

// finish_mixes_kernel<8> from v41_hc.cu, verbatim.
template<int Split = 0>
__global__ void finish_mixes_kernel(const float* partials,
    const float* scale, const float* base, float* pre, float* post, float* comb) {
  const uint64_t row=blockIdx.x;
  const int lane=threadIdx.x, warp=0;
  __shared__ float projected[24];
  if(lane<24) {
    if constexpr (Split > 0) {
      float dot=0, square=0;
#pragma unroll
      for(int part=0;part<Split;++part) {
        const uint64_t at=((row*24+lane)*Split+part)*2;
        dot=__fadd_rn(dot,partials[at]);
        square=__fadd_rn(square,partials[at+1]);
      }
      projected[lane]=dot*rsqrtf(square/20480.0f+1e-20f);
    } else {
      projected[lane]=lane<4 ? pre[row*4+lane] :
          lane<8 ? post[row*4+lane-4] : comb[row*16+lane-8];
    }
  }
  __syncthreads();
  if(threadIdx.x<4) {
    const int j=threadIdx.x;
    const float a=fmaf(projected[j],scale[0],base[j]);
    const float b=fmaf(projected[j+4],scale[1],base[j+4]);
    pre[row*4+j]=1.0f/(1.0f+expf(-a))+1e-6f;
    post[row*4+j]=2.0f/(1.0f+expf(-b));
  }
  if(warp==0) {
    float value=lane<16 ? fmaf(projected[8+lane],scale[2],base[8+lane]) : 0;
    float maximum=fmaxf(value,__shfl_xor_sync(0xffffffffu,value,1));
    maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,2));
    value=expf(value-maximum);
    float sum=value+__shfl_xor_sync(0xffffffffu,value,1);
    sum+=__shfl_xor_sync(0xffffffffu,sum,2);
    value=value/sum+1e-6f;
    sum=value+__shfl_xor_sync(0xffffffffu,value,4);
    sum+=__shfl_xor_sync(0xffffffffu,sum,8);
    value/=sum+1e-6f;
    for(int iteration=1;iteration<20;++iteration) {
      sum=value+__shfl_xor_sync(0xffffffffu,value,1);
      sum+=__shfl_xor_sync(0xffffffffu,sum,2);
      value/=sum+1e-6f;
      sum=value+__shfl_xor_sync(0xffffffffu,value,4);
      sum+=__shfl_xor_sync(0xffffffffu,sum,8);
      value/=sum+1e-6f;
    }
    if(lane<16) comb[row*16+lane]=value;
  }
}

template <int W, int PW, bool FS, bool FIN>
int32_t launch(const void* residual, const void* fn, const void* incoming, const void* norm,
    void* partials, void* normalized, int rows, int rows_per_cta, bool projection, bool norms,
    Finish finish, cudaStream_t stream) {
  constexpr int PT = W * PW;
  const int row_tiles = (rows + rows_per_cta - 1) / rows_per_cta;
  const int projection_ctas = projection ? kParts * (24 / PT) * row_tiles : 0;
  size_t smem = size_t(rows_per_cta) * kPartK * 2 + (FS ? size_t(PT) * kPartK * 4 : 0);
  if (smem < 1024) smem = 1024;
  if (smem > 99 * 1024) return cudaErrorInvalidValue;
  auto kernel = begin_kernel<W, PW, FS, FIN>;
  if (smem > 48 * 1024) {
    const auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    if (status) return status;
  }
  kernel<<<projection_ctas + (norms ? rows : 0), W * 32, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(residual), static_cast<const float*>(fn),
      static_cast<const float*>(incoming), static_cast<const uint16_t*>(norm),
      static_cast<float2*>(partials), static_cast<uint16_t*>(normalized), rows, rows_per_cta,
      projection_ctas, 1e-20f, finish);
  return cudaGetLastError();
}
template <int W>
int32_t launch_p1(const void* residual, const void* fn, const void* incoming, const void* norm,
    void* partials, void* normalized, int rows, int rows_per_cta, bool projection, bool norms,
    cudaStream_t stream) {
  const int row_tiles = (rows + rows_per_cta - 1) / rows_per_cta;
  const int projection_ctas = projection ? kParts * 24 * row_tiles : 0;
  const size_t smem = size_t(kPartK) * 4 + size_t(rows_per_cta) * kPartK * 2;
  if (smem > 99 * 1024) return cudaErrorInvalidValue;
  auto kernel = begin_p1_kernel<W>;
  if (smem > 48 * 1024) {
    const auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    if (status) return status;
  }
  kernel<<<projection_ctas + (norms ? rows : 0), W * 32, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(residual), static_cast<const float*>(fn),
      static_cast<const float*>(incoming), static_cast<const uint16_t*>(norm),
      static_cast<float2*>(partials), static_cast<uint16_t*>(normalized), rows, rows_per_cta,
      projection_ctas, 1e-20f);
  return cudaGetLastError();
}
}  // namespace

// cfg = W*100 + PW*10 + FS selects the projection tiling (FS: fn through shared
// memory); rows_per_cta the row tile. mode 0: begin + finish kernels; 1: begin
// only; 2: finish only; 3: projection CTAs only; 4: norm CTAs only; 5: one
// launch, the last projection CTA of each row finishes it (counters: rows
// uint32, zero before first use, left zero).
extern "C" int32_t hc_begin_sm89(const void* residual, const void* fn, const void* scale,
    const void* base, const void* incoming, const void* norm, void* next_pre, void* post,
    void* comb, void* normalized, void* scratch, uint64_t scratch_bytes, int32_t rows,
    int32_t cfg, int32_t rows_per_cta, int32_t mode, void* counters, void* stream_handle) {
  if (rows < 1 || rows > 4096 || rows_per_cta < 1 || rows_per_cta > 18) return cudaErrorInvalidValue;
  if (scratch_bytes < uint64_t(rows) * 1536) return cudaErrorInvalidValue;
  if (mode == 5 && !counters) return cudaErrorInvalidValue;
  const auto stream = static_cast<cudaStream_t>(stream_handle);
  const Finish finish{static_cast<const float*>(scale), static_cast<const float*>(base),
      static_cast<float*>(next_pre), static_cast<float*>(post), static_cast<float*>(comb),
      static_cast<unsigned*>(counters)};
  // mode 6: begin + two-warp finish; 7: two-warp finish only.
  const bool projection = mode != 4, norms = mode != 3;
  if (mode == 7) {
    finish2_kernel<<<rows, 64, 0, stream>>>(static_cast<const float*>(scratch),
        static_cast<const float*>(scale), static_cast<const float*>(base),
        static_cast<float*>(next_pre), static_cast<float*>(post), static_cast<float*>(comb));
    return cudaGetLastError();
  }
  int32_t status = 0;
  if (mode != 2) {
#define CUTEAFD_LAUNCH(W, PW, FS) (mode == 5 \
    ? launch<W, PW, FS, true>(residual, fn, incoming, norm, scratch, normalized, rows, rows_per_cta, projection, norms, finish, stream) \
    : launch<W, PW, FS, false>(residual, fn, incoming, norm, scratch, normalized, rows, rows_per_cta, projection, norms, finish, stream))
    switch (cfg) {
      case 410: status = CUTEAFD_LAUNCH(4, 1, false); break;
      case 411: status = CUTEAFD_LAUNCH(4, 1, true); break;
      case 810: status = CUTEAFD_LAUNCH(8, 1, false); break;
      case 420: status = CUTEAFD_LAUNCH(4, 2, false); break;
      case 421: status = CUTEAFD_LAUNCH(4, 2, true); break;
      case 1410: status = mode == 5 ? cudaErrorInvalidValue : launch_p1<4>(residual, fn, incoming, norm, scratch, normalized, rows, rows_per_cta, projection, norms, stream); break;
      case 1810: status = mode == 5 ? cudaErrorInvalidValue : launch_p1<8>(residual, fn, incoming, norm, scratch, normalized, rows, rows_per_cta, projection, norms, stream); break;
      default: return cudaErrorInvalidValue;
    }
#undef CUTEAFD_LAUNCH
    if (status) return status;
  }
  if (mode == 6) {
    finish2_kernel<<<rows, 64, 0, stream>>>(static_cast<const float*>(scratch),
        static_cast<const float*>(scale), static_cast<const float*>(base),
        static_cast<float*>(next_pre), static_cast<float*>(post), static_cast<float*>(comb));
    return cudaGetLastError();
  }
  if (mode == 0 || mode == 2) {
    finish_mixes_kernel<8><<<rows, 32, 0, stream>>>(static_cast<const float*>(scratch),
        static_cast<const float*>(scale), static_cast<const float*>(base),
        static_cast<float*>(next_pre), static_cast<float*>(post), static_cast<float*>(comb));
    status = cudaGetLastError();
  }
  return status;
}

extern "C" int32_t hc_post_finish_sm89(const void* sublayer, const void* residual, const void* partials,
    const void* scale, const void* base, void* next_pre, void* post, void* comb, void* output,
    int32_t rows, void* stream) {
  if (rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  post_finish_kernel<<<dim3(5, rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(sublayer), static_cast<const __nv_bfloat16*>(residual),
      static_cast<const float*>(partials), static_cast<const float*>(scale), static_cast<const float*>(base),
      static_cast<float*>(next_pre), static_cast<float*>(post), static_cast<float*>(comb),
      static_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}

// Measurement aid: occupy N SMs (one 1024-thread CTA at 64 registers each fills
// an SM's register file) until released, to time a 114-SM part on a 128-SM card.
namespace {
__global__ void __launch_bounds__(1024, 1) occupy_kernel(const volatile int* flag, float* sink, int heavy) {
  if (heavy) {  // never taken: sizes the register allocation to the full 64
    float acc[80];
    #pragma unroll
    for (int i = 0; i < 80; ++i) acc[i] = threadIdx.x * 1e-3f + i;
    for (int k = 0; k < heavy; ++k) {
      #pragma unroll
      for (int i = 0; i < 80; ++i) acc[i] = fmaf(acc[i], 0.999f, acc[(i + 1) % 80] * 1e-3f);
    }
    float total = 0;
    #pragma unroll
    for (int i = 0; i < 80; ++i) total += acc[i];
    sink[threadIdx.x] = total;
  }
  if (threadIdx.x == 0) {
    while (*flag == 0) __nanosleep(20000);
  }
  __syncthreads();
}
int* occupy_flag = nullptr;
int* occupy_flag_device = nullptr;
float* occupy_sink = nullptr;
cudaStream_t occupy_stream = nullptr;
}  // namespace

extern "C" int32_t occupy_start(int32_t ctas) {
  if (!occupy_flag) {
    auto status = cudaHostAlloc(reinterpret_cast<void**>(&occupy_flag), sizeof(int), cudaHostAllocMapped);
    if (status) return status;
    status = cudaHostGetDevicePointer(reinterpret_cast<void**>(&occupy_flag_device), occupy_flag, 0);
    if (status) return status;
    status = cudaMalloc(reinterpret_cast<void**>(&occupy_sink), 1024 * sizeof(float));
    if (status) return status;
    status = cudaStreamCreateWithFlags(&occupy_stream, cudaStreamNonBlocking);
    if (status) return status;
  }
  *reinterpret_cast<volatile int*>(occupy_flag) = 0;
  occupy_kernel<<<ctas, 1024, 0, occupy_stream>>>(occupy_flag_device, occupy_sink, 0);
  return cudaGetLastError();
}
extern "C" int32_t occupy_stop() {
  *reinterpret_cast<volatile int*>(occupy_flag) = 1;
  return cudaStreamSynchronize(occupy_stream);
}
