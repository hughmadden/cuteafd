// V4.1 mHC boundary for SM89 (RTX 4090 and 4090 D), in today's SM89 decode
// arithmetic at every row count up to kMaxRows (native/CMakeLists.txt builds
// this file for sm_89 only).
//
// cuteafd_v41_hc_begin, one launch:
//  - projection CTAs (K part, projection tile, row tile): the 24 FP32
//    projection partials of each row in the split-8 arithmetic of the
//    hc_project AOT (a lane-strided fma.rn chain over 80 steps per 2560-wide
//    K part, then the shfl_down tree), each fn slice read once per row tile;
//  - one norm CTA per row: the collapse with the incoming coefficients and its
//    RMSNorm, in the arithmetic of pre_kernel + ds41_rmsnorm_cached_bf16_kernel.
// The partials stay in scratch. cuteafd_v41_hc_post_finish then runs
// finish_mixes_kernel<8>'s arithmetic per row (next pre, post, comb) ahead of
// post_vec4_kernel's, so the boundary's finish costs no launch of its own.
// Nothing here bakes in an SM count: the row plan takes the device's at
// initialization, so one build serves both cards.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <mutex>
#include "cuteafd_v41_hc.h"

namespace {
constexpr int kHidden = 5120, kK = 20480, kParts = 8, kPartK = 2560, kSteps = 80;
constexpr int kNormThreads = 256;  // ds41_rmsnorm_cached_bf16_kernel's kBlock
constexpr int kWarps = 4;          // projection and norm CTAs: 128 threads
constexpr int kMaxRows = 128;      // decode, verify and decoder-replay widths
constexpr int kMaxTileRows = 16;   // 80 KB of residual per projection CTA
constexpr uint64_t kPartialBytesPerRow = 24 * kParts * 8;

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

// NR rows of one warp's PW projections over the CTA's K part. Chains run in
// step order from zero, as the AOT lane loop does. SQ adds the residual square
// chain, the same for every projection, and stores it to all 24 slots.
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

// pre_kernel + ds41_rmsnorm_cached_bf16_kernel<5120> for one row; the CTA's
// threads emulate the norm's 256 (each owns columns t, t+256, ...).
__device__ __forceinline__ void norm_row(const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ incoming, const uint16_t* __restrict__ weight,
    uint16_t* __restrict__ out, int row, float* scratch) {
  constexpr int T = kWarps * 32, VT = kNormThreads / T, kPer = kHidden / kNormThreads;
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
  const float inv = rsqrtf(scratch[0] / static_cast<float>(kHidden) + 1e-20f);
  #pragma unroll
  for (int v = 0; v < VT; ++v)
    #pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const int col = tid + v * T + i * kNormThreads;
      out[uint64_t(row) * kHidden + col] = f32_to_bf16(values[v][i] * inv * bf16_to_f32(weight[col]));
    }
}

// Projection CTAs (part, projection tile of 4 * PW, row tile), then one norm
// CTA per row. A row tile's residual part is staged once and shared by the
// CTA's warps; each warp keeps its projections' fn slice in registers.
template <int PW>
__global__ void __launch_bounds__(kWarps * 32, 1)
begin_kernel(const __nv_bfloat16* __restrict__ residual, const float* __restrict__ fn,
             const float* __restrict__ incoming, const uint16_t* __restrict__ weight,
             float2* __restrict__ partials, uint16_t* __restrict__ normalized,
             int rows, int tile_rows, int projection_ctas) {
  extern __shared__ __align__(16) unsigned char smem[];
  if (int(blockIdx.x) >= projection_ctas) {
    norm_row(residual, incoming, weight, normalized, int(blockIdx.x) - projection_ctas,
             reinterpret_cast<float*>(smem));
    return;
  }
  constexpr int PT = kWarps * PW, kTiles = 24 / PT;
  static_assert(kTiles * PT == 24);
  const int cta = blockIdx.x, part = cta % kParts, tile = (cta / kParts) % kTiles;
  const int row0 = (cta / (kParts * kTiles)) * tile_rows;
  const int count = min(tile_rows, rows - row0);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int p0 = tile * PT + warp * PW, k0 = part * kPartK;
  float w[PW][kSteps];
  #pragma unroll
  for (int q = 0; q < PW; ++q)
    #pragma unroll
    for (int j = 0; j < kSteps; ++j) w[q][j] = __ldg(fn + uint64_t(p0 + q) * kK + k0 + 32 * j + lane);
  uint16_t* s_x = reinterpret_cast<uint16_t*>(smem);
  const uint4* source = reinterpret_cast<const uint4*>(residual);
  for (int index = threadIdx.x; index < count * (kPartK / 8); index += kWarps * 32) {
    const int r = index / (kPartK / 8), c = index % (kPartK / 8);
    reinterpret_cast<uint4*>(s_x)[index] = __ldg(source + (uint64_t(row0 + r) * kK + k0) / 8 + c);
  }
  __syncthreads();
  for (int r = 0; r < count; r += 2) {
    const bool square = tile == 0 && ((r >> 1) % kWarps) == warp;
    const uint16_t* x = s_x + r * kPartK;
    if (r + 1 < count) {
      if (square) chains<PW, 2, true>(x, w, lane, partials, row0 + r, p0, part);
      else chains<PW, 2, false>(x, w, lane, partials, row0 + r, p0, part);
    } else {
      if (square) chains<PW, 1, true>(x, w, lane, partials, row0 + r, p0, part);
      else chains<PW, 1, false>(x, w, lane, partials, row0 + r, p0, part);
    }
  }
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
// compute_post_scalar in v41_hc.cu (post_kernel's per-element arithmetic).
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

// Each of a row's 5 CTAs runs finish_mixes_kernel<8>'s arithmetic (warp 0
// Sinkhorn, warp 1 pre and post) into shared memory while its residual loads
// are in flight, then post_vec4_kernel's arithmetic. CTA 0 publishes the row's
// next pre, post and comb.
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
    const float p=1.0f/(1.0f+expf(-a))+1e-6f;
    const float q=2.0f/(1.0f+expf(-b));
    s_post[j] = q;
    if (publish) { pre[row*4+j]=p; post[row*4+j]=q; }
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
    const float c0 = s_comb[0 + d], c1 = s_comb[4 + d], c2 = s_comb[8 + d], c3 = s_comb[12 + d], p = s_post[d];
    uint2 o;
    o.x = post_pair(r0.x, r1.x, r2.x, r3.x, sub.x, c0, c1, c2, c3, p);
    o.y = post_pair(r0.y, r1.y, r2.y, r3.y, sub.y, c0, c1, c2, c3, p);
    st_u2(out + d * 1280, o);
  }
}

// The begin's projection plan: a single row keeps 4 projections per CTA (48
// CTAs, the fn stream on more SMs); wider units 8 per CTA, in row tiles that
// spread one projection wave over the card (four tiles on 114 and 128 SMs)
// up to 64 rows, then 8-row tiles. Measured on an RTX 4090 and on 114 SMs.
struct Plan {
  bool pair;
  int tile_rows;
  int projection_ctas;
  size_t smem;
};

Plan plan(int rows, int sms) {
  Plan p{};
  p.pair = rows > 1;
  if (!p.pair) {
    p.tile_rows = 1;
  } else if (rows <= 64) {
    const int tiles = (sms + 16) / 32 > 1 ? (sms + 16) / 32 : 1;
    p.tile_rows = (rows + tiles - 1) / tiles;
    p.tile_rows = p.tile_rows < 2 ? 2 : p.tile_rows > kMaxTileRows ? kMaxTileRows : p.tile_rows;
  } else {
    p.tile_rows = 8;
  }
  const int tiles_per_part = 24 / (kWarps * (p.pair ? 2 : 1));
  p.projection_ctas = kParts * tiles_per_part * ((rows + p.tile_rows - 1) / p.tile_rows);
  const size_t staged = size_t(p.tile_rows) * kPartK * 2;
  p.smem = staged > kNormThreads * 4 ? staged : kNormThreads * 4;
  return p;
}

std::mutex mutex;
int ready_device[16];  // device id + 1 once initialized
int ready_sms[16];

int device_sms(int device) {
  for (int slot = 0; slot < 16; ++slot)
    if (ready_device[slot] == device + 1) return ready_sms[slot];
  return 0;
}

bool span(const void* pointer, uint64_t bytes, uint64_t alignment, uintptr_t& start, uintptr_t& end) {
  start = reinterpret_cast<uintptr_t>(pointer);
  if (!start || start % alignment || start > UINTPTR_MAX - bytes) return false;
  end = start + bytes;
  return true;
}

// Outputs [first, count) must not overlap any other listed buffer.
bool disjoint(const uintptr_t* starts, const uintptr_t* ends, int count, int first) {
  for (int i = first; i < count; ++i)
    for (int j = 0; j < count; ++j)
      if (i != j && starts[i] < ends[j] && starts[j] < ends[i]) return false;
  return true;
}
}  // namespace

extern "C" int32_t cuteafd_v41_hc_sm89_initialize() {
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status) return status;
  int major = 0, minor = 0, sms = 0;
  status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (!status) status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (!status) status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status) return status;
  if (major != 8 || minor != 9 || sms < 1) return cudaErrorInvalidDevice;
  std::lock_guard<std::mutex> lock(mutex);
  if (device_sms(device)) return cudaSuccess;
  const int bytes = kMaxTileRows * kPartK * 2;
  status = cudaFuncSetAttribute(begin_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (!status) status = cudaFuncSetAttribute(begin_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (status) return status;
  for (int slot = 0; slot < 16; ++slot)
    if (!ready_device[slot]) {
      ready_sms[slot] = sms;
      ready_device[slot] = device + 1;
      return cudaSuccess;
    }
  return cudaErrorInvalidDevice;
}

extern "C" int32_t cuteafd_v41_hc_begin(const void* residual, const void* fn,
    const void* scale, const void* base, const void* incoming, const void* norm,
    void* predicted, void* post, void* comb, void* normalized, void* scratch,
    uint64_t scratch_bytes, int32_t rows, void* stream) {
  if (rows < 1 || rows > kMaxRows || scratch_bytes < uint64_t(rows) * kPartialBytesPerRow)
    return cudaErrorInvalidValue;
  const void* pointers[] = {residual, fn, scale, base, incoming, norm,
                            predicted, post, comb, normalized, scratch};
  const uint64_t r = uint64_t(rows);
  const uint64_t sizes[] = {r * 40960, 24 * 20480 * 4, 12, 96, r * 16, 10240,
                            r * 16, r * 16, r * 64, r * 10240, r * kPartialBytesPerRow};
  uintptr_t starts[11], ends[11];
  for (int i = 0; i < 11; ++i)
    if (!span(pointers[i], sizes[i], 16, starts[i], ends[i])) return cudaErrorInvalidValue;
  if (!disjoint(starts, ends, 11, 6)) return cudaErrorInvalidValue;
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status) return status;
  const int sms = device_sms(device);
  if (!sms) return cudaErrorInvalidDevice;
  const Plan p = plan(rows, sms);
  const dim3 grid(p.projection_ctas + rows);
  const auto s = static_cast<cudaStream_t>(stream);
  const auto* x = static_cast<const __nv_bfloat16*>(residual);
  const auto* w = static_cast<const float*>(fn);
  const auto* in = static_cast<const float*>(incoming);
  const auto* g = static_cast<const uint16_t*>(norm);
  auto* partials = static_cast<float2*>(scratch);
  auto* y = static_cast<uint16_t*>(normalized);
  if (p.pair) begin_kernel<2><<<grid, kWarps * 32, p.smem, s>>>(x, w, in, g, partials, y, rows, p.tile_rows, p.projection_ctas);
  else begin_kernel<1><<<grid, kWarps * 32, p.smem, s>>>(x, w, in, g, partials, y, rows, p.tile_rows, p.projection_ctas);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_hc_post_finish(const uint16_t* sublayer, const uint16_t* residual,
    const void* scratch, uint64_t scratch_bytes, const float* scale, const float* base,
    float* predicted, float* post, float* comb, uint16_t* output, int32_t rows, void* stream) {
  if (rows < 1 || rows > kMaxRows || scratch_bytes < uint64_t(rows) * kPartialBytesPerRow)
    return cudaErrorInvalidValue;
  const void* pointers[] = {sublayer, residual, scratch, scale, base, predicted, post, comb, output};
  const uint64_t r = uint64_t(rows);
  const uint64_t sizes[] = {r * 10240, r * 40960, r * kPartialBytesPerRow, 12, 96, r * 16, r * 16, r * 64, r * 40960};
  const uint64_t alignments[] = {16, 16, 16, 4, 4, 16, 16, 16, 16};
  uintptr_t starts[9], ends[9];
  for (int i = 0; i < 9; ++i)
    if (!span(pointers[i], sizes[i], alignments[i], starts[i], ends[i])) return cudaErrorInvalidValue;
  if (!disjoint(starts, ends, 9, 5)) return cudaErrorInvalidValue;
  int device = -1;
  auto status = cudaGetDevice(&device);
  if (status) return status;
  if (!device_sms(device)) return cudaErrorInvalidDevice;
  post_finish_kernel<<<dim3(5, rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(sublayer), reinterpret_cast<const __nv_bfloat16*>(residual),
      static_cast<const float*>(scratch), scale, base, predicted, post, comb,
      reinterpret_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}
