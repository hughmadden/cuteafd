#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "cuteafd_v41_hc.h"
namespace {
__global__ void pre_kernel(const __nv_bfloat16* residual, const float* pre,
    __nv_bfloat16* output, uint64_t values) {
  const uint64_t i = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= values) return;
  const uint64_t row = i/5120, col = i%5120;
  float sum = 0;
#pragma unroll
  for (int copy = 0; copy < 4; ++copy)
    sum = __fadd_rn(sum, __fmul_rn(pre[row*4+copy], __bfloat162float(residual[(row*4+copy)*5120+col])));
  output[i] = __float2bfloat16_rn(sum);
}
__global__ void post_kernel(const __nv_bfloat16* sublayer, const __nv_bfloat16* residual,
    const float* post, const float* comb, __nv_bfloat16* output, uint64_t values) {
  const uint64_t i = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= values) return;
  const uint64_t row = i/(4*5120), destination = (i/5120)%4, col = i%5120;
  float sum = 0;
#pragma unroll
  for (int source = 0; source < 4; ++source)
    sum = __fadd_rn(sum, __fmul_rn(comb[row*16+source*4+destination],
        __bfloat162float(residual[(row*4+source)*5120+col])));
  const float expansion = __fmul_rn(post[row*4+destination], __bfloat162float(sublayer[row*5120+col]));
  output[i] = __float2bfloat16_rn(__fadd_rn(expansion, sum));
}

struct DeviceCaps {
  int sm_count;
  int l2_bytes;
};

inline DeviceCaps get_device_caps() {
  static DeviceCaps cached[16] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 16) dev = 0;
  if (cached[dev].sm_count == 0) {
    int sm = 0, l2 = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
    cached[dev].sm_count = sm > 0 ? sm : 1;
    cached[dev].l2_bytes = l2 > 0 ? l2 : 0;
  }
  return cached[dev];
}

__device__ __forceinline__ uint2 ld_ca_u2(const uint2* p) {
  uint2 v;
  asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

__device__ __forceinline__ uint2 ld_cs_u2(const uint2* p) {
  uint2 v;
  asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

__device__ __forceinline__ void st_wb_u2(uint2* p, uint2 v) {
  asm volatile("st.global.wb.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(v.x), "r"(v.y));
}

__device__ __forceinline__ void st_cs_u2(uint2* p, uint2 v) {
  asm volatile("st.global.cs.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(v.x), "r"(v.y));
}

__device__ __forceinline__ float bf16_lo(uint32_t u) {
  return __bfloat162float(__ushort_as_bfloat16(uint16_t(u)));
}

__device__ __forceinline__ float bf16_hi(uint32_t u) {
  return __bfloat162float(__ushort_as_bfloat16(uint16_t(u >> 16)));
}

__device__ __forceinline__ __nv_bfloat16 compute_post_scalar(
    float r0, float r1, float r2, float r3, float sub,
    float c0, float c1, float c2, float c3, float p) {
  float sum = 0.0f;
  sum = __fadd_rn(sum, __fmul_rn(c0, r0));
  sum = __fadd_rn(sum, __fmul_rn(c1, r1));
  sum = __fadd_rn(sum, __fmul_rn(c2, r2));
  sum = __fadd_rn(sum, __fmul_rn(c3, r3));
  const float expansion = __fmul_rn(p, sub);
  return __float2bfloat16_rn(__fadd_rn(expansion, sum));
}

__device__ __forceinline__ uint32_t compute_post_pair(
    float r0_a, float r1_a, float r2_a, float r3_a, float sub_a,
    float r0_b, float r1_b, float r2_b, float r3_b, float sub_b,
    float c0, float c1, float c2, float c3, float p) {
  const __nv_bfloat16 b0 = compute_post_scalar(r0_a, r1_a, r2_a, r3_a, sub_a, c0, c1, c2, c3, p);
  const __nv_bfloat16 b1 = compute_post_scalar(r0_b, r1_b, r2_b, r3_b, sub_b, c0, c1, c2, c3, p);
  return uint32_t(__bfloat16_as_ushort(b0)) | (uint32_t(__bfloat16_as_ushort(b1)) << 16);
}

__device__ __forceinline__ uint2 compute_post_u2(
    uint2 r0, uint2 r1, uint2 r2, uint2 r3, uint2 sub,
    float c0, float c1, float c2, float c3, float p) {
  uint2 out;
  out.x = compute_post_pair(
      bf16_lo(r0.x), bf16_lo(r1.x), bf16_lo(r2.x), bf16_lo(r3.x), bf16_lo(sub.x),
      bf16_hi(r0.x), bf16_hi(r1.x), bf16_hi(r2.x), bf16_hi(r3.x), bf16_hi(sub.x),
      c0, c1, c2, c3, p);
  out.y = compute_post_pair(
      bf16_lo(r0.y), bf16_lo(r1.y), bf16_lo(r2.y), bf16_lo(r3.y), bf16_lo(sub.y),
      bf16_hi(r0.y), bf16_hi(r1.y), bf16_hi(r2.y), bf16_hi(r3.y), bf16_hi(sub.y),
      c0, c1, c2, c3, p);
  return out;
}

template <bool StreamWrites>
__global__ __launch_bounds__(256, 5)
void post_vec4_kernel(const __nv_bfloat16* __restrict__ sublayer,
                      const __nv_bfloat16* __restrict__ residual,
                      const float* __restrict__ post,
                      const float* __restrict__ comb,
                      __nv_bfloat16* __restrict__ output,
                      int32_t cached_rows) {
  const uint64_t row = blockIdx.y;
  const uint32_t col = (blockIdx.x * 256 + threadIdx.x) * 4;
  const uint64_t r_idx = row * 20480 + col;
  const uint64_t s_idx = row * 5120 + col;

  uint2 sub, r0, r1, r2, r3;
  if constexpr (!StreamWrites) {
    sub = ld_ca_u2(reinterpret_cast<const uint2*>(sublayer + s_idx));
    r0 = ld_ca_u2(reinterpret_cast<const uint2*>(residual + r_idx + 0 * 5120));
    r1 = ld_ca_u2(reinterpret_cast<const uint2*>(residual + r_idx + 1 * 5120));
    r2 = ld_ca_u2(reinterpret_cast<const uint2*>(residual + r_idx + 2 * 5120));
    r3 = ld_ca_u2(reinterpret_cast<const uint2*>(residual + r_idx + 3 * 5120));
  } else {
    if (int32_t(row) < cached_rows) {
      sub = ld_ca_u2(reinterpret_cast<const uint2*>(sublayer + s_idx));
      r0 = ld_ca_u2(reinterpret_cast<const uint2*>(residual + r_idx + 0 * 5120));
      r1 = ld_ca_u2(reinterpret_cast<const uint2*>(residual + r_idx + 1 * 5120));
      r2 = ld_ca_u2(reinterpret_cast<const uint2*>(residual + r_idx + 2 * 5120));
      r3 = ld_ca_u2(reinterpret_cast<const uint2*>(residual + r_idx + 3 * 5120));
    } else {
      sub = ld_cs_u2(reinterpret_cast<const uint2*>(sublayer + s_idx));
      r0 = ld_cs_u2(reinterpret_cast<const uint2*>(residual + r_idx + 0 * 5120));
      r1 = ld_cs_u2(reinterpret_cast<const uint2*>(residual + r_idx + 1 * 5120));
      r2 = ld_cs_u2(reinterpret_cast<const uint2*>(residual + r_idx + 2 * 5120));
      r3 = ld_cs_u2(reinterpret_cast<const uint2*>(residual + r_idx + 3 * 5120));
    }
  }

  const float* comb_row = comb + row * 16;
  const float* post_row = post + row * 4;
  uint2* out_base = reinterpret_cast<uint2*>(output + r_idx);

  #pragma unroll
  for (int d = 0; d < 4; ++d) {
    const uint2 out_d = compute_post_u2(
        r0, r1, r2, r3, sub,
        comb_row[0 + d], comb_row[4 + d], comb_row[8 + d], comb_row[12 + d], post_row[d]);
    if constexpr (StreamWrites) {
      st_cs_u2(out_base + d * 1280, out_d);
    } else {
      st_wb_u2(out_base + d * 1280, out_d);
    }
  }
}

bool valid(const void* pointer, uint64_t bytes, uint64_t alignment) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  return address && address%alignment == 0 && address <= UINTPTR_MAX-bytes;
}
bool disjoint(const void* a, uint64_t na, const void* b, uint64_t nb) {
  const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return x+na <= y || y+nb <= x;
}
}
extern "C" int32_t cuteafd_v41_hc_pre(const uint16_t* residual, const float* pre,
    uint16_t* collapsed, int32_t rows, void* stream) {
  if (rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t r = uint64_t(rows)*40960, p = uint64_t(rows)*16, c = uint64_t(rows)*10240;
  if (!valid(residual,r,2) || !valid(pre,p,4) || !valid(collapsed,c,2) ||
      !disjoint(residual,r,collapsed,c) || !disjoint(pre,p,collapsed,c)) return cudaErrorInvalidValue;
  pre_kernel<<<(c/2+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(residual),pre,
      reinterpret_cast<__nv_bfloat16*>(collapsed),c/2);
  return cudaGetLastError();
}
extern "C" int32_t cuteafd_v41_hc_post(const uint16_t* sublayer, const uint16_t* residual,
    const float* post, const float* comb, uint16_t* output, int32_t rows, void* stream) {
  if (rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t r = uint64_t(rows)*40960, p = uint64_t(rows)*16, c = uint64_t(rows)*10240;
  const void* inputs[] = {sublayer,residual,post,comb};
  const uint64_t sizes[] = {c,r,p,p*4};
  if (!valid(output,r,2)) return cudaErrorInvalidValue;
  for (int i=0;i<4;++i)
    if (!valid(inputs[i],sizes[i],i<2?2:4) || !disjoint(inputs[i],sizes[i],output,r)) return cudaErrorInvalidValue;
  if (rows <= 16) {
    post_kernel<<<(r/2+255)/256,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(sublayer),reinterpret_cast<const __nv_bfloat16*>(residual),
        post,comb,reinterpret_cast<__nv_bfloat16*>(output),r/2);
  } else {
    const auto s_stream = reinterpret_cast<cudaStream_t>(stream);
    const bool aligned8 = ((reinterpret_cast<uintptr_t>(sublayer) |
                            reinterpret_cast<uintptr_t>(residual) |
                            reinterpret_cast<uintptr_t>(output)) & 7u) == 0;
    if (!aligned8) {
      post_kernel<<<(r/2+255)/256,256,0,s_stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(sublayer),reinterpret_cast<const __nv_bfloat16*>(residual),
          post,comb,reinterpret_cast<__nv_bfloat16*>(output),r/2);
      return cudaGetLastError();
    }
    const DeviceCaps caps = get_device_caps();
    const int64_t l2_budget = (int64_t(caps.l2_bytes) * 85) / 100;
    const int32_t cached_rows = l2_budget > 0 ? int32_t(l2_budget / 51280) : 0;
    const int64_t total_bytes = int64_t(rows) * 92240;
    const dim3 grid(5, uint32_t(rows));
    if (l2_budget > 0 && total_bytes > l2_budget) {
      post_vec4_kernel<true><<<grid, 256, 0, s_stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(sublayer),
          reinterpret_cast<const __nv_bfloat16*>(residual),
          post, comb, reinterpret_cast<__nv_bfloat16*>(output), cached_rows);
    } else {
      post_vec4_kernel<false><<<grid, 256, 0, s_stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(sublayer),
          reinterpret_cast<const __nv_bfloat16*>(residual),
          post, comb, reinterpret_cast<__nv_bfloat16*>(output), rows);
    }
  }
  return cudaGetLastError();
}

namespace {
__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset=16;offset;offset>>=1) value += __shfl_down_sync(0xffffffffu,value,offset);
  return value;
}
__global__ void mixes_kernel(const __nv_bfloat16* residual, const float* fn,
    const float* scale, const float* base, float* pre, float* post, float* comb) {
  const uint64_t row=blockIdx.x;
  const int lane=threadIdx.x%32, warp=threadIdx.x/32;
  float dot[3]={0,0,0}, square=0;
  for (int col=lane;col<20480;col+=32) {
    const float x=__bfloat162float(residual[row*20480+col]);
    if (warp==0) square=fmaf(x,x,square);
#pragma unroll
    for(int j=0;j<3;++j) dot[j]=fmaf(x,fn[uint64_t(warp+j*8)*20480+col],dot[j]);
  }
  __shared__ float projected[24], inverse;
  if(warp==0) {
    square=warp_sum(square);
    if(lane==0) inverse=rsqrtf(square/20480.0f+1e-20f);
  }
#pragma unroll
  for(int j=0;j<3;++j) {
    dot[j]=warp_sum(dot[j]);
    if(lane==0) projected[warp+j*8]=dot[j];
  }
  __syncthreads();
  if(threadIdx.x<4) {
    const int j=threadIdx.x;
    const float a=fmaf(projected[j]*inverse,scale[0],base[j]);
    const float b=fmaf(projected[j+4]*inverse,scale[1],base[j+4]);
    pre[row*4+j]=1.0f/(1.0f+expf(-a))+1e-6f;
    post[row*4+j]=2.0f/(1.0f+expf(-b));
  }
  if(warp==0) {
    float value=lane<16 ? fmaf(projected[8+lane]*inverse,scale[2],base[8+lane]) : 0;
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

__device__ __forceinline__ void store_bf16x4_to_smem(float* smem_ptr, uint2 raw) {
  const float f0 = bf16_lo(raw.x);
  const float f1 = bf16_hi(raw.x);
  const float f2 = bf16_lo(raw.y);
  const float f3 = bf16_hi(raw.y);
  *reinterpret_cast<float4*>(smem_ptr) = make_float4(f0, f1, f2, f3);
}

__device__ __forceinline__ void sinkhorn_epilogue_warp(
    uint64_t row, int lane, float proj_val,
    const float* scale, const float* base,
    float* pre, float* post, float* comb) {
  const float proj_p4 = __shfl_down_sync(0xffffffffu, proj_val, 4);
  const float proj_p8 = __shfl_down_sync(0xffffffffu, proj_val, 8);
  if (lane < 4) {
    const float a = fmaf(proj_val, scale[0], base[lane]);
    const float b = fmaf(proj_p4, scale[1], base[lane + 4]);
    pre[row * 4 + lane] = 1.0f / (1.0f + expf(-a)) + 1e-6f;
    post[row * 4 + lane] = 2.0f / (1.0f + expf(-b));
  }
  float value = lane < 16 ? fmaf(proj_p8, scale[2], base[8 + lane]) : 0.0f;
  float maximum = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, 2));
  value = expf(value - maximum);
  float sum = value + __shfl_xor_sync(0xffffffffu, value, 1);
  sum += __shfl_xor_sync(0xffffffffu, sum, 2);
  value = value / sum + 1e-6f;
  sum = value + __shfl_xor_sync(0xffffffffu, value, 4);
  sum += __shfl_xor_sync(0xffffffffu, sum, 8);
  value /= sum + 1e-6f;
  for (int iteration = 1; iteration < 20; ++iteration) {
    sum = value + __shfl_xor_sync(0xffffffffu, value, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    value /= sum + 1e-6f;
    sum = value + __shfl_xor_sync(0xffffffffu, value, 4);
    sum += __shfl_xor_sync(0xffffffffu, sum, 8);
    value /= sum + 1e-6f;
  }
  if (lane < 16) comb[row * 16 + lane] = value;
}

__global__ __launch_bounds__(256, 6)
void finish_mixes_fast_kernel(const float* __restrict__ scale,
                              const float* __restrict__ base,
                              float* __restrict__ pre,
                              float* __restrict__ post,
                              float* __restrict__ comb,
                              int32_t rows) {
  const int row = int(blockIdx.x) * 8 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  float proj_val = 0.0f;
  if (lane < 24) {
    proj_val = (lane < 4) ? pre[uint64_t(row) * 4 + lane] :
               (lane < 8) ? post[uint64_t(row) * 4 + lane - 4] :
                            comb[uint64_t(row) * 16 + lane - 8];
  }
  sinkhorn_epilogue_warp(uint64_t(row), lane, proj_val, scale, base, pre, post, comb);
}

template <int W_r, int W_p, int R_t, int P_t, int MinBlocksPerSM>
__global__ __launch_bounds__(W_r * W_p * 32, MinBlocksPerSM)
void mixes_pipe_2d_kernel(const __nv_bfloat16* __restrict__ residual,
                          const float* __restrict__ fn,
                          const float* __restrict__ scale,
                          const float* __restrict__ base,
                          float* __restrict__ pre,
                          float* __restrict__ post,
                          float* __restrict__ comb,
                          int32_t rows,
                          int32_t cached_rows) {
  constexpr int W = W_r * W_p;
  constexpr int R_block = W_r * R_t;
  constexpr int P_block = W_p * P_t;
  constexpr int X_LOADS = (R_block + W - 1) / W;
  constexpr int FN_LOADS = (P_block + W - 1) / W;

  const int row_base = int(blockIdx.x) * R_block;
  const int valid_rows = min(R_block, rows - row_base);
  const int proj_base = (P_block < 24) ? (int(blockIdx.y) * P_block) : 0;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int w_r = warp / W_p;
  const int w_p = warp % W_p;

  __shared__ alignas(16) float s_x[2][R_block][128];
  __shared__ alignas(16) float s_fn[2][P_block][128];

  float dot[R_t][P_t];
  #pragma unroll
  for (int r = 0; r < R_t; ++r) {
    #pragma unroll
    for (int p = 0; p < P_t; ++p) {
      dot[r][p] = 0.0f;
    }
  }
  float square[X_LOADS];
  #pragma unroll
  for (int k = 0; k < X_LOADS; ++k) square[k] = 0.0f;

  #pragma unroll
  for (int k = 0; k < X_LOADS; ++k) {
    const int r_idx = warp + k * W;
    if (R_block % W == 0 || r_idx < R_block) {
      const int clamped_row = min(row_base + r_idx, rows - 1);
      const uint2* ptr = reinterpret_cast<const uint2*>(
          residual + uint64_t(clamped_row) * 20480 + lane * 4);
      const uint2 raw = (clamped_row < cached_rows) ? ld_ca_u2(ptr) : ld_cs_u2(ptr);
      store_bf16x4_to_smem(&s_x[0][r_idx][lane * 4], raw);
    }
  }
  #pragma unroll
  for (int k = 0; k < FN_LOADS; ++k) {
    const int p_idx = warp + k * W;
    if (P_block % W == 0 || p_idx < P_block) {
      *reinterpret_cast<float4*>(&s_fn[0][p_idx][lane * 4]) = __ldg(reinterpret_cast<const float4*>(
          fn + uint64_t(proj_base + p_idx) * 20480 + lane * 4));
    }
  }
  __syncthreads();

  #pragma unroll 1
  for (int tile = 0; tile < 160; ++tile) {
    const int stage = tile & 1;
    const int next_tile = tile + 1;
    uint2 next_raw[X_LOADS];
    float4 next_fn[FN_LOADS];

    if (next_tile < 160) {
      const int next_c = next_tile * 128 + lane * 4;
      #pragma unroll
      for (int k = 0; k < X_LOADS; ++k) {
        const int r_idx = warp + k * W;
        if (R_block % W == 0 || r_idx < R_block) {
          const int clamped_row = min(row_base + r_idx, rows - 1);
          const uint2* ptr = reinterpret_cast<const uint2*>(
              residual + uint64_t(clamped_row) * 20480 + next_c);
          next_raw[k] = (clamped_row < cached_rows) ? ld_ca_u2(ptr) : ld_cs_u2(ptr);
        }
      }
      #pragma unroll
      for (int k = 0; k < FN_LOADS; ++k) {
        const int p_idx = warp + k * W;
        if (P_block % W == 0 || p_idx < P_block) {
          next_fn[k] = __ldg(reinterpret_cast<const float4*>(
              fn + uint64_t(proj_base + p_idx) * 20480 + next_c));
        }
      }
    }

    #pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int col_in_tile = s * 32 + lane;
      float x_reg[R_t];
      #pragma unroll
      for (int r = 0; r < R_t; ++r) {
        x_reg[r] = s_x[stage][w_r * R_t + r][col_in_tile];
      }
      #pragma unroll
      for (int k = 0; k < X_LOADS; ++k) {
        const int r_idx = warp + k * W;
        if (R_block % W == 0 || r_idx < R_block) {
          const float x_sq = s_x[stage][r_idx][col_in_tile];
          square[k] = fmaf(x_sq, x_sq, square[k]);
        }
      }
      float w_reg[P_t];
      #pragma unroll
      for (int p = 0; p < P_t; ++p) {
        w_reg[p] = s_fn[stage][w_p * P_t + p][col_in_tile];
      }
      #pragma unroll
      for (int r = 0; r < R_t; ++r) {
        #pragma unroll
        for (int p = 0; p < P_t; ++p) {
          dot[r][p] = fmaf(x_reg[r], w_reg[p], dot[r][p]);
        }
      }
    }

    if (next_tile < 160) {
      #pragma unroll
      for (int k = 0; k < X_LOADS; ++k) {
        const int r_idx = warp + k * W;
        if (R_block % W == 0 || r_idx < R_block) {
          store_bf16x4_to_smem(&s_x[stage ^ 1][r_idx][lane * 4], next_raw[k]);
        }
      }
      #pragma unroll
      for (int k = 0; k < FN_LOADS; ++k) {
        const int p_idx = warp + k * W;
        if (P_block % W == 0 || p_idx < P_block) {
          *reinterpret_cast<float4*>(&s_fn[stage ^ 1][p_idx][lane * 4]) = next_fn[k];
        }
      }
      __syncthreads();
    }
  }

  __syncthreads();
  float* s_inv = &s_x[0][0][0];

  #pragma unroll
  for (int k = 0; k < X_LOADS; ++k) {
    const int r_idx = warp + k * W;
    if (R_block % W == 0 || r_idx < R_block) {
      const float sq = warp_sum(square[k]);
      if (lane == 0) s_inv[r_idx] = rsqrtf(sq / 20480.0f + 1e-20f);
    }
  }

  if constexpr (P_block == 24) {
    float (*s_proj)[24] = reinterpret_cast<float(*)[24]>(&s_fn[0][0][0]);
    #pragma unroll
    for (int r = 0; r < R_t; ++r) {
      #pragma unroll
      for (int p = 0; p < P_t; ++p) {
        const float d = warp_sum(dot[r][p]);
        if (lane == 0) s_proj[w_r * R_t + r][w_p * P_t + p] = d;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < X_LOADS; ++k) {
      const int r_idx = warp + k * W;
      if (r_idx < valid_rows) {
        const uint64_t row = row_base + r_idx;
        const float inv = s_inv[r_idx];
        const float proj_val = (lane < 24) ? (s_proj[r_idx][lane] * inv) : 0.0f;
        sinkhorn_epilogue_warp(row, lane, proj_val, scale, base, pre, post, comb);
      }
    }
  } else {
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < R_t; ++r) {
      const int r_in_block = w_r * R_t + r;
      const float inv = (r_in_block < valid_rows) ? s_inv[r_in_block] : 0.0f;
      #pragma unroll
      for (int p = 0; p < P_t; ++p) {
        const float d = warp_sum(dot[r][p]);
        if (lane == 0 && r_in_block < valid_rows) {
          const uint64_t row = row_base + r_in_block;
          const int proj = proj_base + w_p * P_t + p;
          const float val = d * inv;
          if (proj < 4) pre[row * 4 + proj] = val;
          else if (proj < 8) post[row * 4 + proj - 4] = val;
          else comb[row * 16 + proj - 8] = val;
        }
      }
    }
  }
}

template <int W_P>
__global__ __launch_bounds__(W_P * 32, 48 / W_P)
void mixes_small_2d_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ fn,
    float* __restrict__ pre,
    float* __restrict__ post,
    float* __restrict__ comb) {
  const uint64_t row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int projection = int(blockIdx.y) * W_P + warp;
  const __nv_bfloat16* r_ptr = residual + row * 20480;
  const float* w_ptr = fn + uint64_t(projection) * 20480;
  float dot = 0.0f, square = 0.0f;
  #pragma unroll 16
  for (int col = lane; col < 20480; col += 32) {
    const float x = __bfloat162float(r_ptr[col]);
    dot = fmaf(x, w_ptr[col], dot);
    square = fmaf(x, x, square);
  }
  dot = warp_sum(dot);
  square = warp_sum(square);
  if (lane == 0) {
    dot *= rsqrtf(square / 20480.0f + 1e-20f);
    if (projection < 4) pre[row * 4 + projection] = dot;
    else if (projection < 8) post[row * 4 + projection - 4] = dot;
    else comb[row * 16 + projection - 8] = dot;
  }
}

// Small batches spread the 24 projections across SMs. Each lane retains the
// original sequential FMA/reduction order, including the residual norm. The
// normalized projections temporarily occupy the disjoint output arrays; the
// next kernel consumes all of them before publishing the final coefficients.
__global__ void project_kernel(const __nv_bfloat16* residual, const float* fn,
    float* pre, float* post, float* comb) {
  const uint64_t row=blockIdx.x;
  const int projection=blockIdx.y, lane=threadIdx.x;
  float dot=0, square=0;
#pragma unroll 32
  for(int col=lane;col<20480;col+=32) {
    const float x=__bfloat162float(residual[row*20480+col]);
    dot=fmaf(x,fn[uint64_t(projection)*20480+col],dot);
    square=fmaf(x,x,square);
  }
  dot=warp_sum(dot);
  square=warp_sum(square);
  if(lane==0) {
    dot *= rsqrtf(square/20480.0f+1e-20f);
    if(projection<4) pre[row*4+projection]=dot;
    else if(projection<8) post[row*4+projection-4]=dot;
    else comb[row*16+projection-8]=dot;
  }
}
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

}
extern "C" int32_t cuteafd_v41_hc_mixes(const uint16_t* residual, const float* fn,
    const float* scale, const float* base, float* pre, float* post, float* comb,
    int32_t rows, void* stream) {
  if(rows<1 || rows>4096) return cudaErrorInvalidValue;
  const void* pointers[]={residual,fn,scale,base,pre,post,comb};
  const uint64_t small=uint64_t(rows)*16;
  const uint64_t sizes[]={uint64_t(rows)*40960,24*20480*4,12,96,small,small,small*4};
  for(int i=0;i<7;++i) if(!valid(pointers[i],sizes[i],i==0?2:4)) return cudaErrorInvalidValue;
  for(int i=4;i<7;++i) for(int j=0;j<i;++j)
    if(!disjoint(pointers[i],sizes[i],pointers[j],sizes[j])) return cudaErrorInvalidValue;
  if(rows<=16) {
    project_kernel<<<dim3(rows,24),32,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual),fn,pre,post,comb);
    auto status=cudaGetLastError();
    if(status!=cudaSuccess) return status;
    finish_mixes_kernel<0><<<rows,32,0,reinterpret_cast<cudaStream_t>(stream)>>>(
        nullptr,scale,base,pre,post,comb);
  } else {
    const auto s_stream = reinterpret_cast<cudaStream_t>(stream);
    const bool aligned = ((reinterpret_cast<uintptr_t>(residual) & 7u) == 0) &&
                         ((reinterpret_cast<uintptr_t>(fn) & 15u) == 0);
    if (!aligned) {
      mixes_kernel<<<rows,256,0,s_stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(residual),fn,scale,base,pre,post,comb);
      return cudaGetLastError();
    }
    const DeviceCaps caps = get_device_caps();
    const int64_t l2_avail = (int64_t(caps.l2_bytes) * 86) / 100 - 1966080;
    const int32_t cached_rows = l2_avail > 0 ? int32_t(l2_avail / 40960) : rows;
    const auto* res_bf16 = reinterpret_cast<const __nv_bfloat16*>(residual);
    if (rows <= 32) {
      mixes_small_2d_kernel<2><<<dim3(rows, 12), 64, 0, s_stream>>>(
          res_bf16, fn, pre, post, comb);
      auto status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      finish_mixes_fast_kernel<<<(rows + 7) / 8, 256, 0, s_stream>>>(
          scale, base, pre, post, comb, rows);
    } else if (rows <= 160) {
      mixes_small_2d_kernel<8><<<dim3(rows, 3), 256, 0, s_stream>>>(
          res_bf16, fn, pre, post, comb);
      auto status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      finish_mixes_fast_kernel<<<(rows + 7) / 8, 256, 0, s_stream>>>(
          scale, base, pre, post, comb, rows);
    } else if (rows <= caps.sm_count * 8) {
      mixes_pipe_2d_kernel<4, 2, 4, 6, 3><<<dim3((rows + 15) / 16, 2), 256, 0, s_stream>>>(
          res_bf16, fn, scale, base, pre, post, comb, rows, cached_rows);
      auto status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      finish_mixes_fast_kernel<<<(rows + 7) / 8, 256, 0, s_stream>>>(
          scale, base, pre, post, comb, rows);
    } else if (rows <= caps.sm_count * 12) {
      mixes_pipe_2d_kernel<6, 2, 4, 6, 2><<<dim3((rows + 23) / 24, 2), 384, 0, s_stream>>>(
          res_bf16, fn, scale, base, pre, post, comb, rows, cached_rows);
      auto status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      finish_mixes_fast_kernel<<<(rows + 7) / 8, 256, 0, s_stream>>>(
          scale, base, pre, post, comb, rows);
    } else if (rows <= caps.sm_count * 16) {
      mixes_pipe_2d_kernel<4, 4, 4, 6, 2><<<(rows + 15) / 16, 512, 0, s_stream>>>(
          res_bf16, fn, scale, base, pre, post, comb, rows, cached_rows);
    } else {
      mixes_pipe_2d_kernel<6, 4, 4, 6, 1><<<(rows + 23) / 24, 768, 0, s_stream>>>(
          res_bf16, fn, scale, base, pre, post, comb, rows, cached_rows);
    }
  }
  return cudaGetLastError();
}


// AOT loading belongs to planning, before any stream capture. The fallback
// keeps CUDA-only builds usable without exporting coordinator CuTe kernels.
#ifdef CUTEAFD_HAVE_V41_HC_AOT
extern "C" int32_t cuteafd_v41_hc_project_launch(
    const uint16_t*, const float*, float*, int32_t, void*);
#else
extern "C" int32_t cuteafd_v41_hc_project_initialize() { return 0; }
#endif
extern "C" int32_t cuteafd_v41_hc_mixes_workspace(const uint16_t* residual,
    const float* fn, const float* scale, const float* base,
    float* pre, float* post, float* comb, void* scratch, uint64_t scratch_bytes,
    int32_t rows, void* stream) {
#ifndef CUTEAFD_HAVE_V41_HC_AOT
  return cuteafd_v41_hc_mixes(residual,fn,scale,base,pre,post,comb,rows,stream);
#else
  if(rows<1 || rows>4096) return cudaErrorInvalidValue;
  if(rows>16) return cuteafd_v41_hc_mixes(residual,fn,scale,base,pre,post,comb,rows,stream);
  const uint64_t small=uint64_t(rows)*16, needed=uint64_t(rows)*1536;
  if(scratch_bytes<needed) return cudaErrorInvalidValue;
  const void* pointers[]={residual,fn,scale,base,pre,post,comb,scratch};
  const uint64_t sizes[]={uint64_t(rows)*40960,24*20480*4,12,96,small,small,small*4,needed};
  for(int i=0;i<8;++i)
    if(!valid(pointers[i],sizes[i],i==0 || i==1 || i==7 ? 16 : 4))
      return cudaErrorInvalidValue;
  for(int i=4;i<8;++i) for(int j=0;j<i;++j)
    if(!disjoint(pointers[i],sizes[i],pointers[j],sizes[j])) return cudaErrorInvalidValue;
  auto* partials=static_cast<float*>(scratch);
  int status=cuteafd_v41_hc_project_launch(residual,fn,partials,rows,stream);
  if(status) return status;
  finish_mixes_kernel<8><<<rows,32,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      partials,scale,base,pre,post,comb);
  return cudaGetLastError();
#endif
}
