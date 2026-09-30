// DeepSeek V4 dSpark drafter helpers: target-layer taps, the main
// projection (block-FP8 linear + RMSNorm) and the Markov-biased argmax chain.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

namespace {

// out[row, offset + h] = mean over the four mHC copies of stream[row, :, h].
__global__ void hc_mean_kernel(const __nv_bfloat16* stream, __nv_bfloat16* out, int hidden, int stride, int offset) {
  const uint64_t row = blockIdx.y;
  for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < hidden; h += gridDim.x * blockDim.x) {
    const __nv_bfloat16* s = stream + row * 4 * hidden + h;
    const float sum = __bfloat162float(s[0]) + __bfloat162float(s[hidden]) + __bfloat162float(s[2 * hidden]) +
                      __bfloat162float(s[3 * hidden]);
    out[row * stride + offset + h] = __float2bfloat16(sum * 0.25f);
  }
}

constexpr int kLinearRows = 8;

// y[r, n] = sum_k x[r, k] * w[n, k] * 2^(scale[n/128, k/128] - 127), one warp
// per output feature, up to kLinearRows rows per pass.
__global__ void fp8_linear_kernel(const __nv_bfloat16* x, const __nv_fp8_e4m3* w, const uint8_t* scale, float* y,
                                  int rows, int n_out, int k_in) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (warp >= n_out) return;
  const int k_blocks = k_in / 128;
  float acc[kLinearRows] = {};
  const __nv_fp8_e4m3* row_w = w + static_cast<uint64_t>(warp) * k_in;
  for (int k = lane * 4; k < k_in; k += 32 * 4) {
    const float s = exp2f(float(int(scale[(warp / 128) * k_blocks + k / 128]) - 127));
    float wv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wv[j] = float(row_w[k + j]) * s;
#pragma unroll
    for (int r = 0; r < kLinearRows; ++r) {
      if (r < rows) {
        const __nv_bfloat16* xr = x + static_cast<uint64_t>(r) * k_in + k;
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[r] = fmaf(__bfloat162float(xr[j]), wv[j], acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kLinearRows; ++r) {
    float v = acc[r];
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0 && r < rows) y[static_cast<uint64_t>(r) * n_out + warp] = v;
  }
}

// out[r] = bf16(bf16(y[r]) * rsqrt(mean(bf16(y[r])^2) + eps) * norm), one block per row.
__global__ void rmsnorm_kernel(const float* y, const __nv_bfloat16* norm, __nv_bfloat16* out, int n, float eps) {
  const uint64_t row = blockIdx.x;
  __shared__ float partial[32];
  float sum = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = __bfloat162float(__float2bfloat16(y[row * n + i]));
    sum += v * v;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0;
    for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
    if (threadIdx.x == 0) partial[0] = rsqrtf(sum / n + eps);
  }
  __syncthreads();
  const float inv = partial[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = __bfloat162float(__float2bfloat16(y[row * n + i]));
    out[row * n + i] = __float2bfloat16(v * inv * __bfloat162float(norm[i]));
  }
}

constexpr int kArgmaxBlocks = 256;

// Draft row `step` of every sequence: value[v] = logits[v] + w2[v] . w1[prev],
// block-local argmax (largest value, lowest index on ties).
__global__ void markov_partial_kernel(const float* logits, const __nv_bfloat16* w1, const __nv_bfloat16* w2,
                                      const uint32_t* first, const uint32_t* drafts, float* partial_value,
                                      uint32_t* partial_index, int vocab, int rank, int block, int step) {
  const int seq = blockIdx.y;
  const uint32_t prev = step == 0 ? first[seq] : drafts[seq * block + step - 1];
  extern __shared__ float embed[];
  for (int i = threadIdx.x; i < rank; i += blockDim.x) embed[i] = __bfloat162float(w1[uint64_t(prev) * rank + i]);
  __syncthreads();
  const float* row_logits = logits + (uint64_t(seq) * block + step) * vocab;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  float best = -CUDART_INF_F;
  uint32_t best_index = UINT32_MAX;
  for (int v = blockIdx.x * warps + warp; v < vocab; v += gridDim.x * warps) {
    const __nv_bfloat16* row = w2 + uint64_t(v) * rank;
    float dot = 0;
    for (int i = lane; i < rank; i += 32) dot = fmaf(__bfloat162float(row[i]), embed[i], dot);
    for (int o = 16; o; o >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, o);
    const float value = row_logits[v] + dot;
    if (lane == 0 && (value > best || (value == best && uint32_t(v) < best_index))) {
      best = value;
      best_index = v;
    }
  }
  __shared__ float values[32];
  __shared__ uint32_t indices[32];
  if (lane == 0) {
    values[warp] = best;
    indices[warp] = best_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < warps; ++i)
      if (values[i] > best || (values[i] == best && indices[i] < best_index)) {
        best = values[i];
        best_index = indices[i];
      }
    partial_value[seq * gridDim.x + blockIdx.x] = best;
    partial_index[seq * gridDim.x + blockIdx.x] = best_index;
  }
}

__global__ void markov_final_kernel(const float* partial_value, const uint32_t* partial_index, uint32_t* drafts,
                                    int partials, int block, int step) {
  const int seq = blockIdx.x;
  float best = -CUDART_INF_F;
  uint32_t best_index = UINT32_MAX;
  for (int i = 0; i < partials; ++i) {
    const float v = partial_value[seq * partials + i];
    const uint32_t index = partial_index[seq * partials + i];
    if (v > best || (v == best && index < best_index)) {
      best = v;
      best_index = index;
    }
  }
  drafts[seq * block + step] = best_index;
}

}  // namespace

extern "C" int32_t cuteafd_dsv4_hc_mean(const void* stream, void* out, int32_t rows, int32_t hidden, int32_t stride,
                                        int32_t offset, void* cuda_stream) {
  if (rows < 1 || hidden < 1 || offset + hidden > stride) return cudaErrorInvalidValue;
  dim3 grid((hidden + 255) / 256, rows);
  hc_mean_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const __nv_bfloat16*>(stream), static_cast<__nv_bfloat16*>(out), hidden, stride, offset);
  return cudaGetLastError();
}

// out [rows, n_out] BF16 = RMSNorm(x [rows, k_in] BF16 @ w^T) * norm; w FP8 E4M3
// [n_out, k_in] with UE8M0 128x128 block scales; `work` holds rows * n_out FP32.
extern "C" int32_t cuteafd_dsv4_fp8_linear_rmsnorm(const void* x, const void* w, const void* scale, const void* norm,
                                                   void* out, void* work, int32_t rows, int32_t n_out, int32_t k_in,
                                                   float eps, void* cuda_stream) {
  if (rows < 1 || n_out % 128 || k_in % 128) return cudaErrorInvalidValue;
  auto s = static_cast<cudaStream_t>(cuda_stream);
  for (int first = 0; first < rows; first += kLinearRows) {
    const int count = rows - first < kLinearRows ? rows - first : kLinearRows;
    fp8_linear_kernel<<<(n_out * 32 + 255) / 256, 256, 0, s>>>(
        static_cast<const __nv_bfloat16*>(x) + uint64_t(first) * k_in, static_cast<const __nv_fp8_e4m3*>(w),
        static_cast<const uint8_t*>(scale), static_cast<float*>(work) + uint64_t(first) * n_out, count, n_out, k_in);
  }
  rmsnorm_kernel<<<rows, 512, 0, s>>>(static_cast<const float*>(work), static_cast<const __nv_bfloat16*>(norm),
                                      static_cast<__nv_bfloat16*>(out), n_out, eps);
  return cudaGetLastError();
}

// Workspace bytes cuteafd_dsv4_markov_drafts needs for `sequences`.
extern "C" uint64_t cuteafd_dsv4_markov_workspace(int32_t sequences) {
  return uint64_t(sequences) * kArgmaxBlocks * 8;
}

// drafts [sequences, block] U32: for step i, argmax over the vocabulary of
// logits[seq, i] + w2 @ w1[prev] with prev = first[seq] for i = 0, else the
// previous draft (the dSpark Markov head, greedy).
extern "C" int32_t cuteafd_dsv4_markov_drafts(const void* logits, const void* w1, const void* w2, const void* first,
                                              void* drafts, void* workspace, int32_t sequences, int32_t block,
                                              int32_t vocab, int32_t rank, void* cuda_stream) {
  if (sequences < 1 || block < 1 || rank < 1 || rank > 1024) return cudaErrorInvalidValue;
  auto s = static_cast<cudaStream_t>(cuda_stream);
  auto* values = static_cast<float*>(workspace);
  auto* indices = reinterpret_cast<uint32_t*>(values + uint64_t(sequences) * kArgmaxBlocks);
  for (int step = 0; step < block; ++step) {
    markov_partial_kernel<<<dim3(kArgmaxBlocks, sequences), 256, rank * sizeof(float), s>>>(
        static_cast<const float*>(logits), static_cast<const __nv_bfloat16*>(w1),
        static_cast<const __nv_bfloat16*>(w2), static_cast<const uint32_t*>(first),
        static_cast<const uint32_t*>(drafts), values, indices, vocab, rank, block, step);
    markov_final_kernel<<<sequences, 1, 0, s>>>(values, indices, static_cast<uint32_t*>(drafts), kArgmaxBlocks,
                                                block, step);
  }
  return cudaGetLastError();
}
