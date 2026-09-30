// Vocabulary head for a few rows: logits[r, v] = sum_k x[r, k] * W[v, k] with
// x [rows, K] and W [V, K] BF16, FP32 products and FP32 accumulation on CUDA
// cores (the reference's FP32-promoted projection, as the pedantic cuBLAS
// head computes it). cuBLAS's pedantic FP32 path reads the 1.9 GB GLM head
// once per 1-2 rows (2 rows 1.3 ms, 8 rows 2.6 ms, 16 rows 5.2 ms); this
// reads it once per pass of up to 8 rows at DRAM speed.
//
// Each CTA stages the pass's x rows (8 x K BF16, 96 KiB at K = 6144) in
// shared memory once, then its warps stream vocabulary rows two at a time:
// lane l multiplies 16-byte chunks l, l + 32, ... of both W rows (evict-first
// loads, four chunks in flight) with every x row, and a shuffle tree sums the
// lanes' FP32 partials.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxRows = 8;
constexpr int kMaxSmem = 99 * 1024;

__device__ __forceinline__ void widen(const uint4& v, float (&f)[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
  }
}

template <int R>
__global__ void __launch_bounds__(kThreads) head_rows(const __nv_bfloat16* __restrict__ x,
                                                      const __nv_bfloat16* __restrict__ w, float* __restrict__ out,
                                                      int k, int vocab) {
  extern __shared__ uint4 xs[];  // [R][k / 8]
  const int chunks = k / 8;
  const uint4* xg = reinterpret_cast<const uint4*>(x);
  for (int i = threadIdx.x; i < R * chunks; i += kThreads) xs[i] = xg[i];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int per_lane = chunks / 32;
  const int warps = gridDim.x * (kThreads / 32);
  for (int v0 = blockIdx.x * (kThreads / 32) + warp; v0 < vocab; v0 += 2 * warps) {
    const int v1 = v0 + warps;
    const bool two = v1 < vocab;
    const uint4* w0 = reinterpret_cast<const uint4*>(w + int64_t(v0) * k) + lane;
    const uint4* w1 = reinterpret_cast<const uint4*>(w + int64_t(two ? v1 : v0) * k) + lane;
    float a0[R], a1[R];
#pragma unroll
    for (int r = 0; r < R; ++r) a0[r] = a1[r] = 0.0f;
#pragma unroll 4
    for (int i = 0; i < per_lane; ++i) {
      const uint4 p = __ldcs(w0 + 32 * i), q = __ldcs(w1 + 32 * i);
      float fp[8], fq[8];
      widen(p, fp);
      widen(q, fq);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        float fx[8];
        widen(xs[r * chunks + lane + 32 * i], fx);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          a0[r] = fmaf(fx[j], fp[j], a0[r]);
          a1[r] = fmaf(fx[j], fq[j], a1[r]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        a0[r] += __shfl_xor_sync(0xffffffffu, a0[r], o);
        a1[r] += __shfl_xor_sync(0xffffffffu, a1[r], o);
      }
    }
    if (lane < R) {
      float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        if (r == lane) {
          s0 = a0[r];
          s1 = a1[r];
        }
      }
      out[int64_t(lane) * vocab + v0] = s0;
      if (two) out[int64_t(lane) * vocab + v1] = s1;
    }
  }
}

template <int R>
cudaError_t launch(const __nv_bfloat16* x, const __nv_bfloat16* w, float* out, int k, int vocab, cudaStream_t s) {
  const int bytes = R * k * 2;
  auto kernel = head_rows<R>;
  cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (status != cudaSuccess) return status;
  int device = 0, sms = 0, per_sm = 0;
  if ((status = cudaGetDevice(&device)) != cudaSuccess) return status;
  if ((status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess) return status;
  if ((status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, bytes)) != cudaSuccess) {
    return status;
  }
  kernel<<<sms * (per_sm > 0 ? per_sm : 1), kThreads, bytes, s>>>(x, w, out, k, vocab);
  return cudaGetLastError();
}

}  // namespace

// logits [rows, vocab] FP32 (row stride vocab) = x [rows, width] BF16 times
// W [vocab, width]^T BF16, in passes of up to 8 rows. width must be a multiple
// of 256 with 8 * width * 2 bytes of shared memory available (width <= 6336).
extern "C" int32_t cuteafd_vocab_head_rows(const void* x, const void* weight, float* logits, int32_t rows,
                                           int32_t width, int32_t vocab, void* stream) {
  if (rows < 1 || vocab < 1 || width < 256 || width % 256 || kMaxRows * width * 2 > kMaxSmem ||
      reinterpret_cast<uintptr_t>(x) % 16 || reinterpret_cast<uintptr_t>(weight) % 16) {
    return cudaErrorInvalidValue;
  }
  auto s = static_cast<cudaStream_t>(stream);
  const auto* xb = static_cast<const __nv_bfloat16*>(x);
  const auto* wb = static_cast<const __nv_bfloat16*>(weight);
  for (int first = 0; first < rows; first += kMaxRows) {
    const int n = rows - first < kMaxRows ? rows - first : kMaxRows;
    const auto* xp = xb + int64_t(first) * width;
    float* op = logits + int64_t(first) * vocab;
    cudaError_t status;
    switch (n) {
      case 1: status = launch<1>(xp, wb, op, width, vocab, s); break;
      case 2: status = launch<2>(xp, wb, op, width, vocab, s); break;
      case 3: status = launch<3>(xp, wb, op, width, vocab, s); break;
      case 4: status = launch<4>(xp, wb, op, width, vocab, s); break;
      case 5: status = launch<5>(xp, wb, op, width, vocab, s); break;
      case 6: status = launch<6>(xp, wb, op, width, vocab, s); break;
      case 7: status = launch<7>(xp, wb, op, width, vocab, s); break;
      default: status = launch<8>(xp, wb, op, width, vocab, s); break;
    }
    if (status != cudaSuccess) return status;
  }
  return 0;
}
