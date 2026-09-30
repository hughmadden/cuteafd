// L2 prefetch of weight ranges (after Hugh Madden's glm53f-afd prefetch.cu,
// MIT, v1.1.0 13f682f): while a decode step waits for its routed experts
// (the Spark exchange), one kernel on the step's stream touches one byte of
// every `stride` bytes of the next layer's weights, in the order that layer
// reads them, with `ld.global.cg.L2::256B` (through L2 only, 256-byte
// prefetch size, several loads in flight per thread). The values fold into a
// register stored only to a sink that is always null, which keeps the loads
// without writing anything: every later read sees the same bytes, from L2
// instead of DRAM while the lines last. Every touched address lies inside
// its range.
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kMaxRanges = 16;
constexpr int kUnroll = 4;

struct Ranges {
  const char* ptr[kMaxRanges];
  int64_t lines[kMaxRanges];
  int n;
  int stride;
  uint32_t* sink;  // Always null.
};

__device__ __forceinline__ uint32_t load_cg(const char* p) {
  uint32_t v;
  asm volatile("ld.global.cg.L2::256B.u8 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__global__ void __launch_bounds__(256) l2_prefetch_kernel(const Ranges r) {
  const int64_t step = int64_t(gridDim.x) * blockDim.x;
  const int64_t t0 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t acc = 0;
  for (int s = 0; s < r.n; ++s) {
    const char* base = r.ptr[s];
    const int64_t lines = r.lines[s];
    int64_t i = t0;
    for (; i + (kUnroll - 1) * step < lines; i += kUnroll * step) {
      uint32_t v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) v[u] = load_cg(base + (i + u * step) * r.stride);
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) acc ^= v[u];
    }
    for (; i < lines; i += step) acc ^= load_cg(base + i * r.stride);
  }
  if (r.sink) *r.sink = acc;
}

}  // namespace

// The device's L2 bytes (0 when the query fails).
extern "C" int64_t cuteafd_l2_cache_bytes(void) {
  int device = 0, bytes = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device) != cudaSuccess) return 0;
  return bytes;
}

// The current device's multiprocessor count (0 when the query fails).
extern "C" int32_t cuteafd_sm_count(void) {
  int device = 0, count = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) return 0;
  return count;
}

// Touches `n` (<= 16) device ranges every `stride` (32/64/128/256) bytes from
// `blocks` blocks of 256 threads on `stream`. Writes nothing.
extern "C" int32_t cuteafd_l2_prefetch(const void* const* ptrs, const int64_t* bytes, int32_t n, int32_t stride,
                                       int32_t blocks, void* stream) {
  if (n < 0 || n > kMaxRanges || blocks < 1 || (stride != 32 && stride != 64 && stride != 128 && stride != 256) ||
      (n > 0 && (!ptrs || !bytes)))
    return cudaErrorInvalidValue;
  Ranges r{};
  int k = 0;
  for (int i = 0; i < n; ++i) {
    if (!ptrs[i] || bytes[i] < 0) return cudaErrorInvalidValue;
    if (bytes[i] == 0) continue;
    r.ptr[k] = static_cast<const char*>(ptrs[i]);
    r.lines[k] = (bytes[i] + stride - 1) / stride;
    ++k;
  }
  if (k == 0) return cudaSuccess;
  r.n = k;
  r.stride = stride;
  l2_prefetch_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(r);
  return cudaGetLastError();
}
