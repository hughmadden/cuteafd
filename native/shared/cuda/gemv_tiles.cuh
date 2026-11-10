#pragma once
// Shared by the skinny tensor-core GEMVs whose weights are packed at load in
// mma.m16n8k16 A-fragment order (fp8_gemv.cu, packed_head.cu): one warp per
// (16-row tile, K split), activations as B fragments in passes of kMaxRows
// rows, and a deterministic split-K whose plan depends on the weight's shape
// and the device's SM count, never on the row count.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace cuteafd::gemv_tiles {

constexpr int kWarps = 4;     // warps per CTA
constexpr int kMaxRows = 64;  // activation rows per pass

// Weights are read once per pass: stream them past L1.
__device__ __forceinline__ uint4 load_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ uint2 load_stream(const uint2* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

// The SM count of the device current at the first call, queried once.
inline int sm_count() {
  static int count = 0;
  if (count == 0) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    if (count <= 0) count = 128;
  }
  return count;
}

// K split of an [n, k] weight in 128-wide K blocks: about 16 (tile, split)
// warps per SM, at most one split per block.
inline void split_plan(int n, int k, int* kb_per_split, int* splits) {
  const int tiles = n / 16, kbs = k / 128;
  const int wanted = (sm_count() * 16 + tiles - 1) / tiles;
  int s = wanted < 1 ? 1 : (wanted > kbs ? kbs : wanted);
  *kb_per_split = (kbs + s - 1) / s;
  *splits = (kbs + *kb_per_split - 1) / *kb_per_split;
}

inline size_t align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

}  // namespace cuteafd::gemv_tiles
