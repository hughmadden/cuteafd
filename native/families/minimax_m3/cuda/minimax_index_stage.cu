// MiniMax M3 MSA prefill: the exported prefill scorer reads index keys as
// contiguous rows, while the cache keeps them in 64-row pages (8192 FP8 key
// bytes, then 64 FP32 scales: 8448 bytes). One block per logical page copies
// the request's page j to key rows [64j, 64j + 64) and their scales.
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kPageUnits = 8448 / 16;
constexpr int kKeyUnits = 8192 / 16;

__global__ void stage_index_pages(const uint4* cache, const int32_t* pages, uint4* keys, uint4* scales) {
  const int64_t page = pages[blockIdx.x];
  const uint4* source = cache + page * kPageUnits;
  for (int unit = threadIdx.x; unit < kPageUnits; unit += blockDim.x) {
    const uint4 value = source[unit];
    if (unit < kKeyUnits) keys[int64_t(blockIdx.x) * kKeyUnits + unit] = value;
    else scales[int64_t(blockIdx.x) * (kPageUnits - kKeyUnits) + unit - kKeyUnits] = value;
  }
}

}  // namespace

// `pages` holds `count` physical page ids (device memory, < pool_pages); keys
// receives count * 64 rows of 128 bytes and scales count * 64 floats. All four
// pointers are 16-byte aligned device memory.
extern "C" int32_t cuteafd_minimax_index_stage_async(const void* cache, const int32_t* pages, uint32_t count,
    void* keys, void* scales, void* stream) {
  if (!cache || !pages || !keys || !scales || !count) return cudaErrorInvalidValue;
  for (const void* pointer : {cache, static_cast<const void*>(pages), static_cast<const void*>(keys),
       static_cast<const void*>(scales)}) {
    if (reinterpret_cast<uintptr_t>(pointer) % 16) return cudaErrorInvalidValue;
  }
  stage_index_pages<<<count, 128, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint4*>(cache), pages,
      static_cast<uint4*>(keys), static_cast<uint4*>(scales));
  return cudaGetLastError();
}
