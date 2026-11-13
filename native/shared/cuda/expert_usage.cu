// Per-engine route counters: no native geometry or telemetry globals.
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace {
__global__ void tally(const uint32_t* ids, unsigned long long* counts,
                      uint32_t rows, uint32_t topk, uint32_t experts,
                      uint32_t mask, const int32_t* valid_rows) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t real = valid_rows ? min(rows, static_cast<uint32_t>(max(0, *valid_rows))) : rows;
  if (i < real * topk) {
    const uint32_t expert = ids[i] & mask;
    if (expert < experts) atomicAdd(counts + expert, 1ULL);
  }
}
__global__ void snapshot(unsigned long long* source, unsigned long long* target, uint32_t cells) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // Atomic reads may overlap serving-stream additions, but never tear a counter.
  if (i < cells) target[i] = atomicAdd(source + i, 0ULL);
}
}
extern "C" int32_t cuteafd_expert_usage_tally(const uint32_t* ids, uint64_t* counts,
    uint32_t rows, uint32_t topk, uint32_t experts, uint32_t mask,
    const int32_t* valid_rows, void* stream) {
  if (!ids || !counts || !topk || !experts || rows > UINT_MAX / topk) return cudaErrorInvalidValue;
  if (!rows) return cudaSuccess;
  tally<<<(static_cast<uint64_t>(rows) * topk + 255) / 256, 256, 0,
      static_cast<cudaStream_t>(stream)>>>(ids, reinterpret_cast<unsigned long long*>(counts),
          rows, topk, experts, mask, valid_rows);
  return cudaGetLastError();
}
extern "C" int32_t cuteafd_expert_usage_snapshot(uint64_t* source, uint64_t* target,
    uint32_t cells, void* stream) {
  if (!source || !target || !cells) return cudaErrorInvalidValue;
  snapshot<<<(static_cast<uint64_t>(cells) + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<unsigned long long*>(source), reinterpret_cast<unsigned long long*>(target), cells);
  return cudaGetLastError();
}
