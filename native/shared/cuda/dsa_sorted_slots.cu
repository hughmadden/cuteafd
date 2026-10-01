// Deterministic order for DSA index top-k selections (GLM 5.x, GLM 5.3 Flash).
//
// The radix top-k emits its selected physical slots in shared-atomic order, which varies run to
// run, and the sparse attention then accumulates in that order. Sorting each row's selection by
// logical position (`logical[slot / 64] * 64 + slot % 64`: the slot's page mapped to its page
// in the sequence) makes the order canonical, so equal selections give bit-equal attention
// whatever the run or the physical pages (a prefix-cache restore lands on other pages).
// Negative (padding) entries sort last. One CTA per row; bitonic sort in shared memory.
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kThreads = 1024;
constexpr int kMaxWidth = 2048;

__global__ void sort_slots_kernel(int32_t* slots, const int32_t* logical, int width, int padded) {
  __shared__ unsigned long long keys[kMaxWidth];
  int32_t* row = slots + static_cast<int64_t>(blockIdx.x) * width;
  for (int i = threadIdx.x; i < padded; i += blockDim.x) {
    unsigned long long key = ~0ull;
    if (i < width) {
      const int32_t slot = row[i];
      if (slot >= 0) {
        const uint32_t position = static_cast<uint32_t>(logical[slot >> 6]) * 64u + static_cast<uint32_t>(slot & 63);
        key = (static_cast<unsigned long long>(position) << 32) | static_cast<uint32_t>(slot);
      } else {
        key = (~0ull << 32) | static_cast<uint32_t>(slot);
      }
    }
    keys[i] = key;
  }
  __syncthreads();
  for (int size = 2; size <= padded; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool ascending = (i & size) == 0;
          const unsigned long long a = keys[i], b = keys[j];
          if ((a > b) == ascending) {
            keys[i] = b;
            keys[j] = a;
          }
        }
      }
      __syncthreads();
    }
  }
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    row[i] = static_cast<int32_t>(static_cast<uint32_t>(keys[i]));
  }
}

}  // namespace

extern "C" int32_t cuteafd_dsa_sort_slots(void* slots, const void* logical, int32_t rows, int32_t width,
                                          void* stream) {
  if (rows < 0 || width < 1 || width > kMaxWidth || slots == nullptr || logical == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  int padded = 1;
  while (padded < width) {
    padded <<= 1;
  }
  sort_slots_kernel<<<rows, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<int32_t*>(slots), static_cast<const int32_t*>(logical), width, padded);
  return static_cast<int32_t>(cudaGetLastError());
}
