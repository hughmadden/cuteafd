// cuteafd_copy_engine_regions lands every region exactly (odd sizes and
// alignments, into mapped pinned host memory and device memory) and refuses
// bad arguments; cuteafd_staging_probe reports a sane table and crossover on
// this device and refuses bad arguments.
#include "cuteafd_peer_exchange.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define check(condition) do { if (!(condition)) { \
  std::cerr << "check failed at " << __LINE__ << ": " << #condition << '\n'; \
  std::abort(); \
} } while (false)

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) { std::cout << "requires a CUDA device\n"; return 77; }
  check(cudaSetDevice(0) == cudaSuccess);
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  constexpr size_t kArena = 1 << 20;
  uint8_t* host = nullptr;
  check(cudaHostAlloc(&host, kArena, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess);
  void* alias = nullptr;
  check(cudaHostGetDevicePointer(&alias, host, 0) == cudaSuccess);
  uint8_t *source = nullptr, *device = nullptr;
  check(cudaMalloc(&source, kArena) == cudaSuccess);
  check(cudaMalloc(&device, kArena) == cudaSuccess);
  std::vector<uint8_t> pattern(kArena);
  for (size_t i = 0; i < kArena; ++i) pattern[i] = uint8_t(i * 131 + 7);
  check(cudaMemcpy(source, pattern.data(), kArena, cudaMemcpyHostToDevice) == cudaSuccess);

  // Three regions, odd offsets and sizes, one of them empty, into the mapped
  // arena; then the same into device memory.
  for (int into_device = 0; into_device < 2; ++into_device) {
    uint8_t* base = into_device ? device : static_cast<uint8_t*>(alias);
    std::memset(host, 0xEE, kArena);
    check(cudaMemset(device, 0xEE, kArena) == cudaSuccess);
    void* destinations[4] = {base + 3, base + 4096, base + 70001, base + 200000};
    const void* sources[4] = {source + 11, source + 64, source + 200003, source + 1};
    const uint64_t bytes[4] = {1000, 65536, 77777, 0};
    check(cuteafd_copy_engine_regions(destinations, sources, bytes, 4, stream) == cudaSuccess);
    check(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<uint8_t> landed(kArena);
    if (into_device) check(cudaMemcpy(landed.data(), device, kArena, cudaMemcpyDeviceToHost) == cudaSuccess);
    else std::memcpy(landed.data(), host, kArena);
    const size_t offsets[3] = {3, 4096, 70001}, from[3] = {11, 64, 200003};
    for (int r = 0; r < 3; ++r)
      check(std::memcmp(landed.data() + offsets[r], pattern.data() + from[r], bytes[r]) == 0);
    check(landed[2] == 0xEE && landed[1003] == 0xEE && landed[4095] == 0xEE && landed[69632] == 0xEE);
    check(landed[70001 + 77777] == 0xEE && landed[200000] == 0xEE);
  }
  {
    void* destinations[1] = {alias};
    const void* sources[1] = {source};
    const uint64_t bytes[1] = {16};
    check(cuteafd_copy_engine_regions(destinations, sources, bytes, 0, stream) == cudaErrorInvalidValue);
    check(cuteafd_copy_engine_regions(destinations, sources, bytes, 5, stream) == cudaErrorInvalidValue);
    check(cuteafd_copy_engine_regions(destinations, sources, bytes, 1, nullptr) == cudaErrorInvalidValue);
    void* null_destination[1] = {nullptr};
    check(cuteafd_copy_engine_regions(null_destination, sources, bytes, 1, stream) == cudaErrorInvalidValue);
  }

  // The probe: a table of doubling totals, positive times, a crossover that
  // is one of the probed totals (or none).
  cuteafd_staging_probe_t probe{};
  check(cuteafd_staging_probe(host, alias, kArena, 3, &probe) == cudaSuccess);
  std::printf("copy engine held by stream waits: %s; crossover %llu bytes\n", probe.copy_engine_held ? "yes" : "no",
      static_cast<unsigned long long>(probe.crossover_bytes));
  if (probe.copy_engine_held) {
    check(probe.probes == 9);
    bool crossover_probed = probe.crossover_bytes == 0;
    for (uint32_t p = 0; p < probe.probes; ++p) {
      std::printf("  %7llu bytes: SM %8.2f us, copy engine %8.2f us\n",
          static_cast<unsigned long long>(probe.bytes[p]), probe.sm_us[p], probe.copy_engine_us[p]);
      check(probe.bytes[p] == (2048ull << p));
      check(probe.sm_us[p] > 0 && probe.copy_engine_us[p] > 0);
      if (probe.bytes[p] == probe.crossover_bytes) crossover_probed = true;
      if (probe.crossover_bytes && probe.bytes[p] >= probe.crossover_bytes)
        check(probe.copy_engine_us[p] < probe.sm_us[p]);
    }
    check(crossover_probed);
  } else {
    check(probe.probes == 0 && probe.crossover_bytes == 0);
  }
  cuteafd_staging_probe_t small{};
  check(cuteafd_staging_probe(host, alias, 4096, 1, &small) == cudaSuccess);
  check(!small.copy_engine_held || small.probes == 2);
  check(cuteafd_staging_probe(host, alias, 4095, 1, &small) == cudaErrorInvalidValue);
  check(cuteafd_staging_probe(host, alias, kArena, 0, &small) == cudaErrorInvalidValue);
  check(cuteafd_staging_probe(host, alias, kArena, 5, &small) == cudaErrorInvalidValue);
  check(cuteafd_staging_probe(nullptr, alias, kArena, 1, &small) == cudaErrorInvalidValue);
  check(cuteafd_staging_probe(host, alias, kArena, 1, nullptr) == cudaErrorInvalidValue);

  check(cudaFree(source) == cudaSuccess);
  check(cudaFree(device) == cudaSuccess);
  check(cudaFreeHost(host) == cudaSuccess);
  check(cudaStreamDestroy(stream) == cudaSuccess);
  std::cout << "cuda staging selftest passed\n";
  return 0;
}
