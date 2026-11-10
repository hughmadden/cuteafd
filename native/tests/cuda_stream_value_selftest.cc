// Stream memory operations on mapped pinned host words: the device-side wait
// and publish the shared Spark exchange orders its streams with.
#include "cuteafd_native.h"
#include "cuteafd_peer_exchange.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

static void check(bool condition, const char* what) {
  if (!condition) {
    std::cerr << "FAILED: " << what << "\n";
    std::abort();
  }
}

static bool idle(void* stream) {
  int32_t ready = -1;
  check(cuteafd_cuda_stream_query(stream, &ready) == CUTEAFD_STATUS_OK, "stream query");
  return ready == 1;
}

static void wait_idle(void* stream) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!idle(stream)) {
    check(std::chrono::steady_clock::now() < deadline, "stream released within 5 s");
    std::this_thread::yield();
  }
}

static std::atomic<uint32_t>& at(uint32_t* word) { return *reinterpret_cast<std::atomic<uint32_t>*>(word); }

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "no CUDA device\n";
    return 77;
  }
  cuteafd_host_buffer_t words{};
  check(cuteafd_alloc_host_buffer(4096, &words) == CUTEAFD_STATUS_OK, "mapped pinned words");
  std::memset(words.ptr, 0, words.bytes);
  auto* flag = static_cast<uint32_t*>(words.ptr);       // host -> GPU
  auto* published = flag + 16;                           // GPU -> host
  auto* landing = reinterpret_cast<uint8_t*>(flag + 64); // D2H target
  void* stream = nullptr;
  void* other = nullptr;
  check(cuteafd_cuda_stream_create(&stream) == CUTEAFD_STATUS_OK, "stream");
  check(cuteafd_cuda_stream_create(&other) == CUTEAFD_STATUS_OK, "second stream");
  cuteafd_device_buffer_t device{};
  check(cuteafd_alloc_device_buffer(1 << 20, &device) == CUTEAFD_STATUS_OK, "device buffer");

  // Arguments are validated before any enqueue.
  check(cuteafd_cuda_stream_wait_value32(nullptr, flag, 1) == CUTEAFD_STATUS_INVALID_ARGUMENT, "null stream");
  check(cuteafd_cuda_stream_wait_value32(stream, nullptr, 1) == CUTEAFD_STATUS_INVALID_ARGUMENT, "null word");
  check(cuteafd_cuda_stream_write_value32(stream, reinterpret_cast<uint32_t*>(landing + 1), 1)
      == CUTEAFD_STATUS_INVALID_ARGUMENT, "misaligned word");
  uint32_t pageable = 0;
  check(cuteafd_cuda_stream_wait_value32(stream, &pageable, 1) != CUTEAFD_STATUS_OK, "unmapped word refused");

  std::cerr << "step 1\n";
  // 1. A wait holds the stream, and the work behind it, until the host stores.
  check(cudaMemsetAsync(device.ptr, 0x11, 4096, static_cast<cudaStream_t>(stream)) == cudaSuccess, "memset");
  check(cuteafd_cuda_stream_wait_value32(stream, flag, 1) == CUTEAFD_STATUS_OK, "enqueue wait 1");
  check(cudaMemsetAsync(device.ptr, 0x22, 4096, static_cast<cudaStream_t>(stream)) == cudaSuccess, "memset after");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  check(!idle(stream), "the wait holds the stream before the store");
  unsigned char probe = 0;
  check(cudaMemcpy(&probe, device.ptr, 1, cudaMemcpyDeviceToHost) == cudaSuccess && probe == 0x11,
      "work behind the wait has not run");
  check(idle(other), "an unrelated stream is not held");
  at(flag).store(1, std::memory_order_release);
  wait_idle(stream);
  check(cudaMemcpy(&probe, device.ptr, 1, cudaMemcpyDeviceToHost) == cudaSuccess && probe == 0x22,
      "work behind the wait ran after the store");

  // 2. A larger value also satisfies an earlier target (GEQ), so a
  //    publisher that ran ahead never strands a queued wait.
  at(flag).store(5, std::memory_order_release);
  check(cuteafd_cuda_stream_wait_value32(stream, flag, 3) == CUTEAFD_STATUS_OK, "enqueue wait 3 at 5");
  wait_idle(stream);

  // 3. Cyclic compare across wraparound.
  at(flag).store(0xfffffffeu, std::memory_order_release);
  check(cuteafd_cuda_stream_wait_value32(stream, flag, 1) == CUTEAFD_STATUS_OK, "enqueue wait past wrap");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(!idle(stream), "0xfffffffe does not satisfy 1");
  at(flag).store(1, std::memory_order_release);
  wait_idle(stream);

  // 4. A write publishes only after the stream's earlier D2H copy is visible.
  std::vector<unsigned char> pattern(2048);
  for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<unsigned char>(i * 7 + 3);
  check(cudaMemcpy(device.ptr, pattern.data(), pattern.size(), cudaMemcpyHostToDevice) == cudaSuccess, "upload");
  std::memset(landing, 0, pattern.size());
  check(cuteafd_cuda_stream_wait_value32(stream, flag, 2) == CUTEAFD_STATUS_OK, "gate the copy");
  check(cudaMemcpyAsync(landing, device.ptr, pattern.size(), cudaMemcpyDeviceToHost,
      static_cast<cudaStream_t>(stream)) == cudaSuccess, "D2H");
  check(cuteafd_cuda_stream_write_value32(stream, published, 9) == CUTEAFD_STATUS_OK, "enqueue write");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  check(at(published).load(std::memory_order_acquire) == 0, "write waits for the gated stream");
  at(flag).store(2, std::memory_order_release);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (at(published).load(std::memory_order_acquire) != 9) {
    check(std::chrono::steady_clock::now() < deadline, "write published within 5 s");
  }
  check(std::memcmp(landing, pattern.data(), pattern.size()) == 0, "copy visible when the write is seen");

  // 5. Cross-stream: a host relay forwards one stream's write to another's wait.
  check(cuteafd_cuda_stream_write_value32(other, published, 10) == CUTEAFD_STATUS_OK, "other write");
  check(cuteafd_cuda_stream_wait_value32(stream, flag, 3) == CUTEAFD_STATUS_OK, "relay wait");
  while (at(published).load(std::memory_order_acquire) != 10) {
    check(std::chrono::steady_clock::now() < deadline + std::chrono::seconds(5), "other write within 5 s");
  }
  at(flag).store(3, std::memory_order_release);
  wait_idle(stream);

  // 6. Many queued waits released in order by one rising counter (a decode
  //    step queued ahead of its exchange).
  for (uint32_t layer = 5; layer < 85; ++layer) {
    check(cuteafd_cuda_stream_wait_value32(stream, flag, layer) == CUTEAFD_STATUS_OK, "queued layer wait");
    check(cuteafd_cuda_stream_write_value32(stream, published, layer + 100) == CUTEAFD_STATUS_OK, "layer write");
  }
  for (uint32_t layer = 5; layer < 85; ++layer) {
    at(flag).store(layer, std::memory_order_release);
    while (at(published).load(std::memory_order_acquire) != layer + 100) {
      check(std::chrono::steady_clock::now() < deadline + std::chrono::seconds(30), "layer released");
    }
  }
  wait_idle(stream);

  std::cerr << "steps 1-6 passed\n";
  // Kernels a stream waits in front of must be loaded first: a lazily loaded
  // kernel's first launch can wait for the device to idle (a pending wait
  // never lets it), which hangs the host in the launch.
  check(cuteafd_peer_exchange_initialize() == cudaSuccess, "preload exchange kernels");
  // 7. Kernels behind waits, deep run-ahead (the exchange's staging copies):
  //    80 x [wait(flag >= k), SM copy of row k into one mapped host row,
  //    write(published = k)], all queued first. The host releases k one at a
  //    time and, once k is published, checks the row holds k's data and is
  //    not yet overwritten by k+1 (which must still be held by its wait).
  {
    const size_t row = 4096;
    cuteafd_device_buffer_t rows_device{};
    check(cuteafd_alloc_device_buffer(80 * row, &rows_device) == CUTEAFD_STATUS_OK, "rows buffer");
    std::vector<unsigned char> rows_host(80 * row);
    for (size_t i = 0; i < rows_host.size(); ++i) rows_host[i] = static_cast<unsigned char>((i / row) * 3 + 1);
    check(cudaMemcpy(rows_device.ptr, rows_host.data(), rows_host.size(), cudaMemcpyHostToDevice) == cudaSuccess, "rows");
    cuteafd_host_buffer_t target_buffer{};
    check(cuteafd_alloc_host_buffer(row, &target_buffer) == CUTEAFD_STATUS_OK, "mapped target");
    auto* target = static_cast<volatile unsigned char*>(target_buffer.ptr);
    void* target_alias = nullptr;
    check(cudaHostGetDevicePointer(&target_alias, target_buffer.ptr, 0) == cudaSuccess, "target alias");
    at(flag).store(1000, std::memory_order_release);
    at(published).store(1000, std::memory_order_release);
    const uint32_t depth = std::getenv("SELFTEST_DEPTH") ? uint32_t(std::atoi(std::getenv("SELFTEST_DEPTH"))) : 80;
    std::cerr << "step 7: queueing " << depth << " kernel layers\n";
    for (uint32_t k = 1; k <= depth; ++k) {
      check(cuteafd_cuda_stream_wait_value32(stream, flag, 1000 + k) == CUTEAFD_STATUS_OK, "kernel wait");
      void* destinations[1] = {target_alias};
      const void* sources[1] = {static_cast<unsigned char*>(rows_device.ptr) + (k - 1) * row};
      uint64_t bytes[1] = {row};
      check(cuteafd_sm_copy_regions(destinations, sources, bytes, 1, stream) == cudaSuccess, "sm copy");
      check(cuteafd_cuda_stream_write_value32(stream, published, 1000 + k) == CUTEAFD_STATUS_OK, "kernel write");
    }
    std::cerr << "step 7: queued\n";
    for (uint32_t k = 1; k <= depth; ++k) {
      std::this_thread::sleep_for(std::chrono::microseconds(200));
      // k not released yet: the row must still hold k-1's data.
      if (k > 1) check(target[0] == static_cast<unsigned char>((k - 2) * 3 + 1), "kernel ran ahead of its wait");
      at(flag).store(1000 + k, std::memory_order_release);
      const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (at(published).load(std::memory_order_acquire) != 1000 + k) {
        check(std::chrono::steady_clock::now() < until, "kernel layer published");
      }
      check(target[0] == static_cast<unsigned char>((k - 1) * 3 + 1) &&
            target[row - 1] == static_cast<unsigned char>((k - 1) * 3 + 1), "kernel copy visible at publish");
    }
    wait_idle(stream);
    check(cuteafd_free_device_buffer(&rows_device) == CUTEAFD_STATUS_OK, "free rows");
    check(cuteafd_free_host_buffer(&target_buffer) == CUTEAFD_STATUS_OK, "free target");
  }

  check(cuteafd_free_device_buffer(&device) == CUTEAFD_STATUS_OK, "free device");
  check(cuteafd_cuda_stream_destroy(other) == CUTEAFD_STATUS_OK, "destroy other");
  check(cuteafd_cuda_stream_destroy(stream) == CUTEAFD_STATUS_OK, "destroy stream");
  check(cuteafd_free_host_buffer(&words) == CUTEAFD_STATUS_OK, "free words");
  std::cout << "stream value wait/write order streams against mapped host words\n";
  return 0;
}
