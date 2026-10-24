// cuteafd_host_publish against a host proxy: every published sequence finds its
// wave's three spans complete in pinned, device-mapped host memory (one block at
// decode row counts, many blocks for wide waves), the descriptor words match,
// the stream's completion wait sees the proxy's answer, recorded waves replay
// with fresh sequences, and invalid spans are refused.
#include "cuteafd_peer_exchange.h"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#define check(condition) do { if (!(condition)) { \
  std::cerr << "check failed at " << __LINE__ << ": " << #condition << '\n'; \
  std::abort(); \
} } while (false)

namespace {
// MiniMax: a 6144-wide BF16 wire row and four routes (u32 ids, f32 weights) per row.
constexpr size_t kRow = 12288, kRouteBytes = 16, kRows = 64;
// Mailbox u32 words (cuteafd-transport device_mailbox byte offsets / 4).
constexpr size_t kReady = 0, kDone = 16, kDescriptor = 32, kSend = 48, kRecv = 56, kRoutesAt = 64;
constexpr size_t kMailboxBytes = kRoutesAt * 4 + 2 * kRows * kRouteBytes;

uint8_t pattern(size_t wave, size_t span, size_t i) { return uint8_t((wave * 31 + span * 7 + i * 13) % 251); }
size_t span_bytes(size_t rows, size_t span) { return span < 2 ? rows * kRouteBytes : rows * kRow; }
}  // namespace

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) { std::cout << "requires a CUDA device\n"; return 77; }
  check(cudaSetDevice(0) == cudaSuccess);
  check(cuteafd_peer_exchange_initialize() == cudaSuccess);
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  uint32_t* mailbox = nullptr;
  uint8_t* egress = nullptr;
  check(cudaHostAlloc(reinterpret_cast<void**>(&mailbox), kMailboxBytes, cudaHostAllocPortable | cudaHostAllocMapped)
      == cudaSuccess);
  check(cudaHostAlloc(reinterpret_cast<void**>(&egress), kRows * kRow, cudaHostAllocPortable | cudaHostAllocMapped)
      == cudaSuccess);
  std::memset(mailbox, 0, kMailboxBytes);
  uint8_t* source[3] = {};
  uint32_t* arrivals = nullptr;
  for (auto& s : source) check(cudaMalloc(&s, kRows * kRow) == cudaSuccess);
  check(cudaMalloc(&arrivals, 256) == cudaSuccess);
  check(cudaMemset(arrivals, 0, 256) == cudaSuccess);
  uint8_t* routes = reinterpret_cast<uint8_t*>(mailbox + kRoutesAt);
  uint8_t* destination[3] = {routes, routes + kRows * kRouteBytes, egress};
  auto* ready = reinterpret_cast<std::atomic<uint32_t>*>(mailbox + kReady);
  auto* done = reinterpret_cast<std::atomic<uint32_t>*>(mailbox + kDone);

  const auto prepare = [&](size_t wave, size_t rows) {
    for (size_t s = 0; s < 3; ++s) {
      std::vector<uint8_t> host(span_bytes(rows, s));
      for (size_t i = 0; i < host.size(); ++i) host[i] = pattern(wave, s, i);
      check(cudaMemcpy(source[s], host.data(), host.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    }
  };
  const auto enqueue = [&](size_t layer, size_t rows) {
    const void* src[3] = {source[0], source[1], source[2]};
    void* dst[3] = {destination[0], destination[1], destination[2]};
    const uint64_t bytes[3] = {span_bytes(rows, 0), span_bytes(rows, 1), span_bytes(rows, 2)};
    const uint32_t words[4] = {uint32_t(layer), uint32_t(rows), 1, 4};
    check(cuteafd_host_publish(src, dst, bytes, mailbox + kReady, mailbox + kSend, mailbox + kDescriptor, words,
        arrivals, stream) == cudaSuccess);
    check(cuteafd_peer_wait(mailbox + kDone, mailbox + kRecv, stream) == cudaSuccess);
  };
  // Eager waves of 1, 16 and 64 rows (one block, four, the 32-block cap), then a
  // recorded pair (1 and 7 rows) replayed three times. (pattern, rows) per sequence:
  std::vector<std::pair<size_t, size_t>> plan = {{0, 1}, {1, 16}, {2, 64}};
  for (int replay = 0; replay < 3; ++replay) { plan.push_back({3, 1}); plan.push_back({3, 7}); }
  std::thread proxy([&] {
    for (size_t k = 0; k < plan.size(); ++k) {
      while (ready->load(std::memory_order_acquire) != k + 1) {}
      const auto [wave, rows] = plan[k];
      volatile uint32_t* d = mailbox + kDescriptor;
      check(d[1] == rows && d[2] == 1 && d[3] == 4);
      for (size_t s = 0; s < 3; ++s) {
        for (size_t i = 0; i < span_bytes(rows, s); ++i) check(destination[s][i] == pattern(wave, s, i));
        std::memset(destination[s], 0, span_bytes(rows, s));  // a stale span must fail the next wave
      }
      done->store(uint32_t(k + 1), std::memory_order_release);
    }
  });
  for (size_t w = 0; w < 3; ++w) {
    prepare(w, plan[w].second);
    enqueue(3 + w, plan[w].second);
    check(cudaStreamSynchronize(stream) == cudaSuccess);
  }
  prepare(3, kRows);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  enqueue(10, 1);
  enqueue(11, 7);
  check(cudaStreamEndCapture(stream, &graph) == cudaSuccess);
  check(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess);
  for (int replay = 0; replay < 3; ++replay) check(cudaGraphLaunch(exec, stream) == cudaSuccess);
  check(cudaStreamSynchronize(stream) == cudaSuccess);
  proxy.join();
  check(mailbox[kSend] == plan.size() && mailbox[kRecv] == plan.size());
  uint32_t left = 1;
  check(cudaMemcpy(&left, arrivals, 4, cudaMemcpyDeviceToHost) == cudaSuccess && left == 0);
  // Refused: an unaligned span, a length not in 16-byte units, nothing to publish.
  const void* src[3] = {source[0] + 4, source[1], source[2]};
  void* dst[3] = {destination[0], destination[1], destination[2]};
  const uint32_t words[4] = {0, 1, 1, 4};
  const uint64_t fine[3] = {16, 16, kRow}, odd[3] = {16, 16, 100}, none[3] = {0, 0, 0};
  check(cuteafd_host_publish(src, dst, fine, mailbox + kReady, mailbox + kSend, mailbox + kDescriptor, words,
      arrivals, stream) != cudaSuccess);
  src[0] = source[0];
  check(cuteafd_host_publish(src, dst, odd, mailbox + kReady, mailbox + kSend, mailbox + kDescriptor, words,
      arrivals, stream) != cudaSuccess);
  check(cuteafd_host_publish(src, dst, none, mailbox + kReady, mailbox + kSend, mailbox + kDescriptor, words,
      arrivals, stream) != cudaSuccess);
  check(cudaGraphExecDestroy(exec) == cudaSuccess && cudaGraphDestroy(graph) == cudaSuccess);
  std::cout << "PASS host publish: " << plan.size() << " waves (1, 16, 64 rows eager; 1 and 7 rows replayed x3)\n";
  return 0;
}
