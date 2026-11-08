#include "memory_failure.h"
#include <atomic>

namespace {
std::atomic<cuteafd_memory_failure_callback_t> memory_failure_callback{nullptr};
std::atomic<cuteafd_memory_event_callback_t> memory_event_callback{nullptr};
}
extern "C" void cuteafd_set_memory_failure_callback(cuteafd_memory_failure_callback_t callback) {
  memory_failure_callback.store(callback, std::memory_order_release);
}
extern "C" void cuteafd_memory_failure(const cuteafd_memory_failure_t* failure) {
  if (const auto callback = memory_failure_callback.load(std::memory_order_acquire)) callback(failure);
}

extern "C" void cuteafd_set_memory_event_callback_v3(cuteafd_memory_event_callback_t callback) {
  memory_event_callback.store(callback, std::memory_order_release);
}
extern "C" void cuteafd_memory_event(int32_t space, uintptr_t ptr, size_t bytes, int32_t released, const char* file, int32_t line,
    uint32_t kind, const char* owner) {
  if (const auto callback = memory_event_callback.load(std::memory_order_acquire))
    callback(space, ptr, bytes, released, file, line, kind, owner);
}
