#include "memory_failure.h"
#include <atomic>
#include <cstring>
#include <string>

namespace {
// The most recent tag this thread attempted, retained (owner copied) so a
// failure report raised after the allocator returns can name what was asked for.
thread_local uint32_t attempted_kind = 0;
thread_local std::string attempted_owner;
}
extern "C" int32_t cuteafd_tag_check(uint32_t kind, const char* owner) {
  if (kind == 0 || kind > CUTEAFD_MEMORY_KIND_COUNT || owner == nullptr || owner[0] == '\0') return 0;
  attempted_kind = kind;
  attempted_owner = owner;
  return 1;
}
extern "C" int32_t cuteafd_last_allocation_tag(uint32_t* kind, char* owner, size_t owner_bytes, size_t* required) {
  if (kind == nullptr || required == nullptr || (owner == nullptr && owner_bytes != 0)) return 1;  // INVALID_ARGUMENT
  *kind = attempted_kind;
  *required = attempted_owner.size() + 1;
  if (*required > owner_bytes) return 2;  // BUFFER_TOO_SMALL
  std::memcpy(owner, attempted_owner.c_str(), *required);
  return 0;
}

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
