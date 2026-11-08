#pragma once
#include "cuteafd_memory_kind.h"
#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct cuteafd_memory_failure_t {
  const char* api;
  const char* domain;
  const char* file;
  int32_t line;
  int32_t code;
  size_t bytes;
  int32_t bytes_known;
  uintptr_t handle;
} cuteafd_memory_failure_t;
typedef void (*cuteafd_memory_failure_callback_t)(const cuteafd_memory_failure_t*);
void cuteafd_set_memory_failure_callback(cuteafd_memory_failure_callback_t callback);
void cuteafd_memory_failure(const cuteafd_memory_failure_t* failure);
// An allocation event names the tag it was made under: kind is a
// CUTEAFD_MEMORY_KIND_* value and owner a NUL-terminated name the callback
// copies synchronously (never retained). Releases pass kind 0 and no owner.
typedef void (*cuteafd_memory_event_callback_t)(int32_t space, uintptr_t ptr, size_t bytes, int32_t released,
    const char* file, int32_t line, uint32_t kind, const char* owner);
void cuteafd_set_memory_event_callback_v3(cuteafd_memory_event_callback_t callback);
void cuteafd_memory_event(int32_t space, uintptr_t ptr, size_t bytes, int32_t released, const char* file, int32_t line,
    uint32_t kind, const char* owner);
#ifdef __cplusplus
}

// Evaluate the raw call once and retain its API-domain status before cleanup.
template <typename Status>
inline Status cuteafd_checked_memory_call(const char* api, const char* domain,
    Status status, size_t bytes, int known, uintptr_t handle,
    const char* file, int line) {
  const int code = static_cast<int>(status);
  const bool memory_failure = (strcmp(domain, "cuda") == 0 && code == 2)
      || (strcmp(domain, "cublas") == 0 && code == 3)
      || (strcmp(domain, "libc") == 0 && code == ENOMEM);
  if (memory_failure) {
    const cuteafd_memory_failure_t failure = {
        api, domain, file, line, static_cast<int32_t>(status), bytes, known, handle};
    cuteafd_memory_failure(&failure);
  }
  return status;
}
template <typename Pointer, typename Call>
inline auto cuteafd_tracked_alloc(Pointer ptr, size_t bytes, int space, uint32_t kind, const char* owner,
    Call call, const char* api, const char* file, int line) {
  const auto status = cuteafd_checked_memory_call(api, "cuda", call(), bytes, 1, 0, file, line);
  if (static_cast<int>(status) == 0)
    cuteafd_memory_event(space, reinterpret_cast<uintptr_t>(*ptr), bytes, 0, file, line, kind, owner);
  return status;
}
template <typename Pointer, typename Call>
inline auto cuteafd_tracked_free(Pointer ptr, Call call) {
  const auto status = call();
  if (static_cast<int>(status) == 0)
    cuteafd_memory_event(0, reinterpret_cast<uintptr_t>(ptr), 0, 1, nullptr, 0, 0, nullptr);
  return status;
}
// Every tracked allocation states its tag (kind, owner) like the public allocators.
#define CUTEAFD_TRACK_ALLOC(api, ptr, bytes, space, kind, owner, call) \
  cuteafd_tracked_alloc(ptr, bytes, space, kind, owner, [&]() { return (call); }, api, __FILE__, __LINE__)
#define CUTEAFD_TRACK_FREE(ptr, call) \
  cuteafd_tracked_free(ptr, [&]() { return (call); })
#define CUTEAFD_CHECK_MEMORY(api, domain, call, bytes, known, handle) \
  cuteafd_checked_memory_call(api, domain, (call), bytes, known, \
      reinterpret_cast<uintptr_t>(handle), __FILE__, __LINE__)
#endif

#ifdef __cplusplus
inline void* cuteafd_checked_malloc(size_t bytes, const char* file, int line) {
  void* ptr = ::malloc(bytes);
  if (!ptr) cuteafd_checked_memory_call("malloc", "libc", ENOMEM, bytes, 1, 0, file, line);
  return ptr;
}
inline void* cuteafd_tracked_malloc(size_t bytes, uint32_t kind, const char* owner, const char* file, int line) {
  void* ptr = cuteafd_checked_malloc(bytes, file, line);
  if (ptr) cuteafd_memory_event(3, reinterpret_cast<uintptr_t>(ptr), bytes, 0, file, line, kind, owner);
  return ptr;
}
inline void cuteafd_tracked_host_free(void* ptr) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  ::free(ptr);
  cuteafd_memory_event(3, address, 0, 1, nullptr, 0, 0, nullptr);
}
#define CUTEAFD_TRACK_MALLOC(bytes, kind, owner) cuteafd_tracked_malloc(bytes, kind, owner, __FILE__, __LINE__)
#define CUTEAFD_HOST_FREE(ptr) cuteafd_tracked_host_free(ptr)
#define CUTEAFD_MALLOC(bytes) cuteafd_checked_malloc(bytes, __FILE__, __LINE__)
#endif
