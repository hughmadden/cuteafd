#pragma once
#include <stddef.h>
#include <stdint.h>
// The two device backings a media runtime (vision, audio, PCM tower) owns:
// its immutable weight arena and its scratch arena. Rust reads them after
// create and records each under the tags it chose for the tower; it retires
// each only when the matching released bit says that backing's free succeeded.
#ifdef __cplusplus
extern "C" {
#endif
typedef struct cuteafd_media_backings {
  void* weights;
  uint64_t weight_bytes;
  void* workspace;
  uint64_t workspace_bytes;
} cuteafd_media_backings;
enum {
  CUTEAFD_MEDIA_RELEASED_WEIGHTS = 1u,
  CUTEAFD_MEDIA_RELEASED_WORKSPACE = 2u,
};
// The tags a tower's two backings are allocated under (kind is a
// CUTEAFD_MEMORY_KIND_* value; owners are NUL-terminated and nonempty).
typedef struct cuteafd_media_tags {
  uint32_t weight_kind;
  const char* weight_owner;
  uint32_t workspace_kind;
  const char* workspace_owner;
} cuteafd_media_tags;
// The ledger has one recorder per backing: Rust records both tower backings
// (get_backings, after create, including a partial owner create returns) and
// retires each on its released bit, so tower allocations and frees emit no
// native memory events.

// Create contract for every tower: on failure *out is null when every backing
// was released, and otherwise holds the owner with exactly its still-live
// backings, which the caller records (get_backings) and destroys later.
#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && defined(__CUDACC__)
#include <cuda_runtime.h>
#include "memory_failure.h"
// The tagged allocator the towers use (cudaError_t status): the tag is checked
// and retained as this thread's attempted tag before cudaMalloc, so a tower OOM
// reaches the failure callback naming the tag it asked for.
inline int32_t cuteafd_media_allocate(void** ptr, size_t bytes, uint32_t kind, const char* owner) {
  if (ptr == nullptr || bytes == 0) return static_cast<int32_t>(cudaErrorInvalidValue);
  *ptr = nullptr;
  if (cuteafd_tag_check(kind, owner) == 0) return static_cast<int32_t>(cudaErrorInvalidValue);
  return static_cast<int32_t>(CUTEAFD_CHECK_MEMORY("cudaMalloc", "cuda", cudaMalloc(ptr, bytes), bytes, 1, nullptr));
}
#endif
