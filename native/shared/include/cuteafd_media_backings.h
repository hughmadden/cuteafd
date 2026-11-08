#pragma once
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
#ifdef __cplusplus
}
#endif
