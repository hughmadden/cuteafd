#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace cuteafd {
// Internal test seam; production always supplies cudaDeviceGetAttribute.
using PeerAttributeQuery = decltype(&cudaDeviceGetAttribute);
int32_t peer_copy_with_query(void* destination, const void* source,
    uint64_t bytes, void* stream, PeerAttributeQuery query);
int32_t peer_copy_rows_with_query(void* destination, const void* source,
    uint64_t width, uint64_t rows, uint64_t destination_pitch,
    uint64_t source_pitch, void* stream, PeerAttributeQuery query);
}  // namespace cuteafd
