#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define CUTEAFD_PACK_HD __host__ __device__
#else
#define CUTEAFD_PACK_HD
#endif
/* Shared inverse lane mapping, also exercised by the CPU native fixture. */
static CUTEAFD_PACK_HD inline void cuteafd_expert_pack_coordinates(
    uint64_t index, uint32_t k, int scales, uint32_t* row, uint32_t* col) {
  const uint32_t tile_elements = scales ? 1024 : 4096;
  const uint64_t tile = index / tile_elements;
  const uint32_t local = index % tile_elements;
  const uint32_t k_tiles = (k + 127) / 128;
  const uint32_t nt = tile / k_tiles, kt = tile % k_tiles;
  if (scales) {
    *row = nt * 256 + local / 4;
    *col = kt * 4 + local % 4;
  } else {
    const uint32_t n8i = local & 3;
    const uint32_t combined = (local >> 2) & 31;
    const uint32_t n8c = (local >> 7) & 7;
    const uint32_t k32 = local >> 10;
    *row = nt * 256 + n8c * 32 + n8i * 8 + (combined >> 2);
    *col = kt * 16 + k32 * 4 + (combined & 3);
  }
}
static CUTEAFD_PACK_HD inline uint64_t cuteafd_nvfp4_scale_offset(
    uint64_t row, uint64_t col, uint64_t padded_cols) {
  return (row / 128) * (padded_cols * 128) + (col / 4) * 512
      + (row % 32) * 16 + ((row / 32) % 4) * 4 + col % 4;
}
#undef CUTEAFD_PACK_HD
