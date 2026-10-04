#include "cuteafd_experts.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace {

#ifndef CUTEAFD_V41_EXACT_SPARK_SLICES
#define CUTEAFD_V41_EXACT_SPARK_SLICES 0
#endif
constexpr bool kExactTp4Storage = CUTEAFD_V41_EXACT_SPARK_SLICES != 0;

// N256/K128 lane-major representation consumed by b12x W4A8.
template<bool Gated, bool Scales>
__global__ void pack(const uint8_t* first, const uint8_t* second, uint8_t* output,
    uint32_t n, uint32_t k, uint32_t n_pad, uint64_t count) {
  const uint32_t k_tiles = (k + 127) / 128;
  constexpr uint32_t tile_elements = Scales ? 1024 : 4096;
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t tile = index / tile_elements;
    const uint32_t local = index % tile_elements;
    const uint32_t nt = tile / k_tiles, kt = tile % k_tiles;
    uint32_t row, col;
    if constexpr (Scales) {
      row = nt * 256 + local / 4;
      col = kt * 4 + local % 4;
    } else {
      const uint32_t n8i = local & 3;
      const uint32_t combined = (local >> 2) & 31;
      const uint32_t n8c = (local >> 7) & 7;
      const uint32_t k32 = local >> 10;
      row = nt * 256 + n8c * 32 + n8i * 8 + (combined >> 2);
      col = kt * 16 + k32 * 4 + (combined & 3);
    }
    const uint8_t* source = first;
    if constexpr (Gated) {
      if (row >= n_pad) { row -= n_pad; source = second; }
    }
    constexpr uint32_t divisor = Scales ? 32 : 8;
    const bool valid = row < n && col < k / divisor;
    if constexpr (Scales) {
      output[index] = valid ? source[uint64_t(row) * (k / divisor) + col] : 0;
    } else {
      reinterpret_cast<uint32_t*>(output)[index] = valid
        ? reinterpret_cast<const uint32_t*>(source)[uint64_t(row) * (k / divisor) + col] : 0;
    }
  }
}

// Exact TP4 layout used by both grouped slices and the compact N64-tail path.
// FC1 stores each projection in N128 tiles (last tile N64); FC2 stores
// N128/K128 tiles (last K tile K64). No resident payload or scale padding.
template<bool Gated, bool Scales>
__global__ void pack_tp4_exact(const uint8_t* first, const uint8_t* second,
    uint8_t* output, uint32_t hidden, uint64_t count) {
  constexpr uint32_t intermediate = 576;
  constexpr uint32_t divisor = Scales ? 32 : 8;
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += uint64_t(gridDim.x) * blockDim.x) {
    const uint8_t* source = first;
    uint32_t row, col;
    if constexpr (Gated) {
      const uint64_t half = uint64_t(intermediate) * (hidden / divisor);
      uint64_t local = index;
      if (local >= half) { local -= half; source = second; }
      const uint32_t tile = local / (128 * (hidden / divisor));
      local %= 128 * (hidden / divisor);
      const uint32_t rows = tile == 4 ? 64 : 128;
      if constexpr (Scales) {
        const uint32_t kt = local / (rows * 4);
        local %= rows * 4;
        row = tile * 128 + local / 4;
        col = kt * 4 + local % 4;
      } else {
        const uint32_t kb = local / (rows * 4);
        local %= rows * 4;
        const uint32_t chunk = local / 128;
        const uint32_t lane_word = local % 128;
        row = tile * 128 + chunk * 32 + (lane_word % 4) * 8 + lane_word / 16;
        col = kb * 4 + (lane_word / 4) % 4;
      }
    } else {
      const uint32_t nt = index / (128 * (intermediate / divisor));
      uint32_t local = index % (128 * (intermediate / divisor));
      if constexpr (Scales) {
        const uint32_t kt = local / 512;
        local %= 512;
        const uint32_t cols = kt == 4 ? 2 : 4;
        row = nt * 128 + local / cols;
        col = kt * 4 + local % cols;
      } else {
        const uint32_t kb = local / 512;
        local %= 512;
        row = nt * 128 + (local / 128) * 32 + (local % 4) * 8 + (local % 128) / 16;
        col = kb * 4 + (local / 4) % 4;
      }
    }
    const uint32_t k = Gated ? hidden : intermediate;
    if constexpr (Scales) output[index] = source[uint64_t(row) * (k / divisor) + col];
    else reinterpret_cast<uint32_t*>(output)[index] =
        reinterpret_cast<const uint32_t*>(source)[uint64_t(row) * (k / divisor) + col];
  }
}

bool overlaps(const void* a, uint64_t an, const void* b, uint64_t bn) {
  const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
  return av <= bv ? bv - av < an : av - bv < bn;
}
}

extern "C" int32_t cuteafd_expert_packed_sizes(uint32_t intermediate,
    uint64_t bytes[4]) {
  // Per-rank intermediate extents are multiples of 32 so the K/32 scale axis is
  // exact. Opt-in TP4 uses its exact N64-tail representation; other geometries
  // retain the N256/K128 pack contract.
  const uint64_t hidden = cuteafd_expert_hidden();
  if (!bytes || !intermediate || intermediate % 32 != 0 || intermediate > 8192)
    return cudaErrorInvalidValue;
  const uint64_t padded = kExactTp4Storage && intermediate == 576 && hidden == 5120 ? intermediate : (intermediate + 127) / 128 * 128;
  bytes[0] = padded * hidden;
  bytes[1] = padded * hidden / 16;
  bytes[2] = hidden * padded / 2;
  bytes[3] = hidden * padded / 32;
  return cudaSuccess;
}

extern "C" int32_t cuteafd_pack_expert_async(const uint8_t* const sources[6],
    uint8_t* const destinations[4], uint32_t intermediate, void* stream) {
  uint64_t sizes[4];
  if (!sources || !destinations || cuteafd_expert_packed_sizes(intermediate, sizes))
    return cudaErrorInvalidValue;
  const uint32_t hidden = cuteafd_expert_hidden();
  const uint64_t weight_bytes = uint64_t(intermediate) * hidden / 2;
  const uint64_t source_sizes[] = {weight_bytes, weight_bytes, weight_bytes,
    weight_bytes / 16, weight_bytes / 16, weight_bytes / 16};
  for (int i = 0; i < 6; ++i)
    if (!sources[i] || (i < 3 && reinterpret_cast<uintptr_t>(sources[i]) % 4))
      return cudaErrorInvalidValue;
  for (int i = 0; i < 4; ++i) {
    if (!destinations[i] || reinterpret_cast<uintptr_t>(destinations[i]) % 16)
      return cudaErrorInvalidValue;
    for (int j = 0; j < 6; ++j)
      if (overlaps(destinations[i], sizes[i], sources[j], source_sizes[j])) return cudaErrorInvalidValue;
    for (int j = 0; j < i; ++j)
      if (overlaps(destinations[i], sizes[i], destinations[j], sizes[j])) return cudaErrorInvalidValue;
  }
  const uint32_t padded = (intermediate + 127) / 128 * 128;
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  if (kExactTp4Storage && intermediate == 576 && hidden == 5120) {
    pack_tp4_exact<true, false><<<256, 256, 0, cuda_stream>>>(sources[1], sources[0], destinations[0], hidden, sizes[0] / 4);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    pack_tp4_exact<true, true><<<256, 256, 0, cuda_stream>>>(sources[4], sources[3], destinations[1], hidden, sizes[1]);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    pack_tp4_exact<false, false><<<256, 256, 0, cuda_stream>>>(sources[2], nullptr, destinations[2], hidden, sizes[2] / 4);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    pack_tp4_exact<false, true><<<256, 256, 0, cuda_stream>>>(sources[5], nullptr, destinations[3], hidden, sizes[3]);
    return cudaGetLastError();
  }
  // First half is up (W3), second half gate (W1), padded independently.
  pack<true, false><<<256, 256, 0, cuda_stream>>>(sources[1], sources[0], destinations[0],
      intermediate, hidden, padded, sizes[0] / 4);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<true, true><<<256, 256, 0, cuda_stream>>>(sources[4], sources[3], destinations[1],
      intermediate, hidden, padded, sizes[1]);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<false, false><<<256, 256, 0, cuda_stream>>>(sources[2], nullptr, destinations[2],
      hidden, intermediate, 0, sizes[2] / 4);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  pack<false, true><<<256, 256, 0, cuda_stream>>>(sources[5], nullptr, destinations[3],
      hidden, intermediate, 0, sizes[3]);
  return cudaGetLastError();
}
