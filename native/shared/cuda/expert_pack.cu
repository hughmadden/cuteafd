#include "cuteafd_experts.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include "expert_pack_layout.h"

namespace {

// N256/K128 lane-major representation consumed by b12x W4A8.
template<bool Gated, bool Scales>
__global__ void pack(const uint8_t* first, const uint8_t* second, uint8_t* output,
    uint32_t n, uint32_t k, uint32_t n_pad, uint64_t count) {
  for (uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += uint64_t(gridDim.x) * blockDim.x) {
    uint32_t row, col;
    cuteafd_expert_pack_coordinates(index, k, Scales, &row, &col);
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

bool overlaps(const void* a, uint64_t an, const void* b, uint64_t bn) {
  const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
  return av <= bv ? bv - av < an : av - bv < bn;
}
}

extern "C" int32_t cuteafd_expert_packed_sizes(uint32_t intermediate,
    uint64_t bytes[4]) {
  // Per-rank intermediate extents are multiples of 32 so the K/32 scale axis is
  // exact; the 128 padding below is storage-only (V4.1 TP4 576 -> 640).
  const uint64_t hidden = cuteafd_expert_hidden();
  if (!bytes || !intermediate || intermediate % 32 != 0 || intermediate > 8192)
    return cudaErrorInvalidValue;
  const uint64_t padded = (intermediate + 127) / 128 * 128;
  bytes[0] = padded * hidden;
  bytes[1] = padded * hidden / 16;
  bytes[2] = hidden * padded / 2;
  bytes[3] = hidden * padded / 32;
  return cudaSuccess;
}

// One plane of the already-fused source image. Reuse the exact mount pack
// instantiations; the caller owns one out-of-place scratch across all planes.
extern "C" int32_t cuteafd_pack_expert_plane_async(
    const uint8_t* source, uint64_t source_bytes, uint8_t* destination,
    uint64_t destination_bytes, uint32_t intermediate, uint32_t plane, void* stream) {
  uint64_t sizes[4];
  if (plane >= 4 || !source || !destination ||
      cuteafd_expert_packed_sizes(intermediate, sizes)) return cudaErrorInvalidValue;
  const auto hidden = cuteafd_expert_hidden();
  const auto padded = (intermediate + 127) / 128 * 128;
  const bool gated = plane < 2, scales = plane % 2;
  const uint64_t required = gated ? sizes[plane]
      : uint64_t(hidden) * intermediate / (scales ? 32 : 2);
  if (source_bytes < required || destination_bytes < sizes[plane] ||
      reinterpret_cast<uintptr_t>(source) % 4 ||
      reinterpret_cast<uintptr_t>(destination) % 16 ||
      overlaps(source, required, destination, sizes[plane])) return cudaErrorInvalidValue;
  int device, sms;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) return status;
  if (sms <= 0) return cudaErrorInvalidDevice;
  const uint64_t count = sizes[plane] / (scales ? 1 : 4);
  const unsigned blocks = static_cast<unsigned>(
      std::min<uint64_t>((count + 255) / 256, uint64_t(sms) * 4));
  auto s = static_cast<cudaStream_t>(stream);
  if (gated && !scales)
    pack<false, false><<<blocks, 256, 0, s>>>(source, nullptr, destination,
        2 * padded, hidden, 0, sizes[plane] / 4);
  else if (gated)
    pack<false, true><<<blocks, 256, 0, s>>>(source, nullptr, destination,
        2 * padded, hidden, 0, sizes[plane]);
  else if (!scales)
    pack<false, false><<<blocks, 256, 0, s>>>(source, nullptr, destination,
        hidden, intermediate, 0, sizes[plane] / 4);
  else
    pack<false, true><<<blocks, 256, 0, s>>>(source, nullptr, destination,
        hidden, intermediate, 0, sizes[plane]);
  return cudaGetLastError();
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
