#include "v41_capacity.h"
#include "cuteafd_v41_dspark_cache.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
static_assert(sizeof(cuteafd_v41_kv_write_t) == 24);
namespace {
__global__ void write_window(const __nv_bfloat16* source, const cuteafd_v41_kv_write_t* writes,
    uint8_t* ring, uint32_t source_rows, uint32_t slots) {
  const auto write = writes[blockIdx.y];
  if (!write.token_count || write.slot >= slots || write.source_row >= source_rows ||
      write.token_count > source_rows - write.source_row ||
      write.position > UINT64_MAX - write.token_count || write.reserved) return;
  const uint32_t kept = min(write.token_count, 128u);
  if (blockIdx.x >= kept) return;
  // Ignore overwritten prefix rows so no two blocks write the same ring row.
  const uint64_t offset = uint64_t(write.token_count - kept) + blockIdx.x;
  const uint64_t input_row = uint64_t(write.source_row) + offset;
  const uint64_t output_row = uint64_t(write.slot) * 128 + (write.position + offset) % 128;
  const int col = threadIdx.x;
  const float value = __bfloat162float(source[input_row * 512 + col]);
  float maximum = fabsf(value);
  for (int shift = 16; shift; shift >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, shift));
  maximum = fmaxf(maximum, 1e-4f);
  const uint32_t bits = __float_as_uint(maximum * (1.0f / 448.0f));
  const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
  const float scale = ldexpf(1.0f, exponent);
  const __nv_fp8_e4m3 quantized(value / scale);
  ring[output_row * CUTEAFD_V41_DSPARK_KV_ROW_BYTES + col] = quantized.__x;
  if ((col % 32) == 0)
    ring[output_row * CUTEAFD_V41_DSPARK_KV_ROW_BYTES + 512 + col / 32] = uint8_t(exponent + 127);
}
// The same rows one warp each, four to a 128-thread block: lane l holds
// columns [16l, 16l + 16) (two 16-byte loads), so a 32-column scale group is
// lanes 2j and 2j + 1. Per element the arithmetic is write_window's (the
// group maximum is exact in any order), so the ring bytes are identical; the
// grid has 16x fewer threads, which a commit of a few rows spends launching
// blocks that write nothing. Needs 16-byte aligned source and ring.
constexpr int kRowsPerBlock = 4;
__global__ void write_window_rows(const __nv_bfloat16* source, const cuteafd_v41_kv_write_t* writes,
    uint8_t* ring, uint32_t source_rows, uint32_t slots) {
  const auto write = writes[blockIdx.y];
  if (!write.token_count || write.slot >= slots || write.source_row >= source_rows ||
      write.token_count > source_rows - write.source_row ||
      write.position > UINT64_MAX - write.token_count || write.reserved) return;
  const uint32_t kept = min(write.token_count, 128u);
  const uint32_t row = blockIdx.x * kRowsPerBlock + threadIdx.x / 32;
  if (row >= kept) return;
  const uint32_t lane = threadIdx.x % 32;
  // Ignore overwritten prefix rows so no two warps write the same ring row.
  const uint64_t offset = uint64_t(write.token_count - kept) + row;
  const uint64_t input_row = uint64_t(write.source_row) + offset;
  const uint64_t output_row = uint64_t(write.slot) * 128 + (write.position + offset) % 128;
  const uint4* from = reinterpret_cast<const uint4*>(source + input_row * 512 + lane * 16);
  const uint4 words[2] = {from[0], from[1]};
  const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(words);
  float maximum = 0.0f;
  for (int i = 0; i < 16; ++i) maximum = fmaxf(maximum, fabsf(__bfloat162float(values[i])));
  maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1));
  maximum = fmaxf(maximum, 1e-4f);
  const uint32_t bits = __float_as_uint(maximum * (1.0f / 448.0f));
  const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
  const float scale = ldexpf(1.0f, exponent);
  uint4 packed;
  uint8_t* bytes = reinterpret_cast<uint8_t*>(&packed);
  for (int i = 0; i < 16; ++i) bytes[i] = __nv_fp8_e4m3(__bfloat162float(values[i]) / scale).__x;
  uint8_t* out = ring + output_row * CUTEAFD_V41_DSPARK_KV_ROW_BYTES;
  *reinterpret_cast<uint4*>(out + lane * 16) = packed;
  if ((lane % 2) == 0) out[512 + lane / 2] = uint8_t(exponent + 127);
}
bool span(const void* ptr, uint64_t bytes, uint32_t alignment) {
  const auto start = reinterpret_cast<uintptr_t>(ptr);
  return start && start % alignment == 0 && start <= UINTPTR_MAX - bytes;
}
bool separate(const void* a, uint64_t an, const void* b, uint64_t bn) {
  const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
  return x + an <= y || y + bn <= x;
}
}
extern "C" int32_t cuteafd_v41_dspark_cache_write_fp8(const uint16_t* source,
    const cuteafd_v41_kv_write_t* writes, uint8_t* ring, int32_t source_rows,
    int32_t slots, void* stream) {
  if (source_rows < 1 || source_rows > 4096 || slots < 1 || slots > CUTEAFD_V41_MAX_REQUESTS) return cudaErrorInvalidValue;
  const uint64_t input_bytes = uint64_t(source_rows) * 1024, ring_bytes = uint64_t(slots) * 128 * CUTEAFD_V41_DSPARK_KV_ROW_BYTES;
  if (!span(source,input_bytes,2) || !span(writes,CUTEAFD_V41_WRITE_DESCRIPTOR_BYTES,8) || !span(ring,ring_bytes,1) ||
      !separate(source,input_bytes,writes,CUTEAFD_V41_WRITE_DESCRIPTOR_BYTES) || !separate(source,input_bytes,ring,ring_bytes) ||
      !separate(writes,CUTEAFD_V41_WRITE_DESCRIPTOR_BYTES,ring,ring_bytes)) return cudaErrorInvalidValue;
  if (reinterpret_cast<uintptr_t>(source) % 16 == 0 && reinterpret_cast<uintptr_t>(ring) % 16 == 0) {
    write_window_rows<<<dim3(128 / kRowsPerBlock, CUTEAFD_V41_MAX_REQUESTS), 32 * kRowsPerBlock, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const __nv_bfloat16*>(source), writes, ring,
        source_rows, slots);
    return cudaGetLastError();
  }
  write_window<<<dim3(128,CUTEAFD_V41_MAX_REQUESTS),512,0,reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<const __nv_bfloat16*>(source), writes,
      ring,source_rows,slots);
  return cudaGetLastError();
}
