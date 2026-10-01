#include "cuteafd_exl3_wire.h"
#include "cuteafd_experts.h"
#include "expert_hidden.cuh"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <new>

namespace {
struct Context { int device; };
// Rows are Hidden E4M3 bytes then Hidden/32 UE8M0 scales; Hidden is the process
// expert hidden size (cuteafd_set_expert_hidden), a constant per instantiation.
template<uint64_t Hidden>
__global__ void decode_wire(const uint8_t* input, __nv_bfloat16* output, uint64_t elements) {
  constexpr uint64_t kRow = Hidden + Hidden / 32;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < elements; i += uint64_t(gridDim.x) * blockDim.x) {
    const uint64_t row = i / Hidden, col = i % Hidden;
    __nv_fp8_e4m3 value; value.__x = input[row * kRow + col];
    __nv_fp8_e8m0 scale; scale.__x = input[row * kRow + Hidden + col / 32];
    output[i] = __float2bfloat16_rn(__fmul_rn(static_cast<float>(value), static_cast<float>(scale)));
  }
}
}

extern "C" int32_t cuteafd_exl3_wire_initialize(void** out) {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  auto* context = new(std::nothrow) Context;
  if (!context) return cudaErrorMemoryAllocation;
  auto status = cudaGetDevice(&context->device);
  cudaFuncAttributes attributes{};
  if (status == cudaSuccess) status = cudaFuncGetAttributes(&attributes, decode_wire<5120>);
  if (status != cudaSuccess) { delete context; return status; }
  *out = context; return cudaSuccess;
}
extern "C" void cuteafd_exl3_wire_destroy(void* handle) {
  delete static_cast<Context*>(handle);
}
extern "C" int32_t cuteafd_exl3_wire_decode(void* handle, const uint8_t* input,
    uint64_t input_bytes, uint16_t* output, uint64_t output_bytes, uint32_t rows, void* stream) {
  if (!handle || !input || !output || rows < 1 || rows > 4096) return cudaErrorInvalidValue;
  const uint64_t hidden = cuteafd_expert_hidden();
  const uint64_t in_size = uint64_t(rows) * (hidden + hidden / 32), out_size = uint64_t(rows) * hidden * 2;
  const auto in = reinterpret_cast<uintptr_t>(input), out = reinterpret_cast<uintptr_t>(output);
  if (in % 16 || out % 16 || input_bytes < in_size || output_bytes < out_size ||
      in > UINTPTR_MAX - in_size || out > UINTPTR_MAX - out_size ||
      (in < out + out_size && out < in + in_size)) return cudaErrorInvalidValue;
  int device = -1; auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) return status;
  if (device != static_cast<Context*>(handle)->device) return cudaErrorInvalidDevice;
  const uint64_t elements = uint64_t(rows) * hidden;
  const unsigned blocks = static_cast<unsigned>((elements + 255) / 256);
  CUTEAFD_WITH_HIDDEN(hidden, decode_wire<kHidden><<<blocks < 1024 ? blocks : 1024, 256, 0,
      static_cast<cudaStream_t>(stream)>>>(input, reinterpret_cast<__nv_bfloat16*>(output), elements));
  return cudaGetLastError();
}
