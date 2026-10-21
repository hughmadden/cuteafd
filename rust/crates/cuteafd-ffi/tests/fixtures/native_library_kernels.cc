// CPU stand-ins for load-only native kernels, linked into the fake native DSO:
// the same shared rule the CUDA kernels run, executed sequentially.
#include <stdint.h>

#include "../../../../../native/shared/cuda/fp8_load_quant.h"

extern "C" int32_t cuteafd_fp8_load_quant(const void* jobs, int32_t count, void* stream) {
  (void)stream;
  if (count < 0) return 1;
  const auto* list = static_cast<const cuteafd_load_quant::Job*>(jobs);
  for (int32_t i = 0; i < count; ++i) cuteafd_load_quant::run_job(list[i]);
  return 0;
}

extern "C" int32_t cuteafd_fp8_load_quant_abi(uint64_t* job_bytes, uint64_t* segment_bytes, uint64_t* max_cols) {
  *job_bytes = sizeof(cuteafd_load_quant::Job);
  *segment_bytes = sizeof(cuteafd_load_quant::Segment);
  *max_cols = 512 * 128;
  return 0;
}
