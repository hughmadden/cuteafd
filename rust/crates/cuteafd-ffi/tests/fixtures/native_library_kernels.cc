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

extern "C" int32_t cuteafd_load_widen_bf16_f32(const void* src, void* dst, int64_t n, void* stream) {
  (void)stream;
  for (int64_t i = 0; i < n; ++i)
    static_cast<uint32_t*>(dst)[i] = cuteafd_load_quant::widen_bf16(static_cast<const uint16_t*>(src)[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_kvb_absorb(const void* kv_b, void* uk, void* uv, int64_t h0, int64_t heads,
                                           int64_t nope, int64_t v, int64_t lat, void* stream) {
  (void)stream;
  const auto* source = static_cast<const uint16_t*>(kv_b);
  for (int64_t i = 0; i < heads * lat * nope; ++i)
    static_cast<uint16_t*>(uk)[i] = cuteafd_load_quant::absorb_uk(source, h0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * v * lat; ++i)
    static_cast<uint16_t*>(uv)[i] = cuteafd_load_quant::absorb_uv(source, h0, nope, v, lat, i);
  return 0;
}
