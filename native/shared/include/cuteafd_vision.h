#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Version 1: donated MiMo numerics (FP32 residual, BF16 dense inputs,
 * FP16 rotary Q/K/V). All offsets address an admitted resident weight arena.
 * matrices: BF16; vectors (norm/bias/sinks/inv_freq): FP32. */
#define CUTEAFD_VISION_NUMERICS 1
#define CUTEAFD_VISION_NO_OFFSET UINT64_MAX
#define CUTEAFD_VISION_DEPTH 28

typedef struct {
  uint64_t qkv, qkv_bias, proj, proj_bias, gate_up, gate_up_bias;
  uint64_t down, down_bias, norm1, norm2, key0_bias;
  int32_t window, column_order;
} cuteafd_vision_block;

typedef struct {
  uint32_t abi_version, max_tokens, output_width, reserved;
  uint64_t weight_bytes, patch, merger_norm, merger_fc1, merger_fc2, inv_freq;
  cuteafd_vision_block blocks[CUTEAFD_VISION_DEPTH];
} cuteafd_vision_spec;

typedef struct {
  uint64_t weights, scratch, blas_workspace, device_allocations, encodes;
} cuteafd_vision_ledger;

/* Probe-only observer: stage 0 patch, 1..28 blocks, 29 merger norm, 30 output.
 * data is device FP32 (stage 30 is BF16); column_order applies to block rows.
 * The observer must not retain data past its return, and must propagate errors.
 * Owner remains alive until all queued CUDA work has drained, including errors. */
typedef int32_t (*cuteafd_vision_observer)(void*, int32_t stage, const void* data,
                                         int32_t rows, int32_t width, int32_t column_order);

/* Admission is checked before any CUDA allocation. Native create owns a
 * lowest-priority nonblocking stream; no CUDA allocation/capture in encode. */
int32_t cuteafd_vision_required(const cuteafd_vision_spec*, cuteafd_vision_ledger*);
int32_t cuteafd_vision_create(const cuteafd_vision_spec*, int32_t device,
                             uint64_t admitted_bytes, void** owner);
int32_t cuteafd_vision_upload(void* owner, uint64_t offset, const void*, uint64_t bytes);
/* RGB8 is resized, row-major HWC, width=grid_w*16 and height=grid_h*16.
 * lut is FP32 [3,256], host-provided exact reference normalization. */
int32_t cuteafd_vision_encode(void* owner, const uint8_t* rgb, uint64_t rgb_bytes,
                             const float* lut, int32_t grid_h, int32_t grid_w,
                             uint16_t* output, uint64_t output_bytes,
                             cuteafd_vision_observer observer, void* observer_context);
int32_t cuteafd_vision_get_ledger(void* owner, cuteafd_vision_ledger*);
int32_t cuteafd_vision_destroy(void* owner);

/* Shared toolkit. Inputs FP32, output BF16; bias may be null. Serialize calls
 * on a current-device stream and retain every extent through completion. */
int32_t cuteafd_vision_rmsnorm_bf16(const float*, const float*, uint16_t*, int32_t, int32_t, float, void*);
int32_t cuteafd_vision_layernorm_bf16(const float*, const float*, const float*, uint16_t*, int32_t, int32_t, float, void*);
int32_t cuteafd_vision_swiglu_bf16(const float*, const float*, uint16_t*, int64_t, int32_t, float, void*);
int32_t cuteafd_vision_gelu_bf16(const float*, uint16_t*, int64_t, int32_t, void*);

/* Scatter feature rows into [rows,copies,width] gathered token embeddings.
 * Indices must be unique (enforced by host admission); out-of-range rows are
 * ignored defensively. No allocation, no synchronization. Returns CUDA status. */
int32_t cuteafd_embed_inject(const uint16_t* features, const uint32_t* indices,
                            uint16_t* out, int32_t feature_rows, int32_t rows,
                            int32_t width, int32_t copies, void* stream);
#ifdef __cplusplus
}
#endif
