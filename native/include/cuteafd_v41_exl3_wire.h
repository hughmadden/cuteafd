#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Input rows: H E4M3 bytes followed by H/32 UE8M0 K32 scale bytes, where H is
 * the process expert hidden size (cuteafd_set_expert_hidden; 5120 for V4.1).
 * Output: contiguous BF16[rows,H], rounded after reconstructing FP32 values.
 * Uses CUDA's E4M3/UE8M0 conversion semantics (including subnormal/NaN codes).
 * Initialize before capture, on the owning device. Caller retains distinct,
 * 16-byte-aligned buffers and this handle through completion/graph destruction.
 * Launch performs no allocation or host synchronization; 1 <= rows <= 4096.
 * Return values are CUDA runtime status codes. Destroy on the owning thread. */
int32_t cuteafd_v41_exl3_wire_initialize(void** out);
void cuteafd_v41_exl3_wire_destroy(void* handle);
int32_t cuteafd_v41_exl3_wire_decode(void* handle, const uint8_t* input,
    uint64_t input_bytes, uint16_t* output, uint64_t output_bytes,
    uint32_t rows, void* stream);
#ifdef __cplusplus
}
#endif
