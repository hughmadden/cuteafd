#pragma once
/* Weight-variant row steering, ABI v1 (weight-variants contract section 4).
 *
 * Per row r: v = vidx[r]; a row with v <= 0 or k[v] <= 0 is not read or
 * written. Otherwise, with kv = min(k[v], k_max):
 *   t_j  = sum_h b[v][j][h] * x[h]          (FP32, fixed reduction order)
 *   x[h] = bf16_rne(float(x[h]) - sum_j a[v][j][h] * t_j)
 * a and b are FP32 [V+1][k_max][h] (slot 0 unused), k is int32 [V+1].
 * Deterministic and batch-invariant; no allocation and no host sync, so
 * the launch can be captured in a CUDA graph. steer_rows_ref in
 * cuteafd-core copies the reduction order bit for bit.
 *
 * Shapes: h is a multiple of 128 in 128..8192, ld >= h with ld % 8 == 0,
 * k_max in 1..32; x, a and b are 16-byte aligned. Anything else returns
 * cudaErrorInvalidValue and launches nothing. rows == 0 launches nothing. */
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Loads both programs onto the current device. Call once per device before
 * any graph capture; cuteafd_steer_rows loads lazily only outside capture. */
int cuteafd_steer_rows_load(void);

int cuteafd_steer_rows(
    __nv_bfloat16* x, int64_t ld,
    int32_t rows, int32_t h,
    const int32_t* vidx,
    const float* a, const float* b,
    const int32_t* k,
    int32_t k_max, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
