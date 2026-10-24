#pragma once
/* Weight-variant row steering, ABI v2 (weight-variants contract section 4).
 *
 * Per row r: v = vidx[r]; a row with v <= 0, v > variants or k[v] <= 0 is not read or
 * written. Otherwise, with kv = min(k[v], k_max):
 *   t_j  = sum_h b[v][j][h] * x[h]          (FP32, fixed reduction order)
 *   x[h] = bf16_rne(float(x[h]) - sum_j a[v][j][h] * t_j)
 * a and b are FP32 [V+1][k_max][h] (slot 0 unused), k is int32 [V+1].
 * variants is the actual initialized bound V in 1..7, never allocation capacity.
 * Initialize slot 0 and all holes. Reject invalid indices before any table read.
 * Tables, V, strides, ranks and pointers stay immutable through graph retirement;
 * only vidx changes between replays.
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

/* Loads the steering and delta programs onto the current device. Call once
 * per device before any graph capture; the launches load lazily only
 * outside capture. */
/* ABI identity is available without CUDA initialization. Old unbounded entry
 * points are deliberately not exported. */
#define CUTEAFD_STEER_ROWS_ABI_VERSION 2
int cuteafd_steer_rows_abi_version(void);
int cuteafd_steer_rows_load_v2(void);

int cuteafd_steer_rows_v2(
    __nv_bfloat16* x, int64_t ld,
    int32_t rows, int32_t h,
    const int32_t* vidx,
    const float* a, const float* b,
    const int32_t* k,
    int32_t k_max, int32_t variants, cudaStream_t stream);

/* Stored-delta sites: y[r][i] = bf16_rne(float(y[r][i]) + d[r][i]) on the
 * first h columns when 0 < v = vidx[r] <= variants and on[v] != 0; every other row is
 * not read or written. One FP32 add and one rounding per element.
 * h is a multiple of 8 in 8..65536, ld_y >= h and ld_d >= h are multiples of
 * 8 and 4, y and d are 16-byte aligned; anything else returns
 * cudaErrorInvalidValue. rows == 0 launches nothing. */
int cuteafd_add_rows_v2(
    __nv_bfloat16* y, int64_t ld_y,
    const float* d, int64_t ld_d,
    int32_t rows, int32_t h,
    const int32_t* vidx, const int32_t* on,
    int32_t variants, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
