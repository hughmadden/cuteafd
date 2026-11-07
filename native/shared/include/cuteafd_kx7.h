// KX7 exact Huffman-packed BF16 matrices (native/shared/cuda/kx7.cu; format
// and packer in cuteafd_core::kx7). A pack reaches the native code as one
// small struct: a header of the scalars the kernels need and the pack's
// device buffers. Fields are only ever appended; `struct_bytes` is the
// caller's sizeof, so a later format that needs another buffer or scalar adds
// a field without changing any call site.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// format 11: KX7 v1.1 (cuteafd_core::kx7::V11_LAYOUT / V11_CODE): lane16, K split 4, the 32
// streams of each 8-row block and K slice quad-interleaved (quad stride 512) with ragged tails,
// the two-class code (11-bit primary, ESC + 12-bit rare rank).
#define CUTEAFD_KX7_FORMAT_V11 11u
// format 12: v1.1's stream layout under the Huffman fallback code, for a matrix whose two-class
// ESC cannot fit 16 bits (too many distinct patterns for 4,096 rare slots at a short ESC): a
// canonical Huffman code limited to 16 bits, two-level tables (11-bit primary, sub-tables), at
// most CUTEAFD_KX7_MAX_HUFFMAN_WORDS words; esc_shift and esc_mask are 0.
#define CUTEAFD_KX7_FORMAT_V11_HUFFMAN 12u
#define CUTEAFD_KX7_MAX_HUFFMAN_WORDS 16384u

typedef struct cuteafd_kx7_pack_t {
  uint32_t struct_bytes;        // sizeof(cuteafd_kx7_pack_t) as the caller compiled it
  uint32_t format;              // CUTEAFD_KX7_FORMAT_*
  int32_t n;                    // rows (output features), a multiple of 8
  int32_t k;                    // columns (input features)
  int32_t slices;               // K slices S (4)
  int32_t streams_per_row;      // 4 * S
  int32_t symbols_per_stream;   // K / (4 * S)
  uint32_t quad_stride;         // bytes between a stream's interleaved quads (512)
  uint32_t ragged;              // 1: each group's interleave stops at its shortest stream
  uint32_t esc_shift;           // 31 - esc_len - rare_bits
  uint32_t esc_mask;            // (1 << (rare_bits + 1)) - 2
  uint32_t table_words;         // u32 words of `tables`
  const void* tables;           // u32 decode tables (cuteafd_core::kx7::Kx7Pack::device_tables)
  const void* offsets;          // u32 [n * streams_per_row + 1] quad 0 of each stream, then the total
  const void* lengths;          // u16 [n * streams_per_row] byte length of each stream
  const void* streams;          // u8, 16-byte aligned, then 16 zero bytes
} cuteafd_kx7_pack_t;

// Workspace bytes cuteafd_kx7_linear needs for `rows` rows over an [n, k] pack (0: invalid).
size_t cuteafd_kx7_workspace(int32_t rows, int32_t n, int32_t k);
// Decodes `pack` into BF16 out_bf16 [n, k] row-major (16-byte aligned). Asynchronous on
// `stream`, graph capturable. Returns a cudaError_t.
int32_t cuteafd_kx7_decode(const cuteafd_kx7_pack_t* pack, void* out_bf16, void* stream);
// out [rows, n] BF16 = x [rows, k] BF16 @ W^T (FP32 accumulation) for W = `pack`, through
// `workspace` (>= cuteafd_kx7_workspace bytes). Asynchronous on `stream`, graph capturable.
// Returns 0, a cudaError_t, or for a failed GEMM the native status.
int32_t cuteafd_kx7_linear(const void* x, const cuteafd_kx7_pack_t* pack, void* out, int32_t rows,
                           void* workspace, size_t workspace_bytes, void* stream);

// Decode and verify steps (rows 1..64) of `count` (1..3) packs of one shape and code in one launch:
// out[r * ldo + col0[i] + c] = (x @ W_i^T)[r, c], each element bit for bit cuteafd_kx7_linear's for
// its pack. cudaErrorInvalidValue when the packs cannot share a launch; otherwise a cudaError_t.
int32_t cuteafd_kx7_gemv_group(const void* x, const cuteafd_kx7_pack_t* const* packs, int32_t count,
                               const int32_t* col0, void* out, int32_t ldo, int32_t rows, void* stream);
// cuteafd_kx7_linear's GEMV rows (1..64) with output rows ldo (>= n) elements apart.
int32_t cuteafd_kx7_linear_ld(const void* x, const cuteafd_kx7_pack_t* pack, void* out, int32_t ldo, int32_t rows,
                              void* stream);
// out[r * ldo + c] = (x [rows, k] @ w^T)[r, c] for row-major BF16 w [n, k], rows 1..64 (n % 8 == 0,
// k % 512 == 0), FP32 accumulation, row-count invariant. Returns a cudaError_t.
int32_t cuteafd_bf16_gemv_ld(const void* x, const void* w, void* out, int32_t ldo, int32_t rows, int32_t n,
                             int32_t k, void* stream);

// Test support: the KX7 GEMV's BF16 twin over row-major BF16 w [n, k], for rows 1..64 (the
// GEMV's range): bit for bit what cuteafd_kx7_linear gives for a pack of w at those rows.
int32_t cuteafd_kx7_reference_gemv(const void* x, const void* w, void* out, int32_t rows, int32_t n, int32_t k,
                                   void* stream);

#ifdef __cplusplus
}
#endif
