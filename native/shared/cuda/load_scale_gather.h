// Load-time scale gathers, shared by the CUDA kernel and the CPU fixture:
// per-row x 128-K block FP32 scales gathered from a source grid by a row index
// (`index[r]` the source row, or kGatherZero for a zero row), written row
// major [rows, kb] and/or K-block major [kb, rows]. A pure gather (exact).
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define CUTEAFD_SG_FN __host__ __device__ inline
#else
#define CUTEAFD_SG_FN inline
#endif

namespace cuteafd_scale_gather {
constexpr uint32_t kGatherZero = 0xffffffffu;

// Element `i` of the [rows, kb] result: source grid [*, grid_cols], columns
// col0..col0 + kb of row index[r].
CUTEAFD_SG_FN float gathered(const float* grid, const uint32_t* index, int64_t grid_cols, int64_t col0,
                             int64_t r, int64_t b) {
  const uint32_t source = index[r];
  return source == kGatherZero ? 0.0f : grid[int64_t(source) * grid_cols + col0 + b];
}
}  // namespace cuteafd_scale_gather
