// CUDA load-time scale gather against its shared definition (the CPU fixture's
// rule): row-major and K-block-major outputs, a column window and zero rows.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "../shared/cuda/load_scale_gather.h"

extern "C" int32_t cuteafd_load_scale_gather(const void* grid, const void* index, void* row_major, void* k_major,
                                             int64_t rows, int64_t kb, int64_t grid_cols, int64_t col0, void* stream);

#define CHECK(x) do { if ((x) != cudaSuccess) { std::fprintf(stderr, "%s failed\n", #x); return 1; } } while (0)

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  const int64_t grid_rows = 37, grid_cols = 48, rows = 300, kb = 40, col0 = 5;
  std::vector<float> grid(grid_rows * grid_cols);
  for (size_t i = 0; i < grid.size(); ++i) grid[i] = 1.0f / float(i + 3) + float(i % 7) * 0.25f;
  std::vector<uint32_t> index(rows);
  for (int64_t r = 0; r < rows; ++r) index[r] = r % 11 == 0 ? cuteafd_scale_gather::kGatherZero : uint32_t((r * 7) % grid_rows);
  float *d_grid, *d_row, *d_k;
  uint32_t* d_index;
  CHECK(cudaMalloc(&d_grid, grid.size() * 4));
  CHECK(cudaMalloc(&d_index, index.size() * 4));
  CHECK(cudaMalloc(&d_row, rows * kb * 4));
  CHECK(cudaMalloc(&d_k, rows * kb * 4));
  CHECK(cudaMemcpy(d_grid, grid.data(), grid.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_index, index.data(), index.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_row, 0xff, rows * kb * 4));
  CHECK(cudaMemset(d_k, 0xff, rows * kb * 4));
  if (cuteafd_load_scale_gather(d_grid, d_index, d_row, d_k, rows, kb, grid_cols, col0, nullptr)) return 1;
  CHECK(cudaDeviceSynchronize());
  std::vector<float> row(rows * kb), k(rows * kb);
  CHECK(cudaMemcpy(row.data(), d_row, row.size() * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(k.data(), d_k, k.size() * 4, cudaMemcpyDeviceToHost));
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t b = 0; b < kb; ++b) {
      const float want = cuteafd_scale_gather::gathered(grid.data(), index.data(), grid_cols, col0, r, b);
      if (std::memcmp(&row[r * kb + b], &want, 4) || std::memcmp(&k[b * rows + r], &want, 4)) {
        std::fprintf(stderr, "mismatch at row %ld block %ld\n", long(r), long(b));
        return 1;
      }
    }
  if (cuteafd_load_scale_gather(d_grid, d_index, nullptr, nullptr, rows, kb, grid_cols, col0, nullptr) == 0) return 1;
  if (cuteafd_load_scale_gather(d_grid, d_index, d_row, nullptr, rows, kb, grid_cols, grid_cols - 1, nullptr) == 0)
    return 1;
  cudaFree(d_grid); cudaFree(d_index); cudaFree(d_row); cudaFree(d_k);
  std::printf("PASS load scale gather: %ld rows x %ld blocks exact, window and zero rows\n", long(rows), long(kb));
  return 0;
}
