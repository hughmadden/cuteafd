// GPU check of cuteafd_fp8_grid_kmajor against its definition, including a
// row count that ends inside a 128-row block.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" int32_t cuteafd_fp8_grid_kmajor(const void* grid, void* kmajor, int64_t rows, int64_t kb, void* stream);

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  for (const int64_t rows : {2624, 128, 1}) {
    const int64_t kb = 48, grid_rows = (rows + 127) / 128;
    std::vector<float> grid(grid_rows * kb), expected(rows * kb), actual(rows * kb);
    for (size_t i = 0; i < grid.size(); ++i) grid[i] = 0.25f + float(i) / 7.0f;
    for (int64_t b = 0; b < kb; ++b)
      for (int64_t r = 0; r < rows; ++r) expected[b * rows + r] = grid[(r / 128) * kb + b];
    float *g = nullptr, *k = nullptr;
    if (cudaMalloc(&g, grid.size() * 4) || cudaMalloc(&k, actual.size() * 4)) return 1;
    cudaMemcpy(g, grid.data(), grid.size() * 4, cudaMemcpyHostToDevice);
    if (cuteafd_fp8_grid_kmajor(g, k, rows, kb, nullptr) != 0 || cudaDeviceSynchronize() != cudaSuccess) return 1;
    cudaMemcpy(actual.data(), k, actual.size() * 4, cudaMemcpyDeviceToHost);
    if (std::memcmp(actual.data(), expected.data(), actual.size() * 4) != 0) {
      std::fprintf(stderr, "K-major grid differs for %lld rows\n", (long long)rows);
      return 1;
    }
    cudaFree(g);
    cudaFree(k);
  }
  std::printf("GPU K-major grid widening exact\n");
  return 0;
}
