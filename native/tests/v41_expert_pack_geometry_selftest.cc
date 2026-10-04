// Byte-exact native expert packing for every role and all four TP4 bands.
// Exact TP4 reference scatters source coordinates into resident tiles and
// verifies the four rank bands cover the full logical weights exactly once.
#include "cuteafd_experts.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kHidden = 5120;
constexpr uint32_t kFullIntermediate = 2304;

struct Case {
  const char* name;
  uint32_t intermediate;  // per-rank LOGICAL width passed to the packer
  uint32_t offset;        // source row/column offset of this rank's band
};

const Case kCases[] = {
    {"tp6", 384, 0},
    {"tp3", 768, 0},
    {"tp2", 1152, 0},
    {"tp4", 576, 0},
    {"full", 2304, 0},
    {"band4/0", 576, 0},
    {"band4/1", 576, 576},
    {"band4/2", 576, 1152},
    {"band4/3", 576, 1728},
};

void require(bool condition, const std::string& message) {
  if (!condition) {
    std::fprintf(stderr, "v41 expert pack geometry selftest: %s\n",
                 message.c_str());
    std::exit(1);
  }
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "v41 expert pack geometry selftest: %s failed: %s\n",
                 what, cudaGetErrorString(status));
    std::exit(1);
  }
}

uint8_t pattern(uint64_t index, uint32_t salt) {
  uint64_t value = index * 0x9e3779b97f4a7c15ull + uint64_t(salt) * 0x100000001b3ull;
  value ^= value >> 29;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 32;
  return static_cast<uint8_t>(value & 0xffu);
}

std::vector<uint8_t> make_source(size_t bytes, uint32_t salt) {
  std::vector<uint8_t> source(bytes);
  for (size_t i = 0; i < bytes; ++i) source[i] = pattern(i, salt);
  return source;
}

// Direct port of the device `pack<Gated,Scales>` index math.
std::vector<uint8_t> reference_pack(bool gated, bool scales, const uint8_t* first,
    const uint8_t* second, uint32_t n, uint32_t k, uint32_t n_pad, uint64_t count) {
  const uint32_t k_tiles = (k + 127) / 128;
  const uint32_t tile_elements = scales ? 1024 : 4096;
  const uint32_t divisor = scales ? 32 : 8;
  std::vector<uint8_t> out(size_t(count) * (scales ? 1 : 4), 0);
  for (uint64_t index = 0; index < count; ++index) {
    const uint64_t tile = index / tile_elements;
    const uint32_t local = static_cast<uint32_t>(index % tile_elements);
    const uint32_t nt = static_cast<uint32_t>(tile / k_tiles);
    const uint32_t kt = static_cast<uint32_t>(tile % k_tiles);
    uint32_t row = 0, col = 0;
    if (scales) {
      row = nt * 256 + local / 4;
      col = kt * 4 + local % 4;
    } else {
      const uint32_t n8i = local & 3;
      const uint32_t combined = (local >> 2) & 31;
      const uint32_t n8c = (local >> 7) & 7;
      const uint32_t k32 = local >> 10;
      row = nt * 256 + n8c * 32 + n8i * 8 + (combined >> 2);
      col = kt * 16 + k32 * 4 + (combined & 3);
    }
    const uint8_t* source = first;
    if (gated && row >= n_pad) {
      row -= n_pad;
      source = second;
    }
    const bool valid = row < n && col < k / divisor;
    if (scales) {
      out[index] = valid ? source[uint64_t(row) * (k / divisor) + col] : 0;
    } else {
      uint32_t value = 0;
      if (valid)
        std::memcpy(&value, source + (uint64_t(row) * (k / divisor) + col) * 4, 4);
      out[size_t(index) * 4 + 0] = static_cast<uint8_t>(value & 0xffu);
      out[size_t(index) * 4 + 1] = static_cast<uint8_t>((value >> 8) & 0xffu);
      out[size_t(index) * 4 + 2] = static_cast<uint8_t>((value >> 16) & 0xffu);
      out[size_t(index) * 4 + 3] = static_cast<uint8_t>((value >> 24) & 0xffu);
    }
  }
  return out;
}

// w1/w3 are [intermediate, hidden] FP4; s1/s3 are [intermediate, hidden/32].
std::vector<uint8_t> band_rows(const std::vector<uint8_t>& full, uint32_t offset,
    uint32_t rows, uint32_t row_bytes) {
  const uint64_t start = uint64_t(offset) * row_bytes;
  return std::vector<uint8_t>(full.begin() + start,
                              full.begin() + start + uint64_t(rows) * row_bytes);
}

// w2 is [hidden, intermediate] FP4; s2 is [hidden, intermediate/32]. Both are
// sliced on their LAST axis, which is why a band offset is divided by the
// element rate (2 for FP4 bytes, 32 for K32 scales).
std::vector<uint8_t> band_cols(const std::vector<uint8_t>& full, uint32_t rows,
    uint32_t row_bytes_full, uint32_t col_start, uint32_t cols) {
  std::vector<uint8_t> out(uint64_t(rows) * cols);
  for (uint32_t row = 0; row < rows; ++row)
    std::memcpy(out.data() + uint64_t(row) * cols,
                full.data() + uint64_t(row) * row_bytes_full + col_start, cols);
  return out;
}

// Independent inverse map: scatter logical source words into exact resident
// tiles and prove there are neither collisions nor unused slots.
std::vector<uint8_t> exact_reference(bool gated, bool scales,
    const uint8_t* first, const uint8_t* second, uint64_t bytes) {
  constexpr uint32_t intermediate = 576;
  const uint32_t n = gated ? intermediate : kHidden;
  const uint32_t k = gated ? kHidden : intermediate;
  const uint32_t divisor = scales ? 32 : 8;
  std::vector<uint8_t> output(bytes);
  std::vector<uint8_t> covered(bytes / (scales ? 1 : 4), 0);
  for (uint32_t half = 0; half < (gated ? 2u : 1u); ++half)
    for (uint32_t row = 0; row < n; ++row)
      for (uint32_t col = 0; col < k / divisor; ++col) {
        uint64_t dest;
        const uint32_t rows = gated && row / 128 == 4 ? 64 : 128;
        const uint32_t cols = !gated && col / (128 / divisor) == 4 ? 2 : 4;
        const uint64_t base = uint64_t(half) * n * (k / divisor) + uint64_t(row / 128) * 128 * (k / divisor);
        if (scales) {
          dest = base + (gated ? (col / 4) * rows * 4 : (col / 4) * 512)
                 + (row % 128) * (gated ? 4 : cols) + col % 4;
        } else {
          dest = base + (col / 4) * rows * 4 + (row % 128 / 32) * 128
                 + (row % 8) * 16 + (col % 4) * 4 + (row % 32 / 8);
        }
        require(dest < covered.size() && !covered[dest], "exact pack is not a bijection");
        covered[dest] = 1;
        const uint8_t* source = half ? second : first;
        const size_t size = scales ? 1 : 4;
        std::memcpy(output.data() + dest * size, source + (uint64_t(row) * (k / divisor) + col) * size, size);
      }
  for (auto value : covered) require(value == 1, "exact pack has holes");
  return output;
}

}  // namespace

int main() {
  int devices = 0;
  const bool required = std::getenv("CUTEAFD_REQUIRE_CUDA") != nullptr;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
    if (required) {
      std::fprintf(stderr,
                   "v41 expert pack geometry selftest: CUTEAFD_REQUIRE_CUDA is set "
                   "but no CUDA device is present\n");
      return 1;
    }
    std::printf("v41 expert pack geometry selftest: no CUDA device, skipping\n");
    return 77;
  }
  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  const uint64_t full_weight_bytes = uint64_t(kFullIntermediate) * kHidden / 2;
  const uint64_t full_scale_bytes = full_weight_bytes / 16;
  const std::vector<uint8_t> full_w1 = make_source(full_weight_bytes, 11);
  const std::vector<uint8_t> full_w3 = make_source(full_weight_bytes, 12);
  const std::vector<uint8_t> full_w2 = make_source(full_weight_bytes, 13);
  const std::vector<uint8_t> full_s1 = make_source(full_scale_bytes, 14);
  const std::vector<uint8_t> full_s3 = make_source(full_scale_bytes, 15);
  const std::vector<uint8_t> full_s2 = make_source(full_scale_bytes, 16);

  // Union coverage over the four band cases in the un-padded global layouts.
  // W13 has two logical halves (up then gate): 2*2304 rows x 640 words.
  std::vector<uint8_t> w13_covered(size_t(2) * kFullIntermediate * (kHidden / 8), 0);
  std::vector<uint8_t> w2_covered(size_t(kHidden) * (kFullIntermediate / 8), 0);

  for (const Case& item : kCases) {
    const uint32_t intermediate = item.intermediate;
    const std::string name = item.name;
    uint64_t sizes[4] = {};
    require(cuteafd_expert_packed_sizes(intermediate, sizes) == cudaSuccess,
            name + ": packer rejected the extent");

    const uint64_t weight_bytes = uint64_t(intermediate) * kHidden / 2;
    const uint64_t scale_bytes = weight_bytes / 16;
    const std::vector<uint8_t> w1 = band_rows(full_w1, item.offset, intermediate, kHidden / 2);
    const std::vector<uint8_t> w3 = band_rows(full_w3, item.offset, intermediate, kHidden / 2);
    const std::vector<uint8_t> s1 = band_rows(full_s1, item.offset, intermediate, kHidden / 32);
    const std::vector<uint8_t> s3 = band_rows(full_s3, item.offset, intermediate, kHidden / 32);
    const std::vector<uint8_t> w2 = band_cols(full_w2, kHidden, kFullIntermediate / 2,
                                             item.offset / 2, intermediate / 2);
    const std::vector<uint8_t> s2 = band_cols(full_s2, kHidden, kFullIntermediate / 32,
                                             item.offset / 32, intermediate / 32);
    require(w1.size() == weight_bytes && s2.size() == scale_bytes,
            name + ": band slice extents are wrong");

    const uint8_t* sources[6] = {w1.data(), w3.data(), w2.data(),
                                 s1.data(), s3.data(), s2.data()};
    uint8_t* destinations[4] = {};
    for (int i = 0; i < 4; ++i)
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&destinations[i]), sizes[i]),
                 "cudaMalloc destination");
    require(cuteafd_pack_expert_async(sources, destinations, intermediate,
                                         nullptr) == cudaSuccess,
            name + ": pack launch failed");
    check_cuda(cudaStreamSynchronize(nullptr), "synchronize");

    const uint32_t kernel_intermediate = static_cast<uint32_t>(sizes[0] / kHidden);
    std::vector<uint8_t> expected_w13 = reference_pack(
        /*gated=*/true, /*scales=*/false, w3.data(), w1.data(), intermediate,
        kHidden, kernel_intermediate, sizes[0] / 4);
    std::vector<uint8_t> expected_s13 = reference_pack(
        /*gated=*/true, /*scales=*/true, s3.data(), s1.data(), intermediate,
        kHidden, kernel_intermediate, sizes[1]);
    std::vector<uint8_t> expected_w2 = reference_pack(
        /*gated=*/false, /*scales=*/false, w2.data(), nullptr, kHidden,
        intermediate, 0, sizes[2] / 4);
    std::vector<uint8_t> expected_s2 = reference_pack(
        /*gated=*/false, /*scales=*/true, s2.data(), nullptr, kHidden,
        intermediate, 0, sizes[3]);
    if (intermediate == 576 && kernel_intermediate == 576) {
      require(sizes[0] == 2 * weight_bytes && sizes[1] == 2 * scale_bytes &&
              sizes[2] == weight_bytes && sizes[3] == scale_bytes, "TP4 must store exact planes");
      expected_w13 = exact_reference(true, false, w3.data(), w1.data(), sizes[0]);
      expected_s13 = exact_reference(true, true, s3.data(), s1.data(), sizes[1]);
      expected_w2 = exact_reference(false, false, w2.data(), nullptr, sizes[2]);
      expected_s2 = exact_reference(false, true, s2.data(), nullptr, sizes[3]);
    }
    const std::vector<uint8_t>* expected[4] = {&expected_w13, &expected_s13,
                                               &expected_w2, &expected_s2};

    std::array<std::vector<uint8_t>, 4> actual;
    for (int i = 0; i < 4; ++i) {
      actual[i].resize(sizes[i]);
      check_cuda(cudaMemcpy(actual[i].data(), destinations[i], sizes[i],
                            cudaMemcpyDeviceToHost),
                 "copy destination");
      for (size_t byte = 0; byte < actual[i].size(); ++byte)
        if (actual[i][byte] != (*expected[i])[byte]) {
          std::fprintf(stderr,
                       "%s destination %d byte %zu actual=%02x expected=%02x\n",
                       name.c_str(), i, byte, actual[i][byte], (*expected[i])[byte]);
          std::exit(1);
        }
      check_cuda(cudaFree(destinations[i]), "cudaFree destination");
    }

    std::printf("%-8s intermediate=%-4u kernel=%-4u w13=%llu s13=%llu w2=%llu s2=%llu\n",
                name.c_str(), intermediate, kernel_intermediate,
                static_cast<unsigned long long>(sizes[0]),
                static_cast<unsigned long long>(sizes[1]),
                static_cast<unsigned long long>(sizes[2]),
                static_cast<unsigned long long>(sizes[3]));

    // TP4 four-rank logical partition: the four 576-wide bands must claim every
    // un-padded global word exactly once, and each packed word must equal the
    // band's own source word.
    if (name.rfind("band4/", 0) == 0) {
      // Source coordinates are the inverse of exact_reference's bijection.
      for (uint32_t half = 0; half < 2; ++half)
        for (uint32_t row = 0; row < intermediate; ++row)
          for (uint32_t col = 0; col < kHidden / 8; ++col) {
            const auto index = (uint64_t(half) * kFullIntermediate + item.offset + row) * (kHidden / 8) + col;
            require(!w13_covered[index], "TP4 W13 overlaps");
            w13_covered[index] = 1;
          }
      for (uint32_t row = 0; row < kHidden; ++row)
        for (uint32_t col = 0; col < intermediate / 8; ++col) {
          const auto index = uint64_t(row) * (kFullIntermediate / 8) + item.offset / 8 + col;
          require(!w2_covered[index], "TP4 W2 overlaps");
          w2_covered[index] = 1;
        }
    }
  }

  size_t missing_w13 = 0;
  for (uint8_t value : w13_covered) missing_w13 += (value == 0);
  size_t missing_w2 = 0;
  for (uint8_t value : w2_covered) missing_w2 += (value == 0);
  require(missing_w13 == 0 && missing_w2 == 0,
          "band4 partition left " + std::to_string(missing_w13) + " W13 and " +
              std::to_string(missing_w2) + " W2 words uncovered");
  std::printf("band4 partition: %zu W13 + %zu W2 words covered exactly once\n",
              w13_covered.size(), w2_covered.size());

  std::printf("v41 expert pack geometry selftest: ok (%zu geometries, incl. TP4 storage contract)\n",
              sizeof(kCases) / sizeof(kCases[0]));
  return 0;
}
