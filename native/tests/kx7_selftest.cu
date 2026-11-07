// Standalone harness for native/shared/cuda/kx7.cu, the KX7 v1.1 exact packs
// of BF16 matrices (format and packer: cuteafd_core::kx7). It reads pack file
// sets written on the host by cuteafd-core's packer (Kx7Pack::dump, with
// source.bf16.bin and device_tables.bin; the committed fixtures are
// native/tests/fixtures/kx7, made by the daemon's ignored test
// kx7_dump_selftest_fixtures, which also writes the GLM 5.3 Flash q/k/v
// [8192, 4096] and o [4096, 8192] shapes on request) or by the reference
// tool's dump_pack, and checks:
//   1. the decode tables built here from table.bin and rare.bin equal the
//      packer's device_tables.bin word for word, and a host walk of every
//      stream (quads gathered from the ragged interleave, the two-class lookup)
//      gives back the source;
//   2. cuteafd_kx7_decode reproduces every source weight bit for bit;
//   3. cuteafd_kx7_linear equals, at 1..64 rows, the fused GEMV's BF16 twin over the source weights
//      (and is width-invariant); above 64 rows, the BF16 cuBLAS linear over the source weights
//      bit for bit at 1 to 300 rows, and a CUDA graph of it replays bit for bit;
//   4. invalid packs (format, header size, shape, alignment, tables,
//      workspace) are refused.
// It ends with one PASS/FAIL line: exit 0 PASS, 1 FAIL, 77 without a CUDA
// device. --host-only runs part 1 without touching CUDA.
//
// The native build makes it (ctest -R kx7) linked to cuteafd_native, with the
// committed fixtures; run it on a GPU host:
//   ./cuteafd_kx7_selftest [--host-only] [PACK_DIR...]
// Standalone, from the repository root against a built library:
//   nvcc -std=c++17 -O3 -arch=sm_120f -Inative/shared/include -o kx7_selftest \
//     native/tests/kx7_selftest.cu -Lnative/build -lcuteafd_native
//   ./kx7_selftest native/tests/fixtures/kx7/*
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <map>
#include <vector>

#include "cuteafd_kx7.h"
#include "cuteafd_native.h"

#ifndef CUTEAFD_KX7_FIXTURE_DIR
#define CUTEAFD_KX7_FIXTURE_DIR "native/tests/fixtures/kx7"
#endif

namespace {

int failures = 0;

void fail(const char* what, const std::string& where, long long a = 0, long long b = 0) {
  ++failures;
  if (failures <= 40) std::fprintf(stderr, "FAIL: %s [%s] (%lld, %lld)\n", what, where.c_str(), a, b);
}

void check(bool ok, const char* what, const std::string& where, long long a = 0, long long b = 0) {
  if (!ok) fail(what, where, a, b);
}

#define CUDA_MUST(expr)                                                                             \
  do {                                                                                              \
    const cudaError_t err_ = (expr);                                                                \
    if (err_ != cudaSuccess) {                                                                      \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err_)); \
      std::exit(1);                                                                                 \
    }                                                                                               \
  } while (0)

#define MUST(expr)                                                                    \
  do {                                                                                \
    const int32_t status_ = int32_t(expr);                                            \
    if (status_ != 0) {                                                               \
      std::fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, status_); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

// ---------------------------------------------------------------- pack files

std::vector<uint8_t> slurp(const std::string& path, bool required = true) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (required) {
      std::fprintf(stderr, "cannot read %s\n", path.c_str());
      std::exit(1);
    }
    return {};
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

template <typename T>
std::vector<T> words(const std::vector<uint8_t>& bytes) {
  std::vector<T> out(bytes.size() / sizeof(T));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
  return out;
}

// The integer after `"key":` in header.json (-1 if absent).
long long json_int(const std::string& json, const std::string& key) {
  const size_t at = json.find("\"" + key + "\":");
  if (at == std::string::npos) return -1;
  size_t p = at + key.size() + 3;
  while (p < json.size() && (json[p] == ' ' || json[p] == '[')) ++p;
  return std::strtoll(json.c_str() + p, nullptr, 10);
}

struct Pack {
  std::string name;
  int n = 0, k = 0, slices = 0, pbits = 0, rare_bits = 0, esc_len = 0, esc_index = -1;
  bool ragged = false, huffman = false;
  std::vector<uint16_t> table, rare, lengths, source;
  std::vector<uint32_t> offsets, lut, packer_lut;
  std::vector<uint8_t> streams;  // with 16 zero bytes after
};

// Two-class decode tables (cuteafd_core::kx7::Kx7Code::device_tables, the reference build_lut):
// 2^pbits entries (leaf pattern << 16 | len, ESC 0xFF << 8 | (len + rare_bits)), then the rare
// patterns by rank as u16 pairs over 2^rare_bits slots.
std::vector<uint32_t> build_lut(const Pack& p) {
  std::vector<uint32_t> w(size_t(1) << p.pbits, 0);
  uint32_t code = 0, prev = 0;
  size_t i = 16;
  for (uint32_t l = 1; l <= 16; ++l) {
    for (uint32_t c = 0; c < p.table[l - 1]; ++c, ++i) {
      code <<= l - prev;
      prev = l;
      const bool esc = int(i - 16) == p.esc_index;
      const uint32_t entry = esc ? (0xFFu << 8 | (l + uint32_t(p.rare_bits))) : (uint32_t(p.table[i]) << 16 | l);
      std::fill(w.begin() + (code << (p.pbits - l)), w.begin() + ((code + 1) << (p.pbits - l)), entry);
      ++code;
    }
  }
  const size_t start = w.size();
  w.resize(start + ((size_t(1) << p.rare_bits) + 1) / 2, 0);
  for (size_t r = 0; r < p.rare.size(); ++r) w[start + r / 2] |= uint32_t(p.rare[r]) << (16 * (r % 2));
  return w;
}

// Huffman fallback decode tables (the reference build_lut's two-level form): 2^pbits primary
// entries (leaf pattern << 16 | len, or a sub-table index << 16 | (32 - its bits) << 8 | 0xFF),
// then the sub-tables of the codes longer than pbits, in prefix order.
std::vector<uint32_t> build_lut_huffman(const Pack& p) {
  const int P = p.pbits;
  std::vector<uint8_t> lens(65536, 0);
  std::vector<uint32_t> codes(65536, 0);
  uint32_t code = 0, prev = 0;
  size_t i = 16;
  for (uint32_t l = 1; l <= 16; ++l) {
    for (uint32_t c = 0; c < p.table[l - 1]; ++c, ++i) {
      code <<= l - prev;
      prev = l;
      lens[p.table[i]] = uint8_t(l);
      codes[p.table[i]] = code++;
    }
  }
  std::vector<uint32_t> w(size_t(1) << P, 0);
  std::map<uint32_t, int> maxlen_of;
  for (int sym = 0; sym < 65536; ++sym) {
    const int l = lens[sym];
    if (!l) continue;
    if (l <= P) {
      std::fill(w.begin() + (codes[sym] << (P - l)), w.begin() + ((codes[sym] + 1) << (P - l)), uint32_t(sym) << 16 | l);
    } else {
      const uint32_t prefix = codes[sym] >> (l - P);
      maxlen_of[prefix] = std::max(maxlen_of[prefix], l);
    }
  }
  std::map<uint32_t, uint32_t> base_of;
  for (const auto& [prefix, maxlen] : maxlen_of) {
    const int bits = maxlen - P;
    base_of[prefix] = uint32_t(w.size());
    w[prefix] = uint32_t(w.size()) << 16 | uint32_t(32 - bits) << 8 | 0xFFu;
    w.resize(w.size() + (size_t(1) << bits), 0);
  }
  for (int sym = 0; sym < 65536; ++sym) {
    const int l = lens[sym];
    if (l <= P) continue;
    const uint32_t prefix = codes[sym] >> (l - P);
    const int bits = maxlen_of[prefix] - P;
    const uint32_t rem = codes[sym] & ((1u << (l - P)) - 1u);
    std::fill(w.begin() + base_of[prefix] + (rem << (bits - (l - P))),
              w.begin() + base_of[prefix] + ((rem + 1) << (bits - (l - P))), uint32_t(sym) << 16 | l);
  }
  return w;
}

Pack load(const std::string& dir) {
  Pack p;
  p.name = dir.substr(dir.find_last_of('/') + 1);
  const std::vector<uint8_t> header = slurp(dir + "/header.json");
  const std::string j(header.begin(), header.end());
  const size_t shape = j.find("\"shape\":");
  p.n = int(json_int(j, "shape"));
  p.k = int(std::strtoll(j.c_str() + j.find(',', shape) + 1, nullptr, 10));
  p.slices = int(json_int(j, "ksplit"));
  p.pbits = int(json_int(j, "pbits"));
  p.rare_bits = int(json_int(j, "rare_bits"));
  p.esc_len = int(json_int(j, "esc_len"));
  p.esc_index = int(json_int(j, "esc_index"));
  p.ragged = j.find("\"ragged\": true") != std::string::npos;
  p.huffman = j.find("\"code\": \"huffman\"") != std::string::npos;
  if (p.huffman) p.pbits = 11;  // the device tables' primary width for the Huffman fallback (KX7_PBITS)
  const bool layout = json_int(j, "quad_stride") == 512 && json_int(j, "interleave_rows") == 8 && p.ragged &&
                      p.slices == 4;
  const bool code = p.huffman || (j.find("\"code\": \"twoclass\"") != std::string::npos && p.pbits == 11 &&
                                  p.rare_bits == 12);
  if (!layout || !code) {
    std::fprintf(stderr, "%s: not a KX7 v1.1 pack (lane16 ksplit 4, ragged 8-row interleave; twoclass p11 r12 or "
                 "its Huffman fallback)\n", dir.c_str());
    std::exit(1);
  }
  p.table = words<uint16_t>(slurp(dir + "/table.bin"));
  if (!p.huffman) p.rare = words<uint16_t>(slurp(dir + "/rare.bin"));
  p.offsets = words<uint32_t>(slurp(dir + "/offsets.bin"));
  p.lengths = words<uint16_t>(slurp(dir + "/lengths.bin"));
  p.source = words<uint16_t>(slurp(dir + "/source.bf16.bin"));
  p.packer_lut = words<uint32_t>(slurp(dir + "/device_tables.bin", false));
  p.streams = slurp(dir + "/streams.bin");
  p.streams.resize(p.streams.size() + 16, 0);
  p.lut = p.huffman ? build_lut_huffman(p) : build_lut(p);
  return p;
}

// The bytes of stream `id` from where the ragged 8-row interleave puts them.
std::vector<uint8_t> gather(const Pack& p, size_t id) {
  const size_t spr = 4 * size_t(p.slices), row = id / spr, slice = (id % spr) / 4;
  const size_t block = row / 8, lane = 4 * (row % 8) + id % 4;
  size_t nq[32], qmin = SIZE_MAX;
  for (size_t l = 0; l < 32; ++l) {
    nq[l] = (p.lengths[(8 * block + l / 4) * spr + slice * 4 + l % 4] + 15) / 16;
    qmin = std::min(qmin, nq[l]);
  }
  size_t before = 0;
  for (size_t l = 0; l < lane; ++l) before += nq[l] - qmin;
  const size_t base = p.offsets[id] - lane * 16;
  std::vector<uint8_t> out;
  for (size_t q = 0; q < nq[lane]; ++q) {
    const size_t at = q < qmin ? p.offsets[id] + q * 512 : base + qmin * 512 + (before + q - qmin) * 16;
    out.insert(out.end(), p.streams.begin() + at, p.streams.begin() + at + 16);
  }
  return out;
}

// The host walk: a 64-bit buffer refilled with big-endian words, the two-class lookup.
bool host_part(const Pack& p) {
  const int before = failures;
  check(p.packer_lut.empty() || p.packer_lut == p.lut, "tables built here equal the packer's device tables", p.name,
        (long long)p.lut.size(), (long long)p.packer_lut.size());
  check(p.offsets.size() == size_t(p.n) * 4 * p.slices + 1 && p.source.size() == size_t(p.n) * p.k,
        "file sizes agree with the shape", p.name);
  if (failures != before) return false;
  const size_t spr = 4 * size_t(p.slices), width = size_t(p.k) / p.slices;
  const uint16_t* rare = reinterpret_cast<const uint16_t*>(p.lut.data() + (size_t(1) << p.pbits));
  size_t bad = 0;
  for (size_t id = 0; id < size_t(p.n) * spr; ++id) {
    const std::vector<uint8_t> bytes = gather(p, id);
    uint64_t buf = 0;
    uint32_t nbits = 0;
    size_t next = 0;
    const size_t row = id / spr, slice = (id % spr) / 4, t = id % 4;
    for (size_t c = 0; c < width / 64; ++c) {
      for (size_t i = 0; i < 16; ++i) {
        if (nbits < 32) {
          uint32_t word = 0;
          for (int b = 0; b < 4; ++b) word = word << 8 | (next + b < bytes.size() ? bytes[next + b] : 0);
          buf |= uint64_t(word) << (32 - nbits);
          nbits += 32;
          next += 4;
        }
        const uint32_t peek = uint32_t(buf >> 48);
        uint32_t e = p.lut[peek >> (16 - p.pbits)];
        uint32_t sym = e >> 16;
        if (p.huffman && (e & 0xff) == 0xff) {  // a code past pbits: its sub-table
          const uint32_t sub = ((peek << 16) << p.pbits) >> ((e >> 8) & 0xff);
          e = p.lut[(e >> 16) + sub];
          sym = e >> 16;
        } else if (!p.huffman && ((e >> 8) & 0xff) == 0xff) {
          sym = rare[(peek >> (16 - p.esc_len - p.rare_bits)) & ((1u << p.rare_bits) - 1)];
        }
        buf <<= (e & 0xff);
        nbits -= (e & 0xff);
        bad += sym != p.source[row * p.k + slice * width + 64 * c + 16 * t + i];
      }
    }
  }
  check(bad == 0, "the host walk gives back the source", p.name, (long long)bad);
  std::printf("host %-24s [%d, %d]: %zu rare patterns, %.3f stream bits a weight, %s\n", p.name.c_str(), p.n, p.k,
              p.rare.size(), double(p.streams.size() - 16) * 8.0 / (double(p.n) * p.k), bad ? "MISMATCH" : "exact");
  return failures == before;
}

// ---------------------------------------------------------------- device

struct DevicePack {
  void* buffers[4] = {nullptr, nullptr, nullptr, nullptr};
  cuteafd_kx7_pack_t header{};
  explicit DevicePack(const Pack& p) {
    const void* host[4] = {p.lut.data(), p.offsets.data(), p.lengths.data(), p.streams.data()};
    const size_t bytes[4] = {p.lut.size() * 4, p.offsets.size() * 4, p.lengths.size() * 2, p.streams.size()};
    for (int i = 0; i < 4; ++i) {
      CUDA_MUST(cudaMalloc(&buffers[i], bytes[i]));
      CUDA_MUST(cudaMemcpy(buffers[i], host[i], bytes[i], cudaMemcpyHostToDevice));
    }
    header = {uint32_t(sizeof(cuteafd_kx7_pack_t)), p.huffman ? CUTEAFD_KX7_FORMAT_V11_HUFFMAN : CUTEAFD_KX7_FORMAT_V11,
              p.n, p.k, p.slices, 4 * p.slices, p.k / (4 * p.slices), 512u, p.ragged ? 1u : 0u,
              p.huffman ? 0u : uint32_t(31 - p.esc_len - p.rare_bits), p.huffman ? 0u : (1u << (p.rare_bits + 1)) - 2u,
              uint32_t(p.lut.size()), buffers[0], buffers[1], buffers[2], buffers[3]};
  }
  ~DevicePack() {
    for (void* b : buffers) cudaFree(b);
  }
};

void device_part(const Pack& p, cudaStream_t s) {
  DevicePack d(p);
  const size_t n = size_t(p.n), k = size_t(p.k);
  // 2. decode
  void* out = nullptr;
  CUDA_MUST(cudaMalloc(&out, n * k * 2));
  CUDA_MUST(cudaMemsetAsync(out, 0xff, n * k * 2, s));
  MUST(cuteafd_kx7_decode(&d.header, out, s));
  std::vector<uint16_t> got(n * k);
  CUDA_MUST(cudaMemcpyAsync(got.data(), out, got.size() * 2, cudaMemcpyDeviceToHost, s));
  CUDA_MUST(cudaStreamSynchronize(s));
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != p.source[i] && bad++ == 0) first = i;
  }
  check(bad == 0, "the GPU decode gives back the source", p.name, (long long)bad, (long long)first);
  cudaFree(out);
  // 3. linear against the BF16 cuBLAS linear over the source, and a graph of it
  const int max_rows = 300;
  uint64_t r = 0x9e3779b97f4a7c15ull ^ k;
  std::vector<uint16_t> x(size_t(max_rows) * k);
  for (auto& v : x) {
    r = r * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t bits = uint32_t(r >> 33);
    v = uint16_t(((bits & 1) << 15) | ((118 + (bits >> 1) % 12) << 7) | ((bits >> 5) & 127));
  }
  const size_t ws_bytes = cuteafd_kx7_workspace(max_rows, p.n, p.k);
  check(ws_bytes >= n * k * 2, "the workspace holds the decoded matrix", p.name, (long long)ws_bytes);
  void *dx, *dw, *ref, *res, *ws;
  CUDA_MUST(cudaMalloc(&dx, x.size() * 2));
  CUDA_MUST(cudaMalloc(&dw, n * k * 2));
  CUDA_MUST(cudaMalloc(&ref, size_t(max_rows) * n * 2));
  CUDA_MUST(cudaMalloc(&res, size_t(max_rows) * n * 2));
  CUDA_MUST(cudaMalloc(&ws, ws_bytes));
  CUDA_MUST(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CUDA_MUST(cudaMemcpy(dw, p.source.data(), n * k * 2, cudaMemcpyHostToDevice));
  std::vector<uint16_t> a(size_t(max_rows) * n), b(a.size());
  // Rows 1..64 run the fused GEMV: bit for bit its BF16 twin over the source weights. Wider steps
  // decode once and run the BF16 cuBLAS GEMM: bit for bit that GEMM over the source weights.
  for (int rows : {1, 2, 7, 8, 9, 16, 17, 31, 32, 33, 48, 63, 64, 65, 128, 129, 300}) {
    const size_t bytes = size_t(rows) * n * 2;
    const bool fused = rows <= 64 && p.n % 32 == 0 && p.k % 256 == 0;
    if (fused) {
      CUDA_MUST(cudaError_t(cuteafd_kx7_reference_gemv(dx, dw, ref, rows, int32_t(n), int32_t(k), s)));
    } else {
      MUST(cuteafd_cuda_linear_bf16_cublas_async(static_cast<const uint16_t*>(dx), static_cast<const uint16_t*>(dw),
                                                 nullptr, static_cast<uint16_t*>(ref), size_t(rows), k, n, s));
    }
    MUST(cuteafd_kx7_linear(dx, &d.header, res, rows, ws, ws_bytes, s));
    CUDA_MUST(cudaMemcpyAsync(a.data(), ref, bytes, cudaMemcpyDeviceToHost, s));
    CUDA_MUST(cudaMemcpyAsync(b.data(), res, bytes, cudaMemcpyDeviceToHost, s));
    CUDA_MUST(cudaStreamSynchronize(s));
    check(std::memcmp(a.data(), b.data(), bytes) == 0,
          fused ? "kx7 GEMV equals its BF16 twin" : "kx7 linear equals the BF16 cuBLAS linear", p.name, rows);
  }
  // A row's bits do not depend on the step's width inside the GEMV (1..64 rows): row 0 of every
  // width equals row 0 of a one-row call.
  if (p.n % 32 == 0 && p.k % 256 == 0) {
    std::vector<uint16_t> one(n), wide(size_t(64) * n);
    MUST(cuteafd_kx7_linear(dx, &d.header, res, 1, ws, ws_bytes, s));
    CUDA_MUST(cudaMemcpyAsync(one.data(), res, n * 2, cudaMemcpyDeviceToHost, s));
    CUDA_MUST(cudaStreamSynchronize(s));
    for (int rows : {2, 16, 17, 32, 33, 64}) {
      MUST(cuteafd_kx7_linear(dx, &d.header, res, rows, ws, ws_bytes, s));
      CUDA_MUST(cudaMemcpyAsync(wide.data(), res, n * 2, cudaMemcpyDeviceToHost, s));
      CUDA_MUST(cudaStreamSynchronize(s));
      check(std::memcmp(one.data(), wide.data(), n * 2) == 0, "row 0 is invariant to the step width", p.name, rows);
    }
  }
  const int rows = 64;
  const size_t bytes = size_t(rows) * n * 2;
  MUST(cuteafd_kx7_linear(dx, &d.header, res, rows, ws, ws_bytes, s));
  CUDA_MUST(cudaMemcpyAsync(a.data(), res, bytes, cudaMemcpyDeviceToHost, s));
  CUDA_MUST(cudaStreamSynchronize(s));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_MUST(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
  MUST(cuteafd_kx7_linear(dx, &d.header, res, rows, ws, ws_bytes, s));
  CUDA_MUST(cudaStreamEndCapture(s, &graph));
  CUDA_MUST(cudaGraphInstantiate(&exec, graph, 0));
  for (int replay = 0; replay < 2; ++replay) {
    CUDA_MUST(cudaMemsetAsync(res, 0, bytes, s));
    CUDA_MUST(cudaGraphLaunch(exec, s));
    CUDA_MUST(cudaMemcpyAsync(b.data(), res, bytes, cudaMemcpyDeviceToHost, s));
    CUDA_MUST(cudaStreamSynchronize(s));
    check(std::memcmp(a.data(), b.data(), bytes) == 0, "a graph replay equals the eager linear", p.name, replay);
  }
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  // 4. refusals
  auto refused = [&](const char* what, void (*edit)(cuteafd_kx7_pack_t&)) {
    cuteafd_kx7_pack_t bad = d.header;
    edit(bad);
    check(cuteafd_kx7_decode(&bad, ws, s) != 0 && cuteafd_kx7_linear(dx, &bad, res, 1, ws, ws_bytes, s) != 0, what,
          p.name);
  };
  refused("an unknown format", [](cuteafd_kx7_pack_t& h) { h.format = 1; });
  refused("a short header", [](cuteafd_kx7_pack_t& h) { h.struct_bytes = 40; });
  refused("rows off the 8-row blocks", [](cuteafd_kx7_pack_t& h) { h.n -= 4; });
  refused("K off the slices' chunks", [](cuteafd_kx7_pack_t& h) { h.k += 64; });
  refused("streams per row off the slices", [](cuteafd_kx7_pack_t& h) { h.streams_per_row += 1; });
  refused("another quad stride", [](cuteafd_kx7_pack_t& h) { h.quad_stride = 16; });
  refused("tables of another size", [](cuteafd_kx7_pack_t& h) { h.table_words += 1; });
  refused("misaligned streams", [](cuteafd_kx7_pack_t& h) { h.streams = static_cast<const uint8_t*>(h.streams) + 4; });
  refused("missing offsets", [](cuteafd_kx7_pack_t& h) { h.offsets = nullptr; });
  check(cuteafd_kx7_decode(nullptr, ws, s) != 0, "a null pack is refused", p.name);
  check(cuteafd_kx7_decode(&d.header, static_cast<uint8_t*>(ws) + 2, s) != 0, "a misaligned output is refused", p.name);
  check(cuteafd_kx7_linear(dx, &d.header, res, 1, ws, n * k, s) != 0, "a short workspace is refused", p.name);
  check(cuteafd_kx7_workspace(1, p.n, p.k + 1) == 0 && cuteafd_kx7_workspace(0, p.n, p.k) == 0,
        "invalid workspace shapes", p.name);
  CUDA_MUST(cudaGetLastError());
  for (void* ptr : {dx, dw, ref, res, ws}) cudaFree(ptr);
  std::printf("device %-22s [%d, %d]: decode exact, GEMV bit for bit its BF16 twin (1..64 rows, width-invariant), linear bit for bit cuBLAS (65..300), graph\n",
              p.name.c_str(), p.n, p.k);
}

}  // namespace

int main(int argc, char** argv) {
  bool host_only = false;
  std::vector<std::string> dirs;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--host-only") == 0) host_only = true;
    else dirs.push_back(argv[i]);
  }
  if (dirs.empty()) {
    for (const char* name : {"fp8_born_32x1024", "rare_heavy_16x1024"}) {
      dirs.push_back(std::string(CUTEAFD_KX7_FIXTURE_DIR) + "/" + name);
    }
  }
  std::vector<Pack> packs;
  for (const std::string& dir : dirs) {
    packs.push_back(load(dir));
    host_part(packs.back());
  }
  if (!host_only) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      std::printf("SKIP: no CUDA device\n");
      return 77;
    }
    cudaStream_t s;
    CUDA_MUST(cudaStreamCreate(&s));
    for (const Pack& p : packs) device_part(p, s);
    cudaStreamDestroy(s);
  }
  std::printf("%s: kx7 selftest (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
