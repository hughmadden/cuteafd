// Standalone harness for native/shared/cuda/packed_head.cu, the exact 12-bit
// BF16 vocabulary head. On random head-shaped BF16 matrices (k = 5120; per-row
// exponents falling off geometrically below a per-row top over spans of 11..31,
// a few values outside each row's 16-exponent window, some rows with none) it
// checks, per row count n in {1024, 4096, 64640} (split-K and single-split
// plans on any current GPU; --vocab adds the full 129280-row head):
//   1. the pack verifies itself, matches a host packer byte for byte and
//      decodes every value back bit for bit (plus the golden digest shared with
//      cuteafd-core's CPU reference);
//   2. every weight through the GEMV bit for bit (one-hot activation rows), and
//      exact logits against an FP64 reference of the exact BF16 weights within
//      FP32 reordering tolerance;
//   3. each row's logits bit-identical across activation row counts 1, 2, 5,
//      6, 12, 48, 64 and 100 (two passes), alone or in a batch, FP32 or BF16;
//   4. coarse logits within the bound their weights' coarse errors imply;
//   5. verification refusing a corrupted coarse byte, fine nibble, base,
//      table value, table column and row pointer, and an overflowing table.
// It ends with one PASS/FAIL line: exit 0 PASS, 1 FAIL, 77 without a CUDA
// device. --host-only runs the host packer and golden digest without CUDA.
// From the repository root (-arch=sm_121 on a GB10; compiling needs no GPU):
//
//   nvcc -std=c++17 -O3 -Xcompiler -fopenmp -arch=sm_120f -o packed_head_selftest native/tests/packed_head_selftest.cu native/shared/cuda/packed_head.cu -lgomp
//   ./packed_head_selftest [--vocab]
//
// The native build also makes it (ctest -R packed_head), without OpenMP: the
// host reference then runs on one core.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int32_t cuteafd_packed_head_pack(const void* w, void* const* planes, int64_t capacity, int32_t n,
                                            int32_t k, void* status, void* stream);
extern "C" int32_t cuteafd_packed_head_verify(const void* w, const void* const* planes, int64_t capacity, int32_t n,
                                              int32_t k, void* status, void* stream);
extern "C" size_t cuteafd_packed_head_workspace(int32_t rows, int32_t k, int32_t n);
extern "C" int32_t cuteafd_packed_head_linear(const void* x, const void* const* planes, void* out, int32_t out_f32,
                                              int32_t rows, int32_t k, int32_t n, int32_t coarse, void* workspace,
                                              size_t workspace_bytes, void* stream);

namespace {

constexpr int kK = 5120;
constexpr uint64_t kGoldenDigest = 0xbcf7e770276c4482ull;  // cuteafd_core::packed_head::GOLDEN_DIGEST
enum Plane { kCoarse, kFine, kBase, kRowPtr, kColumns, kValues, kPlanes };

int failures = 0;

void fail(const char* what, long long a = 0, long long b = 0) {
  ++failures;
  if (failures <= 40) std::fprintf(stderr, "FAIL: %s (%lld, %lld)\n", what, a, b);
}

void check(bool ok, const char* what, long long a = 0, long long b = 0) {
  if (!ok) fail(what, a, b);
}

#define CUDA_MUST(expr)                                                                        \
  do {                                                                                         \
    const cudaError_t err_ = (expr);                                                           \
    if (err_ != cudaSuccess) {                                                                 \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err_)); \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

#define MUST(expr)                                                                      \
  do {                                                                                  \
    const int32_t status_ = (expr);                                                     \
    if (status_ != 0) {                                                                 \
      std::fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, status_);   \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

// --- the format, on the host (cuteafd-core's packed_head reference) ---------

uint32_t exponent_of(uint32_t bits) { return (bits >> 7) & 0xffu; }
bool in_window(uint32_t bits, uint32_t e0) { return exponent_of(bits) >= e0 && exponent_of(bits) <= e0 + 15; }

uint32_t clamp_to_window(uint32_t bits, uint32_t e0) {
  const uint32_t e = exponent_of(bits), sign = bits & 0x8000u;
  if (e < e0) return sign | (e0 << 7);
  if (e > e0 + 15) return sign | ((e0 + 15) << 7) | 0x7fu;
  return bits;
}

uint32_t decode_exact(uint32_t coarse, uint32_t fine, uint32_t e0) {
  return ((coarse & 0x80u) << 8) | (((coarse & 0x7fu) << 4) + (e0 << 7)) | (fine & 0xfu);
}

uint32_t decode_coarse(uint32_t coarse, uint32_t e0) { return decode_exact(coarse, 8, e0); }

float bf16_float(uint32_t bits) {
  const uint32_t b = bits << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

uint32_t float_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

// Round to nearest even (finite values only), as __float2bfloat16.
uint16_t bf16_rn(float v) {
  const uint32_t b = float_bits(v);
  return uint16_t((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

// Row `row`, column `col`: coarse byte offset, fine u32 word, nibble shift.
void fragment_position(int row, int col, int k, size_t* byte, size_t* word, int* shift) {
  const int g = row & 7, upper = (row >> 3) & 1, kk = col & 31;
  const int step = kk >> 4, half = (kk >> 3) & 1, within = kk & 7;
  const int lane = g * 4 + (within >> 1), h = within & 1, j = half * 2 + upper;
  const size_t index = (size_t(row >> 4) * (k / 32) + (col >> 5)) * 32 + lane;
  *byte = index * 16 + step * 8 + j * 2 + h;
  *word = index * 2 + step;
  *shift = 4 * j + 16 * h;
}

struct HostPacked {
  int n = 0, k = 0;
  std::vector<uint8_t> coarse, base;
  std::vector<uint32_t> fine, row_ptr, columns;
  std::vector<uint16_t> values;
  uint32_t max_row = 0;

  uint32_t coarse_at(int r, int c) const {
    size_t byte, word;
    int shift;
    fragment_position(r, c, k, &byte, &word, &shift);
    return coarse[byte];
  }
  uint32_t fine_at(int r, int c) const {
    size_t byte, word;
    int shift;
    fragment_position(r, c, k, &byte, &word, &shift);
    return (fine[word] >> shift) & 0xfu;
  }
};

HostPacked host_pack(const std::vector<uint16_t>& w, int n, int k) {
  HostPacked p;
  p.n = n;
  p.k = k;
  p.coarse.assign(size_t(n) * k, 0);
  p.fine.assign(size_t(n) * k / 8, 0);
  p.base.assign(n, 0);
  p.row_ptr.assign(n + 1, 0);
  for (int r = 0; r < n; ++r) {
    const uint16_t* row = w.data() + size_t(r) * k;
    uint32_t histogram[256] = {};
    for (int c = 0; c < k; ++c) ++histogram[exponent_of(row[c])];
    uint32_t best = 0, e0 = 0;
    for (uint32_t e = 0; e <= 240; ++e) {
      uint32_t count = 0;
      for (int i = 0; i < 16; ++i) count += histogram[e + i];
      if (count > best) {
        best = count;
        e0 = e;
      }
    }
    p.base[r] = uint8_t(e0);
    p.max_row = std::max(p.max_row, uint32_t(k) - best);
    for (int c = 0; c < k; ++c) {
      const uint32_t coded = clamp_to_window(row[c], e0);
      size_t byte, word;
      int shift;
      fragment_position(r, c, k, &byte, &word, &shift);
      p.coarse[byte] = uint8_t(((coded >> 8) & 0x80u) | ((exponent_of(coded) - e0) << 3) | ((coded >> 4) & 7u));
      p.fine[word] |= (coded & 0xfu) << shift;
      if (!in_window(row[c], e0)) {
        p.columns.push_back(uint32_t(c));
        p.values.push_back(row[c]);
      }
    }
    p.row_ptr[r + 1] = uint32_t(p.columns.size());
  }
  return p;
}

uint64_t digest(const uint8_t* coarse, const uint8_t* fine, const uint8_t* base, const uint8_t* row_ptr,
                const uint8_t* columns, const uint8_t* values, int n, int k, size_t exceptions) {
  uint64_t hash = 0xcbf29ce484222325ull;
  auto eat = [&hash](const uint8_t* p, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 0x100000001b3ull;
  };
  eat(coarse, size_t(n) * k);
  eat(fine, size_t(n) * k / 2);
  eat(base, n);
  eat(row_ptr, (size_t(n) + 1) * 4);
  eat(columns, exceptions * 4);
  eat(values, exceptions * 2);
  return hash;
}

uint64_t host_digest(const HostPacked& p) {
  return digest(p.coarse.data(), reinterpret_cast<const uint8_t*>(p.fine.data()), p.base.data(),
                reinterpret_cast<const uint8_t*>(p.row_ptr.data()), reinterpret_cast<const uint8_t*>(p.columns.data()),
                reinterpret_cast<const uint8_t*>(p.values.data()), p.n, p.k, p.columns.size());
}

// --- inputs -----------------------------------------------------------------

uint32_t lcg(uint64_t& s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return uint32_t(s >> 33);
}

// cuteafd_core::packed_head::golden_input: [32, 256].
std::vector<uint16_t> golden_input() {
  uint64_t state = 0x2545f4914f6cdd1dull;
  std::vector<uint16_t> w(32 * 256);
  for (int r = 0; r < 32; ++r) {
    for (int c = 0; c < 256; ++c) {
      const uint32_t v = lcg(state);
      uint32_t d = (v >> 8) % 23;
      if (r % 4 == 0) d %= 12;
      const uint32_t e = 120 + uint32_t(r % 3) - d;
      w[r * 256 + c] = uint16_t((v & 1) << 15 | e << 7 | ((v >> 1) & 0x7f));
    }
  }
  w[5 * 256 + 3] = 0x0000;
  w[6 * 256 + 7] = 0x0001;
  w[7 * 256 + 9] = 0x7f80;
  w[9 * 256 + 200] = 0xffc1;
  return w;
}

// Random signs and mantissas; exponents fall off geometrically below a per-row
// top, as |w| of a Gaussian does, truncated to the row's span. Kind 0 rows stay
// inside 15 exponents (no exceptions); kind 1 keep their natural tail (spans
// 16..31); kind 2 add 1..6 values far below the bulk; kind 3 also 1..2 above it,
// and one in eight of those a zero and a negative subnormal.
std::vector<uint16_t> head_matrix(int n, int k, uint64_t seed) {
  std::vector<uint16_t> w(size_t(n) * k);
#pragma omp parallel for schedule(static)
  for (int r = 0; r < n; ++r) {
    uint64_t s = seed ^ (0x9e3779b97f4a7c15ull * uint64_t(r + 1));
    uint16_t* row = w.data() + size_t(r) * k;
    const int kind = r % 4;
    const uint32_t top = 124 - r % 6;
    const uint32_t span = kind == 0 ? 11 + r % 5 : 16 + lcg(s) % 16;
    auto value = [&s](uint32_t e) {
      const uint32_t v = lcg(s);
      return uint16_t((v & 1) << 15 | e << 7 | ((v >> 1) & 0x7f));
    };
    for (int c = 0; c < k; ++c) {
      uint32_t d = uint32_t(__builtin_ctz(lcg(s) | 0x80000000u));  // P(d) = 2^-(d + 1)
      if (d > span - 1) d = span - 1;
      row[c] = value(top - d);
    }
    if (kind >= 2) {
      for (uint32_t i = 0, extra = 1 + lcg(s) % 6; i < extra; ++i) row[lcg(s) % k] = value(top - 16 - lcg(s) % (span - 15));
    }
    if (kind == 3) {
      for (uint32_t i = 0, extra = 1 + lcg(s) % 2; i < extra; ++i) row[lcg(s) % k] = value(top + 1 + lcg(s) % 12);
      if (r % 32 == 3) {
        row[lcg(s) % k] = 0x0000;
        row[lcg(s) % k] = 0x8003;
      }
    }
  }
  return w;
}

// Activations about N(0, 1), in BF16.
std::vector<uint16_t> activations(int rows, int k, uint64_t seed) {
  std::vector<uint16_t> x(size_t(rows) * k);
  uint64_t s = seed;
  for (auto& v : x) {
    float t = 0.0f;
    for (int i = 0; i < 12; ++i) t += float(lcg(s)) / 2147483648.0f;
    v = bf16_rn(t - 6.0f);
  }
  return x;
}

// --- device buffers ---------------------------------------------------------

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit Buf(size_t n = 0) { alloc(n); }
  void alloc(size_t n) {
    if (p) CUDA_MUST(cudaFree(p));
    p = nullptr;
    bytes = n;
    if (n) CUDA_MUST(cudaMalloc(&p, n));
  }
  void upload(const void* src, size_t n, size_t offset = 0) {
    CUDA_MUST(cudaMemcpy(static_cast<uint8_t*>(p) + offset, src, n, cudaMemcpyHostToDevice));
  }
  void download(void* dst, size_t n, size_t offset = 0) const {
    CUDA_MUST(cudaMemcpy(dst, static_cast<const uint8_t*>(p) + offset, n, cudaMemcpyDeviceToHost));
  }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  ~Buf() {
    if (p) cudaFree(p);
  }
};

// One packed head on the device.
struct DeviceHead {
  int n, k;
  int64_t capacity;
  Buf coarse, fine, base, row_ptr, columns, values, status;
  void* planes[kPlanes];

  DeviceHead(int n_, int k_, int64_t capacity_)
      : n(n_), k(k_), capacity(capacity_), coarse(size_t(n_) * k_), fine(size_t(n_) * k_ / 2), base(n_),
        row_ptr((size_t(n_) + 1) * 4), columns(size_t(std::max<int64_t>(capacity_, 1)) * 4),
        values(size_t(std::max<int64_t>(capacity_, 1)) * 2), status(32) {
    void* p[kPlanes] = {coarse.p, fine.p, base.p, row_ptr.p, columns.p, values.p};
    std::memcpy(planes, p, sizeof(planes));
  }
  const void* const* view() const { return planes; }

  struct Status {
    unsigned long long exceptions, bad_rows, first_bad_row, max_row;
  };
  Status read_status() const {
    Status s;
    CUDA_MUST(cudaDeviceSynchronize());
    status.download(&s, sizeof(s));
    return s;
  }
  Status pack(const Buf& w) {
    MUST(cuteafd_packed_head_pack(w.p, planes, capacity, n, k, status.p, nullptr));
    return read_status();
  }
  Status verify(const Buf& w) {
    MUST(cuteafd_packed_head_verify(w.p, view(), capacity, n, k, status.p, nullptr));
    return read_status();
  }
};

void linear(const DeviceHead& head, const void* x, void* out, bool out_f32, int rows, bool coarse, Buf& workspace,
            cudaStream_t stream = nullptr) {
  MUST(cuteafd_packed_head_linear(x, head.view(), out, out_f32, rows, head.k, head.n, coarse, workspace.p,
                                  workspace.bytes, stream));
}

int sm_count() {
  int device = 0, count = 0;
  CUDA_MUST(cudaGetDevice(&device));
  CUDA_MUST(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
  return count;
}

int splits_of(int n, int k) {
  const int tiles = n / 16, kbs = k / 128;
  const int wanted = (sm_count() * 16 + tiles - 1) / tiles;
  const int s = wanted < 1 ? 1 : (wanted > kbs ? kbs : wanted);
  const int per = (kbs + s - 1) / s;
  return (kbs + per - 1) / per;
}

// --- the checks -------------------------------------------------------------

struct Totals {
  unsigned long long exceptions = 0, max_row = 0;
  double exact_error = 0.0, coarse_ratio = 0.0, coarse_relative = 0.0;
  long long onehot_values = 0, invariant_rows = 0;
  int refused = 0, corruptions = 0;
  std::string cases;
};

// 1. The golden input packs on the GPU to the CPU reference's bytes.
void golden_check() {
  const std::vector<uint16_t> w = golden_input();
  const HostPacked host = host_pack(w, 32, 256);
  check(host_digest(host) == kGoldenDigest, "golden host digest differs from cuteafd-core");
  Buf wd(w.size() * 2);
  wd.upload(w.data(), wd.bytes);
  DeviceHead head(32, 256, 32 * 256);
  const DeviceHead::Status s = head.pack(wd);
  check(s.exceptions == host.columns.size() && s.bad_rows == 0, "golden pack status", (long long)s.exceptions,
        (long long)s.bad_rows);
  const size_t exceptions = std::min<size_t>(s.exceptions, 32 * 256);
  std::vector<uint8_t> coarse(head.coarse.bytes), fine(head.fine.bytes), base(32), row_ptr(33 * 4);
  std::vector<uint8_t> columns(exceptions * 4), values(exceptions * 2);
  head.coarse.download(coarse.data(), coarse.size());
  head.fine.download(fine.data(), fine.size());
  head.base.download(base.data(), base.size());
  head.row_ptr.download(row_ptr.data(), row_ptr.size());
  head.columns.download(columns.data(), columns.size());
  head.values.download(values.data(), values.size());
  check(digest(coarse.data(), fine.data(), base.data(), row_ptr.data(), columns.data(), values.data(), 32, 256,
               exceptions) == kGoldenDigest,
        "golden GPU pack digest differs from cuteafd-core");
}

void run_case(int n, uint64_t seed, Totals& totals, bool corrupt, bool time) {
  const int k = kK;
  const int splits = splits_of(n, k);
  std::printf("case n=%d k=%d splits=%d\n", n, k, splits);
  std::fflush(stdout);
  const std::vector<uint16_t> w = head_matrix(n, k, seed);
  const HostPacked host = host_pack(w, n, k);
  const int64_t capacity = (int64_t(n) * k + 1023) / 1024;
  const size_t exceptions = host.columns.size();
  totals.exceptions += exceptions;
  totals.max_row = std::max<unsigned long long>(totals.max_row, host.max_row);

  // 1. Pack, self-verification and byte-for-byte agreement with the host packer.
  Buf wd(w.size() * 2);
  wd.upload(w.data(), wd.bytes);
  DeviceHead head(n, k, capacity);
  const DeviceHead::Status s = head.pack(wd);
  check(s.exceptions == exceptions, "pack exception count", (long long)s.exceptions, (long long)exceptions);
  check(s.bad_rows == 0 && s.first_bad_row == ~0ull, "pack self-verification", (long long)s.bad_rows,
        (long long)s.first_bad_row);
  check(s.max_row == host.max_row, "pack most exceptions in a row", (long long)s.max_row, host.max_row);
  check(int64_t(exceptions) <= capacity, "test head overflows its table", (long long)exceptions, capacity);
  {
    std::vector<uint8_t> coarse(head.coarse.bytes), base(n);
    std::vector<uint32_t> fine(head.fine.bytes / 4), row_ptr(n + 1), columns(exceptions);
    std::vector<uint16_t> values(exceptions);
    head.coarse.download(coarse.data(), coarse.size());
    head.fine.download(fine.data(), head.fine.bytes);
    head.base.download(base.data(), n);
    head.row_ptr.download(row_ptr.data(), row_ptr.size() * 4);
    head.columns.download(columns.data(), exceptions * 4);
    head.values.download(values.data(), exceptions * 2);
    check(coarse == host.coarse, "coarse plane differs from the host packer");
    check(fine == host.fine, "fine plane differs from the host packer");
    check(base == host.base, "bases differ from the host packer");
    check(row_ptr == host.row_ptr, "row pointers differ from the host packer");
    check(columns == host.columns && values == host.values, "exception table differs from the host packer");
    long long wrong = 0;
#pragma omp parallel for reduction(+ : wrong) schedule(static)
    for (int r = 0; r < n; ++r) {
      uint32_t at = row_ptr[r];
      for (int c = 0; c < k; ++c) {
        size_t byte, word;
        int shift;
        fragment_position(r, c, k, &byte, &word, &shift);
        uint32_t v = decode_exact(coarse[byte], (fine[word] >> shift) & 0xfu, base[r]);
        if (at < row_ptr[r + 1] && columns[at] == uint32_t(c)) {
          wrong += v != clamp_to_window(w[size_t(r) * k + c], base[r]);
          v = values[at++];
        }
        wrong += v != w[size_t(r) * k + c];
      }
      wrong += at != row_ptr[r + 1];
    }
    check(wrong == 0, "downloaded pack does not decode to the source", wrong);
  }

  Buf workspace(cuteafd_packed_head_workspace(100, k, n));
  check(workspace.bytes > 0 && cuteafd_packed_head_workspace(64, k, n) <= workspace.bytes, "workspace size");
  check(cuteafd_packed_head_workspace(0, k, n) == 0 && cuteafd_packed_head_workspace(1, k, n + 8) == 0,
        "workspace accepts invalid shapes");
  {
    Buf x1(size_t(k) * 2), o1(size_t(n) * 4);
    check(cuteafd_packed_head_linear(x1.p, head.view(), o1.p, 1, 1, k, n, 0, workspace.p,
                                     cuteafd_packed_head_workspace(1, k, n) - 1, nullptr) != 0,
          "linear accepts a short workspace");
  }

  // 2a. Every weight through the GEMV: one-hot activation rows give each
  // output exactly one product (the coded weight, then its exception's fix-up).
  {
    Buf xd(size_t(64) * k * 2), od(size_t(64) * n * 4);
    std::vector<uint16_t> x(size_t(64) * k);
    std::vector<float> out(size_t(64) * n);
    long long wrong[2] = {0, 0};
    for (int c0 = 0; c0 < k; c0 += 64) {
      std::fill(x.begin(), x.end(), 0);
      for (int m = 0; m < 64; ++m) x[size_t(m) * k + c0 + m] = 0x3f80;  // 1.0
      xd.upload(x.data(), xd.bytes);
      for (int coarse = 0; coarse < 2; ++coarse) {
        linear(head, xd.p, od.p, true, 64, coarse, workspace);
        CUDA_MUST(cudaDeviceSynchronize());
        od.download(out.data(), od.bytes);
        long long bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(static)
        for (int r = 0; r < n; ++r) {
          for (int m = 0; m < 64; ++m) {
            const int c = c0 + m;
            const uint32_t e0 = host.base[r];
            float want;
            if (coarse) {
              want = bf16_float(decode_coarse(host.coarse_at(r, c), e0));
            } else {
              want = bf16_float(w[size_t(r) * k + c]);
              for (uint32_t i = host.row_ptr[r]; i < host.row_ptr[r + 1]; ++i) {
                if (host.columns[i] != uint32_t(c)) continue;
                const float coded = bf16_float(clamp_to_window(host.values[i], e0));
                want = std::fma(bf16_float(host.values[i]) - coded, 1.0f, coded);
              }
            }
            bad += !(out[size_t(m) * n + r] == want);
          }
        }
        wrong[coarse] += bad;
      }
    }
    check(wrong[0] == 0, "one-hot exact GEMV differs from the exact weights", wrong[0]);
    check(wrong[1] == 0, "one-hot coarse GEMV differs from the coarse weights", wrong[1]);
    totals.onehot_values += 2ll * n * k;
  }

  // 2b-4. Random activations: reference, BF16 outputs, invariance, coarse bound.
  const int rows = 100;
  const std::vector<uint16_t> x = activations(rows, k, seed ^ 0x5bd1e995u);
  Buf xd(x.size() * 2);
  xd.upload(x.data(), xd.bytes);
  Buf o64(size_t(64) * n * 4), obf(size_t(64) * n * 2), ocoarse(size_t(64) * n * 4), oo(size_t(rows) * n * 4);
  linear(head, xd.p, o64.p, true, 64, false, workspace);
  linear(head, xd.p, obf.p, false, 64, false, workspace);
  linear(head, xd.p, ocoarse.p, true, 64, true, workspace);
  CUDA_MUST(cudaDeviceSynchronize());
  std::vector<float> exact(size_t(64) * n), coarse(size_t(64) * n);
  std::vector<uint16_t> exact_bf16(size_t(64) * n);
  o64.download(exact.data(), o64.bytes);
  obf.download(exact_bf16.data(), obf.bytes);
  ocoarse.download(coarse.data(), ocoarse.bytes);
  {
    long long bad = 0;
    for (size_t i = 0; i < exact.size(); ++i) bad += exact_bf16[i] != bf16_rn(exact[i]);
    check(bad == 0, "BF16 logits are not the FP32 logits rounded", bad);
  }
  {
    const int checked = n <= 4096 ? 64 : 16;
    std::vector<float> xf(size_t(checked) * k);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = bf16_float(x[i]);
    double worst_exact = 0.0, worst_ratio = 0.0, relative_sum = 0.0;
    long long bad_exact = 0, bad_coarse = 0;
#pragma omp parallel for reduction(max : worst_exact, worst_ratio) reduction(+ : relative_sum, bad_exact, bad_coarse) \
    schedule(dynamic, 16)
    for (int r = 0; r < n; ++r) {
      std::vector<float> wf(k), cf(k);
      for (int c = 0; c < k; ++c) {
        wf[c] = bf16_float(w[size_t(r) * k + c]);
        cf[c] = bf16_float(decode_coarse(host.coarse_at(r, c), host.base[r]));
      }
      for (int m = 0; m < checked; ++m) {
        const float* xr = xf.data() + size_t(m) * k;
        double ref = 0.0, magnitude = 0.0, coarse_bound = 0.0, coarse_magnitude = 0.0;
        for (int c = 0; c < k; ++c) {
          ref += double(wf[c]) * xr[c];
          magnitude += std::fabs(double(wf[c]) * xr[c]);
          coarse_bound += std::fabs(double(cf[c]) - wf[c]) * std::fabs(xr[c]);
          coarse_magnitude += std::fabs(double(cf[c]) * xr[c]);
        }
        const double tolerance = std::ldexp(magnitude, -15) + 1e-30;
        const double error = std::fabs(double(exact[size_t(m) * n + r]) - ref);
        bad_exact += error > tolerance;
        worst_exact = std::max(worst_exact, error / (magnitude + 1e-30));
        const double bound = coarse_bound + std::ldexp(coarse_magnitude, -15) + 1e-30;
        const double coarse_error = std::fabs(double(coarse[size_t(m) * n + r]) - ref);
        bad_coarse += coarse_error > bound;
        worst_ratio = std::max(worst_ratio, coarse_error / bound);
        relative_sum += coarse_error / (magnitude + 1e-30);
      }
    }
    check(bad_exact == 0, "exact logits outside FP32 reordering tolerance (2^-15 of sum |w x|)", bad_exact);
    check(bad_coarse == 0, "coarse logits outside their weight-error bound", bad_coarse);
    totals.exact_error = std::max(totals.exact_error, worst_exact);
    totals.coarse_ratio = std::max(totals.coarse_ratio, worst_ratio);
    totals.coarse_relative = std::max(totals.coarse_relative, relative_sum / (double(n) * checked));
  }
  {
    // Row counts within one pass, single rows alone, and two passes.
    auto same = [&](const Buf& got, const std::vector<float>& want, int first, int count, const char* what) {
      std::vector<float> host_out(size_t(count) * n);
      CUDA_MUST(cudaDeviceSynchronize());
      got.download(host_out.data(), host_out.size() * 4);
      const bool ok = std::memcmp(host_out.data(), want.data() + size_t(first) * n, host_out.size() * 4) == 0;
      check(ok, what, first, count);
      totals.invariant_rows += ok ? count : 0;
    };
    for (int coarse_mode = 0; coarse_mode < 2; ++coarse_mode) {
      const std::vector<float>& want = coarse_mode ? coarse : exact;
      for (const int count : {1, 2, 5, 6, 12, 48, 64}) {
        linear(head, xd.p, oo.p, true, count, coarse_mode, workspace);
        same(oo, want, 0, count, coarse_mode ? "coarse logits change with the row count" : "exact logits change with the row count");
      }
      for (const int m : {0, 7, 8, 13, 47, 63}) {
        linear(head, static_cast<uint16_t*>(xd.p) + size_t(m) * k, oo.p, true, 1, coarse_mode, workspace);
        same(oo, want, m, 1, coarse_mode ? "coarse logits of a row alone differ" : "exact logits of a row alone differ");
      }
      // 100 rows: two passes (64 + 36); the second pass equals those rows alone.
      std::vector<float> all(size_t(rows) * n), tail(size_t(36) * n);
      linear(head, xd.p, oo.p, true, rows, coarse_mode, workspace);
      CUDA_MUST(cudaDeviceSynchronize());
      oo.download(all.data(), all.size() * 4);
      check(std::memcmp(all.data(), want.data(), size_t(64) * n * 4) == 0, "first pass of 100 rows differs");
      linear(head, static_cast<uint16_t*>(xd.p) + size_t(64) * k, oo.p, true, 36, coarse_mode, workspace);
      CUDA_MUST(cudaDeviceSynchronize());
      oo.download(tail.data(), tail.size() * 4);
      check(std::memcmp(all.data() + size_t(64) * n, tail.data(), tail.size() * 4) == 0,
            "second pass of 100 rows differs from those rows alone");
      // BF16 logits: the same at one row and at 64.
      Buf one(size_t(n) * 2);
      linear(head, static_cast<uint16_t*>(xd.p) + size_t(13) * k, one.p, false, 1, coarse_mode, workspace);
      std::vector<uint16_t> one_host(n);
      CUDA_MUST(cudaDeviceSynchronize());
      one.download(one_host.data(), one.bytes);
      if (!coarse_mode) {
        check(std::memcmp(one_host.data(), exact_bf16.data() + size_t(13) * n, size_t(n) * 2) == 0,
              "BF16 logits of a row alone differ");
      } else {
        long long bad = 0;
        for (int r = 0; r < n; ++r) bad += one_host[r] != bf16_rn(coarse[size_t(13) * n + r]);
        check(bad == 0, "coarse BF16 logits of a row alone differ", bad);
      }
    }
  }

  // 5. Corruptions the verification must refuse, each restored after.
  if (corrupt && exceptions > 0) {
    int clean = -1, busy = -1;
    for (int r = 0; r + 1 < n && (clean < 0 || busy < 0); ++r) {
      const uint32_t count = host.row_ptr[r + 1] - host.row_ptr[r];
      if (count == 0 && clean < 0) clean = r;
      if (count >= 2 && busy < 0) busy = r;
    }
    check(clean >= 0 && busy >= 0, "no clean or exception-heavy row to corrupt", clean, busy);
    auto expect_refused = [&](const char* what, int first, unsigned long long rows_bad) {
      const DeviceHead::Status v = head.verify(wd);
      const bool ok = v.bad_rows == rows_bad && v.first_bad_row == (unsigned long long)first;
      check(ok, what, (long long)v.bad_rows, (long long)v.first_bad_row);
      totals.refused += ok;
      ++totals.corruptions;
    };
    if (clean >= 0 && busy >= 0) {
      size_t byte, word;
      int shift;
      fragment_position(clean, 5, k, &byte, &word, &shift);
      uint8_t b = host.coarse[byte] ^ 1u;
      head.coarse.upload(&b, 1, byte);
      expect_refused("corrupted coarse byte accepted", clean, 1);
      head.coarse.upload(&host.coarse[byte], 1, byte);
      fragment_position(clean, 77, k, &byte, &word, &shift);
      uint32_t f = host.fine[word] ^ (1u << shift);
      head.fine.upload(&f, 4, word * 4);
      expect_refused("corrupted fine nibble accepted", clean, 1);
      head.fine.upload(&host.fine[word], 4, word * 4);
      b = uint8_t(host.base[clean] + 1);
      head.base.upload(&b, 1, clean);
      expect_refused("corrupted base accepted", clean, 1);
      head.base.upload(&host.base[clean], 1, clean);
      const uint32_t at = host.row_ptr[busy];
      uint16_t value = host.values[at] ^ 1u;
      head.values.upload(&value, 2, size_t(at) * 2);
      expect_refused("corrupted table value accepted", busy, 1);
      head.values.upload(&host.values[at], 2, size_t(at) * 2);
      uint32_t column = host.columns[at] + 1;
      head.columns.upload(&column, 4, size_t(at) * 4);
      expect_refused("corrupted table column accepted", busy, 1);
      head.columns.upload(&host.columns[at], 4, size_t(at) * 4);
      uint32_t end = host.row_ptr[busy + 1] - 1;
      head.row_ptr.upload(&end, 4, size_t(busy + 1) * 4);
      expect_refused("corrupted row pointer accepted", busy, 2);
      head.row_ptr.upload(&host.row_ptr[busy + 1], 4, size_t(busy + 1) * 4);
      const DeviceHead::Status restored = head.verify(wd);
      check(restored.bad_rows == 0, "restored head fails verification", (long long)restored.bad_rows);
      // A table one entry short: every row fails.
      DeviceHead small(n, k, int64_t(exceptions) - 1);
      const DeviceHead::Status overflow = small.pack(wd);
      const bool ok = overflow.exceptions == exceptions && overflow.bad_rows == (unsigned long long)n;
      check(ok, "overflowing table accepted", (long long)overflow.exceptions, (long long)overflow.bad_rows);
      totals.refused += ok;
      ++totals.corruptions;
    }
  }

  // Timings (information): median of 20 after 3 warm-up calls.
  if (time) {
    cudaStream_t stream;
    CUDA_MUST(cudaStreamCreate(&stream));
    for (int coarse_mode = 0; coarse_mode < 2; ++coarse_mode) {
      for (const int count : {1, 8, 64}) {
        std::vector<float> ms;
        for (int i = 0; i < 23; ++i) {
          cudaEvent_t a, b;
          CUDA_MUST(cudaEventCreate(&a));
          CUDA_MUST(cudaEventCreate(&b));
          CUDA_MUST(cudaEventRecord(a, stream));
          linear(head, xd.p, oo.p, true, count, coarse_mode, workspace, stream);
          CUDA_MUST(cudaEventRecord(b, stream));
          CUDA_MUST(cudaEventSynchronize(b));
          float dt = 0.0f;
          CUDA_MUST(cudaEventElapsedTime(&dt, a, b));
          CUDA_MUST(cudaEventDestroy(a));
          CUDA_MUST(cudaEventDestroy(b));
          if (i >= 3) ms.push_back(dt);
        }
        std::sort(ms.begin(), ms.end());
        const double bytes = double(n) * k * (coarse_mode ? 1.0 : 1.5);
        std::printf("  %s rows=%-3d %.3f ms  %.0f GB/s of planes\n", coarse_mode ? "coarse" : "exact ", count,
                    ms[ms.size() / 2], bytes / (ms[ms.size() / 2] * 1e6));
      }
    }
    CUDA_MUST(cudaStreamDestroy(stream));
  }
  char line[64];
  std::snprintf(line, sizeof(line), "%s%d", totals.cases.empty() ? "" : ",", n);
  totals.cases += line;
  std::printf("  exceptions %zu (most in a row %u)\n", exceptions, host.max_row);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  bool vocab = false, host_only = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--vocab")) vocab = true;
    else if (!std::strcmp(argv[i], "--host-only")) host_only = true;
    else {
      std::fprintf(stderr, "usage: %s [--vocab] [--host-only]\n", argv[0]);
      return 2;
    }
  }
  if (host_only) {
    // The host packer against cuteafd-core's golden digest, and a head-shaped
    // matrix's round trip through it.
    const HostPacked golden = host_pack(golden_input(), 32, 256);
    check(host_digest(golden) == kGoldenDigest, "golden host digest differs from cuteafd-core");
    const std::vector<uint16_t> w = head_matrix(1024, kK, 0x1234);
    const HostPacked host = host_pack(w, 1024, kK);
    long long wrong = 0;
    for (int r = 0; r < 1024; ++r) {
      uint32_t at = host.row_ptr[r];
      for (int c = 0; c < kK; ++c) {
        uint32_t v = decode_exact(host.coarse_at(r, c), host.fine_at(r, c), host.base[r]);
        if (at < host.row_ptr[r + 1] && host.columns[at] == uint32_t(c)) v = host.values[at++];
        wrong += v != w[size_t(r) * kK + c];
      }
    }
    check(wrong == 0, "host round trip", wrong);
    int none = 0;
    for (int r = 0; r < 1024; ++r) none += host.row_ptr[r] == host.row_ptr[r + 1];
    std::printf("packed_head_selftest --host-only %s: golden %016llx, 1024 x %d head: %zu exceptions, most %u in a "
                "row, %d rows without\n",
                failures ? "FAIL" : "PASS", (unsigned long long)host_digest(golden), kK, host.columns.size(),
                host.max_row, none);
    return failures ? 1 : 0;
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("packed_head_selftest SKIP: no CUDA device\n");
    return 77;
  }
  cudaDeviceProp prop;
  CUDA_MUST(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s, sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  check(cuteafd_packed_head_pack(nullptr, nullptr, 0, 8, kK, nullptr, nullptr) != 0, "pack accepts n = 8");
  golden_check();
  Totals totals;
  run_case(1024, 0x1234, totals, true, false);
  run_case(4096, 0x5678, totals, false, false);
  run_case(64640, 0x9abc, totals, false, true);
  if (vocab) run_case(129280, 0xdef0, totals, false, true);
  std::printf(
      "packed_head_selftest %s: n=%s on %d SMs; %llu exceptions (most %llu in a row); %lld weights bit-exact "
      "through the exact and coarse GEMVs; exact logits max |err| %.2e of sum|wx| (tolerance %.2e); %lld row "
      "outputs invariant to row count; coarse error max %.2f of its bound (mean %.2e of sum|wx|); %d/%d "
      "corruptions refused; golden digest %s\n",
      failures ? "FAIL" : "PASS", totals.cases.c_str(), prop.multiProcessorCount, totals.exceptions, totals.max_row,
      totals.onehot_values / 2, totals.exact_error, std::ldexp(1.0, -15), totals.invariant_rows, totals.coarse_ratio,
      totals.coarse_relative, totals.refused, totals.corruptions, failures ? "?" : "ok");
  return failures ? 1 : 0;
}
