// Standalone harness for native/shared/cuda/fp8_gemv.cu's W8A8 route
// (cuteafd_fp8_linear, mode 2). It is compiled against one revision of
// fp8_gemv.cu at a time, so two builds' outputs can be diffed: the rows <= 64
// route must be bit-identical across revisions, and the wide route is compared
// against both the other revision and an FP64 host reference of the same
// arithmetic (dequantized E4M3 weights x quantized E4M3 activations, under the
// same per-row 128-wide block scales the kernels derive).
//
//   nvcc -std=c++17 -O3 -fopenmp -arch=sm_120f -Xptxas -v \
//     -o fp8_gemv_selftest native/tests/fp8_gemv_wide_selftest.cu \
//     native/shared/cuda/fp8_gemv.cu -lcudart
//   ./fp8_gemv_selftest --rule 1 --dump /tmp/old
//
// It declares the same extern "C" entry points and mirrors the pack rules on
// the host. Compiling needs no GPU; the numerics and the timings do.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

extern "C" int32_t cuteafd_fp8_w8a16_pack(const void* w, void* packed, void* scale, int32_t n, int32_t k,
                                          int32_t rule, void* stream);
extern "C" size_t cuteafd_fp8_linear_workspace(int32_t rows, int32_t k, int32_t n, int32_t mode);
extern "C" int32_t cuteafd_fp8_linear(const void* x, const void* packed, const void* scale, void* out,
                                      int32_t out_f32, int32_t rows, int32_t k, int32_t n, int32_t mode,
                                      void* workspace, size_t workspace_bytes, void* stream);

void die(const char* what, int line) {
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, line, what);
  std::exit(2);
}

#define MUST(expr)                                                       \
  do {                                                                   \
    const int32_t status_ = (expr);                                       \
    if (status_ != 0) {                                                   \
      std::fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, status_); \
      std::exit(2);                                                       \
    }                                                                     \
  } while (0)

#define CUDA_MUST(expr)                                        \
  do {                                                         \
    const cudaError_t err_ = (expr);                           \
    if (err_ != cudaSuccess) die(cudaGetErrorString(err_), __LINE__); \
  } while (0)

namespace {

constexpr int kModeWide = 1;
constexpr int kModeW8a8 = 2;

// --- host arithmetic the kernels' quantization mirrors ----------------------

uint32_t fbits(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  return b;
}

float from_bits(uint32_t b) {
  float x;
  std::memcpy(&x, &b, 4);
  return x;
}

// RN to BF16 (the checkpoint's weight and activation dtype), ties to even.
float bf16_rn(float v) { return from_bits((fbits(v) + 0x7fffu + ((fbits(v) >> 16) & 1u)) & 0xffff0000u); }

// E4M3 decode; 0x7f/0xff (the NaN codes) are the format's only non-finite.
float e4m3_value(int code) {
  const int sign = (code >> 7) & 1, exp = (code >> 3) & 15, man = code & 7;
  if (exp == 15 && man == 7) return NAN;
  const float v = exp == 0 ? std::ldexp(man / 8.0f, -6) : std::ldexp(1.0f + man / 8.0f, exp - 7);
  return sign ? -v : v;
}

// Round to nearest even, saturating at the format's largest finite value:
// what __nv_cvt_float2_to_fp8x2(..., __NV_SATFINITE, __NV_E4M3) does.
uint8_t e4m3_rn(float x) {
  if (std::isnan(x)) return 0x7f;
  const uint32_t sign = fbits(x) & 0x80000000u;
  const float a = std::fabs(x);
  if (a > 448.0f) return uint8_t((sign ? 0x80 : 0) | 0x7e);
  int code;
  if (a < std::ldexp(1.0f, -6)) {
    code = int(std::lrintf(a * 512.0f));  // subnormal quantum 2^-9
  } else {
    int e;
    const float frac = frexpf(a, &e) * 2.0f - 1.0f;
    int m = int(std::lrintf(frac * 8.0f)), ef = e - 1 + 7;
    if (m == 8) {
      m = 0;
      ++ef;
    }
    if (ef > 15) return uint8_t((sign ? 0x80 : 0) | 0x7e);
    code = (ef << 3) | m;
  }
  return uint8_t((sign ? 0x80 : 0) | code);
}

// fp8_gemv.cu's pow2_scale, bit for bit.
float pow2_scale(float amax) {
  const uint32_t b = fbits(amax);
  int x = int(b >> 23) - 135 + int((b & 0x7FFFFFu) > 0x600000u);
  x = x < -126 ? -126 : (x > 127 ? 127 : x);
  return from_bits(uint32_t(x + 127) << 23);
}

// e4m3_rn against a nearest-search decoder over the whole format: every finite
// value, plus random values across the exponent range, positive and negative.
void check_e4m3() {
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  auto next = [&seed]() {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(seed >> 32);
  };
  std::vector<float> probes;
  for (int i = 0; i < 30000; ++i) {
    const float u = float(next()) / float(0xffffffffu);
    probes.push_back(std::ldexp(1.0f + u, int(next() % 20) - 12));
  }
  for (int c = 0; c < 256; ++c) probes.push_back(e4m3_value(c));
  for (const float a : probes) {
    for (const int sgn : {1, -1}) {
      const float x = sgn * a;
      if (!std::isfinite(x)) continue;
      double best = 1e30;
      float nearest = 0.0f;
      for (int c = 0; c < 256; ++c) {
        const float v = e4m3_value(c);
        if (std::isnan(v) || (v != 0.0f && std::signbit(v) != std::signbit(x))) continue;
        const double d = std::fabs(double(v) - double(x));
        if (d < best - 1e-30) {
          best = d;
          nearest = v;
        } else if (d <= best + 1e-30 && !(c & 1)) {  // tie: even mantissa wins
          nearest = v;
        }
      }
      const float want = std::fabs(x) > 448.0f ? float(sgn) * 448.0f : nearest;
      const float got = e4m3_value(e4m3_rn(x));
      if (got != want) {
        std::fprintf(stderr, "e4m3_rn(%g) = %g, nearest-search %g\n", x, got, want);
        std::exit(2);
      }
    }
  }
}

// --- device buffers ---------------------------------------------------------

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;

  void alloc(size_t n) {
    if (p) CUDA_MUST(cudaFree(p));
    bytes = n;
    CUDA_MUST(cudaMalloc(&p, n ? n : 1));
  }
  void upload(const void* src, size_t n) { CUDA_MUST(cudaMemcpy(p, src, n, cudaMemcpyHostToDevice)); }
  void download(void* dst, size_t n) const { CUDA_MUST(cudaMemcpy(dst, p, n, cudaMemcpyDeviceToHost)); }
  ~Buf() {
    if (p) cudaFree(p);
  }
};

// --- shapes, data and the reference ----------------------------------------

struct Case {
  int k, n;
};

const Case kCases[] = {{20480, 4096}, {4096, 2048}};  // GLM 5.3 Flash DFlash2 fc and one KV slice
const int kRows[] = {1, 8, 16, 64, 65, 128, 129, 1024, 2048};

float gauss(uint64_t& s) {
  float t = 0.0f;
  for (int i = 0; i < 12; ++i) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    t += float(s >> 40) / float(1 << 24);
  }
  return t - 6.0f;
}

// The run's quantized tensors, in the kernels' own scheme: weights per output
// row and 128-wide K block under `rule`, activations per row and 128-wide K
// block at amax / 448. E4M3 values as floats (exact in double, and in the
// 128-wide block sums).
struct HostQuant {
  std::vector<float> qw, sw, qx, sx;

  static std::vector<float> widen(const std::vector<uint16_t>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
      const uint32_t b = uint32_t(v[i]) << 16;
      std::memcpy(&f[i], &b, 4);
    }
    return f;
  }

  void weights(const std::vector<uint16_t>& w, int n, int k, int rule) {
    const int kbs = k / 128;
    const std::vector<float> wf = widen(w);
    qw.assign(size_t(n) * k, 0.0f);
    sw.assign(size_t(n) * kbs, 0.0f);
    for (int col = 0; col < n; ++col) {
      for (int b = 0; b < kbs; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 128; ++i) amax = std::fmax(amax, std::fabs(wf[size_t(col) * k + b * 128 + i]));
        const float s = amax > 0.0f ? (rule == 1 ? pow2_scale(amax) : amax / 448.0f) : 1.0f;
        sw[size_t(col) * kbs + b] = s;
        for (int i = 0; i < 128; ++i) {
          qw[size_t(col) * k + b * 128 + i] = e4m3_value(e4m3_rn(wf[size_t(col) * k + b * 128 + i] / s));
        }
      }
    }
  }

  void activations(const std::vector<uint16_t>& x, int rows, int k) {
    const int kbs = k / 128;
    const std::vector<float> xf = widen(x);
    qx.assign(size_t(rows) * k, 0.0f);
    sx.assign(size_t(rows) * kbs, 0.0f);
    for (int m = 0; m < rows; ++m) {
      for (int b = 0; b < kbs; ++b) {
        float amax = 0.0f;
        for (int i = 0; i < 128; ++i) amax = std::fmax(amax, std::fabs(xf[size_t(m) * k + b * 128 + i]));
        const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
        sx[size_t(m) * kbs + b] = s;
        for (int i = 0; i < 128; ++i) {
          qx[size_t(m) * k + b * 128 + i] = e4m3_value(e4m3_rn(xf[size_t(m) * k + b * 128 + i] / s));
        }
      }
    }
  }
};

// y [rows, n] = sum over K blocks of (weight block scale x activation block
// scale) x (the block's dequantized E4M3 products), in double throughout: the
// arithmetic the W8A8 tensors compute, without their tiling or FP32 adds.
void reference(const HostQuant& q, int rows, int n, int k, std::vector<double>& y) {
  const int kbs = k / 128;
  const int RB = 64, CB = 64;
  y.assign(size_t(rows) * n, 0.0);
#pragma omp parallel for schedule(dynamic)
  for (int m0 = 0; m0 < rows; m0 += RB) {
    const int mr = std::min(RB, rows - m0);
    for (int c0 = 0; c0 < n; c0 += CB) {
      const int cr = std::min(CB, n - c0);
      double acc[RB][CB];
      for (int a = 0; a < mr; ++a)
        for (int c = 0; c < cr; ++c) acc[a][c] = 0.0;
      for (int b = 0; b < kbs; ++b) {
        for (int a = 0; a < mr; ++a) {
          const float* xr = q.qx.data() + size_t(m0 + a) * k + b * 128;
          const double sxv = q.sx[size_t(m0 + a) * kbs + b];
          for (int c = 0; c < cr; ++c) {
            const float* wr = q.qw.data() + size_t(c0 + c) * k + b * 128;
            double s = 0.0;
            for (int i = 0; i < 128; ++i) s += double(wr[i]) * double(xr[i]);
            acc[a][c] += s * double(q.sw[size_t(c0 + c) * kbs + b]) * sxv;
          }
        }
      }
      for (int a = 0; a < mr; ++a)
        for (int c = 0; c < cr; ++c) y[size_t(m0 + a) * n + c0 + c] = acc[a][c];
    }
  }
}

struct Metrics {
  double abs_err = 0.0, ref_max = 0.0, rel_at_max = 0.0, rel_guard = 0.0;
};

Metrics compare(const std::vector<float>& got, const std::vector<double>& ref) {
  Metrics m;
  size_t at = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::fabs(ref[i]) > m.ref_max) {
      m.ref_max = std::fabs(ref[i]);
      at = i;
    }
  }
  const double guard = 0.01 * m.ref_max;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(double(got[i]) - ref[i]);
    m.abs_err = std::fmax(m.abs_err, d);
    if (std::fabs(ref[i]) >= guard) m.rel_guard = std::fmax(m.rel_guard, d / std::fabs(ref[i]));
  }
  m.rel_at_max = std::fabs(double(got[at]) - ref[at]) / m.ref_max;
  return m;
}

// --- timing and kernel counts ----------------------------------------------

double median_ms(int iters, const std::function<void(cudaStream_t)>& call) {
  std::vector<float> ms;
  cudaStream_t s;
  CUDA_MUST(cudaStreamCreate(&s));
  for (int i = 0; i < iters; ++i) {
    cudaEvent_t a, b;
    CUDA_MUST(cudaEventCreate(&a));
    CUDA_MUST(cudaEventCreate(&b));
    CUDA_MUST(cudaEventRecord(a, s));
    call(s);
    CUDA_MUST(cudaEventRecord(b, s));
    CUDA_MUST(cudaEventSynchronize(b));
    float dt = 0.0f;
    CUDA_MUST(cudaEventElapsedTime(&dt, a, b));
    CUDA_MUST(cudaEventDestroy(a));
    CUDA_MUST(cudaEventDestroy(b));
    ms.push_back(dt);
  }
  CUDA_MUST(cudaStreamDestroy(s));
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

// One node per kernel launch: stream capture records the launch chain without
// running it.
int kernel_count(const std::function<void(cudaStream_t)>& call) {
  cudaStream_t s;
  CUDA_MUST(cudaStreamCreate(&s));
  CUDA_MUST(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  call(s);
  cudaGraph_t g = nullptr;
  CUDA_MUST(cudaStreamEndCapture(s, &g));
  size_t nodes = 0;
  CUDA_MUST(cudaGraphGetNodes(g, nullptr, &nodes));
  CUDA_MUST(cudaGraphDestroy(g));
  CUDA_MUST(cudaStreamDestroy(s));
  return int(nodes);
}

// --- the wide path's contract, GPU against GPU ------------------------------

uint32_t lcg(uint64_t& s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return uint32_t(s >> 33);
}

std::vector<uint16_t> bf16_rows(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = uint16_t(fbits(bf16_rn(v[i])) >> 16);
  return o;
}

// {-1, 0, 1} / 64 per row and 128-wide K block, the block's first entry 1 / 64:
// at either scale rule every weight code is exact (448 and 0, or 256 and 0 at
// the pow2 scale).
std::vector<uint16_t> ternary_weights(uint64_t& s, int n, int k) {
  std::vector<float> v(size_t(n) * k);
  for (int r = 0; r < n; ++r) {
    for (int b = 0; b < k / 128; ++b) {
      for (int i = 0; i < 128; ++i) {
        const int t = i == 0 ? 1 : int(lcg(s) % 3) - 1;
        v[size_t(r) * k + b * 128 + i] = float(t) / 64.0f;
      }
    }
  }
  return bf16_rows(v);
}

// {-1, 0, 1} * 448 * 2^j per row and 128-wide K block, the block's first entry
// 448 * 2^j: each block's amax is its scale, so every quantized activation and
// every product is exact and both routes sum exactly the same values, whatever
// their accumulation order.
std::vector<uint16_t> ternary_x(uint64_t& s, int rows, int k) {
  std::vector<float> v(size_t(rows) * k);
  for (int m = 0; m < rows; ++m) {
    for (int b = 0; b < k / 128; ++b) {
      const float sc = std::ldexp(448.0f, int(lcg(s) % 10) - 12);
      for (int i = 0; i < 128; ++i) {
        const int t = i == 0 ? 1 : int(lcg(s) % 3) - 1;
        v[size_t(m) * k + b * 128 + i] = float(t) * sc;
      }
    }
  }
  return bf16_rows(v);
}

// E4M3 values times 2^j per row and 128-wide K block, the block's first entry
// 448 * 2^j: activations the W8A8 quantization holds exactly, so the routes'
// only difference is their accumulation order.
std::vector<uint16_t> exact_x(uint64_t& s, int rows, int k) {
  std::vector<float> v(size_t(rows) * k);
  for (int m = 0; m < rows; ++m) {
    for (int b = 0; b < k / 128; ++b) {
      const float e = std::ldexp(1.0f, int(lcg(s) % 10) - 12);
      for (int i = 0; i < 128; ++i) {
        int code = int(lcg(s) & 0xff);
        if ((code & 0x7f) == 0x7f) code ^= 1;  // the format's NaN codes
        const float val = e4m3_value(code & 0x7f) * e;
        v[size_t(m) * k + b * 128 + i] = (code & 0x80) ? -val : val;
      }
      v[size_t(m) * k + b * 128] = (lcg(s) & 1) ? 448.0f * e : -448.0f * e;
    }
  }
  return bf16_rows(v);
}

std::vector<float> widen(const std::vector<uint16_t>& h) {
  std::vector<float> f(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    const uint32_t b = uint32_t(h[i]) << 16;
    std::memcpy(&f[i], &b, 4);
  }
  return f;
}

// Where a wide call's numerics differ between its own rows, on generic
// operands: mode 2 against mode 0 (wide W8A16), reported per row region. A
// route that picks the activation format per pass leaves a trailing pass of at
// most kW8a16Rows rows on the f16 path, so those rows alone carry no activation
// quantization; a route that keeps every row on E4M3 has one numerics
// throughout. Informational: both are sound, and the brief allows the wide
// path's bits to differ from the old route's.
void tail_probe(int rule, const Case& c, void* scratch, size_t scratch_bytes) {
  const int k = c.k, n = c.n;
  const int rows_all[] = {65, 129, 136, 200, 1024, 2048};
  uint64_t ws = 0x2468ace0ull ^ (uint64_t(k) << 17) ^ uint64_t(n);
  std::vector<uint16_t> w(size_t(n) * k);
  for (size_t i = 0; i < w.size(); ++i) w[i] = uint16_t(fbits(bf16_rn(gauss(ws) * 0.05f)) >> 16);
  Buf w_dev, packed, scale;
  w_dev.alloc(w.size() * 2);
  w_dev.upload(w.data(), w.size() * 2);
  packed.alloc(size_t(n) * k);
  scale.alloc(size_t(n) * (k / 128) * 4);
  if (cuteafd_fp8_w8a16_pack(w_dev.p, packed.p, scale.p, n, k, rule, nullptr) != cudaSuccess) return;
  CUDA_MUST(cudaDeviceSynchronize());
  std::printf("\ntail k=%d n=%d (rule %d): |mode2 - mode0| per row region, generic operands\n", k, n, rule);
  for (const int rows : rows_all) {
    uint64_t xs = 0x31415926ull ^ (uint64_t(k) << 19) ^ uint64_t(rows);
    std::vector<float> xf(size_t(rows) * k);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = gauss(xs);
    const std::vector<uint16_t> xh = bf16_rows(xf);
    Buf x_dev, o0, o2;
    x_dev.alloc(xh.size() * 2);
    x_dev.upload(xh.data(), xh.size() * 2);
    o0.alloc(size_t(rows) * n * 2);
    o2.alloc(size_t(rows) * n * 2);
    for (int mode = 0; mode < 2; ++mode) {
      void* out = mode ? o2.p : o0.p;
      if (cuteafd_fp8_linear(x_dev.p, packed.p, scale.p, out, 0, rows, k, n, mode ? kModeW8a8 : kModeWide, scratch,
                             scratch_bytes, nullptr) != cudaSuccess) {
        std::printf("tail k=%5d n=%4d M=%4d  call failed\n", k, n, rows);
        return;
      }
      CUDA_MUST(cudaDeviceSynchronize());
    }
    std::vector<uint16_t> h0(size_t(rows) * n), h2(h0.size());
    o0.download(h0.data(), h0.size() * 2);
    o2.download(h2.data(), h2.size() * 2);
    const std::vector<float> r0 = widen(h0), r2 = widen(h2);
    const int wide_rows = rows < 128 ? rows : 128;
    double wide_gap = 0.0, tail_gap = 0.0;
    size_t wide_bits = 0, tail_bits = 0;
    for (int m = 0; m < rows; ++m) {
      for (int j = 0; j < n; ++j) {
        const size_t i = size_t(m) * n + j;
        const double d = std::fabs(double(r2[i]) - double(r0[i]));
        if (m < wide_rows) {
          wide_gap = std::fmax(wide_gap, d);
          wide_bits += (h2[i] != h0[i]);
        } else {
          tail_gap = std::fmax(tail_gap, d);
          tail_bits += (h2[i] != h0[i]);
        }
      }
    }
    std::printf("tail k=%5d n=%4d M=%4d  rows 0..%d  max %.3e  %zu/%zu bits differ   rows %d..%d  max %.3e  "
                "%zu/%zu bits differ\n",
                k, n, rows, wide_rows - 1, wide_gap, wide_bits, size_t(wide_rows) * n, wide_rows, rows - 1, tail_gap,
                tail_bits, size_t(rows - wide_rows) * n);
  }
}

// The properties the draft path holds to, at every row count from one draft
// block to the whole TAP_ROWS context, on operand sets whose arithmetic is
// exact -- compared GPU against GPU, since the FP64 reference above shares the
// kernels' scheme but not their accumulation:
//   mode 1 (wide W8A16) is bit-identical to mode 0 everywhere;
//   mode 2 (W8A8) is bit-identical to mode 0 on ternary operands everywhere;
//   mode 2's BF16 output is its FP32 output rounded, everywhere;
//   mode 2 stays inside the accumulation-rounding bound of mode 0 on
//   E4M3-exact activations.
int contract_check(int rule, const Case& c, void* scratch, size_t scratch_bytes) {
  const int k = c.k, n = c.n;
  const int rows_all[] = {8, 64, 65, 96, 127, 128, 129, 136, 200, 1024, 2048};
  uint64_t s = 0x5a5a5a5aull ^ (uint64_t(k) << 21) ^ uint64_t(n);
  const std::vector<uint16_t> w3 = ternary_weights(s, n, k);
  Buf w3_dev, packed3, scale3, xt, xe, o0, o1, o2, o2f;
  w3_dev.alloc(w3.size() * 2);
  w3_dev.upload(w3.data(), w3.size() * 2);
  packed3.alloc(size_t(n) * k);
  scale3.alloc(size_t(n) * (k / 128) * 4);
  MUST(cuteafd_fp8_w8a16_pack(w3_dev.p, packed3.p, scale3.p, n, k, rule, nullptr));
  CUDA_MUST(cudaDeviceSynchronize());
  int bad = 0;
  std::printf("\ncontract k=%d n=%d (rule %d)\n", k, n, rule);
  for (const int rows : rows_all) {
    const std::vector<uint16_t> xt_h = ternary_x(s, rows, k);
    const std::vector<uint16_t> xe_h = exact_x(s, rows, k);
    xt.alloc(xt_h.size() * 2);
    xt.upload(xt_h.data(), xt_h.size() * 2);
    xe.alloc(xe_h.size() * 2);
    xe.upload(xe_h.data(), xe_h.size() * 2);
    o0.alloc(size_t(rows) * n * 2);
    o1.alloc(size_t(rows) * n * 2);
    o2.alloc(size_t(rows) * n * 2);
    o2f.alloc(size_t(rows) * n * 4);
    int status = 0;
    auto run = [&](int mode, const void* xp, void* out, int out_f32) {
      status = cuteafd_fp8_linear(xp, packed3.p, scale3.p, out, out_f32, rows, k, n, mode, scratch, scratch_bytes,
                                  nullptr);
      CUDA_MUST(cudaDeviceSynchronize());
    };
    for (int set = 0; set < 2; ++set) {
      const void* xp = set ? xe.p : xt.p;
      run(0, xp, o0.p, 0);
      run(1, xp, o1.p, 0);
      run(2, xp, o2.p, 0);
      run(2, xp, o2f.p, 1);
      std::vector<uint16_t> h0(size_t(rows) * n), h1(h0.size()), h2(h0.size());
      std::vector<float> f2(h0.size());
      o0.download(h0.data(), h0.size() * 2);
      o1.download(h1.data(), h1.size() * 2);
      o2.download(h2.data(), h2.size() * 2);
      o2f.download(f2.data(), f2.size() * 4);
      const std::vector<float> r0 = widen(h0), r2 = widen(h2);
      size_t wide_diff = 0, w8a8_diff = 0, unrounded = 0;
      double gap = 0.0, ref_max = 0.0;
      for (size_t i = 0; i < h0.size(); ++i) {
        wide_diff += (h1[i] != h0[i]);
        w8a8_diff += (h2[i] != h0[i]);
        unrounded += (h2[i] != uint16_t(fbits(bf16_rn(f2[i])) >> 16));
        gap = std::max(gap, std::fabs(double(r2[i]) - double(r0[i])));
        ref_max = std::max(ref_max, std::fabs(double(r0[i])));
      }
      const double bound = ref_max * 2e-3;
      const bool ok = status == cudaSuccess && wide_diff == 0 && unrounded == 0 &&
                      (set ? gap <= bound : w8a8_diff == 0);
      bad += !ok;
      if (set) {
        std::printf("contract k=%5d n=%4d M=%4d e4m3-exact  mode1==mode0 %s  bf16==round(fp32) %s  "
                    "|mode2-mode0| %.3e <= %.3e %s\n",
                    k, n, rows, wide_diff ? "NO" : "yes", unrounded ? "NO" : "yes", gap, bound,
                    gap <= bound ? "yes" : "NO");
      } else {
        std::printf("contract k=%5d n=%4d M=%4d ternary     mode1==mode0 %s  bf16==round(fp32) %s  "
                    "mode2==mode0 %s\n",
                    k, n, rows, wide_diff ? "NO" : "yes", unrounded ? "NO" : "yes", w8a8_diff ? "NO" : "yes");
      }
    }
  }
  tail_probe(rule, c, scratch, scratch_bytes);
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  int rule = 1;
  bool contract_only = false;
  std::string dump;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--rule" && i + 1 < argc) rule = std::atoi(argv[++i]);
    else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    else if (a == "--contract") contract_only = true;
    else {
      std::fprintf(stderr, "usage: %s [--rule 0|1] [--dump DIR] [--contract]\n", argv[0]);
      return 2;
    }
  }
  check_e4m3();
  std::printf("e4m3_rn verified against a nearest-search decoder\n\n");

  int dev = 0;
  CUDA_MUST(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_MUST(cudaGetDeviceProperties(&prop, dev));
  std::printf("device %s, %d SMs, rule %d\n\n", prop.name, prop.multiProcessorCount, rule);

  // The daemon admits one scratch for every shape at TAP_ROWS (2048) rows.
  size_t scratch_bytes = 0;
  for (const Case& c : kCases) {
    scratch_bytes = std::max(scratch_bytes, cuteafd_fp8_linear_workspace(2048, c.k, c.n, kModeW8a8));
  }
  Buf scratch;
  scratch.alloc(scratch_bytes);
  CUDA_MUST(cudaMemset(scratch.p, 0, scratch_bytes));

  int bad = 0;
  for (const Case& c : kCases) {
    const int k = c.k, n = c.n;
    if (contract_only) {
      bad += contract_check(rule, c, scratch.p, scratch.bytes);
      continue;
    }
    const int kbs = k / 128;
    // Weights: BF16 [n, k], one pack call, one packed copy for every row count.
    uint64_t w_seed = 0x1234567ull ^ (uint64_t(k) << 17) ^ uint64_t(n);
    std::vector<uint16_t> w(size_t(n) * k);
    for (size_t i = 0; i < w.size(); ++i) {
      w[i] = uint16_t(fbits(bf16_rn(gauss(w_seed) * 0.05f)) >> 16);  // ~N(0, 1/400): outputs O(1) at k = 20480
    }
    Buf w_dev, packed, scale, x_dev, out_dev;
    w_dev.alloc(w.size() * 2);
    w_dev.upload(w.data(), w.size() * 2);
    packed.alloc(size_t(n) * k);
    scale.alloc(size_t(n) * kbs * 4);
    MUST(cuteafd_fp8_w8a16_pack(w_dev.p, packed.p, scale.p, n, k, rule, nullptr));
    CUDA_MUST(cudaDeviceSynchronize());

    HostQuant q;
    q.weights(w, n, k, rule);

    for (const int rows : kRows) {
      uint64_t x_seed = 0x89abcdefull ^ (uint64_t(k) << 19) ^ uint64_t(rows);
      std::vector<uint16_t> x(size_t(rows) * k);
      for (size_t i = 0; i < x.size(); ++i) x[i] = uint16_t(fbits(bf16_rn(gauss(x_seed))) >> 16);
      x_dev.alloc(x.size() * 2);
      x_dev.upload(x.data(), x.size() * 2);
      out_dev.alloc(size_t(rows) * n * 2);
      CUDA_MUST(cudaMemset(out_dev.p, 0x7f, size_t(rows) * n * 2));  // NaN patterns stay visible
      const int32_t status = cuteafd_fp8_linear(x_dev.p, packed.p, scale.p, out_dev.p, 0, rows, k, n, kModeW8a8,
                                                scratch.p, scratch.bytes, nullptr);
      CUDA_MUST(cudaDeviceSynchronize());
      if (status != cudaSuccess) {
        std::fprintf(stderr, "k=%d n=%d rows=%d: status %d\n", k, n, rows, status);
        ++bad;
        continue;
      }
      std::vector<uint16_t> out(size_t(rows) * n);
      out_dev.download(out.data(), out.size() * 2);
      std::vector<float> got(out.size());
      for (size_t i = 0; i < out.size(); ++i) {
        const uint32_t b = uint32_t(out[i]) << 16;
        std::memcpy(&got[i], &b, 4);
      }
      q.activations(x, rows, k);
      std::vector<double> ref;
      reference(q, rows, n, k, ref);
      const Metrics m = compare(got, ref);
      std::printf("k=%5d n=%4d M=%4d  ref_max=%9.4f  max_abs=%.3e  rel_at_max=%.3e  max_rel=%9.3e\n", k, n, rows,
                  m.ref_max, m.abs_err, m.rel_at_max, m.rel_guard);
      if (!dump.empty()) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/out_k%d_n%d_m%d.f32", dump.c_str(), k, n, rows);
        FILE* f = std::fopen(path, "wb");
        if (!f) {
          std::fprintf(stderr, "cannot write %s\n", path);
          return 2;
        }
        std::fwrite(got.data(), 4, got.size(), f);
        std::fclose(f);
      }
    }

    bad += contract_check(rule, c, scratch.p, scratch.bytes);

    // Timing and kernel counts at the widest row count.
    const int rows = 2048;
    uint64_t x_seed = 0x89abcdefull ^ (uint64_t(k) << 19) ^ uint64_t(rows);
    std::vector<uint16_t> x(size_t(rows) * k);
    for (size_t i = 0; i < x.size(); ++i) x[i] = uint16_t(fbits(bf16_rn(gauss(x_seed))) >> 16);
    x_dev.alloc(x.size() * 2);
    x_dev.upload(x.data(), x.size() * 2);
    out_dev.alloc(size_t(rows) * n * 2);
    const auto call = [&](cudaStream_t s) {
      cuteafd_fp8_linear(x_dev.p, packed.p, scale.p, out_dev.p, 0, rows, k, n, kModeW8a8, scratch.p, scratch.bytes, s);
    };
    call(nullptr);
    CUDA_MUST(cudaDeviceSynchronize());
    const double one = median_ms(51, call);
    std::printf("\nbench k=%5d n=%4d M=%d: %.0f us, %d kernels\n", k, n, rows, one * 1000.0, kernel_count(call));

    if (k == kCases[0].k) {
      // 1 x fc + 5 x kv: the projection kernels of one 2048-row context wave.
      const Case kvk = kCases[1];
      uint64_t s2 = 0x1234567ull ^ (uint64_t(kvk.k) << 17) ^ uint64_t(kvk.n);
      std::vector<uint16_t> w2(size_t(kvk.n) * kvk.k);
      for (size_t i = 0; i < w2.size(); ++i) w2[i] = uint16_t(fbits(bf16_rn(gauss(s2) * 0.05f)) >> 16);
      Buf w2_dev, packed_kv, scale_kv, x_kv, out_kv;
      w2_dev.alloc(w2.size() * 2);
      w2_dev.upload(w2.data(), w2.size() * 2);
      packed_kv.alloc(size_t(kvk.n) * kvk.k);
      scale_kv.alloc(size_t(kvk.n) * (kvk.k / 128) * 4);
      MUST(cuteafd_fp8_w8a16_pack(w2_dev.p, packed_kv.p, scale_kv.p, kvk.n, kvk.k, rule, nullptr));
      CUDA_MUST(cudaDeviceSynchronize());
      uint64_t x2s = 0x89abcdefull ^ (uint64_t(kvk.k) << 19) ^ uint64_t(rows);
      std::vector<uint16_t> x2(size_t(rows) * kvk.k);
      for (size_t i = 0; i < x2.size(); ++i) x2[i] = uint16_t(fbits(bf16_rn(gauss(x2s))) >> 16);
      x_kv.alloc(x2.size() * 2);
      x_kv.upload(x2.data(), x2.size() * 2);
      out_kv.alloc(size_t(rows) * kvk.n * 2);
      const auto wave = [&](cudaStream_t s) {
        cuteafd_fp8_linear(x_dev.p, packed.p, scale.p, out_dev.p, 0, rows, k, n, kModeW8a8, scratch.p, scratch.bytes, s);
        for (int i = 0; i < 5; ++i) {
          cuteafd_fp8_linear(x_kv.p, packed_kv.p, scale_kv.p, out_kv.p, 0, rows, kvk.k, kvk.n, kModeW8a8, scratch.p,
                             scratch.bytes, s);
        }
      };
      wave(nullptr);
      CUDA_MUST(cudaDeviceSynchronize());
      const double total = median_ms(51, wave);
      std::printf("bench 1x fc + 5x kv M=2048: %.0f us, %d kernels\n", total * 1000.0, kernel_count(wave));
    }
  }
  std::printf("\n%s\n", bad ? "FAILED" : "ok");
  return bad ? 1 : 0;
}
