/* V4.1 index top-k: the radix-select path against the original sort-and-merge
 * path and an exact CPU oracle.
 *
 * Includes the kernel source, so one process runs both paths through the same
 * host entry (select<K, Limit>(..., path), path 0 = merge, 1 = radix). Every
 * call of every chain must give bit-identical selected positions on both paths
 * and the oracle; carries must hold the same key multiset (their order is the
 * path's own). Chains cover reset 0 and 1, a reset in the middle and carries
 * handed from one path to the other.
 *
 * Inputs: random FP32, BF16-rounded scores (score ties), few distinct and all
 * equal scores (position tie-breaks), signed zeros, NaN payloads, +-inf,
 * denormals, FLT_MAX, invalid positions (Limit, 2^32 + small, UINT64_MAX),
 * shuffled positions, short rows, fewer valid keys than K, exact-K ties at
 * the boundary, and duplicate keys (outside the unique-position contract; both
 * paths still agree with the multiset oracle).
 *
 *   v41_index_topk_selftest            correctness suite
 *   v41_index_topk_selftest --time     plus per-call times of both paths
 */
#include "../families/deepseek_v41/cuda/v41_index_topk.cu"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace topk_test {

#define CHECK_CUDA(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

constexpr int kMaxCandidates = 16384;

uint64_t host_encode(uint32_t bits, uint64_t pos, uint64_t limit) {
  float score;
  std::memcpy(&score, &bits, 4);
  if (pos >= limit || std::isnan(score) || score == -INFINITY) return 0;
  const uint32_t b = score == 0.f ? 0u : bits;
  const uint32_t ordered = (b & 0x80000000u) ? ~b : (b ^ 0x80000000u);
  return (uint64_t(ordered) << 32) | uint32_t(~uint32_t(pos));
}

uint32_t fbits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
uint32_t bf16_round(uint32_t b) {  // round to nearest even BF16, kept as FP32 bits
  if ((b & 0x7f800000u) == 0x7f800000u) return b & 0xffff0000u;
  return (b + 0x7fffu + ((b >> 16) & 1u)) & 0xffff0000u;
}

struct Oracle {
  int K;
  uint64_t limit;
  std::vector<std::vector<uint64_t>> carry;
  Oracle(int k, uint64_t l, int rows) : K(k), limit(l), carry(rows, std::vector<uint64_t>(k, 0)) {}
  void call(const std::vector<uint32_t>& bits, const std::vector<uint64_t>& pos, int rows, int count, bool reset,
            std::vector<int32_t>& out) {
    out.assign(size_t(rows) * K, -1);
    std::vector<uint64_t> keys;
    for (int r = 0; r < rows; ++r) {
      keys.clear();
      for (int c = 0; c < count; ++c) keys.push_back(host_encode(bits[size_t(r) * count + c], pos[size_t(r) * count + c], limit));
      if (!reset) keys.insert(keys.end(), carry[r].begin(), carry[r].end());
      while (keys.size() < size_t(K)) keys.push_back(0);
      std::sort(keys.begin(), keys.end(), std::greater<uint64_t>());
      carry[r].assign(keys.begin(), keys.begin() + K);
      std::vector<uint32_t> p;
      for (uint64_t k : carry[r]) if (k) p.push_back(~uint32_t(k));
      std::sort(p.begin(), p.end());
      for (size_t i = 0; i < p.size(); ++i) out[size_t(r) * K + i] = int32_t(p[i]);
    }
  }
};

template <int K, uint64_t Limit>
struct Device {
  int rows;
  float* scores = nullptr;
  uint64_t* positions = nullptr;
  uint64_t* carry = nullptr;
  void* scratch = nullptr;
  uint64_t scratch_bytes = 0;
  int32_t* output = nullptr;
  explicit Device(int r) : rows(r) {
    scratch_bytes = uint64_t(rows) * ((kMaxCandidates + 2 * K - 1) / (2 * K)) * K * 16;
    CHECK_CUDA(cudaMalloc(&scores, size_t(rows) * kMaxCandidates * 4));
    CHECK_CUDA(cudaMalloc(&positions, size_t(rows) * kMaxCandidates * 8));
    CHECK_CUDA(cudaMalloc(&carry, size_t(rows) * K * 8));
    CHECK_CUDA(cudaMalloc(&scratch, scratch_bytes));
    CHECK_CUDA(cudaMalloc(&output, size_t(rows) * K * 4));
    CHECK_CUDA(cudaMemset(carry, 0xa5, size_t(rows) * K * 8));  // reset=1 must never read it
  }
  ~Device() { cudaFree(scores); cudaFree(positions); cudaFree(carry); cudaFree(scratch); cudaFree(output); }
  void upload(const std::vector<uint32_t>& bits, const std::vector<uint64_t>& pos) {
    CHECK_CUDA(cudaMemcpy(scores, bits.data(), bits.size() * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(positions, pos.data(), pos.size() * 8, cudaMemcpyHostToDevice));
  }
  int32_t call(int path, int count, int reset, cudaStream_t stream = nullptr) {
    return select<K, Limit>(scores, positions, carry, scratch, scratch_bytes, output, rows, count, reset, stream, path);
  }
  void read(std::vector<int32_t>& out, std::vector<uint64_t>& c) {
    out.resize(size_t(rows) * K);
    c.resize(size_t(rows) * K);
    CHECK_CUDA(cudaMemcpy(out.data(), output, out.size() * 4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(c.data(), carry, c.size() * 8, cudaMemcpyDeviceToHost));
  }
};

// One chunk of candidate scores and positions for every row.
struct Chunk {
  std::vector<uint32_t> bits;
  std::vector<uint64_t> pos;
  int count;
  int reset;
};

enum Dist { kUniform, kBf16, kFew, kEqual, kSpecial, kInvalid, kSparse, kNegative, kBoundary, kDuplicate };
const char* dist_name(Dist d) {
  static const char* n[] = {"uniform", "bf16", "few", "equal", "special", "invalid", "sparse", "negative",
                            "boundary", "duplicate"};
  return n[d];
}

uint32_t special_bits(std::mt19937_64& g) {
  static const uint32_t s[] = {0x00000000u, 0x80000000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0xffffffffu,
                               0x7fffffffu, 0x7f800000u, 0xff800000u, 0x00000001u, 0x80000001u, 0x007fffffu,
                               0x807fffffu, 0x7f7fffffu, 0xff7fffffu, 0x3f800000u, 0xbf800000u, 0x3f800000u};
  return s[g() % (sizeof(s) / sizeof(s[0]))];
}

// Builds `chunks` chunks of `count` candidates per row. Positions are unique
// per row across the chain except for kDuplicate.
std::vector<Chunk> make_chain(Dist d, int rows, int count, int chunks, uint64_t limit, int K, uint64_t seed,
                              bool shuffled, int reset_at) {
  std::mt19937_64 g(seed);
  std::normal_distribution<float> normal(0.f, 1.f);
  std::vector<Chunk> out(chunks);
  const uint64_t span = uint64_t(count) * chunks;
  std::vector<std::vector<uint64_t>> order(rows);
  for (int r = 0; r < rows; ++r) {
    auto& o = order[r];
    if (shuffled && span <= limit) {
      // Unique positions anywhere below the (power-of-two) limit, scattered:
      // an odd multiplier is a bijection modulo the limit.
      const uint64_t a = g() | 1, b = g();
      o.resize(span);
      for (uint64_t i = 0; i < span; ++i) o[i] = (a * i + b) & (limit - 1);
    } else {
      o.resize(span);
      const uint64_t base = span < limit ? g() % (limit - span + 1) : 0;
      for (uint64_t i = 0; i < span; ++i) o[i] = base + i;  // tile-like ascending, may pass the limit
    }
  }
  const float few[] = {0.5f, 1.25f, -3.f};
  for (int k = 0; k < chunks; ++k) {
    Chunk& c = out[k];
    c.count = count;
    c.reset = (k == 0 || k == reset_at) ? 1 : 0;
    c.bits.resize(size_t(rows) * count);
    c.pos.resize(size_t(rows) * count);
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < count; ++i) {
        const size_t at = size_t(r) * count + i;
        uint64_t p = order[r][size_t(k) * count + i];
        uint32_t b = fbits(normal(g));
        switch (d) {
          case kUniform: break;
          case kBf16: b = bf16_round(fbits(std::fabs(normal(g)) * 0.25f)); break;
          case kFew: b = fbits(few[g() % 3]); break;
          case kEqual: b = fbits(0.75f); break;
          case kSpecial: if (g() % 3) b = special_bits(g); break;
          case kInvalid: {
            const uint64_t bad[] = {limit, limit + 1, UINT64_MAX, (uint64_t(1) << 32) + (p & 0xffff),
                                    UINT64_MAX - 7, (uint64_t(1) << 63) | p};
            if (g() % 4 == 0) p = bad[g() % 6];
            b = bf16_round(b);
            break;
          }
          case kSparse: if (g() % 64) b = (g() & 1) ? 0xff800000u : 0x7fc00001u; break;
          case kNegative: b = fbits(-std::fabs(normal(g)) - 1e-3f); break;
          case kBoundary: {
            // Exactly K-1, K or K+1 keys per row share the top score.
            const int tied = K - 1 + (r % 3);
            b = (size_t(k) * count + i) < size_t(tied) ? fbits(8.f) : bf16_round(fbits(normal(g)));
            break;
          }
          case kDuplicate:
            b = bf16_round(fbits(normal(g)));
            if (i % 5 == 1) { p = c.pos[at - 1]; if (g() & 1) b = c.bits[at - 1]; }
            break;
        }
        c.bits[at] = b;
        c.pos[at] = p;
      }
    }
  }
  return out;
}

int failures = 0;
long long checked_calls = 0;

void compare_carry(const char* what, int rows, int K, std::vector<uint64_t> a, const std::vector<std::vector<uint64_t>>& oracle) {
  for (int r = 0; r < rows; ++r) {
    std::vector<uint64_t> x(a.begin() + size_t(r) * K, a.begin() + size_t(r + 1) * K), y = oracle[r];
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    if (x != y) {
      if (failures < 20) std::fprintf(stderr, "FAIL %s: row %d carry multiset differs\n", what, r);
      ++failures;
      return;
    }
  }
}

bool same_output(const char* what, const std::vector<int32_t>& a, const std::vector<int32_t>& b, int K) {
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      if (failures < 20)
        std::fprintf(stderr, "FAIL %s: row %zu rank %zu: %d vs %d\n", what, i / K, i % K, a[i], b[i]);
      ++failures;
      return false;
    }
  }
  return true;
}

template <int K, uint64_t Limit>
void run_chain(Dist d, int rows, int count, int chunks, bool shuffled, int reset_at, uint64_t seed, int mode) {
  // mode 0: merge and radix side by side; 1: alternate paths, carry handed over.
  char what[160];
  std::snprintf(what, sizeof(what), "K=%d %s rows=%d count=%d chunks=%d%s reset_at=%d mode=%d", K, dist_name(d),
                rows, count, chunks, shuffled ? " shuffled" : "", reset_at, mode);
  const auto chain = make_chain(d, rows, count, chunks, Limit, K, seed, shuffled, reset_at);
  Oracle oracle(K, Limit, rows);
  Device<K, Limit> merge(rows), radix(rows);
  std::vector<int32_t> expect, out_m, out_r;
  std::vector<uint64_t> carry_m, carry_r;
  for (int k = 0; k < chunks; ++k) {
    const Chunk& c = chain[k];
    oracle.call(c.bits, c.pos, rows, c.count, c.reset, expect);
    if (mode == 0) {
      merge.upload(c.bits, c.pos);
      radix.upload(c.bits, c.pos);
      CHECK_CUDA(cudaError_t(merge.call(0, c.count, c.reset)));
      CHECK_CUDA(cudaError_t(radix.call(1, c.count, c.reset)));
      CHECK_CUDA(cudaDeviceSynchronize());
      merge.read(out_m, carry_m);
      radix.read(out_r, carry_r);
      same_output(what, out_m, expect, K);
      same_output(what, out_r, expect, K);
      compare_carry(what, rows, K, carry_m, oracle.carry);
      compare_carry(what, rows, K, carry_r, oracle.carry);
    } else {
      radix.upload(c.bits, c.pos);
      CHECK_CUDA(cudaError_t(radix.call(k % 2, c.count, c.reset)));
      CHECK_CUDA(cudaDeviceSynchronize());
      radix.read(out_r, carry_r);
      same_output(what, out_r, expect, K);
      compare_carry(what, rows, K, carry_r, oracle.carry);
    }
    ++checked_calls;
  }
}

template <int K, uint64_t Limit>
void suite(uint64_t seed) {
  const int counts512[] = {1, 7, 511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2048, 4095, 4096, 4097, 9000, 16383, 16384};
  const int counts2048[] = {1, 100, 2047, 2048, 2049, 4096, 6000, 16384};
  const int* counts = K == 512 ? counts512 : counts2048;
  const int ncounts = K == 512 ? int(sizeof(counts512) / 4) : int(sizeof(counts2048) / 4);
  const Dist dists[] = {kUniform, kBf16, kFew, kEqual, kSpecial, kInvalid, kSparse, kNegative, kBoundary, kDuplicate};
  // Every distribution at every short-row count (one row, two chunks).
  for (Dist d : dists)
    for (int i = 0; i < ncounts; ++i) run_chain<K, Limit>(d, 1, counts[i], 2, false, -1, seed++, 0);
  // Row counts, chains with a mid-chain reset, shuffled positions, path hand-over.
  const int rows[] = {1, 6, 48, 130};
  for (Dist d : dists) {
    for (int r : rows) {
      const int full = K == 512 ? 16384 : 2048;
      run_chain<K, Limit>(d, r, full, K == 512 ? 4 : 8, false, -1, seed++, 0);
      run_chain<K, Limit>(d, r, K == 512 ? 4500 : 600, 5, d != kInvalid, 3, seed++, 0);
      run_chain<K, Limit>(d, r, full, 4, false, -1, seed++, 1);
    }
  }
}

// ---- timing ----
float time_graph(cudaGraphExec_t exec, cudaStream_t s, int replays) {
  cudaEvent_t a, b;
  CHECK_CUDA(cudaEventCreate(&a));
  CHECK_CUDA(cudaEventCreate(&b));
  CHECK_CUDA(cudaGraphLaunch(exec, s));
  CHECK_CUDA(cudaStreamSynchronize(s));
  float best = 1e30f;
  for (int rep = 0; rep < 3; ++rep) {
    CHECK_CUDA(cudaEventRecord(a, s));
    for (int i = 0; i < replays; ++i) CHECK_CUDA(cudaGraphLaunch(exec, s));
    CHECK_CUDA(cudaEventRecord(b, s));
    CHECK_CUDA(cudaEventSynchronize(b));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return best;
}

template <int K, uint64_t Limit>
void time_case(Dist d, int rows, int count, int calls, const char* label) {
  // `calls` chained calls (reset on the first) in one graph, like one
  // selection's tiles; reports microseconds per selection and per call.
  // Each call has its own candidates (unique positions across the chain).
  const auto chain = make_chain(d, rows, count, calls, Limit, K, 99, false, -1);
  std::vector<Device<K, Limit>*> dev;
  for (int c = 0; c < calls; ++c) {
    dev.push_back(new Device<K, Limit>(rows));
    dev[c]->upload(chain[c].bits, chain[c].pos);
  }
  cudaStream_t s;
  CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  float us[2];
  for (int path = 0; path < 2; ++path) {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    const int selections = 8;
    CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int sel = 0; sel < selections; ++sel)
      for (int c = 0; c < calls; ++c)
        CHECK_CUDA(cudaError_t(select<K, Limit>(dev[c]->scores, dev[c]->positions, dev[0]->carry, dev[0]->scratch,
            dev[0]->scratch_bytes, dev[0]->output, rows, count, c == 0, s, path)));
    CHECK_CUDA(cudaStreamEndCapture(s, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    const int replays = 20;
    us[path] = time_graph(exec, s, replays) * 1000.f / (replays * selections);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }
  std::printf("time K=%d %-8s %-22s rows=%3d count=%5d calls=%d  merge %8.1f us  radix %7.1f us  (%5.1fx)  per call %6.1f -> %5.1f\n",
              K, dist_name(d), label, rows, count, calls, us[0], us[1], us[0] / us[1], us[0] / calls, us[1] / calls);
  std::fflush(stdout);
  cudaStreamDestroy(s);
  for (auto* p : dev) delete p;
}

void timing() {
  const int rows[] = {1, 6, 48};
  for (Dist d : {kBf16, kUniform, kEqual}) {
    for (int r : rows) {
      time_case<512, 1048576>(d, r, 2048, 1, "4K ratio2 (1 call)");
      time_case<512, 1048576>(d, r, 4096, 1, "4K ratio1 (1 call)");
      time_case<512, 1048576>(d, r, 16384, 1, "32K r2 / follower (1)");
      time_case<512, 1048576>(d, r, 16384, 2, "32K source20 (2)");
      time_case<512, 1048576>(d, r, 16384, 4, "128K ratio2 (4)");
      time_case<512, 1048576>(d, r, 16384, 8, "128K source20 (8)");
      time_case<2048, 131072>(d, r, 512, 1, "4K blocks (1)");
      time_case<2048, 131072>(d, r, 2048, 2, "32K blocks (2)");
      time_case<2048, 131072>(d, r, 2048, 8, "128K blocks (8)");
    }
  }
}

#ifdef CUTEAFD_V41_TOPK_PHASES
template <int K, uint64_t Limit>
void phase_case(Dist d, int rows, int count, int reset, const char* label) {
  // Block 0's clock64 at each phase of one radix call, after a warm-up call.
  const auto chain = make_chain(d, rows, count, 2, Limit, K, 77, false, -1);
  Device<K, Limit> dev(rows);
  dev.upload(chain[0].bits, chain[0].pos);
  CHECK_CUDA(cudaError_t(dev.call(1, count, 1)));
  if (!reset) dev.upload(chain[1].bits, chain[1].pos);
  unsigned long long zero[32] = {};
  float best = 1e30f;
  unsigned long long ph[32];
  for (int rep = 0; rep < 5; ++rep) {
    CHECK_CUDA(cudaMemcpyToSymbol(topk_phase, zero, sizeof(zero)));
    cudaEvent_t a, b;
    CHECK_CUDA(cudaEventCreate(&a));
    CHECK_CUDA(cudaEventCreate(&b));
    CHECK_CUDA(cudaEventRecord(a));
    CHECK_CUDA(cudaError_t(dev.call(1, count, reset)));
    CHECK_CUDA(cudaEventRecord(b));
    CHECK_CUDA(cudaEventSynchronize(b));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    if (ms < best) {
      best = ms;
      CHECK_CUDA(cudaMemcpyFromSymbol(ph, topk_phase, sizeof(ph)));
    }
  }
  int rate_khz = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&rate_khz, cudaDevAttrClockRate, 0));
  const double ghz = 2.7;  // nominal loaded SM clock; clock64 counts cycles
  auto us = [&](int i, int j) { return ph[i] && ph[j] ? double(ph[j] - ph[i]) / (ghz * 1e3) : 0.0; };
  int last = 1;
  std::printf("phases K=%d %-8s %-22s rows=%3d count=%5d reset=%d event %.1f us | load %.2f", K, dist_name(d), label,
              rows, count, reset, best * 1000.f, us(0, 1));
  for (int p = 0; p < 10; ++p)
    if (ph[2 + p]) { std::printf(" pass%d %.2f", p, us(last, 2 + p)); last = 2 + p; }
  std::printf(" | scan %.2f select %.2f sort %.2f write %.2f | in-kernel %.2f us, passes %llu\n", us(last, 12),
              us(12, 13), us(13, 14), us(14, 15), us(0, 15), ph[20]);
  std::fflush(stdout);
}

void phases() {
  for (Dist d : {kBf16, kUniform, kEqual, kFew}) {
    for (int r : {1, 48}) {
      phase_case<512, 1048576>(d, r, 2048, 1, "4K ratio2");
      phase_case<512, 1048576>(d, r, 16384, 1, "16K first tile");
      phase_case<512, 1048576>(d, r, 16384, 0, "16K carried tile");
      phase_case<2048, 131072>(d, r, 512, 1, "blocks 512 first");
      phase_case<2048, 131072>(d, r, 2048, 0, "blocks 2048 carried");
    }
  }
}
#endif

}  // namespace topk_test

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("SKIP: no CUDA device\n");
    return 77;
  }
  topk_test::suite<512, 1048576>(1000);
  topk_test::suite<2048, 131072>(2000);
  std::printf("%s: %lld chained calls, %d failures\n", topk_test::failures ? "FAILED" : "passed",
              topk_test::checked_calls, topk_test::failures);
  if (topk_test::failures) return 1;
  if (argc > 1 && std::string(argv[1]) == "--time") topk_test::timing();
#ifdef CUTEAFD_V41_TOPK_PHASES
  if (argc > 1 && std::string(argv[1]) == "--phases") topk_test::phases();
#endif
  return 0;
}
