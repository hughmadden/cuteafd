// CPU stand-ins for load-only native kernels, linked into the fake native DSO:
// the same shared rule the CUDA kernels run, executed sequentially.
#include <stdint.h>

#include "../../../../../native/shared/cuda/fp8_load_quant.h"

#include <cmath>
#include <algorithm>

// Mirror fp8_gemv.cu's per-row scales and 16x32 A-fragment packing.
// Allocation-only and lifetime tests retain the C fixture's no-math mode.
extern "C" int fixture_fp8_w8a16_pack(const void* source, void* packed, void* scales,
                                       int n, int k, int rule) {
  using namespace cuteafd_load_quant;
  if (!source || !packed || !scales || n <= 0 || k <= 0 || n % 16 || k % 128 || rule < 0 || rule > 2) return 1;
  const auto* w = static_cast<const uint16_t*>(source);
  auto* q = static_cast<uint8_t*>(packed);
  auto* scale = static_cast<float*>(scales);
  for (int r = 0; r < n; ++r) for (int kb = 0; kb < k / 128; ++kb) {
    float maximum = 0.0f;
    for (int c = 0; c < 128; ++c) maximum = fmax32(maximum, fabs32(bf16(w[size_t(r) * k + kb * 128 + c])));
    const float sa = amax_scale(maximum);
    const uint32_t bits = f32_bits(maximum);
    const int exponent = std::clamp(int(bits >> 23) - 135 + int((bits & 0x7FFFFFu) > 0x600000u), -126, 127);
    const float sp = maximum > 0 ? bits_f32(uint32_t(exponent + 127) << 23) : 1.0f;
    float chosen = rule == 1 ? sp : sa;
    if (rule == 2) {
      float ea[32] = {}, ep[32] = {};
      for (int lane = 0; lane < 32; ++lane) for (int j = 0; j < 4; ++j) {
        const float x = bf16(w[size_t(r) * k + kb * 128 + lane * 4 + j]);
        const float da = fsub(fmul(e4m3_value(e4m3(fdiv(x, sa))), sa), x);
        const float dp = fsub(fmul(e4m3_value(e4m3(fdiv(x, sp))), sp), x);
        ea[lane] = fadd(ea[lane], fmul(da, da));
        ep[lane] = fadd(ep[lane], fmul(dp, dp));
      }
      for (int offset = 16; offset; offset >>= 1) {
        float next_a[32], next_p[32];
        for (int lane = 0; lane < 32; ++lane) {
          next_a[lane] = fadd(ea[lane], ea[lane ^ offset]);
          next_p[lane] = fadd(ep[lane], ep[lane ^ offset]);
        }
        memcpy(ea, next_a, sizeof(ea)); memcpy(ep, next_p, sizeof(ep));
      }
      chosen = ep[0] < ea[0] ? sp : sa;
    }
    scale[size_t(r) * (k / 128) + kb] = chosen;
  }
  for (int rt = 0; rt < n / 16; ++rt) for (int kb = 0; kb < k / 128; ++kb)
    for (int kt = 0; kt < 4; ++kt) for (int lane = 0; lane < 32; ++lane) {
      const int g = lane / 4, c = lane % 4 * 2;
      const size_t at = ((size_t(rt) * (k / 32) + kb * 4 + kt) * 32 + lane) * 16;
      for (int step = 0; step < 2; ++step) for (int j = 0; j < 8; ++j) {
        const int row = rt * 16 + g + ((j % 4 >= 2) ? 8 : 0);
        const int col = kb * 128 + kt * 32 + step * 16 + c + (j >= 4 ? 8 : 0) + j % 2;
        q[at + step * 8 + j] = e4m3(fdiv(bf16(w[size_t(row) * k + col]), scale[size_t(row) * (k / 128) + kb]));
      }
    }
  return 0;
}

// Preserve the warp's accumulation/reduction order for deferred-load tests.
static int32_t markov_norms(const void* source, void* output, int32_t rows, int32_t cols, bool slack) {
  if (!source || !output || rows < 1 || cols != (slack ? 512 : 256)) return 1;
  const auto* values = static_cast<const uint16_t*>(source);
  for (int32_t r = 0; r < rows; ++r) {
    float lanes[32] = {};
    for (int32_t lane = 0; lane < 32; ++lane)
      for (int32_t c = lane; c < cols; c += 32) {
        const uint32_t bits = uint32_t(values[int64_t(r) * cols + c]) << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        lanes[lane] = std::fma(value, value, lanes[lane]);
      }
    for (int32_t offset = 16; offset; offset >>= 1) {
      float next[32];
      for (int32_t lane = 0; lane < 32; ++lane) next[lane] = lanes[lane] + lanes[lane ^ offset];
      memcpy(lanes, next, sizeof(lanes));
    }
    const float norm = std::sqrt(lanes[0]);
    static_cast<float*>(output)[r] = slack ? std::fma(norm, 1.0001f, 1e-6f) : norm;
  }
  return 0;
}

extern "C" int32_t cuteafd_glmf_dspark_markov_norms(const void* source, void* output, int32_t rows,
                                                  int32_t cols, void* stream) {
  (void)stream;
  return markov_norms(source, output, rows, cols, false);
}

extern "C" int32_t cuteafd_ml4_dspark_markov_norms(const void* source, void* output, int32_t rows,
                                                 int32_t cols, void* stream) {
  (void)stream;
  return markov_norms(source, output, rows, cols, true);
}

// A synthetic device count keeps owner construction CPU-only.
extern "C" int32_t cuteafd_sm_count() { return 1; }

extern "C" int32_t cuteafd_fp8_load_quant(const void* jobs, int32_t count, void* stream) {
  (void)stream;
  if (count < 0) return 1;
  const auto* list = static_cast<const cuteafd_load_quant::Job*>(jobs);
  for (int32_t i = 0; i < count; ++i) cuteafd_load_quant::run_job(list[i]);
  return 0;
}

extern "C" int32_t cuteafd_fp8_load_quant_abi(uint64_t* job_bytes, uint64_t* segment_bytes, uint64_t* max_cols) {
  *job_bytes = sizeof(cuteafd_load_quant::Job);
  *segment_bytes = sizeof(cuteafd_load_quant::Segment);
  *max_cols = 512 * 128;
  return 0;
}

extern "C" int32_t cuteafd_load_widen_bf16_f32(const void* src, void* dst, int64_t n, void* stream) {
  (void)stream;
  for (int64_t i = 0; i < n; ++i)
    static_cast<uint32_t*>(dst)[i] = cuteafd_load_quant::widen_bf16(static_cast<const uint16_t*>(src)[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_narrow_f32_bf16(const void* src, void* dst, int64_t n, int32_t sms, void* stream) {
  (void)stream;
  if (n < 1 || sms < 1) return 1;
  for (int64_t i = 0; i < n; ++i)
    static_cast<uint16_t*>(dst)[i] = cuteafd_load_quant::narrow_f32(static_cast<const uint32_t*>(src)[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_narrow_i64_i32(const void* src, void* dst, int64_t n,
                                               uint32_t* invalid, void* stream) {
  (void)stream;
  if (!src || !dst || !invalid || n < 1) return 1;
  for (int64_t i = 0; i < n; ++i)
    if (!cuteafd_load_quant::narrow_i64(static_cast<const int64_t*>(src)[i], static_cast<int32_t*>(dst) + i))
      *invalid = 1;
  return 0;
}

extern "C" int32_t cuteafd_bf16_f32_add_async(const uint16_t* rank0, const uint16_t* rank1,
                                              float* dst, size_t n, int32_t sms, void* stream) {
  (void)stream;
  if (n < 1 || sms < 1) return 1;
  for (size_t i = 0; i < n; ++i)
    dst[i] = cuteafd_load_quant::partial_add_bf16_f32(rank0[i], rank1[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_kvb_absorb(const void* kv_b, void* uk, void* uv, int64_t h0, int64_t heads,
                                           int64_t nope, int64_t v, int64_t lat, void* stream) {
  (void)stream;
  const auto* source = static_cast<const uint16_t*>(kv_b);
  for (int64_t i = 0; i < heads * lat * nope; ++i)
    static_cast<uint16_t*>(uk)[i] = cuteafd_load_quant::absorb_uk(source, h0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * v * lat; ++i)
    static_cast<uint16_t*>(uv)[i] = cuteafd_load_quant::absorb_uv(source, h0, nope, v, lat, i);
  return 0;
}

extern "C" int32_t cuteafd_load_kvb_split_fp8(const void* kv_b, const void* grid, void* uk, void* uk_scale, void* uv,
                                              void* uv_scale, int64_t row0, int64_t heads, int64_t nope, int64_t v,
                                              int64_t lat, void* stream) {
  (void)stream;
  namespace lq = cuteafd_load_quant;
  const auto* q = static_cast<const uint8_t*>(kv_b);
  const auto* g = static_cast<const float*>(grid);
  for (int64_t i = 0; i < heads * lat * nope; ++i) static_cast<uint8_t*>(uk)[i] = lq::absorb_uk_fp8(q, 0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * v * lat; ++i) static_cast<uint8_t*>(uv)[i] = lq::absorb_uv_fp8(q, 0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * lat * (nope / 64); ++i)
    static_cast<float*>(uk_scale)[i] = lq::absorb_uk_scale(g, row0, nope, v, lat, i);
  for (int64_t i = 0; i < heads * v * (lat / 64); ++i)
    static_cast<float*>(uv_scale)[i] = lq::absorb_uv_scale(g, row0, nope, v, lat, i);
  return 0;
}

extern "C" int32_t cuteafd_load_bf16_product_f32(const void* a, const void* b, void* dst, int64_t n, void* stream) {
  (void)stream;
  for (int64_t i = 0; i < n; ++i)
    static_cast<uint32_t*>(dst)[i] = cuteafd_load_quant::product_bf16_f32(static_cast<const uint16_t*>(a)[i],
                                                                          static_cast<const uint16_t*>(b)[i]);
  return 0;
}

extern "C" int32_t cuteafd_load_trellis_tiles(const void* src, void* dst, int64_t rows, int64_t k0, int64_t tiles,
                                              int64_t tile_bytes, int32_t side, int64_t k16, void* stream) {
  (void)stream;
  if (rows < 1 || k0 < 0 || tiles < 1 || tile_bytes < 16 || tile_bytes % 16 != 0 || (side != 0 && side != 1)
      || k0 + rows > k16 || tiles % 8 != 0) return 1;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < tiles; ++c)
      memcpy(static_cast<uint8_t*>(dst) + cuteafd_load_quant::trellis_tile(side, k0 + r, c, tiles, k16) * tile_bytes,
             static_cast<const uint8_t*>(src) + (r * tiles + c) * tile_bytes, size_t(tile_bytes));
  return 0;
}

#include "../../../../../native/shared/cuda/load_scale_gather.h"

extern "C" int32_t cuteafd_load_scale_gather(const void* grid, const void* index, void* row_major, void* k_major,
                                             int64_t rows, int64_t kb, int64_t grid_cols, int64_t col0, void* stream) {
  (void)stream;
  if (!grid || !index || rows < 1 || kb < 1 || col0 < 0 || col0 + kb > grid_cols || (!row_major && !k_major)) return 1;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t b = 0; b < kb; ++b) {
      const float v = cuteafd_scale_gather::gathered(static_cast<const float*>(grid),
          static_cast<const uint32_t*>(index), grid_cols, col0, r, b);
      if (row_major) static_cast<float*>(row_major)[r * kb + b] = v;
      if (k_major) static_cast<float*>(k_major)[b * rows + r] = v;
    }
  return 0;
}

// Allocation-only workspace contracts (one synthetic SM); no draft math runs.
static uint64_t fixture_align256(uint64_t n) { return (n + 255) / 256 * 256; }
extern "C" size_t cuteafd_fp8_linear_workspace(int32_t rows, int32_t k, int32_t n, int32_t mode) {
  if (n < 16 || k < 128 || mode < 0 || mode > 2) return 0;
  const int tiles = n / 16, blocks = k / 128;
  const int wanted = std::min(std::max((16 + tiles - 1) / tiles, 1), blocks);
  const int per_split = (blocks + wanted - 1) / wanted;
  const int splits = (blocks + per_split - 1) / per_split;
  const int cap = mode == 0 ? 64 : 128;
  return fixture_align256(2 * cap * 4) + fixture_align256((k + 255) / 256 * cap * 4)
      + fixture_align256(uint64_t(k) * cap * 2)
      + (mode == 2 ? fixture_align256(k / 128 * cap * 4) : 0)
      + (splits > 1 ? fixture_align256(uint64_t(splits) * std::min(rows, cap) * n * 4) : 0);
}
extern "C" size_t cuteafd_fp8_w8a16_workspace(int32_t rows, int32_t k, int32_t n) {
  return cuteafd_fp8_linear_workspace(rows, k, n, 0);
}
static uint64_t fixture_draft_attention(int sequences, int kv_heads, int max_keys, int dim) {
  const int partitions = std::min((max_keys + 31) / 32, (2 + kv_heads * 2 - 1) / (kv_heads * 2));
  const int scalar = (std::min(max_keys, 1040) + 127) / 128 * (dim == 64 ? 32 : 16);
  return uint64_t(sequences) * kv_heads * std::max(2 * partitions * 32, scalar) * (dim + 2) * 4;
}
extern "C" uint64_t cuteafd_glm_dflash_attention_workspace(int32_t sequences, int32_t kv_heads, int32_t max_keys) {
  return fixture_draft_attention(sequences, kv_heads, max_keys, 128);
}
extern "C" uint64_t cuteafd_glm_dflash_attention_workspace_d64(int32_t sequences, int32_t kv_heads, int32_t max_keys) {
  return fixture_draft_attention(sequences, kv_heads, max_keys, 64);
}
extern "C" uint64_t cuteafd_glm_dflash_topk_workspace(int32_t rows) { return uint64_t(rows) * 64 * 16 * 8; }

// Independent sequential implementation of the exact packed-head planes.
extern "C" int fixture_packed_head_pack(const void* source, void* const* planes,
                                         int64_t capacity, int n, int k, void* status) {
  if (!source || !planes || !status || n <= 0 || n % 16 || k <= 0 || k % 128 || capacity < 0) return 1;
  const auto* w = static_cast<const uint16_t*>(source);
  auto* coarse = static_cast<uint8_t*>(planes[0]);
  auto* fine = static_cast<uint32_t*>(planes[1]);
  auto* bases = static_cast<uint8_t*>(planes[2]);
  auto* row_ptr = static_cast<uint32_t*>(planes[3]);
  auto* columns = static_cast<uint32_t*>(planes[4]);
  auto* values = static_cast<uint16_t*>(planes[5]);
  auto* words = static_cast<uint64_t*>(status);
  memset(fine, 0, size_t(n) * k / 2);
  words[0] = words[1] = words[3] = 0; words[2] = UINT64_MAX;
  row_ptr[0] = 0;
  for (int r = 0; r < n; ++r) {
    int histogram[256] = {}, best = -1, base = 0;
    for (int c = 0; c < k; ++c) ++histogram[(w[size_t(r) * k + c] >> 7) & 255];
    for (int e = 0; e <= 240; ++e) {
      int count = 0;
      for (int i = 0; i < 16; ++i) count += histogram[e + i];
      if (count > best) { best = count; base = e; }
    }
    bases[r] = uint8_t(base);
    uint64_t row_exceptions = 0;
    for (int c = 0; c < k; ++c) {
      const uint16_t bits = w[size_t(r) * k + c];
      const int exponent = (bits >> 7) & 255;
      uint16_t coded = bits;
      if (exponent < base || exponent >= base + 16) {
        coded = uint16_t((bits & 0x8000) | (exponent < base ? base << 7 : ((base + 15) << 7) | 127));
        if (words[0] < uint64_t(capacity)) { columns[words[0]] = uint32_t(c); values[words[0]] = bits; }
        ++words[0]; ++row_exceptions;
      }
      const int within = c & 7, j = ((c >> 3) & 1) * 2 + ((r >> 3) & 1);
      const size_t lane = (r & 7) * 4 + (within >> 1);
      const size_t index = ((size_t(r >> 4) * (k / 32) + (c >> 5)) * 32 + lane);
      const int step = (c >> 4) & 1, h = within & 1;
      coarse[index * 16 + step * 8 + j * 2 + h] = uint8_t(((coded >> 8) & 128)
          | ((((coded >> 7) & 255) - base) << 3) | ((coded >> 4) & 7));
      fine[index * 2 + step] |= uint32_t(coded & 15) << (4 * j + 16 * h);
    }
    row_ptr[r + 1] = uint32_t(words[0]);
    words[3] = std::max(words[3], row_exceptions);
  }
  if (words[0] > uint64_t(capacity)) { words[1] = uint64_t(n); words[2] = 0; }
  return 0;
}
