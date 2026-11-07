// KX7: BF16 matrices held as an exact code over their 16-bit patterns
// (cuteafd_core::kx7, format v1.1), for GLM 5.3 Flash's FP8-born KDA
// projections. Decode and verify steps (1..64 rows) run the fused GEMV
// (kx7_gemv.cuh): each lane decodes its stream straight into MMA fragments, bit
// for bit the GEMV's BF16 twin over the source weights and invariant to the
// step's width. Prefill and wider steps decode every stream into a BF16
// workspace and run the BF16 cuBLAS GEMM over it, bit for bit that GEMM over the
// source weights.
//
// The pack (built and verified on the host by cuteafd_core::kx7) reaches this
// file as a cuteafd_kx7_pack_t (cuteafd_kx7.h): its header scalars and device
// buffers. Format v1.1 (CUTEAFD_KX7_FORMAT_V11): lane16 with K split S = 4, so
// stream s * 4 + t of row r (id r * 4 S + s * 4 + t) holds, for each 64-wide
// chunk c of K slice s, inputs s * K / S + 64 c + 16 t + i, i = 0..15. The 32
// streams of each 8-row block and slice (lane l = 4 (row % 8) + t) are cut
// into 16-byte quads, quad q of lane l at the group base + (q * 32 + l) * 16
// up to the group's shortest stream, the ragged tails then in lane order;
// offsets give each stream's quad 0 and lengths its bytes. The code is two-
// class: an 11-bit primary table resolves every frequent pattern and ESC; a
// rare pattern is ESC plus a 12-bit rank into the u16 table after the
// primary one (cuteafd_core::kx7::Kx7Pack::device_tables).
//
// The decode is the reference device decoder (kx7_decode.cuh) that the KX7
// GEMV runs, walked as the GEMV's warps walk it: one warp per (8-row block,
// K slice) group, its 32 lanes that group's 32 streams; each lane writes its
// 16 symbols of every 64-wide chunk into row 8 * block + lane / 4 at columns
// slice * K / S + 64 c + 16 (lane % 4) + i, as two 16-byte stores.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "cuteafd_kx7.h"
#include "cuteafd_native.h"
#include "kx7_decode.cuh"
#include "kx7_gemv.cuh"

namespace {

// Groups (warps) per CTA: the decode tables are copied to shared memory once per CTA.
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kInterleaveRows = 8;
constexpr uint32_t kQuadStride = 512;
// v1.1's two-class tables: 2^11 primary entries, then 2^12 u16 rare patterns two to a word.
constexpr uint32_t kV11TableWords = (1u << KX7_PBITS) + (1u << 12) / 2;
// Shared memory a kernel may take without opting in.
constexpr size_t kDefaultSmem = 48 * 1024;
static_assert(kV11TableWords * 4 + size_t(kWarps) * KX7_RING_BYTES <= kDefaultSmem,
              "the two-class decode stays within the default shared memory");

bool huffman(const cuteafd_kx7_pack_t* p) { return p->format == CUTEAFD_KX7_FORMAT_V11_HUFFMAN; }

// Raises `kernel`'s dynamic shared memory limit to `bytes` the first time a launch needs more than
// the default (a Huffman pack's larger tables); `granted` is the launch site's own record.
// cudaFuncSetAttribute is not a stream operation, so this is safe inside a capture.
template <typename Kernel>
cudaError_t allow_smem(Kernel kernel, size_t bytes, size_t& granted) {
  if (bytes <= kDefaultSmem || bytes <= granted) return cudaSuccess;
  int device = 0, limit = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (e != cudaSuccess) return e;
  if (bytes > size_t(limit)) return cudaErrorInvalidValue;
  e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes));
  if (e == cudaSuccess) granted = bytes;
  return e;
}
// The fused GEMV takes decode and verify steps up to this many rows (four 16-row activation
// tiles); wider steps and prefill decode once and run the BF16 GEMM.
constexpr int32_t kGemvRows = 64;
// The GEMV's K warps are the pack's K slices.
constexpr int kGemvSlices = 4;

bool aligned(const void* p, uintptr_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; }

size_t align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

// The fields of a v1.1 pack this code reads (the header grows only by appending).
constexpr size_t kPackV11Bytes = offsetof(cuteafd_kx7_pack_t, streams) + sizeof(void*);

bool valid_v11(const cuteafd_kx7_pack_t* p) {
  if (!p || p->struct_bytes < kPackV11Bytes ||
      (p->format != CUTEAFD_KX7_FORMAT_V11 && p->format != CUTEAFD_KX7_FORMAT_V11_HUFFMAN)) {
    return false;
  }
  const bool shape = p->n > 0 && p->n % kInterleaveRows == 0 && p->k > 0 && p->slices >= 1 && p->slices <= 256 &&
                     p->k % (64 * p->slices) == 0 && p->streams_per_row == 4 * p->slices &&
                     p->symbols_per_stream == p->k / (4 * p->slices) &&
                     uint64_t(p->n) * uint64_t(p->streams_per_row) < (uint64_t(1) << 31);
  const bool tables = huffman(p) ? p->table_words > (1u << KX7_PBITS) && p->table_words <= CUTEAFD_KX7_MAX_HUFFMAN_WORDS
                                  : p->table_words == kV11TableWords;
  return shape && p->quad_stride == kQuadStride && p->ragged <= 1 && tables &&
         p->tables && p->offsets && p->lengths && p->streams && aligned(p->tables, 4) && aligned(p->offsets, 4) &&
         aligned(p->lengths, 2) && aligned(p->streams, 16);
}

template <bool TWOCLASS>
__global__ void __launch_bounds__(kThreads)
    decode_kernel(const uint32_t* __restrict__ lut_g, int lut_words, const uint32_t* __restrict__ offsets,
                  const uint16_t* __restrict__ lengths, const uint8_t* __restrict__ streams, uint16_t* __restrict__ out,
                  int k, int slices, int streams_per_row, uint32_t stride, int ragged, uint32_t esc_shift,
                  uint32_t esc_mask, int groups) {
  extern __shared__ __align__(16) uint32_t smem[];
  uint32_t* lut = smem;
  uint32_t* ring = smem + lut_words;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  kx7_load_tables(lut, lut_g, lut_words, tid, kThreads);
  __syncthreads();
  const int group = blockIdx.x * kWarps + warp;
  if (group >= groups) return;  // a whole warp: the ragged start's warp reductions stay full
  const int block = group / slices, slice = group % slices;
  const int row = kInterleaveRows * block + lane / 4, t = lane % 4;
  const int width = k / slices;
  const size_t sidx = size_t(row) * streams_per_row + slice * 4 + t;
  Kx7Stream s[1];
  uint32_t* const ring_lane[1] = {ring + warp * (KX7_RING_BYTES / 4) + lane};
  kx7_init(s[0], streams, offsets[sidx], lengths[sidx], stride, ragged != 0, ring_lane[0]);
  uint16_t* dst = out + size_t(row) * k + size_t(slice) * width + 16 * t;
  for (int c = 0; c < width / 64; ++c) {
    uint32_t w[1][8];
    kx7_decode16_multi<1, TWOCLASS>(s, ring_lane, lut, esc_shift, esc_mask, stride, w);
    uint4* at = reinterpret_cast<uint4*>(dst + 64 * c);
    at[0] = make_uint4(w[0][0], w[0][1], w[0][2], w[0][3]);
    at[1] = make_uint4(w[0][4], w[0][5], w[0][6], w[0][7]);
  }
}

// The GEMV's CTAs own 32 weight rows in every configuration.
constexpr int32_t kGemvCols = 32;

// Whether a call takes the fused GEMV: a decode/verify width, the GEMV's K split, whole CTAs.
bool gemv_applies(int32_t rows, int32_t n, int32_t slices) {
  return rows >= 1 && rows <= kGemvRows && slices == kGemvSlices && n % kGemvCols == 0;
}

// A grouped launch's packs as the kernel reads them; `col0[i]` is pack i's first output column.
kx7gemv::Kx7GemvGroup group_of(const cuteafd_kx7_pack_t* const* packs, int32_t count, const int32_t* col0) {
  kx7gemv::Kx7GemvGroup g{};
  g.packs = count;
  g.streams_per_row = packs[0]->streams_per_row;
  g.stride = packs[0]->quad_stride;
  g.ragged = int(packs[0]->ragged);
  for (int i = 0; i < count; ++i) {
    const cuteafd_kx7_pack_t* p = packs[i];
    g.pack[i] = kx7gemv::Kx7GemvPack{static_cast<const uint8_t*>(p->streams), static_cast<const uint32_t*>(p->offsets),
                                     static_cast<const uint16_t*>(p->lengths), static_cast<const uint32_t*>(p->tables),
                                     int(p->table_words), p->esc_shift, p->esc_mask, p->n, col0[i]};
  }
  return g;
}

// One GEMV configuration <KS, RG, G, MT> launched over a group of packs of one shape and code.
template <int RG, int G, int MT, bool TWOCLASS>
int32_t launch_gemv(const cuteafd_kx7_pack_t* const* packs, int32_t count, const int32_t* col0, const void* x,
                    void* out, int32_t ldo, int32_t rows, cudaStream_t s) {
  using Geo = kx7gemv::Geometry<kGemvSlices, RG, G, MT>;
  static_assert(Geo::kx7_smem(int(kV11TableWords)) <= int(kDefaultSmem), "two-class GEMVs need no opt-in");
  static_assert(Geo::COLS == kGemvCols, "every GEMV configuration owns kGemvCols weight rows per CTA");
  int ctas = 0;
  uint32_t lut_words = 0;
  for (int i = 0; i < count; ++i) {
    if (packs[i]->n % Geo::COLS != 0) return cudaErrorInvalidValue;
    ctas += packs[i]->n / Geo::COLS;
    lut_words = packs[i]->table_words > lut_words ? packs[i]->table_words : lut_words;
  }
  const size_t smem = size_t(Geo::kx7_smem(int(lut_words)));
  static size_t granted = kDefaultSmem;
  const cudaError_t allowed = allow_smem(kx7gemv::kx7_gemv_kernel<kGemvSlices, RG, G, MT, TWOCLASS>, smem, granted);
  if (allowed != cudaSuccess) return allowed;
  kx7gemv::kx7_gemv_kernel<kGemvSlices, RG, G, MT, TWOCLASS><<<ctas, Geo::THREADS, smem, s>>>(
      static_cast<const uint16_t*>(x), rows, packs[0]->k, ldo, group_of(packs, count, col0),
      static_cast<__nv_bfloat16*>(out));
  return cudaGetLastError();
}

// The GEMV for `rows` <= kGemvRows: one, two or four activation tiles, the configurations measured
// best on an RTX 5090 for GLM 5.3 Flash's [8192, 4096] q/k/v and [4096, 8192] o_proj. Every
// configuration has K split 4 and the same per-element K order and fold, so a row's bits do not
// depend on the row count or on which packs share the launch.
template <bool TWOCLASS>
int32_t gemv_code(const cuteafd_kx7_pack_t* const* packs, int32_t count, const int32_t* col0, const void* x,
                  void* out, int32_t ldo, int32_t rows, cudaStream_t s) {
  if (rows <= 16) return launch_gemv<4, 1, 1, TWOCLASS>(packs, count, col0, x, out, ldo, rows, s);
  if (rows <= 32) return launch_gemv<4, 1, 2, TWOCLASS>(packs, count, col0, x, out, ldo, rows, s);
  if (packs[0]->n >= 8192) return launch_gemv<4, 1, 4, TWOCLASS>(packs, count, col0, x, out, ldo, rows, s);
  return launch_gemv<2, 2, 4, TWOCLASS>(packs, count, col0, x, out, ldo, rows, s);
}

int32_t gemv(const cuteafd_kx7_pack_t* const* packs, int32_t count, const int32_t* col0, const void* x, void* out,
             int32_t ldo, int32_t rows, cudaStream_t s) {
  return huffman(packs[0]) ? gemv_code<false>(packs, count, col0, x, out, ldo, rows, s)
                           : gemv_code<true>(packs, count, col0, x, out, ldo, rows, s);
}

// Whether `count` packs can share one GEMV launch writing [rows, ldo] rows: valid v1.1 packs of one
// shape, K split, layout and code, each a GEMV width, their columns inside a row of `ldo`.
bool groupable(const cuteafd_kx7_pack_t* const* packs, int32_t count, const int32_t* col0, int32_t ldo,
               int32_t rows) {
  if (!packs || !col0 || count < 1 || count > kx7gemv::kMaxGroup) return false;
  for (int i = 0; i < count; ++i) {
    const cuteafd_kx7_pack_t* p = packs[i];
    if (!valid_v11(p) || !gemv_applies(rows, p->n, p->slices) || col0[i] < 0 || col0[i] % 2 != 0 ||
        int64_t(col0[i]) + p->n > ldo || p->k != packs[0]->k || p->format != packs[0]->format ||
        p->streams_per_row != packs[0]->streams_per_row || p->ragged != packs[0]->ragged) {
      return false;
    }
  }
  return true;
}

}  // namespace

// Bytes cuteafd_kx7_linear needs as `workspace` for `rows` activation rows over an [n, k]
// pack (0 for an invalid shape): the decoded BF16 matrix. The fused kernels will need less; a
// caller sizes its workspace from this call.
extern "C" size_t cuteafd_kx7_workspace(int32_t rows, int32_t n, int32_t k) {
  if (rows < 1 || n < 1 || k < 1 || k % 64 != 0) return 0;
  return align256(size_t(n) * size_t(k) * 2);
}

// Decodes `pack` into BF16 `out_bf16` [n, k] row-major (16-byte aligned). Asynchronous on
// `stream`; no allocation, no host synchronization (the header is read here, at launch), so it
// can be captured into a CUDA graph. Returns a cudaError_t. Format specific: a later format is
// another kernel selected here by `pack->format`.
extern "C" int32_t cuteafd_kx7_decode(const cuteafd_kx7_pack_t* pack, void* out_bf16, void* stream) {
  if (!valid_v11(pack) || !out_bf16 || !aligned(out_bf16, 16)) return cudaErrorInvalidValue;
  const int groups = (pack->n / kInterleaveRows) * pack->slices;
  const int blocks = (groups + kWarps - 1) / kWarps;
  const size_t smem = size_t(pack->table_words) * 4 + size_t(kWarps) * KX7_RING_BYTES;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  auto launch = [&](auto kernel, size_t& granted) -> cudaError_t {
    const cudaError_t allowed = allow_smem(kernel, smem, granted);
    if (allowed != cudaSuccess) return allowed;
    kernel<<<blocks, kThreads, smem, s>>>(
        static_cast<const uint32_t*>(pack->tables), int(pack->table_words), static_cast<const uint32_t*>(pack->offsets),
        static_cast<const uint16_t*>(pack->lengths), static_cast<const uint8_t*>(pack->streams),
        static_cast<uint16_t*>(out_bf16), pack->k, pack->slices, pack->streams_per_row, pack->quad_stride,
        int(pack->ragged), pack->esc_shift, pack->esc_mask, groups);
    return cudaSuccess;
  };
  static size_t granted_two_class = kDefaultSmem, granted_huffman = kDefaultSmem;
  const cudaError_t launched = huffman(pack) ? launch(decode_kernel<false>, granted_huffman)
                                             : launch(decode_kernel<true>, granted_two_class);
  if (launched != cudaSuccess) return launched;
  return cudaGetLastError();
}

// out [rows, n] BF16 = x [rows, k] BF16 @ W^T for W = `pack` [n, k] (FP32 accumulation): the
// fused GEMV at 1..64 rows, else decode + the BF16 cuBLAS GEMM. `workspace` holds `workspace_bytes` >= cuteafd_kx7_workspace and
// is the caller's for the call's duration (one is enough for every pack launched in order on
// one stream). Asynchronous on `stream` and graph capturable. Returns 0, a cudaError_t, or for
// a failed GEMM the native status.
extern "C" int32_t cuteafd_kx7_linear(const void* x, const cuteafd_kx7_pack_t* pack, void* out, int32_t rows,
                                      void* workspace, size_t workspace_bytes, void* stream) {
  if (!x || !out || !workspace || rows < 1 || !valid_v11(pack)) return cudaErrorInvalidValue;
  const int32_t n = pack->n, k = pack->k;
  if (workspace_bytes < cuteafd_kx7_workspace(rows, n, k)) return cudaErrorInvalidValue;
  // Decode and verify steps: the fused GEMV, each lane decoding its stream straight into MMA
  // fragments (no workspace pass). Wider steps and prefill: decode once, then the BF16 GEMM.
  if (gemv_applies(rows, pack->n, pack->slices)) {
    const int32_t col0 = 0;
    return gemv(&pack, 1, &col0, x, out, n, rows, static_cast<cudaStream_t>(stream));
  }
  const int32_t decoded = cuteafd_kx7_decode(pack, workspace, stream);
  if (decoded != 0) return decoded;
  const cuteafd_status_t gemm = cuteafd_cuda_linear_bf16_cublas_async(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(workspace), nullptr, static_cast<uint16_t*>(out),
      size_t(rows), size_t(k), size_t(n), stream);
  return gemm == CUTEAFD_STATUS_OK ? 0 : int32_t(gemm);
}

// Test support: the GEMV's BF16 twin over row-major BF16 weights w [n, k] (the same structure and
// configurations as the KX7 GEMV for `rows` <= 64), so a test can show the KX7 GEMV equals it bit for
// bit. Returns a cudaError_t; cudaErrorInvalidValue outside the GEMV's row range.
extern "C" int32_t cuteafd_kx7_reference_gemv(const void* x, const void* w, void* out, int32_t rows, int32_t n,
                                               int32_t k, void* stream) {
  if (!x || !w || !out || !gemv_applies(rows, n, kGemvSlices) || k % (64 * kGemvSlices) != 0) {
    return cudaErrorInvalidValue;
  }
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  auto run = [&](auto geo, auto kernel) -> int32_t {
    using Geo = decltype(geo);
    if (n % Geo::COLS != 0) return cudaErrorInvalidValue;
    kernel<<<n / Geo::COLS, Geo::THREADS, Geo::bf16_smem(), s>>>(static_cast<const uint16_t*>(x), rows, n, k,
                                                                 static_cast<const uint16_t*>(w), n,
                                                                 static_cast<__nv_bfloat16*>(out));
    return cudaGetLastError();
  };
  using kx7gemv::Geometry;
  using kx7gemv::bf16_gemv_kernel;
  if (rows <= 16) return run(Geometry<kGemvSlices, 4, 1, 1>{}, bf16_gemv_kernel<kGemvSlices, 4, 1, 1>);
  if (rows <= 32) return run(Geometry<kGemvSlices, 4, 1, 2>{}, bf16_gemv_kernel<kGemvSlices, 4, 1, 2>);
  if (n >= 8192) return run(Geometry<kGemvSlices, 4, 1, 4>{}, bf16_gemv_kernel<kGemvSlices, 4, 1, 4>);
  return run(Geometry<kGemvSlices, 2, 2, 4>{}, bf16_gemv_kernel<kGemvSlices, 2, 2, 4>);
}

// Decode and verify steps of several packs of one shape and code at once (GLM Flash's q, k and v):
// out[r * ldo + col0[i] + c] = (x [rows, k] @ W_i^T)[r, c] for pack i, rows 1..64, as one GEMV
// launch whose packs' CTAs run side by side. Each output element is bit for bit what
// cuteafd_kx7_linear gives for its pack alone. Returns cudaErrorInvalidValue when the packs cannot
// share a launch (the caller then runs cuteafd_kx7_linear_ld per pack); otherwise a cudaError_t.
extern "C" int32_t cuteafd_kx7_gemv_group(const void* x, const cuteafd_kx7_pack_t* const* packs, int32_t count,
                                          const int32_t* col0, void* out, int32_t ldo, int32_t rows, void* stream) {
  if (!x || !out || !groupable(packs, count, col0, ldo, rows)) return cudaErrorInvalidValue;
  return gemv(packs, count, col0, x, out, ldo, rows, static_cast<cudaStream_t>(stream));
}

// cuteafd_kx7_linear with the output rows `ldo` elements apart (ldo >= n): the GEMV's rows only
// (1..64); a wider call returns cudaErrorInvalidValue.
extern "C" int32_t cuteafd_kx7_linear_ld(const void* x, const cuteafd_kx7_pack_t* pack, void* out, int32_t ldo,
                                         int32_t rows, void* stream) {
  const int32_t col0 = 0;
  if (!x || !out || !groupable(&pack, 1, &col0, ldo, rows)) return cudaErrorInvalidValue;
  return gemv(&pack, 1, &col0, x, out, ldo, rows, static_cast<cudaStream_t>(stream));
}

// A BF16 matrix's decode-row GEMV beside KX7 ones: out[r * ldo + c] = (x [rows, k] @ w^T)[r, c] for
// row-major BF16 w [n, k], rows 1..64, FP32 accumulation, every row's bits independent of the row
// count (K split 8 over one 8-column group per CTA: a short matrix still spreads over n / 8 CTAs).
// Returns a cudaError_t (cudaErrorInvalidValue outside n % 8 == 0, k % 512 == 0, 1..64 rows).
extern "C" int32_t cuteafd_bf16_gemv_ld(const void* x, const void* w, void* out, int32_t ldo, int32_t rows,
                                        int32_t n, int32_t k, void* stream) {
  constexpr int kSlices = 8;
  if (!x || !w || !out || rows < 1 || rows > kGemvRows || n < 8 || n % 8 != 0 || k % (64 * kSlices) != 0 ||
      ldo < n || !aligned(x, 16) || !aligned(w, 16)) {
    return cudaErrorInvalidValue;
  }
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  auto run = [&](auto geo, auto kernel) -> int32_t {
    using Geo = decltype(geo);
    kernel<<<n / Geo::COLS, Geo::THREADS, Geo::bf16_smem(), s>>>(static_cast<const uint16_t*>(x), rows, n, k,
                                                                 static_cast<const uint16_t*>(w), ldo,
                                                                 static_cast<__nv_bfloat16*>(out));
    return cudaGetLastError();
  };
  using kx7gemv::Geometry;
  using kx7gemv::bf16_gemv_kernel;
  if (rows <= 16) return run(Geometry<kSlices, 1, 1, 1>{}, bf16_gemv_kernel<kSlices, 1, 1, 1>);
  if (rows <= 32) return run(Geometry<kSlices, 1, 1, 2>{}, bf16_gemv_kernel<kSlices, 1, 1, 2>);
  return run(Geometry<kSlices, 1, 1, 4>{}, bf16_gemv_kernel<kSlices, 1, 1, 4>);
}
