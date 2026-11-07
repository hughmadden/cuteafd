// KX7: BF16 matrices held as an exact code over their 16-bit patterns
// (cuteafd_core::kx7, format v1.1), for GLM 5.3 Flash's FP8-born KDA
// projections. This file is v0, correct first: a decode of every stream into a
// BF16 workspace and the BF16 cuBLAS GEMM the BF16 path runs over it, so a KX7
// linear is bit for bit that GEMM over the source weights.
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

namespace {

// Groups (warps) per CTA: the decode tables are copied to shared memory once per CTA.
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kInterleaveRows = 8;
constexpr uint32_t kQuadStride = 512;
// v1.1's two-class tables: 2^11 primary entries, then 2^12 u16 rare patterns two to a word.
constexpr uint32_t kV11TableWords = (1u << KX7_PBITS) + (1u << 12) / 2;
constexpr size_t kSmemBytes = kV11TableWords * 4 + size_t(kWarps) * KX7_RING_BYTES;
static_assert(kSmemBytes <= 48 * 1024, "the decode stays within the default shared memory");
// Row classes the fused KX7 kernels will take, at the dispatch point in cuteafd_kx7_linear.
constexpr int32_t kGemvRows = 16;
constexpr int32_t kTileRows = 128;

bool aligned(const void* p, uintptr_t bytes) { return reinterpret_cast<uintptr_t>(p) % bytes == 0; }

size_t align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

// The fields of a v1.1 pack this code reads (the header grows only by appending).
constexpr size_t kPackV11Bytes = offsetof(cuteafd_kx7_pack_t, streams) + sizeof(void*);

bool valid_v11(const cuteafd_kx7_pack_t* p) {
  if (!p || p->struct_bytes < kPackV11Bytes || p->format != CUTEAFD_KX7_FORMAT_V11) return false;
  const bool shape = p->n > 0 && p->n % kInterleaveRows == 0 && p->k > 0 && p->slices >= 1 && p->slices <= 256 &&
                     p->k % (64 * p->slices) == 0 && p->streams_per_row == 4 * p->slices &&
                     p->symbols_per_stream == p->k / (4 * p->slices) &&
                     uint64_t(p->n) * uint64_t(p->streams_per_row) < (uint64_t(1) << 31);
  return shape && p->quad_stride == kQuadStride && p->ragged <= 1 && p->table_words == kV11TableWords &&
         p->tables && p->offsets && p->lengths && p->streams && aligned(p->tables, 4) && aligned(p->offsets, 4) &&
         aligned(p->lengths, 2) && aligned(p->streams, 16);
}

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
    kx7_decode16_multi<1, true>(s, ring_lane, lut, esc_shift, esc_mask, stride, w);
    uint4* at = reinterpret_cast<uint4*>(dst + 64 * c);
    at[0] = make_uint4(w[0][0], w[0][1], w[0][2], w[0][3]);
    at[1] = make_uint4(w[0][4], w[0][5], w[0][6], w[0][7]);
  }
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
  decode_kernel<<<blocks, kThreads, kSmemBytes, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t*>(pack->tables), int(pack->table_words), static_cast<const uint32_t*>(pack->offsets),
      static_cast<const uint16_t*>(pack->lengths), static_cast<const uint8_t*>(pack->streams),
      static_cast<uint16_t*>(out_bf16), pack->k, pack->slices, pack->streams_per_row, pack->quad_stride,
      int(pack->ragged), pack->esc_shift, pack->esc_mask, groups);
  return cudaGetLastError();
}

// out [rows, n] BF16 = x [rows, k] BF16 @ W^T for W = `pack` [n, k] (FP32 accumulation, the
// BF16 path's cuBLAS GEMM). `workspace` holds `workspace_bytes` >= cuteafd_kx7_workspace and
// is the caller's for the call's duration (one is enough for every pack launched in order on
// one stream). Asynchronous on `stream` and graph capturable. Returns 0, a cudaError_t, or for
// a failed GEMM the native status.
extern "C" int32_t cuteafd_kx7_linear(const void* x, const cuteafd_kx7_pack_t* pack, void* out, int32_t rows,
                                      void* workspace, size_t workspace_bytes, void* stream) {
  if (!x || !out || !workspace || rows < 1 || !valid_v11(pack)) return cudaErrorInvalidValue;
  const int32_t n = pack->n, k = pack->k;
  if (workspace_bytes < cuteafd_kx7_workspace(rows, n, k)) return cudaErrorInvalidValue;
  // ---- Dispatch point (the fused kernels land here, bit for bit the BF16 programs they
  // replace, with no workspace pass):
  //   rows <= kGemvRows (16):        the KX7 GEMV, each lane decoding its stream into bf16x2
  //                                  MMA fragments (MmaFp8Gemv's grid, K order and fold);
  //   kGemvRows < rows <= kTileRows (17..128): the KX7 tile GEMM, producer warps decoding the
  //                                  next stage's weight tile into shared memory;
  //   rows > kTileRows (prefill):    decode once, then the BF16 GEMM below.
  // Until they land every row count takes the v0 path: decode, then the GEMM.
  (void)kGemvRows;
  (void)kTileRows;
  const int32_t decoded = cuteafd_kx7_decode(pack, workspace, stream);
  if (decoded != 0) return decoded;
  const cuteafd_status_t gemm = cuteafd_cuda_linear_bf16_cublas_async(
      static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(workspace), nullptr, static_cast<uint16_t*>(out),
      size_t(rows), size_t(k), size_t(n), stream);
  return gemm == CUTEAFD_STATUS_OK ? 0 : int32_t(gemm);
}
