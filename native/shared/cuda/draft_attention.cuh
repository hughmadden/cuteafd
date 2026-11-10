#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuteafd::draft_attention {
constexpr int queries_per_slice = 32;

// One arena covers scalar 128-key chunks and SM-sized WMMA partitions;
// smaller launches reuse it without allocating or changing pointer lifetimes.
uint64_t workspace(int sequences, int kv_heads, int slices, int max_keys, int head_dim);
int launch(const void* q, const void* k_block, const void* v_block,
           const void* k_ring, const void* v_ring, const void* seq_slots,
           const void* ctx_lengths, const void* ctx_ends, const void* sinks,
           void* out, void* workspace, int sequences, int block_rows,
           int heads, int kv_heads, int ring, int max_keys, int window,
           bool causal, int head_dim, int slices, bool fp8, float scale, cudaStream_t stream,
           const int32_t* pages = nullptr, int table_width = 0);
}  // namespace cuteafd::draft_attention
