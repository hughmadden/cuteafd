#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuteafd::draft_attention {
constexpr int queries_per_slice = 32;

// The capacity arena covers every smaller launch: partitions are sized for
// one sequence, while launches reduce splitting as the batch grows.
uint64_t workspace(int sequences, int kv_heads, int slices, int max_keys, int head_dim);
int launch(const void* q, const void* k_block, const void* v_block,
           const void* k_ring, const void* v_ring, const void* seq_slots,
           const void* ctx_lengths, const void* ctx_ends, const void* sinks,
           void* out, void* workspace, int sequences, int block_rows,
           int heads, int kv_heads, int ring, int max_keys, int window,
           bool causal, int head_dim, int slices, float scale, cudaStream_t stream);
}  // namespace cuteafd::draft_attention
