#include "common.h"
#include "cuteafd_vision.h"

namespace {

__global__ void embedding_lookup_f32_kernel(const float* embedding, const uint32_t* token_ids,
                                            float* out, size_t rows, size_t vocab,
                                            size_t hidden) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / hidden;
  const size_t col = idx % hidden;
  const size_t token_id = static_cast<size_t>(token_ids[row]);
  if (token_id >= vocab) {
    out[idx] = 0.0f;
    return;
  }
  out[idx] = embedding[token_id * hidden + col];
}

__global__ void embedding_lookup_bf16_kernel(const uint16_t* embedding, const uint32_t* token_ids,
                                             uint16_t* out, size_t rows, size_t vocab,
                                             size_t hidden) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * hidden;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / hidden;
  const size_t col = idx % hidden;
  const size_t token_id = static_cast<size_t>(token_ids[row]);
  if (token_id >= vocab) {
    out[idx] = 0;
    return;
  }
  out[idx] = embedding[token_id * hidden + col];
}

cuteafd_status_t validate_embedding_lookup_args(const float* embedding, const uint32_t* token_ids,
                                              const float* out, size_t rows, size_t vocab,
                                              size_t hidden) {
  if (embedding == nullptr || token_ids == nullptr || out == nullptr) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || vocab == 0 || hidden == 0) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(vocab, hidden, &ignored) || !checked_mul(rows, hidden, &ignored)) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  return CUTEAFD_STATUS_OK;
}

cuteafd_status_t validate_embedding_lookup_bf16_args(const uint16_t* embedding,
                                                   const uint32_t* token_ids,
                                                   const uint16_t* out, size_t rows, size_t vocab,
                                                   size_t hidden) {
  if (embedding == nullptr || token_ids == nullptr || out == nullptr) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || vocab == 0 || hidden == 0) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(vocab, hidden, &ignored) || !checked_mul(rows, hidden, &ignored)) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  return CUTEAFD_STATUS_OK;
}

cuteafd_status_t validate_bf16_graph_embedding_lookup_buffers(
    cuteafd_device_buffer_t embedding, cuteafd_device_buffer_t token_ids, cuteafd_device_buffer_t out,
    size_t rows, size_t vocab, size_t hidden) {
  const cuteafd_status_t valid = validate_embedding_lookup_bf16_args(
      static_cast<const uint16_t*>(embedding.ptr), static_cast<const uint32_t*>(token_ids.ptr),
      static_cast<const uint16_t*>(out.ptr), rows, vocab, hidden);
  if (valid != CUTEAFD_STATUS_OK) {
    return valid;
  }
  size_t embedding_values = 0;
  size_t output_values = 0;
  if (!checked_mul(vocab, hidden, &embedding_values) ||
      !checked_mul(rows, hidden, &output_values)) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  size_t embedding_bytes = 0;
  size_t token_bytes = 0;
  size_t output_bytes = 0;
  if (!checked_mul(embedding_values, sizeof(uint16_t), &embedding_bytes) ||
      !checked_mul(rows, sizeof(uint32_t), &token_bytes) ||
      !checked_mul(output_values, sizeof(uint16_t), &output_bytes)) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  if (embedding.bytes < embedding_bytes || token_ids.bytes < token_bytes ||
      out.bytes < output_bytes) {
    return CUTEAFD_STATUS_BUFFER_TOO_SMALL;
  }
  return CUTEAFD_STATUS_OK;
}

__global__ void embed_inject_kernel(const uint16_t* features, const uint32_t* indices,
                                    uint16_t* out, int rows, int width, int copies) {
  const size_t feature = blockIdx.x;
  const uint32_t row = indices[feature];
  if (row >= uint32_t(rows)) return;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const uint16_t value = features[feature * width + col];
    for (int copy = 0; copy < copies; ++copy)
      out[(size_t(row) * copies + copy) * width + col] = value;
  }
}

}  // namespace

extern "C" int32_t cuteafd_embed_inject(const uint16_t* features, const uint32_t* indices,
                                       uint16_t* out, int32_t feature_rows, int32_t rows,
                                       int32_t width, int32_t copies, void* stream) {
  if (!features || !indices || !out || feature_rows < 1 || feature_rows > rows ||
      rows < 1 || rows > 16384 || width < 1 || width > 16384 || copies < 1 || copies > 4)
    return cudaErrorInvalidValue;
  embed_inject_kernel<<<feature_rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      features, indices, out, rows, width, copies);
  return cudaGetLastError();
}

extern "C" cuteafd_status_t cuteafd_cuda_graph_update_embedding_lookup_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    cuteafd_device_buffer_t embedding, cuteafd_device_buffer_t token_ids, cuteafd_device_buffer_t out,
    size_t rows, size_t vocab, size_t hidden) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  const cuteafd_status_t valid =
      validate_bf16_graph_embedding_lookup_buffers(embedding, token_ids, out, rows, vocab, hidden);
  if (valid != CUTEAFD_STATUS_OK) {
    return valid;
  }

  cudaGraphNode_t node = nullptr;
  const cuteafd_status_t node_status = find_kernel_node_by_index(cuda_graph, kernel_node_index, &node);
  if (node_status != CUTEAFD_STATUS_OK) {
    return node_status;
  }

  cudaKernelNodeParams existing = {};
  cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
  if (err != cudaSuccess) {
    return CUTEAFD_STATUS_INTERNAL_ERROR;
  }
  if (existing.func != reinterpret_cast<void*>(embedding_lookup_bf16_kernel)) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* embedding_ptr = static_cast<const uint16_t*>(embedding.ptr);
  const uint32_t* token_ids_ptr = static_cast<const uint32_t*>(token_ids.ptr);
  uint16_t* out_ptr = static_cast<uint16_t*>(out.ptr);
  void* args[] = {
      &embedding_ptr,
      &token_ids_ptr,
      &out_ptr,
      &rows,
      &vocab,
      &hidden,
  };
  const int threads = 256;
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(embedding_lookup_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(block_count), 1, 1);
  params.blockDim = dim3(threads, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  err = cudaGraphKernelNodeSetParams(node, &params);
  if (err != cudaSuccess) {
    return CUTEAFD_STATUS_INTERNAL_ERROR;
  }
  err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec), node,
                                         &params);
  if (err != cudaSuccess) {
    return CUTEAFD_STATUS_INTERNAL_ERROR;
  }
  return CUTEAFD_STATUS_OK;
}

extern "C" cuteafd_status_t cuteafd_cuda_embedding_lookup_f32_async(
    const float* embedding, const uint32_t* token_ids, float* out, size_t rows, size_t vocab,
    size_t hidden, void* cuda_stream) {
  const cuteafd_status_t valid =
      validate_embedding_lookup_args(embedding, token_ids, out, rows, vocab, hidden);
  if (valid != CUTEAFD_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  embedding_lookup_f32_kernel<<<blocks, threads, 0, stream>>>(embedding, token_ids, out, rows,
                                                              vocab, hidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" cuteafd_status_t cuteafd_cuda_embedding_lookup_f32(const float* embedding,
                                                          const uint32_t* token_ids, float* out,
                                                          size_t rows, size_t vocab,
                                                          size_t hidden) {
  const cuteafd_status_t status =
      cuteafd_cuda_embedding_lookup_f32_async(embedding, token_ids, out, rows, vocab, hidden,
                                            nullptr);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" cuteafd_status_t cuteafd_cuda_embedding_lookup_bf16_async(
    const uint16_t* embedding, const uint32_t* token_ids, uint16_t* out, size_t rows, size_t vocab,
    size_t hidden, void* cuda_stream) {
  const cuteafd_status_t valid =
      validate_embedding_lookup_bf16_args(embedding, token_ids, out, rows, vocab, hidden);
  if (valid != CUTEAFD_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * hidden;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return CUTEAFD_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  embedding_lookup_bf16_kernel<<<blocks, threads, 0, stream>>>(embedding, token_ids, out, rows,
                                                               vocab, hidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" cuteafd_status_t cuteafd_cuda_embedding_lookup_bf16(const uint16_t* embedding,
                                                           const uint32_t* token_ids,
                                                           uint16_t* out, size_t rows,
                                                           size_t vocab, size_t hidden) {
  const cuteafd_status_t status =
      cuteafd_cuda_embedding_lookup_bf16_async(embedding, token_ids, out, rows, vocab, hidden,
                                             nullptr);
  if (status != CUTEAFD_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

