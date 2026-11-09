#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace cuteafd {
// Optional inspection must not leave a recoverable error for the next kernel's
// cudaGetLastError(). Preserve the returned status for diagnostic reporting.
inline cudaError_t diagnostic_cuda(cudaError_t status) {
  if (status != cudaSuccess) (void)cudaGetLastError();
  return status;
}

using GraphEdgesQuery = decltype(&cudaGraphGetEdges);
inline cudaError_t inspect_graph_topology(cudaGraph_t graph, size_t* count,
    size_t* width, int* spans_devices, GraphEdgesQuery query_edges = cudaGraphGetEdges) {
  *count = *width = 0;
  *spans_devices = 0;
  auto status = diagnostic_cuda(cudaGraphGetNodes(graph, nullptr, count));
  if (status != cudaSuccess) return status;
  std::vector<cudaGraphNode_t> nodes(*count);
  status = diagnostic_cuda(cudaGraphGetNodes(graph, nodes.data(), count));
  if (status != cudaSuccess) return status;
  // CUDA 13 requires endpoints whenever edgeData is supplied, even for a
  // zero-capacity count query. Null edgeData rejects programmatic dependencies.
  size_t edges = 0;
  cudaGraphNode_t dummy_from = nullptr, dummy_to = nullptr;
  cudaGraphEdgeData dummy_data{};
  status = diagnostic_cuda(query_edges(graph, &dummy_from, &dummy_to, &dummy_data, &edges));
  if (status != cudaSuccess) return status;
  std::vector<cudaGraphNode_t> from(edges), to(edges);
  std::vector<cudaGraphEdgeData> data(edges);
  if (edges) {
    status = diagnostic_cuda(query_edges(graph, from.data(), to.data(), data.data(), &edges));
    if (status != cudaSuccess) return status;
  }
  std::unordered_map<cudaGraphNode_t, size_t> index;
  for (size_t n = 0; n < *count; ++n) index[nodes[n]] = n;
  std::vector<size_t> incoming(*count, 0);
  std::vector<std::vector<size_t>> outgoing(*count);
  for (size_t e = 0; e < edges; ++e) {
    if (!index.count(from[e]) || !index.count(to[e])) return cudaErrorInvalidValue;
    outgoing[index[from[e]]].push_back(index[to[e]]);
    ++incoming[index[to[e]]];
  }
  std::vector<size_t> ready;
  for (size_t n = 0; n < *count; ++n) if (!incoming[n]) ready.push_back(n);
  size_t frontier = ready.size();
  for (size_t n = 0; n < ready.size(); ++n) {
    for (size_t next : outgoing[ready[n]]) if (--incoming[next] == 0) ready.push_back(next);
    frontier = std::max(frontier, ready.size() - n - 1);
  }
  if (ready.size() != *count) return cudaErrorInvalidValue;
  int device = -1;
  status = diagnostic_cuda(cudaGetDevice(&device));
  if (status != cudaSuccess) return status;
  for (auto node : nodes) {
    cudaGraphNodeType type;
    status = diagnostic_cuda(cudaGraphNodeGetType(node, &type));
    if (status != cudaSuccess) return status;
    if (type != cudaGraphNodeTypeMemcpy) continue;
    cudaMemcpy3DParms params{};
    status = diagnostic_cuda(cudaGraphMemcpyNodeGetParams(node, &params));
    if (status != cudaSuccess) return status;
    for (void* ptr : {params.srcPtr.ptr, params.dstPtr.ptr}) {
      if (!ptr) continue;
      cudaPointerAttributes attributes{};
      status = diagnostic_cuda(cudaPointerGetAttributes(&attributes, ptr));
      if (status != cudaSuccess) return status;
      if (attributes.type == cudaMemoryTypeDevice && attributes.device != device) *spans_devices = 1;
    }
  }
  *width = frontier;
  return cudaSuccess;
}
}  // namespace cuteafd
