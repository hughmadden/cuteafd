#pragma once
#include "memory_failure.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>
#include <memory>
#include <new>
#include <pthread.h>
#include <sys/syscall.h>
#include <string>
#include <unistd.h>
#include <thread>
#include <unordered_map>

namespace cuteafd {
// One pool per host execution thread/device: no concurrent handle mutation.
// Streams/plans retain private workspaces for graph pointer stability. Every
// GEMM rebinds stream BEFORE workspace (SetStream resets it), math and atomics.
struct BlasOwner {
  const int device;
  const std::thread::id thread = std::this_thread::get_id();
  const long thread_id = syscall(SYS_gettid);
  cublasHandle_t gemm = nullptr;
  cublasLtHandle_t lt = nullptr;
  std::string gemm_owner;
  std::string lt_owner;
  explicit BlasOwner(int id) : device(id) {}
  void log(const char* event, const char* backend) const noexcept {
    char name[16] = {};
    pthread_getname_np(pthread_self(), name, sizeof(name));
    const auto& allocation_owner = backend[0] == 'g' ? gemm_owner : lt_owner;
    std::fprintf(stderr, "blas_owner event=%s backend=%s device=%d thread_id=%ld thread_name=%s owner=%p kind=workspace allocation_owner=%s\n",
        event, backend, device, thread_id, name, static_cast<const void*>(this), allocation_owner.c_str());
  }
  ~BlasOwner() {
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return;
    if (gemm && cublasDestroy(gemm) == CUBLAS_STATUS_SUCCESS) log("destroy", "gemm");
    if (lt && cublasLtDestroy(lt) == CUBLAS_STATUS_SUCCESS) log("destroy", "lt");
    if (previous != device) cudaSetDevice(previous);
  }
};
inline cublasStatus_t check_blas_owner(const std::shared_ptr<BlasOwner>& owner) {
  if (!owner || owner->thread != std::this_thread::get_id()) return CUBLAS_STATUS_INVALID_VALUE;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return CUBLAS_STATUS_NOT_INITIALIZED;
  return device == owner->device ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}
inline cublasStatus_t borrow_blas(bool tensor, std::shared_ptr<BlasOwner>& out,
    const char* allocation_owner) noexcept {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return CUBLAS_STATUS_NOT_INITIALIZED;
  try {
    static thread_local std::unordered_map<int, std::weak_ptr<BlasOwner>> owners;
    auto& slot = owners[device];
    auto owner = slot.lock();
    if (!owner) owner = std::make_shared<BlasOwner>(device);
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    if ((tensor && !owner->lt) || (!tensor && !owner->gemm)) {
      // cuBLAS owns opaque device allocations. Retain their attempted tag for
      // faults; the calibrated admission owner, not a fake pointer, prices them.
      if (!cuteafd_tag_check(CUTEAFD_MEMORY_KIND_WORKSPACE, allocation_owner))
        return CUBLAS_STATUS_INVALID_VALUE;
      auto& name = tensor ? owner->lt_owner : owner->gemm_owner;
      name = allocation_owner;
      status = tensor
          ? CUTEAFD_CHECK_MEMORY("cublasLtCreate", "cublas", cublasLtCreate(&owner->lt), 0, 0, nullptr)
          : CUTEAFD_CHECK_MEMORY("cublasCreate", "cublas", cublasCreate(&owner->gemm), 0, 0, nullptr);
      if (status == CUBLAS_STATUS_SUCCESS) owner->log("create", tensor ? "lt" : "gemm");
    }
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    slot = owner;
    out = std::move(owner);
    return CUBLAS_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return CUBLAS_STATUS_ALLOC_FAILED;
  }
}
inline cublasStatus_t bind_blas(const std::shared_ptr<BlasOwner>& owner, cudaStream_t stream,
    void* workspace, size_t bytes, cublasMath_t math = CUBLAS_DEFAULT_MATH,
    cublasAtomicsMode_t atomics = CUBLAS_ATOMICS_ALLOWED) {
  auto status = check_blas_owner(owner);
  if (status != CUBLAS_STATUS_SUCCESS || !owner->gemm) return CUBLAS_STATUS_INVALID_VALUE;
  status = cublasSetStream(owner->gemm, stream);
  if (status == CUBLAS_STATUS_SUCCESS) status = cublasSetWorkspace(owner->gemm, workspace, bytes);
  if (status == CUBLAS_STATUS_SUCCESS) status = cublasSetMathMode(owner->gemm, math);
  if (status == CUBLAS_STATUS_SUCCESS) status = cublasSetAtomicsMode(owner->gemm, atomics);
  return status;
}
}  // namespace cuteafd
