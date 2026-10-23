#include "cuteafd_vision_attention_internal.h"
#include "vision_attention_d64.h"
#include "vision_attention_d72.h"
#include "vision_attention_d80.h"
#include <mutex>
#include <new>

namespace {
cuteafd_vision_attention_d64_Kernel_Module_t module64{};
cuteafd_vision_attention_d72_Kernel_Module_t module72{};
cuteafd_vision_attention_d80_Kernel_Module_t module80{};
std::mutex mutex;
uint32_t references[3]{};
bool loaded[3][32]{};
struct Attention { int dim, device; };
}
extern "C" int32_t cuteafd_vision_attention_create(int32_t dim, void** out) {
  if(!out || (dim!=64 && dim!=72 && dim!=80))return cudaErrorInvalidValue;
  *out=nullptr;
  int device=-1;auto status=cudaGetDevice(&device);
  if(status)return status;
  if(device<0 || device>=32)return cudaErrorInvalidDevice;
  auto* owner=new(std::nothrow)Attention{dim,device};
  if(!owner)return cudaErrorMemoryAllocation;
  std::lock_guard<std::mutex> guard(mutex);
  int index=(dim-64)/8, result=0;
  auto* library=dim==80?&module80.module:dim==72?&module72.module:&module64.module;
  if(!*library) {
    void* args[]={&library,&result};
    if(dim==80)_mlir_cuteafd_vision_attention_d80_cuda_init(args);
    else if(dim==72)_mlir_cuteafd_vision_attention_d72_cuda_init(args);
    else _mlir_cuteafd_vision_attention_d64_cuda_init(args);
  }
  if(!result && !loaded[index][device]) {
    void* args[]={&library,&device,&result};
    if(dim==80)_mlir_cuteafd_vision_attention_d80_cuda_load_to_device(args);
    else if(dim==72)_mlir_cuteafd_vision_attention_d72_cuda_load_to_device(args);
    else _mlir_cuteafd_vision_attention_d64_cuda_load_to_device(args);
    if(!result)loaded[index][device]=true;
  }
  if(result) { delete owner; return result; }
  ++references[index];*out=owner;return 0;
}
extern "C" int32_t cuteafd_vision_attention_destroy(void* opaque) {
  if(!opaque)return 0;
  auto* owner=static_cast<Attention*>(opaque);
  // The caller has drained its stream before releasing its module reference.
  std::lock_guard<std::mutex> guard(mutex);
  int index=(owner->dim-64)/8,result=0;
  if(--references[index]==0) {
    auto* library=index==2?&module80.module:index==1?&module72.module:&module64.module;
    result=cudaLibraryUnload(*library);
    if(!result) { *library=nullptr;for(auto& value:loaded[index])value=false; }
  }
  delete owner;return result;
}
extern "C" int32_t cuteafd_vision_attention_launch(void* opaque,void* q,void* k,void* v,
  void* out,void* lse,void* cu_seqlens,float scale,void* stream) {
  if(!opaque)return cudaErrorInitializationError;
  auto* owner=static_cast<Attention*>(opaque);int device=-1;
  auto status=cudaGetDevice(&device);if(status)return status;
  if(device!=owner->device)return cudaErrorInvalidDevice;
  if(owner->dim==80)return cute_dsl_cuteafd_vision_attention_d80_wrapper(&module80,
    q,k,v,out,lse,cu_seqlens,cu_seqlens,nullptr,scale,static_cast<cudaStream_t>(stream));
  if(owner->dim==72)return cute_dsl_cuteafd_vision_attention_d72_wrapper(&module72,
    q,k,v,out,lse,cu_seqlens,cu_seqlens,nullptr,scale,static_cast<cudaStream_t>(stream));
  return cute_dsl_cuteafd_vision_attention_d64_wrapper(&module64,
    q,k,v,out,lse,cu_seqlens,cu_seqlens,nullptr,scale,static_cast<cudaStream_t>(stream));
}
