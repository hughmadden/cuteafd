// FP32 native MiMo audio qualification path. CUDA library plans and caller-
// admitted CuTe modules are resolved once; encode owns no allocation/capture.
#include "cuteafd_audio.h"
#include "cuteafd_audio_support_internal.h"
#include "audio_tables.h"
#include "audio_identity.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <type_traits>

namespace {
constexpr uint64_t BLAS_BYTES=4ULL<<20, FFT_BYTES=64ULL<<20;
constexpr int BOOKS[20]={1024,1024,256,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128};
uint64_t align256(uint64_t n) { return (n+255)&~uint64_t(255); }
int codes_for(int frames) { return ((frames+1)/2+1)/2; }
int total_codes(int samples) {
  int frames=samples/240+1;
  return frames/6000*1500+codes_for(frames%6000);
}
struct Owner {
  cuteafd_audio_spec s{};
  cuteafd_audio_ledger ledger{};
  int device=0, mel_capacity=0, codec_capacity=0, patch_capacity=0;
  cudaStream_t stream=nullptr;
  cublasHandle_t blas=nullptr;
  cufftHandle fft=0;
  void* support=nullptr;
  unsigned char *weights=nullptr,*arena=nullptr;
  float *pcm=nullptr,*hann=nullptr,*filter=nullptr,*codec_rope=nullptr,*patch_rope=nullptr;
  float *frames=nullptr,*complex=nullptr,*magnitude=nullptr,*mel=nullptr,*columns=nullptr;
  float *x=nullptr,*norm=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*packed_q=nullptr,*packed_k=nullptr,*packed_v=nullptr;
  float *attention=nullptr,*projected=nullptr,*skip=nullptr,*fc1=nullptr,*fc2=nullptr,*scores=nullptr;
  float *residual=nullptr,*dot=nullptr,*square=nullptr,*book_square=nullptr,*local=nullptr,*output=nullptr;
  int32_t* codes=nullptr;
  void *blas_workspace=nullptr,*fft_workspace=nullptr;
  bool uploaded=false;
};
int drain(Owner* o,int first) { auto e=cudaStreamSynchronize(o->stream);return first?first:int(e); }
int bs(cublasStatus_t s) { return s==CUBLAS_STATUS_SUCCESS?0:-int(s); }
int fs(cufftResult s) { return s==CUFFT_SUCCESS?0:-1000-int(s); }
bool extent(const cuteafd_audio_spec& s,uint64_t offset,uint64_t count) {
  return offset%256==0 && offset<=s.weight_bytes && count<=s.weight_bytes-offset;
}
bool valid(const cuteafd_audio_spec& s) {
  if(s.abi_version!=1 || s.numerics!=1 || s.max_samples<=480 || s.max_samples>7200000 ||
      (s.output_width!=4096 && s.output_width!=6144) || !s.weight_bytes || s.weight_bytes>(3ULL<<30))return false;
  const uint64_t matrix=1024ULL*1024*4,vector=1024*4;
  if(!extent(s,s.conv1,1024ULL*128*3*4) || !extent(s,s.conv1b,vector) ||
     !extent(s,s.conv2,matrix*3) || !extent(s,s.conv2b,vector) || !extent(s,s.downsample,matrix*2) ||
     !extent(s,s.norm,vector) || !extent(s,s.normb,vector) || !extent(s,s.downnorm,vector) || !extent(s,s.downnormb,vector))return false;
  for(const auto& b:s.codec) {
    const uint64_t offsets[]={b.q,b.qb,b.k,b.v,b.vb,b.o,b.ob,b.norm1,b.norm1b,b.norm2,b.norm2b,b.fc1,b.fc1b,b.fc2,b.fc2b};
    const uint64_t sizes[]={matrix,vector,matrix,matrix,vector,matrix,vector,vector,vector,vector,vector,matrix*4,vector*4,matrix*4,vector};
    for(int i=0;i<15;++i)if(!extent(s,offsets[i],sizes[i]))return false;
  }
  for(int i=0;i<20;++i)if(!extent(s,s.codebooks[i],uint64_t(BOOKS[i])*1024*4) || !extent(s,s.speech[i],1280ULL*1024*4))return false;
  for(const auto& b:s.patch) {
    const uint64_t offsets[]={b.norm1,b.norm2,b.q,b.qb,b.k,b.kb,b.v,b.vb,b.o,b.gate,b.up,b.down};
    const uint64_t sizes[]={vector,vector,matrix,vector,matrix,vector,matrix,vector,matrix,matrix*4,matrix*4,matrix*4};
    for(int i=0;i<12;++i)if(!extent(s,offsets[i],sizes[i]))return false;
  }
  return extent(s,s.patch_norm,vector) && extent(s,s.projection1,16384ULL*4096*4) && extent(s,s.projection2,uint64_t(s.output_width)*16384*4);
}
uint64_t scratch(Owner* o) {
  int frames=o->s.max_samples/240+1;
  int m=o->mel_capacity=std::min(6000,frames);
  int c=o->codec_capacity=(m+1)/2;
  int p=o->patch_capacity=((total_codes(o->s.max_samples)+3)/4)*4;
  int n=std::max(m,p);
  uint64_t cursor=0;
  auto take=[&](auto& ptr,uint64_t bytes) {
    using P=typename std::remove_reference<decltype(ptr)>::type;
    ptr=o->arena?reinterpret_cast<P>(o->arena+cursor):nullptr;
    cursor+=align256(bytes);
  };
  take(o->pcm,uint64_t(o->s.max_samples)*4);take(o->hann,960*4);take(o->filter,481*128*4);
  take(o->codec_rope,uint64_t(c)*64*4);take(o->patch_rope,4*64*4);
  take(o->frames,uint64_t(m)*960*4);take(o->complex,uint64_t(m)*481*8);
  take(o->magnitude,uint64_t(m)*481*4);take(o->mel,uint64_t(m)*128*4);
  take(o->columns,uint64_t(m)*3072*4);
  for(auto ptr:{&o->x,&o->norm,&o->q,&o->k,&o->v,&o->packed_q,&o->packed_k,&o->packed_v,&o->attention,&o->projected,&o->skip})take(*ptr,uint64_t(n)*1024*4);
  take(o->fc1,uint64_t(n)*4096*4);
  take(o->scores,std::max(uint64_t(c)*c*16,uint64_t(p)*4*16)*4);
  take(o->residual,uint64_t(c)*1024*4);take(o->dot,uint64_t(c)*1024*4);
  take(o->square,uint64_t(c)*4);take(o->book_square,20*1024*4);
  take(o->codes,uint64_t(total_codes(o->s.max_samples))*20*4);
  take(o->local,uint64_t(p)*1024*4);take(o->output,uint64_t(p/4)*o->s.output_width*4);
  // The projection's16384-wide temporary reuses an arena extent sized exactly
  // for its maximum groups, independent of the live clip length.
  take(o->fc2,std::max(uint64_t(n)*4096,uint64_t(p/4)*16384)*4);
  take(o->blas_workspace,BLAS_BYTES);take(o->fft_workspace,FFT_BYTES);
  return cursor;
}
float* w(Owner* o,uint64_t offset) { return reinterpret_cast<float*>(o->weights+offset); }
int support(Owner* o,int operation,int width,const float* x,const float* weight,const float* bias,
            float* out,const float* aux,int32_t* codes,int rows,int length=1,int offset=0,float parameter=1.f) {
  return cuteafd_audio_support_launch(o->support,operation,width,const_cast<float*>(x),const_cast<float*>(weight),
      const_cast<float*>(bias),out,const_cast<float*>(aux),codes,rows,length,offset,parameter,o->stream);
}
int linear(Owner* o,const float* x,uint64_t weight,int m,int n,int k,float* y) {
  const float one=1,zero=0;
  return bs(cublasSgemm(o->blas,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&one,w(o,weight),k,x,k,&zero,y,n));
}
int bias(Owner* o,float* x,uint64_t offset,int rows,int width,bool gelu=false) {
  return support(o,gelu?AUDIO_BIAS_GELU:AUDIO_BIAS,width,x,nullptr,w(o,offset),x,nullptr,nullptr,rows);
}
int norm(Owner* o,float* x,uint64_t weight,uint64_t bias,int rows,bool rms=false) {
  return support(o,rms?AUDIO_RMS_NORM:AUDIO_LAYER_NORM,1024,x,w(o,weight),rms?nullptr:w(o,bias),o->norm,nullptr,nullptr,rows,1,0,rms?1e-6f:1e-5f);
}
int add(Owner* o,float* x,const float* y,int rows) {
  return support(o,AUDIO_ADD,1024,x,nullptr,nullptr,x,y,nullptr,rows);
}
int attention(Owner* o,int rows,int length,int window,const float* rotary) {
  int e=support(o,AUDIO_ROPE_PACK,1024,o->q,rotary,nullptr,o->packed_q,nullptr,nullptr,rows,length);if(e)return e;
  e=support(o,AUDIO_ROPE_PACK,1024,o->k,rotary,nullptr,o->packed_k,nullptr,nullptr,rows,length);if(e)return e;
  e=support(o,AUDIO_PACK_HEADS,1024,o->v,nullptr,nullptr,o->packed_v,nullptr,nullptr,rows,length);if(e)return e;
  const float one=1,zero=0;
  int batches=rows/length*16;
  e=bs(cublasSgemmStridedBatched(o->blas,CUBLAS_OP_T,CUBLAS_OP_N,length,length,64,&one,
      o->packed_k,64,int64_t(length)*64,o->packed_q,64,int64_t(length)*64,&zero,o->scores,length,int64_t(length)*length,batches));if(e)return e;
  e=support(o,AUDIO_SOFTMAX,1024,o->scores,nullptr,nullptr,o->scores,nullptr,nullptr,batches*length,length,window,.125f);if(e)return e;
  e=bs(cublasSgemmStridedBatched(o->blas,CUBLAS_OP_N,CUBLAS_OP_N,64,length,length,&one,
      o->packed_v,64,int64_t(length)*64,o->scores,length,int64_t(length)*length,&zero,o->attention,64,int64_t(length)*64,batches));if(e)return e;
  return support(o,AUDIO_UNPACK_HEADS,1024,o->attention,nullptr,nullptr,o->projected,nullptr,nullptr,rows,length);
}
int observe(Owner* o,cuteafd_audio_observer cb,void* ctx,int stage,int global,const float* data,int rows,int width) {
  if(!cb)return 0;
  int e=drain(o,0);return e?e:cb(ctx,stage,global,data,rows,width);
}
int encode_segment(Owner* o,int samples,int mel_frames,int global_frame,int code_offset,cuteafd_audio_observer cb,void* ctx) {
  int e=support(o,AUDIO_FRAME,960,o->pcm,o->hann,nullptr,o->frames,nullptr,nullptr,o->mel_capacity,samples,global_frame);if(e)return e;
  e=fs(cufftExecR2C(o->fft,o->frames,reinterpret_cast<cufftComplex*>(o->complex)));if(e)return e;
  e=support(o,AUDIO_MAGNITUDE,481,o->complex,nullptr,nullptr,o->magnitude,nullptr,nullptr,mel_frames);if(e)return e;
  const float one=1,zero=0;
  e=bs(cublasSgemm(o->blas,CUBLAS_OP_N,CUBLAS_OP_N,128,mel_frames,481,&one,o->filter,128,o->magnitude,481,&zero,o->mel,128));if(e)return e;
  e=support(o,AUDIO_LOG,128,o->mel,nullptr,nullptr,o->mel,nullptr,nullptr,mel_frames);if(e)return e;
  e=observe(o,cb,ctx,0,global_frame,o->mel,mel_frames,128);if(e)return e;
  // The official tokenizer batches all segments at the longest segment length.
  // Its biased conv1 produces a nonzero first padded row, read by odd conv2 tails.
  int batch_mel=std::min(6000,samples/240+1);
  int conv1_rows=mel_frames+((mel_frames%2 && mel_frames<batch_mel)?1:0);
  e=support(o,AUDIO_IM2COL_384,384,o->mel,nullptr,nullptr,o->columns,nullptr,nullptr,conv1_rows,mel_frames,1);if(e)return e;
  e=linear(o,o->columns,o->s.conv1,conv1_rows,1024,384,o->x);if(e)return e;
  e=bias(o,o->x,o->s.conv1b,conv1_rows,1024,true);if(e)return e;
  e=observe(o,cb,ctx,1,global_frame,o->x,mel_frames,1024);if(e)return e;
  int rows=(mel_frames+1)/2;
  e=support(o,AUDIO_IM2COL_3072,3072,o->x,nullptr,nullptr,o->columns,nullptr,nullptr,rows,conv1_rows,1);if(e)return e;
  e=linear(o,o->columns,o->s.conv2,rows,1024,3072,o->x);if(e)return e;
  e=bias(o,o->x,o->s.conv2b,rows,1024,true);if(e)return e;
  e=observe(o,cb,ctx,2,global_frame,o->x,rows,1024);if(e)return e;
  for(int layer=0;layer<24;++layer) {
    const auto& b=o->s.codec[layer];
    e=norm(o,o->x,b.norm1,b.norm1b,rows);if(e)return e;
    e=linear(o,o->norm,b.q,rows,1024,1024,o->q);if(e)return e;
    e=bias(o,o->q,b.qb,rows,1024);if(e)return e;
    e=linear(o,o->norm,b.k,rows,1024,1024,o->k);if(e)return e;
    e=linear(o,o->norm,b.v,rows,1024,1024,o->v);if(e)return e;
    e=bias(o,o->v,b.vb,rows,1024);if(e)return e;
    e=attention(o,rows,rows,layer%2==0?128:0,o->codec_rope);if(e)return e;
    e=linear(o,o->projected,b.o,rows,1024,1024,o->norm);if(e)return e;
    e=bias(o,o->norm,b.ob,rows,1024);if(e)return e;
    e=add(o,o->x,o->norm,rows);if(e)return e;
    e=norm(o,o->x,b.norm2,b.norm2b,rows);if(e)return e;
    e=linear(o,o->norm,b.fc1,rows,4096,1024,o->fc1);if(e)return e;
    e=bias(o,o->fc1,b.fc1b,rows,4096,true);if(e)return e;
    e=linear(o,o->fc1,b.fc2,rows,1024,4096,o->norm);if(e)return e;
    e=bias(o,o->norm,b.fc2b,rows,1024);if(e)return e;
    e=add(o,o->x,o->norm,rows);if(e)return e;
    if(layer==2) { e=cudaMemcpyAsync(o->skip,o->x,size_t(rows)*1024*4,cudaMemcpyDeviceToDevice,o->stream);if(e)return e; }
  e=observe(o,cb,ctx,3+layer,global_frame,o->x,rows,1024);if(e)return e;
  }
  e=add(o,o->x,o->skip,rows);if(e)return e;
  e=norm(o,o->x,o->s.norm,o->s.normb,rows);if(e)return e;
  e=observe(o,cb,ctx,27,global_frame,o->norm,rows,1024);if(e)return e;
  int codes=(rows+1)/2, down_rows=rows;
  // Official unpacking repeats the last hidden row in a shorter segment's
  // batch padding; only padding beyond the common batch extent is zero.
  if(rows%2 && rows<(batch_mel+1)/2) {
    e=cudaMemcpyAsync(o->norm+rows*1024,o->norm+(rows-1)*1024,1024*4,cudaMemcpyDeviceToDevice,o->stream);if(e)return e;
    ++down_rows;
  }
  e=support(o,AUDIO_IM2COL_2048,2048,o->norm,nullptr,nullptr,o->columns,nullptr,nullptr,codes,down_rows);if(e)return e;
  e=linear(o,o->columns,o->s.downsample,codes,1024,2048,o->x);if(e)return e;
  e=support(o,AUDIO_GELU,1024,o->x,nullptr,nullptr,o->x,nullptr,nullptr,codes);if(e)return e;
  e=norm(o,o->x,o->s.downnorm,o->s.downnormb,codes);if(e)return e;
  e=observe(o,cb,ctx,28,global_frame,o->norm,codes,1024);if(e)return e;
  e=cudaMemcpyAsync(o->residual,o->norm,size_t(codes)*1024*4,cudaMemcpyDeviceToDevice,o->stream);if(e)return e;
  for(int book=0;book<20;++book) {
    e=support(o,AUDIO_SUM_SQUARE,1024,o->residual,nullptr,nullptr,o->square,nullptr,nullptr,codes);if(e)return e;
    e=linear(o,o->residual,o->s.codebooks[book],codes,BOOKS[book],1024,o->dot);if(e)return e;
    e=support(o,AUDIO_RVQ_SELECT,1024,o->dot,w(o,o->s.codebooks[book]),o->square,o->residual,
        o->book_square+book*1024,o->codes+code_offset*20,codes,BOOKS[book],book);if(e)return e;
  }
  return 0;
}
int encode_patch(Owner* o,int codes,cuteafd_audio_observer cb,void* ctx) {
  int rows=((codes+3)/4)*4;
  int e=cudaMemsetAsync(o->local,0,size_t(rows)*1024*4,o->stream);if(e)return e;
  for(int book=0;book<20;++book) {
    e=support(o,AUDIO_SPEECH_ADD,1024,nullptr,w(o,o->s.speech[book]),nullptr,o->local,nullptr,o->codes,rows,codes,book);if(e)return e;
  }
  e=observe(o,cb,ctx,29,0,o->local,rows,1024);if(e)return e;
  for(const auto& b:o->s.patch) {
    e=norm(o,o->local,b.norm1,0,rows,true);if(e)return e;
    e=linear(o,o->norm,b.q,rows,1024,1024,o->q);if(e)return e;
    e=bias(o,o->q,b.qb,rows,1024);if(e)return e;
    e=linear(o,o->norm,b.k,rows,1024,1024,o->k);if(e)return e;
    e=bias(o,o->k,b.kb,rows,1024);if(e)return e;
    e=linear(o,o->norm,b.v,rows,1024,1024,o->v);if(e)return e;
    e=bias(o,o->v,b.vb,rows,1024);if(e)return e;
    e=attention(o,rows,4,-1,o->patch_rope);if(e)return e;
    e=linear(o,o->projected,b.o,rows,1024,1024,o->norm);if(e)return e;
    e=add(o,o->local,o->norm,rows);if(e)return e;
    e=norm(o,o->local,b.norm2,0,rows,true);if(e)return e;
    e=linear(o,o->norm,b.gate,rows,4096,1024,o->fc1);if(e)return e;
    e=linear(o,o->norm,b.up,rows,4096,1024,o->fc2);if(e)return e;
    e=support(o,AUDIO_SILU_PRODUCT,4096,o->fc1,nullptr,nullptr,o->fc1,o->fc2,nullptr,rows);if(e)return e;
    e=linear(o,o->fc1,b.down,rows,1024,4096,o->norm);if(e)return e;
    e=add(o,o->local,o->norm,rows);if(e)return e;
  }
  e=norm(o,o->local,o->s.patch_norm,0,rows,true);if(e)return e;
  e=observe(o,cb,ctx,30,0,o->norm,rows,1024);if(e)return e;
  e=linear(o,o->norm,o->s.projection1,rows/4,16384,4096,o->fc2);if(e)return e;
  e=support(o,AUDIO_GELU,16384,o->fc2,nullptr,nullptr,o->fc2,nullptr,nullptr,rows/4);if(e)return e;
  e=linear(o,o->fc2,o->s.projection2,rows/4,o->s.output_width,16384,o->output);if(e)return e;
  return observe(o,cb,ctx,31,0,o->output,rows/4,o->s.output_width);
}
}
extern "C" int32_t cuteafd_audio_required(const cuteafd_audio_spec* s,cuteafd_audio_ledger* out) {
  if(!s || !out || !valid(*s))return cudaErrorInvalidValue;
  Owner o;o.s=*s;
  uint64_t bytes=scratch(&o);
  *out={s->weight_bytes,bytes-BLAS_BYTES-FFT_BYTES,BLAS_BYTES,FFT_BYTES,0,0};return 0;
}
extern "C" int32_t cuteafd_audio_destroy(void* opaque) {
  if(!opaque)return 0;
  auto* o=static_cast<Owner*>(opaque);
  int previous=-1;cudaGetDevice(&previous);int first=cudaSetDevice(o->device);
  if(o->stream)first=drain(o,first);
  if(o->support) { int e=cuteafd_audio_support_destroy(o->support);if(!first)first=e; }
  if(o->fft) { int e=fs(cufftDestroy(o->fft));if(!first)first=e; }
  if(o->blas) { int e=bs(cublasDestroy(o->blas));if(!first)first=e; }
  if(o->weights) { int e=cudaFree(o->weights);if(!first)first=e; }
  if(o->arena) { int e=cudaFree(o->arena);if(!first)first=e; }
  if(o->stream) { int e=cudaStreamDestroy(o->stream);if(!first)first=e; }
  if(previous>=0 && previous!=o->device)cudaSetDevice(previous);
  delete o;return first;
}
extern "C" int32_t cuteafd_audio_create(const cuteafd_audio_spec* s,int32_t device,uint64_t admitted,void** out) {
  if(!out)return cudaErrorInvalidValue;*out=nullptr;
  cuteafd_audio_ledger ledger{};int e=cuteafd_audio_required(s,&ledger);if(e)return e;
  if(admitted<ledger.weights+ledger.scratch+ledger.blas_workspace+ledger.fft_workspace)return cudaErrorMemoryAllocation;
  e=cudaSetDevice(device);if(e)return e;
  int major=0,minor=0;
  e=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);if(e)return e;
  e=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);if(e)return e;
  if(major*10+minor!=CUTEAFD_AUDIO_AOT_SM)return cudaErrorInvalidDevice;
  auto* o=new(std::nothrow)Owner;if(!o)return cudaErrorMemoryAllocation;
  o->s=*s;o->device=device;o->ledger=ledger;
  auto fail=[&](int rc){cuteafd_audio_destroy(o);return rc;};
  int low,high;e=cudaDeviceGetStreamPriorityRange(&low,&high);if(e)return fail(e);
  e=cudaStreamCreateWithPriority(&o->stream,cudaStreamNonBlocking,low);if(e)return fail(e);
  e=cudaMalloc(reinterpret_cast<void**>(&o->weights),ledger.weights);if(e)return fail(e);++o->ledger.device_allocations;
  e=cudaMalloc(reinterpret_cast<void**>(&o->arena),ledger.scratch+BLAS_BYTES+FFT_BYTES);if(e)return fail(e);++o->ledger.device_allocations;
  scratch(o);
  e=cuteafd_audio_support_create(&o->support);if(e)return fail(e);
  e=bs(cublasCreate(&o->blas));if(e)return fail(e);
  e=bs(cublasSetMathMode(o->blas,CUBLAS_PEDANTIC_MATH));if(e)return fail(e);
  e=bs(cublasSetAtomicsMode(o->blas,CUBLAS_ATOMICS_NOT_ALLOWED));if(e)return fail(e);
  e=bs(cublasSetStream(o->blas,o->stream));if(e)return fail(e);
  e=bs(cublasSetWorkspace(o->blas,o->blas_workspace,BLAS_BYTES));if(e)return fail(e);
  e=fs(cufftCreate(&o->fft));if(e)return fail(e);
  e=fs(cufftSetAutoAllocation(o->fft,0));if(e)return fail(e);
  int n=960;size_t workspace=0;
  e=fs(cufftMakePlanMany(o->fft,1,&n,nullptr,1,960,nullptr,1,481,CUFFT_R2C,o->mel_capacity,&workspace));if(e)return fail(e);
  if(workspace>FFT_BYTES)return fail(cudaErrorMemoryAllocation);
  e=fs(cufftSetWorkArea(o->fft,o->fft_workspace));if(e)return fail(e);
  e=fs(cufftSetStream(o->fft,o->stream));if(e)return fail(e);
  *out=o;return 0;
}
extern "C" int32_t cuteafd_audio_upload(void* opaque,const void* weights,uint64_t bytes,
    const float* hann,const float* filter,const float* codec_rope,const float* patch_rope) {
  if(!opaque || !weights)return cudaErrorInvalidValue;
  if((hann || filter || codec_rope || patch_rope) && !(hann && filter && codec_rope && patch_rope))return cudaErrorInvalidValue;
  // Null table pointers select the attested offline constants. Non-null tables
  // remain a qualification-only oracle-swap path, never a serving sidecar.
  if(!hann) {
    hann=cuteafd_audio_hann;filter=cuteafd_audio_filterbank;
    codec_rope=cuteafd_audio_codec_rotary;patch_rope=cuteafd_audio_patch_rotary;
  }
  auto* o=static_cast<Owner*>(opaque);
  if(bytes!=o->s.weight_bytes || o->uploaded)return cudaErrorInvalidValue;
  int device=-1;auto e=cudaGetDevice(&device);if(e)return e;if(device!=o->device)return cudaErrorInvalidDevice;
  auto copy=[&](void* dst,const void* src,size_t n){return int(cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice,o->stream));};
  int rc=copy(o->weights,weights,bytes);
  if(!rc)rc=copy(o->hann,hann,960*4);
  if(!rc)rc=copy(o->filter,filter,481*128*4);
  if(!rc)rc=copy(o->codec_rope,codec_rope,size_t(o->codec_capacity)*64*4);
  if(!rc)rc=copy(o->patch_rope,patch_rope,4*64*4);
  for(int book=0;book<20 && !rc;++book)rc=support(o,AUDIO_SUM_SQUARE,1024,w(o,o->s.codebooks[book]),nullptr,nullptr,o->book_square+book*1024,nullptr,nullptr,BOOKS[book]);
  rc=drain(o,rc);o->uploaded=rc==0;return rc;
}
extern "C" int32_t cuteafd_audio_encode(void* opaque,const float* pcm,uint32_t samples,
    float* output,uint64_t output_bytes,int32_t* codes,uint64_t code_bytes,cuteafd_audio_observer cb,void* ctx) {
  if(!opaque || !pcm || !output)return cudaErrorInvalidValue;
  auto* o=static_cast<Owner*>(opaque);
  if(!o->uploaded || samples<=480 || samples>o->s.max_samples)return cudaErrorInvalidValue;
  int count=total_codes(samples),tokens=(count+3)/4;
  if(output_bytes!=uint64_t(tokens)*o->s.output_width*4 ||
      (codes && code_bytes!=uint64_t(count)*20*4) || (!codes && code_bytes))return cudaErrorInvalidValue;
  for(uint32_t i=0;i<samples;++i)if(!std::isfinite(pcm[i]))return cudaErrorInvalidValue;
  int device=-1;auto status=cudaGetDevice(&device);if(status)return status;
  if(device!=o->device)return cudaErrorInvalidDevice;
  int e=cudaMemcpyAsync(o->pcm,pcm,size_t(samples)*4,cudaMemcpyHostToDevice,o->stream);
  int remaining=samples/240+1,global=0,offset=0;
  while(remaining>0 && !e) {
    int frames=std::min(6000,remaining);
    e=encode_segment(o,samples,frames,global,offset,cb,ctx);
    remaining-=frames;global+=frames;offset+=codes_for(frames);
  }
  if(!e)e=encode_patch(o,count,cb,ctx);
  if(!e)e=cudaMemcpyAsync(output,o->output,output_bytes,cudaMemcpyDeviceToHost,o->stream);
  if(!e && codes)e=cudaMemcpyAsync(codes,o->codes,code_bytes,cudaMemcpyDeviceToHost,o->stream);
  e=drain(o,e);if(!e)++o->ledger.encodes;return e;
}
extern "C" int32_t cuteafd_audio_get_ledger(void* opaque,cuteafd_audio_ledger* out) {
  if(!opaque || !out)return cudaErrorInvalidValue;*out=static_cast<Owner*>(opaque)->ledger;return 0;
}
extern "C" int32_t cuteafd_audio_backend(char* out,uint64_t capacity) {
  if(!out || capacity<128)return cudaErrorInvalidValue;
  int fft=0;auto e=cufftGetVersion(&fft);if(e)return fs(e);
  int size=std::snprintf(out,capacity,"mimo_audio_fp32_v1/cuda%d/cufft%d/cublas%d.%d.%d/cute_aot_sm%d/export%s",
      CUDART_VERSION,fft,CUBLAS_VER_MAJOR,CUBLAS_VER_MINOR,CUBLAS_VER_PATCH,CUTEAFD_AUDIO_AOT_SM,CUTEAFD_AUDIO_EXPORT_SHA256);
  return size<0 || uint64_t(size)>=capacity ? cudaErrorInvalidValue : 0;
}
