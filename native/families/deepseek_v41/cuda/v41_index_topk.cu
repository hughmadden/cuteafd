#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "cuteafd_v41_index_topk.h"
namespace {
bool valid(const void* p,uint64_t n,uint64_t alignment) {
  const auto a=reinterpret_cast<uintptr_t>(p);
  return a && a%alignment==0 && a<=UINTPTR_MAX-n;
}
bool disjoint(const void* a,uint64_t na,const void* b,uint64_t nb) {
  const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
  return x+na<=y || y+nb<=x;
}
template<uint64_t Limit>
__device__ uint64_t encode(float score,uint64_t pos) {
  if(pos>=Limit || isnan(score) || score==-CUDART_INF_F)return 0;
  // Treat both signed zeros as equal scores.
  const uint32_t bits=score==0.f?0u:__float_as_uint(score);
  const uint32_t ordered=bits&0x80000000u?~bits:bits^0x80000000u;
  return (uint64_t(ordered)<<32)|uint32_t(~uint32_t(pos));
}
template<int K> using Sort=cub::BlockRadixSort<uint64_t,256,K/128>;
template<int K,uint64_t Limit>
__global__ void tiles(const float* scores,const uint64_t* positions,uint64_t* out,int count,int blocks) {
  __shared__ typename Sort<K>::TempStorage storage;
  uint64_t keys[K/128];const uint64_t row=blockIdx.y;
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int col=blockIdx.x*(2*K)+threadIdx.x*(K/128)+i;
    keys[i]=col<count?encode<Limit>(scores[row*count+col],positions[row*count+col]):0;
  }
  Sort<K>(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int rank=threadIdx.x*(K/128)+i;
    if(rank<K)out[(row*blocks+blockIdx.x)*K+rank]=keys[i];
  }
}
template<int K>
__global__ void merge(const uint64_t* in,uint64_t* out,int in_blocks,int out_blocks) {
  __shared__ typename Sort<K>::TempStorage storage;
  uint64_t keys[K/128];const uint64_t row=blockIdx.y;
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int col=blockIdx.x*(2*K)+threadIdx.x*(K/128)+i;
    keys[i]=col<in_blocks*K?in[row*in_blocks*K+col]:0;
  }
  Sort<K>(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int rank=threadIdx.x*(K/128)+i;
    if(rank<K)out[(row*out_blocks+blockIdx.x)*K+rank]=keys[i];
  }
}
template<int K,uint64_t Limit>
__global__ void finish(const uint64_t* selected,uint64_t* carry,int32_t* output,int reset) {
  __shared__ typename Sort<K>::TempStorage storage;
  uint64_t keys[K/128];const uint64_t row=blockIdx.x;
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int col=threadIdx.x*(K/128)+i;
    keys[i]=col<K?selected[row*K+col]:(reset?0:carry[row*K+col-K]);
  }
  Sort<K>(storage).SortDescending(keys);
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int rank=threadIdx.x*(K/128)+i;
    if(rank<K)carry[row*K+rank]=keys[i];
    // Rank-sort the selected logical positions with the same storage. Unselected
    // entries are UINT64_MAX, so only the first K outputs need publishing.
    keys[i]=rank<K && keys[i]!=0?uint32_t(~uint32_t(keys[i])):UINT64_MAX;
  }
  __syncthreads();
  // Logical positions use only 20 (token) or 17 (block) bits. Include
  // one sentinel bit so UINT64_MAX sorts after every valid position.
  static_assert(Limit==1048576 || Limit==131072);
  Sort<K>(storage).Sort(keys,0,Limit==1048576?21:18);
  #pragma unroll
  for(int i=0;i<K/128;++i) {
    const int rank=threadIdx.x*(K/128)+i;
    if(rank<K)output[row*K+rank]=keys[i]==UINT64_MAX?-1:int32_t(keys[i]);
  }
}
// Radix select: one CTA per query row finds the K-th largest key of
// candidates + carry (zero keys pad, as the merge path's zero fill does) and
// keeps every larger key plus enough copies of the K-th to reach K. Keys are
// unique apart from zero, so the selected set is the merge path's and the
// output is bit-identical. Selected keys are compacted in column order; when
// the candidates' positions ascend (tile and expand produce them so) and the
// carry is this path's own (kept in output order), one merge orders them,
// otherwise a block radix sort does. Both paths read a carry as a set, so
// either accepts the other's.
constexpr int kSelectThreads=1024,kActiveCap=2048;
#ifdef CUTEAFD_V41_TOPK_PHASES
__device__ unsigned long long topk_phase[32];
#define TOPK_PHASE(i) do {if(blockIdx.x==0 && threadIdx.x==0)topk_phase[i]=clock64();} while(0)
#define TOPK_NOTE(i,v) do {if(blockIdx.x==0 && threadIdx.x==0)topk_phase[i]=(v);} while(0)
#else
#define TOPK_PHASE(i) do {} while(0)
#define TOPK_NOTE(i,v) do {} while(0)
#endif
// Position of a compacted entry: (position << 32) | ordered score, bit 63
// marking an entry that came from the carry.
__device__ __forceinline__ uint64_t shr(uint64_t x,int s) {return s>=64?0:x>>s;}
__device__ __forceinline__ uint32_t position_of(uint64_t v) {return uint32_t(v>>32)&0x7fffffffu;}
struct Partials {uint32_t any_lo[32],any_hi[32],all_lo[32],all_hi[32],n[32];};
__device__ __forceinline__ void warp_fold(uint64_t& any,uint64_t& all,uint32_t& n) {
#if __CUDA_ARCH__>=800
  any=(uint64_t(__reduce_or_sync(0xffffffffu,uint32_t(any>>32)))<<32)|__reduce_or_sync(0xffffffffu,uint32_t(any));
  all=(uint64_t(__reduce_and_sync(0xffffffffu,uint32_t(all>>32)))<<32)|__reduce_and_sync(0xffffffffu,uint32_t(all));
  n=__reduce_add_sync(0xffffffffu,n);
#else
  #pragma unroll
  for(int d=16;d;d>>=1) {
    any|=__shfl_xor_sync(0xffffffffu,any,d);
    all&=__shfl_xor_sync(0xffffffffu,all,d);
    n+=__shfl_xor_sync(0xffffffffu,n,d);
  }
#endif
}
// Every warp folds the warp partials itself, so no broadcast barrier
// follows. Callers separate two folds by at least one barrier.
template<int Warps>
__device__ __forceinline__ void block_fold(uint64_t& any,uint64_t& all,uint32_t& n,Partials& p,int lane,int warp) {
  warp_fold(any,all,n);
  if(lane==0) {
    p.any_lo[warp]=uint32_t(any);p.any_hi[warp]=uint32_t(any>>32);
    p.all_lo[warp]=uint32_t(all);p.all_hi[warp]=uint32_t(all>>32);p.n[warp]=n;
  }
  __syncthreads();
  any=0;all=~0ull;n=0;
  if(lane<Warps) {
    any=(uint64_t(p.any_hi[lane])<<32)|p.any_lo[lane];
    all=(uint64_t(p.all_hi[lane])<<32)|p.all_lo[lane];
    n=p.n[lane];
  }
  warp_fold(any,all,n);
}
// The bin holding the need-th largest key of a 256-bin histogram (each lane
// owns 8 bins, highest first), the count above it and its size.
__device__ __forceinline__ void find_bin(const uint32_t* histogram,uint32_t need,int lane,
    int& bin,uint32_t& above,uint32_t& size) {
  const uint4 upper=*reinterpret_cast<const uint4*>(&histogram[252-8*lane]);
  const uint4 lower=*reinterpret_cast<const uint4*>(&histogram[248-8*lane]);
  const uint32_t bins[8]={upper.w,upper.z,upper.y,upper.x,lower.w,lower.z,lower.y,lower.x};
  uint32_t sum=0;
  #pragma unroll
  for(int j=0;j<8;++j)sum+=bins[j];
  uint32_t inclusive=sum;
  #pragma unroll
  for(int d=1;d<32;d<<=1) {
    const uint32_t up=__shfl_up_sync(0xffffffffu,inclusive,d);
    if(lane>=d)inclusive+=up;
  }
  const int owner=__ffs(__ballot_sync(0xffffffffu,inclusive>=need))-1;
  above=inclusive-sum;size=0;bin=0;bool found=false;
  #pragma unroll
  for(int j=0;j<8;++j) {
    if(!found && above+bins[j]>=need) {found=true;bin=255-8*lane-j;size=bins[j];}
    else if(!found) above+=bins[j];
  }
  above=__shfl_sync(0xffffffffu,above,owner);
  size=__shfl_sync(0xffffffffu,size,owner);
  bin=__shfl_sync(0xffffffffu,bin,owner);
}
// Entries of a[lo,hi) (ascending positions) below p, or at most p.
__device__ __forceinline__ uint32_t rank_in(const uint64_t* a,uint32_t lo,uint32_t hi,uint32_t p,bool inclusive) {
  uint32_t first=lo,n=hi-lo;
  while(n) {
    const uint32_t half=n/2,q=position_of(a[first+half]);
    if(q<p || (inclusive && q==p)) {first+=half+1;n-=half+1;}
    else n=half;
  }
  return first-lo;
}
// Items keys per thread stay in registers: 1024 threads hold up to 9 each
// within their 64-register budget, the widest calls run 512 threads of 36.
template<int K,uint64_t Limit,int Items,int Threads>
__global__ void __launch_bounds__(Threads,1) radix_select(const float* scores,const uint64_t* positions,
    uint64_t* carry,int32_t* output,int count,int reset) {
  constexpr int Warps=Threads/32,SortItems=(K+Threads-1)/Threads,Segments=Items*Warps;
  constexpr int PerThread=(Segments+Threads-1)/Threads;
  constexpr int PositionBits=Limit==1048576?21:18;
  static_assert(Limit==1048576 || Limit==131072);
  static_assert(Threads%32==0 && Threads>=256 && Warps<=32);
  using RowSort=cub::BlockRadixSort<uint64_t,Threads,SortItems>;
  using Scan=cub::BlockScan<uint32_t,Threads>;
  __shared__ union {
    typename RowSort::TempStorage sort;
    struct {
      typename Scan::TempStorage scan;
      uint64_t selected[K];
      uint64_t active[kActiveCap];
      uint32_t segments[Segments];
      alignas(16) uint32_t histogram[2][256];
      Partials partials;
      uint32_t gathered;
    } select;
  } shared;
  const int tid=threadIdx.x,lane=tid%32,warp=tid/32;
  const uint64_t row=blockIdx.x;
  // Item i is column i*Threads+tid: a candidate below count, then carry
  // entries. Comparing i*Threads with these per-thread bounds keeps no
  // per-item column live past the loads.
  const int fresh=count-tid,filled=count+(reset?0:K)-tid;
  TOPK_PHASE(0);
  uint64_t keys[Items];
  uint64_t any=0,all=~0ull;uint32_t valid=0;
  {
    const float* row_scores=scores+row*count+tid;
    const uint64_t* row_positions=positions+row*count+tid;
    const uint64_t* row_carry=carry+row*K;
    #pragma unroll
    for(int i=0;i<Items;++i) {
      uint64_t key=0;
      if(i*Threads<fresh)key=encode<Limit>(row_scores[i*Threads],row_positions[i*Threads]);
      else if(i*Threads<filled)key=row_carry[i*Threads-fresh];
      keys[i]=key;
      if(key) {any|=key;all&=key;++valid;}
    }
  }
  if(tid<256)shared.select.histogram[0][tid]=0;
  if(tid==0)shared.select.gathered=0;
  block_fold<Warps>(any,all,valid,shared.select.partials,lane,warp);
  TOPK_PHASE(1);
  // Selection: every key above the bin [lo, hi], plus the first `need` keys in
  // it in column order. Those ties are either all taken or identical keys, so
  // which copies are taken never changes the output.
  uint64_t lo=0,hi=0;uint32_t need=0;
  if(valid>uint32_t(K)) {
    // MSB-first 8-bit digits over the bits where the active keys still
    // differ: constant runs (BF16 score tails, high position bits) cost nothing.
    // The active bin is keys k with (k>>shift)==prefix; the bounds are formed
    // once at the end (both forms keep the widest variants spill-free).
    int shift=64;uint64_t prefix=0;
    need=K;
    int active=-1;  // keys gathered to shared memory once the threshold bin fits
    for(int pass=0;;++pass) {
      const uint64_t diff=any^all;
      if(!diff) {shift=0;prefix=all;break;}
      const int high=63-__clzll(diff),low=high>7?high-7:0,width=high-low+1;
      const uint32_t mask=(1u<<width)-1;
      uint32_t* histogram=shared.select.histogram[pass&1];
      if(active<0) {
        #pragma unroll
        for(int i=0;i<Items;++i) {
          const uint64_t key=keys[i];
          if(key && shr(key,shift)==prefix)atomicAdd(&histogram[uint32_t(key>>low)&mask],1u);
        }
      } else {
        for(int s=tid;s<active;s+=Threads) {
          const uint64_t key=shared.select.active[s];
          if(shr(key,shift)==prefix)atomicAdd(&histogram[uint32_t(key>>low)&mask],1u);
        }
      }
      __syncthreads();
      int bin;uint32_t above,size;
      find_bin(histogram,need,lane,bin,above,size);
      // Bits above `high` are common to the active keys; the bin fixes the rest down to `low`.
      prefix=(high==63?0:(all>>(high+1))<<width)|uint64_t(bin);
      shift=low;need-=above;
      TOPK_PHASE(2+(pass<9?pass:9));
      TOPK_NOTE(20,pass+1);
      if(size==need || low==0)break;
      // The next active set's common bits; histogram[next] was last read
      // before the barrier above.
      if(tid<256)shared.select.histogram[(pass+1)&1][tid]=0;
      any=0;all=~0ull;
      if(active<0 && size<=uint32_t(kActiveCap)) {
        // Gather the threshold bin once it fits: later passes read only it.
        #pragma unroll
        for(int i=0;i<Items;++i) {
          const uint64_t key=keys[i];
          const bool hit=key && shr(key,shift)==prefix;
          const uint32_t m=__ballot_sync(0xffffffffu,hit);
          if(m) {
            const int leader=__ffs(m)-1;
            uint32_t base=0;
            if(lane==leader)base=atomicAdd(&shared.select.gathered,uint32_t(__popc(m)));
            base=__shfl_sync(0xffffffffu,base,leader);
            if(hit) {shared.select.active[base+__popc(m&((1u<<lane)-1))]=key;any|=key;all&=key;}
          }
        }
        active=int(size);
        TOPK_NOTE(21,size);
      } else if(active<0) {
        #pragma unroll
        for(int i=0;i<Items;++i) {
          const uint64_t key=keys[i];
          if(key && shr(key,shift)==prefix) {any|=key;all&=key;}
        }
      } else {
        for(int s=tid;s<active;s+=Threads) {
          const uint64_t key=shared.select.active[s];
          if(shr(key,shift)==prefix) {any|=key;all&=key;}
        }
      }
      uint32_t unused=0;
      block_fold<Warps>(any,all,unused,shared.select.partials,lane,warp);
    }
    lo=shift>=64?0:prefix<<shift;hi=shift>=64?~0ull:lo|((uint64_t(1)<<shift)-1);
  }
  // Column order (col = i*Threads + tid) is item-major, then warp, then lane:
  // count each (item, warp) segment, scan the segments, then place.
  #pragma unroll
  for(int i=0;i<Items;++i) {
    const uint64_t key=keys[i];
    const uint32_t up=__popc(__ballot_sync(0xffffffffu,key>hi));
    const uint32_t tie=__popc(__ballot_sync(0xffffffffu,key && key>=lo && key<=hi));
    if(lane==0)shared.select.segments[i*Warps+warp]=up|(tie<<16);
  }
  __syncthreads();
  uint32_t segment[PerThread],totals;
  #pragma unroll
  for(int j=0;j<PerThread;++j)segment[j]=tid*PerThread+j<Segments?shared.select.segments[tid*PerThread+j]:0;
  Scan(shared.select.scan).ExclusiveSum(segment,segment,totals);
  #pragma unroll
  for(int j=0;j<PerThread;++j)if(tid*PerThread+j<Segments)shared.select.segments[tid*PerThread+j]=segment[j];
  __syncthreads();
  TOPK_PHASE(12);
  const uint32_t taken=totals&65535u,ties=totals>>16,selected=taken+(ties<need?ties:need);
  const uint32_t before=(1u<<lane)-1;
  #pragma unroll
  for(int i=0;i<Items;++i) {
    const uint64_t key=keys[i];
    const bool up=key>hi,tie=key && key>=lo && key<=hi;
    const uint32_t ups=__ballot_sync(0xffffffffu,up),tied=__ballot_sync(0xffffffffu,tie);
    const uint32_t offset=shared.select.segments[i*Warps+warp];
    const uint32_t a=(offset&65535u)+__popc(ups&before),t=(offset>>16)+__popc(tied&before);
    uint32_t slot=K;
    if(up)slot=a+(t<need?t:need);
    else if(tie && t<need)slot=a+t;
    if(slot<uint32_t(K))shared.select.selected[slot]=(uint64_t(i*Threads>=fresh)<<63)|
        (uint64_t(~uint32_t(key))<<32)|(key>>32);
  }
  for(uint32_t s=selected+tid;s<uint32_t(K);s+=Threads)shared.select.selected[s]=UINT64_MAX;
  __syncthreads();
  TOPK_PHASE(13);
  // Candidates' selections precede the carry's; each run must ascend.
  uint32_t candidates=0;bool ordered=true;
  #pragma unroll
  for(int j=0;j<SortItems;++j) {
    const uint32_t s=j*Threads+tid;
    bool candidate=false;
    if(s<selected) {
      const uint64_t v=shared.select.selected[s];
      candidate=!(v>>63);
      if(s+1<selected) {
        const uint64_t w=shared.select.selected[s+1];
        if((v>>63)==(w>>63) && position_of(v)>=position_of(w))ordered=false;
      }
    }
    candidates+=__syncthreads_count(candidate);
  }
  ordered=__syncthreads_and(ordered);
  TOPK_NOTE(22,ordered);
  if(ordered) {
    #pragma unroll
    for(int j=0;j<SortItems;++j) {
      const uint32_t s=j*Threads+tid;
      if(s<selected) {
        const uint64_t v=shared.select.selected[s];const uint32_t p=position_of(v);
        const uint32_t rank=s<candidates?s+rank_in(shared.select.selected,candidates,selected,p,false):
            s-candidates+rank_in(shared.select.selected,0,candidates,p,true);
        output[row*K+rank]=int32_t(p);
        carry[row*K+rank]=(uint64_t(uint32_t(v))<<32)|uint32_t(~p);
      } else if(s<uint32_t(K)) {
        output[row*K+s]=-1;carry[row*K+s]=0;
      }
    }
    TOPK_PHASE(14);
  } else {
    uint64_t sorted[SortItems];
    #pragma unroll
    for(int j=0;j<SortItems;++j) {
      const int s=tid*SortItems+j;
      sorted[j]=s<K?shared.select.selected[s]:UINT64_MAX;
    }
    __syncthreads();
    // Positions use 20 (token) or 17 (block) bits plus the sentinel bit.
    RowSort(shared.sort).SortBlockedToStriped(sorted,32,32+PositionBits);
    TOPK_PHASE(14);
    #pragma unroll
    for(int j=0;j<SortItems;++j) {
      const int rank=j*Threads+tid;
      if(rank<K) {
        const uint64_t v=sorted[j];const uint32_t p=position_of(v);
        output[row*K+rank]=v==UINT64_MAX?-1:int32_t(p);
        carry[row*K+rank]=v==UINT64_MAX?0:(uint64_t(uint32_t(v))<<32)|uint32_t(~p);
      }
    }
  }
  TOPK_PHASE(15);
}
template<int K,uint64_t Limit,int Items>
cudaError_t launch_radix(const float* scores,const uint64_t* positions,uint64_t* carry,int32_t* output,
    int queries,int candidates,int reset,cudaStream_t stream) {
  constexpr int Threads=Items>9?512:kSelectThreads;
  static_assert(Items<=9 || 16384+K<=Items*512);
  radix_select<K,Limit,Items,Threads><<<queries,Threads,0,stream>>>(scores,positions,carry,output,candidates,reset);
  return cudaGetLastError();
}
template<int K,uint64_t Limit>
cudaError_t select_radix(const float* scores,const uint64_t* positions,uint64_t* carry,int32_t* output,
    int queries,int candidates,int reset,cudaStream_t stream) {
  const int total=candidates+(reset?0:K);
  constexpr int Widest=(16384+K+511)/512;  // 33 for K=512, 36 for K=2048
  if(total<=2*kSelectThreads)return launch_radix<K,Limit,2>(scores,positions,carry,output,queries,candidates,reset,stream);
  if(total<=5*kSelectThreads)return launch_radix<K,Limit,5>(scores,positions,carry,output,queries,candidates,reset,stream);
  if(total<=9*kSelectThreads)return launch_radix<K,Limit,9>(scores,positions,carry,output,queries,candidates,reset,stream);
  return launch_radix<K,Limit,Widest>(scores,positions,carry,output,queries,candidates,reset,stream);
}
// CUTEAFD_V41_INDEX_TOPK=merge selects the original sort-and-merge path (A/B).
bool merge_path() {
  static const bool merge=[] {
    const char* v=getenv("CUTEAFD_V41_INDEX_TOPK");
    return v && strcmp(v,"merge")==0;
  }();
  return merge;
}
}
template<int K,uint64_t Limit>
static int32_t select(const float* scores,const uint64_t* positions,
    uint64_t* carry,void* scratch,uint64_t scratch_bytes,int32_t* output,
    int32_t queries,int32_t candidates,int32_t reset,void* stream,int path=-1) {
  if(queries<1 || queries>4096 || candidates<1 || candidates>16384 || (reset!=0 && reset!=1))return cudaErrorInvalidValue;
  const int blocks=(candidates+2*K-1)/(2*K);
  const uint64_t count=uint64_t(queries)*candidates,half=uint64_t(queries)*blocks*K;
  const uint64_t bytes[]={count*4,count*8,uint64_t(queries)*K*8,half*16,uint64_t(queries)*K*4};
  const void* ptrs[]={scores,positions,carry,scratch,output};
  const int alignment[]={4,8,8,8,4};
  if(scratch_bytes<bytes[3])return cudaErrorInvalidValue;
  for(int i=0;i<5;++i) {
    if(!valid(ptrs[i],bytes[i],alignment[i]))return cudaErrorInvalidValue;
    for(int j=0;j<i;++j)if(!disjoint(ptrs[i],bytes[i],ptrs[j],bytes[j]))return cudaErrorInvalidValue;
  }
  auto cuda_stream=reinterpret_cast<cudaStream_t>(stream);
  if(path<0)path=merge_path()?0:1;
  if(path==1)return select_radix<K,Limit>(scores,positions,carry,output,queries,candidates,reset,cuda_stream);
  auto* a=static_cast<uint64_t*>(scratch);auto* b=a+half;
  tiles<K,Limit><<<dim3(blocks,queries),256,0,cuda_stream>>>(scores,positions,a,candidates,blocks);
  auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
  for(int current=blocks;current>1;) {
    const int next=(current+1)/2;
    merge<K><<<dim3(next,queries),256,0,cuda_stream>>>(a,b,current,next);
    status=cudaGetLastError();if(status!=cudaSuccess)return status;
    auto* tmp=a;a=b;b=tmp;current=next;
  }
  finish<K,Limit><<<queries,256,0,cuda_stream>>>(a,carry,output,reset);
  return cudaGetLastError();
}

extern "C" int32_t cuteafd_v41_index_top512(const float* scores,const uint64_t* positions,
    uint64_t* carry,void* scratch,uint64_t scratch_bytes,int32_t* output,
    int32_t queries,int32_t candidates,int32_t reset,void* stream) {
  return select<512,1048576>(scores,positions,carry,scratch,scratch_bytes,output,queries,candidates,reset,stream);
}
extern "C" int32_t cuteafd_v41_index_top2048_blocks(const float* scores,const uint64_t* positions,
    uint64_t* carry,void* scratch,uint64_t scratch_bytes,int32_t* output,
    int32_t queries,int32_t candidates,int32_t reset,void* stream) {
  return select<2048,131072>(scores,positions,carry,scratch,scratch_bytes,output,queries,candidates,reset,stream);
}
