// CPU-only admission test; link with vision_runtime.o, cuBLAS and cudart.
#include "cuteafd_vision.h"
#include "cuteafd_vision_attention_internal.h"
#include <cstdio>
#include <cstdlib>

// Admission must never reach an attention owner or device launch.
extern "C" int32_t cuteafd_vision_attention_create(int32_t,void**) { std::abort(); }
extern "C" int32_t cuteafd_vision_attention_destroy(void*) { std::abort(); }
extern "C" int32_t cuteafd_vision_attention_launch(void*,void*,void*,void*,void*,void*,void*,float,void*) { std::abort(); }

int main() {
  cuteafd_vision_spec spec{};
  spec.abi_version=3;spec.reserved=4;spec.max_tokens=4096;spec.output_width=6144;
  spec.hidden=1280;spec.depth=32;spec.heads=16;spec.kv_heads=16;spec.head_dim=80;
  spec.intermediate=5120;spec.patch_size=14;spec.merger_width=24576;spec.norm_eps=1e-5f;
  uint64_t cursor=0;
  auto put=[&](uint64_t bytes) { uint64_t offset=cursor;cursor+=(bytes+255)&~uint64_t(255);return offset; };
  spec.patch=put(1280ULL*1176*4);
  spec.merger_norm=put(1280*4);spec.merger_norm_bias=put(1280*4);spec.inv_freq=put(13*4);
  for(int i=0;i<32;++i) {
    auto& b=i<28?spec.blocks[i]:spec.blocks_extra[i-28];
    b.qkv=put(3840ULL*1280*2);b.qkv_bias=put(3840*4);
    b.proj=put(1280ULL*1280*2);b.proj_bias=put(1280*4);
    b.gate_up=put(5120ULL*1280*2);b.gate_up_bias=put(5120*4);
    b.down=put(1280ULL*5120*2);b.down_bias=put(1280*4);
    b.norm1=put(1280*4);b.norm2=put(1280*4);b.key0_bias=CUTEAFD_VISION_NO_OFFSET;
    (i<28?spec.norm1_bias[i]:spec.norm1_bias_extra[i-28])=put(1280*4);
    (i<28?spec.norm2_bias[i]:spec.norm2_bias_extra[i-28])=put(1280*4);
  }
  spec.merger_extra[0]=put(6144ULL*1280*2);spec.merger_extra[1]=put(6144*4);
  spec.merger_extra[2]=put(6144ULL*6144*2);spec.merger_extra[3]=put(6144*4);
  spec.merger_fc1=put(6144ULL*24576*2);spec.merger_fc1_bias=put(6144*4);
  spec.merger_fc2=put(6144ULL*6144*2);spec.merger_fc2_bias=put(6144*4);
  spec.weight_bytes=cursor;
  unsigned checks=0;
  auto check=[&](bool result,const char* name) {
    ++checks;if(!result) { std::fprintf(stderr,"FAIL: %s\n",name);std::exit(1); }
  };
  cuteafd_vision_ledger ledger{};
  check(cuteafd_vision_required(&spec,&ledger)==0,"valid MiniMax header");
  check(ledger.weights==cursor,"resident weight bytes");
  check(ledger.scratch==758189312,"4096-token scratch bytes");
  check(ledger.blas_workspace==4194304,"BLAS workspace bytes");
  check(ledger.device_allocations==0 && ledger.encodes==0,"admission does not allocate or encode");
  auto reject=[&](cuteafd_vision_spec candidate,const char* name) {
    check(cuteafd_vision_required(&candidate,&ledger)!=0,name);
  };
  auto candidate=spec;candidate.head_dim=78;reject(candidate,"rotary width is not attention width");
  candidate=spec;candidate.depth=28;reject(candidate,"all32 blocks required");
  candidate=spec;candidate.max_tokens=4097;reject(candidate,"attention capacity");
  candidate=spec;candidate.norm_eps=1e-6f;reject(candidate,"pre-LN epsilon");
  candidate=spec;candidate.patch=candidate.weight_bytes-1280ULL*1176*2;
  reject(candidate,"convolution requires FP32 extent");
  candidate=spec;candidate.merger_fc1=candidate.weight_bytes-6144ULL*6144*2;
  reject(candidate,"merge projection requires four patches");
  for(int i=0;i<4;++i) {
    candidate=spec;candidate.blocks_extra[i].qkv=CUTEAFD_VISION_NO_OFFSET;
    reject(candidate,"suffix block matrix extent");
    candidate=spec;candidate.norm1_bias_extra[i]=candidate.weight_bytes;
    reject(candidate,"suffix layer norm bias extent");
  }
  candidate=spec;candidate.blocks[0].window=1;reject(candidate,"full attention only");
  candidate=spec;candidate.blocks_extra[3].column_order=1;reject(candidate,"merge-window order");
  candidate=spec;candidate.merger_norm_bias+=1;reject(candidate,"aligned pre-LN bias");
  candidate=spec;candidate.weight_bytes-=256;reject(candidate,"truncated weight arena");
  check(cuteafd_vision_required(nullptr,&ledger)!=0,"null spec");
  check(cuteafd_vision_required(&spec,nullptr)!=0,"null ledger");
  std::printf("PASS: %u CPU MiniMax admission checks\n",checks);
  return 0;
}
