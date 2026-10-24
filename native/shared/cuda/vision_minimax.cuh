// Official MiniMax tower: FP32 Conv3d, BF16 CLIP blocks and two GELU projectors.
constexpr int MM_HIDDEN=1280, MM_INTER=5120, MM_PATCH=1176, MM_OUTPUT=6144;
constexpr int MM_CHUNK=1024;

bool valid_minimax(const cuteafd_vision_spec& s) {
  if(s.abi_version!=3 || s.reserved!=4 || s.geometry_reserved || !s.max_tokens || s.max_tokens>4096 ||
     s.hidden!=1280 || s.depth!=32 || s.heads!=16 || s.kv_heads!=16 || s.head_dim!=80 ||
     s.intermediate!=5120 || s.patch_size!=14 || s.merger_width!=24576 || s.output_width!=6144 ||
     s.norm_eps!=1e-5f || !s.weight_bytes || s.weight_bytes>(2ULL<<30))return false;
  const uint64_t offsets[]={s.patch,s.merger_norm,s.merger_norm_bias,s.inv_freq,
    s.merger_extra[0],s.merger_extra[1],s.merger_extra[2],s.merger_extra[3],
    s.merger_fc1,s.merger_fc1_bias,s.merger_fc2,s.merger_fc2_bias};
  const size_t bytes[]={1280ULL*1176*4,1280*4,1280*4,13*4,
    6144ULL*1280*2,6144*4,6144ULL*6144*2,6144*4,
    6144ULL*24576*2,6144*4,6144ULL*6144*2,6144*4};
  for(int i=0;i<12;++i)if(!extent(s,offsets[i],bytes[i]))return false;
  for(int i=0;i<32;++i) {
    const auto& b=i<28?s.blocks[i]:s.blocks_extra[i-28];
    uint64_t n1=i<28?s.norm1_bias[i]:s.norm1_bias_extra[i-28];
    uint64_t n2=i<28?s.norm2_bias[i]:s.norm2_bias_extra[i-28];
    const uint64_t off[]={b.qkv,b.qkv_bias,b.proj,b.proj_bias,b.gate_up,b.gate_up_bias,
      b.down,b.down_bias,b.norm1,b.norm2,n1,n2};
    const size_t len[]={3840ULL*1280*2,3840*4,1280ULL*1280*2,1280*4,5120ULL*1280*2,
      5120*4,1280ULL*5120*2,1280*4,1280*4,1280*4,1280*4,1280*4};
    for(int j=0;j<12;++j)if(!extent(s,off[j],len[j]))return false;
    if(b.window || b.column_order || b.key0_bias!=UINT64_MAX)return false;
  }
  return true;
}

size_t scratch_minimax(Owner* o,size_t max_tokens,int output_width) {
  size_t cursor=0,n=max_tokens*4;
  auto take=[&](auto& ptr,size_t bytes) {
    using P=typename std::remove_reference<decltype(ptr)>::type;
    ptr=o->arena?reinterpret_cast<P>(o->arena+cursor):nullptr;
    cursor+=aligned(bytes);
  };
  take(o->x,n*MM_HIDDEN*4);take(o->norm,n*MM_HIDDEN*2);
  take(o->patches_f32,n*MM_PATCH*4);
  take(o->q,n*MM_HIDDEN*2);take(o->k,n*MM_HIDDEN*2);take(o->v,n*MM_HIDDEN*2);
  take(o->attn,n*MM_HIDDEN*2);take(o->output,max_tokens*output_width*2);
  take(o->projected,n*MM_OUTPUT*2);
  take(o->qkv,MM_CHUNK*MM_HIDDEN*3*4);take(o->proj,MM_CHUNK*MM_HIDDEN*4);
  take(o->gu,MM_CHUNK*MM_INTER*4);take(o->mlp,MM_CHUNK*MM_INTER*2);
  take(o->fc1,MM_CHUNK*MM_OUTPUT*4);take(o->gelu,MM_CHUNK*MM_OUTPUT*2);
  take(o->fc2,MM_CHUNK*MM_OUTPUT*4);
  take(o->rgb,n*2*14*14*3);take(o->lut,768*4);take(o->hw_row,n*3*4);
  take(o->lse,size_t(16384)*16*4);take(o->cu_seqlens,2*4);
  take(o->workspace,BLAS_BYTES);return cursor;
}

__global__ void minimax_grid(int32_t* thw,int n,int gh,int gw) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
  int spatial=p%(gh*gw),u=spatial/4,r=spatial%4;
  thw[p*3]=p/(gh*gw);
  thw[p*3+1]=(u/(gw/2))*2+r/2;
  thw[p*3+2]=(u%(gw/2))*2+r%2;
}
__global__ void minimax_patches(const uint8_t* rgb,const float* lut,float* out,
  int n,int gh,int gw,bool still) {
  size_t total=size_t(n)*MM_PATCH;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x) {
    int p=i/MM_PATCH,d=i%MM_PATCH,c=d/(2*196),temporal=(d/196)%2,pixel=d%196;
    int spatial=p%(gh*gw),u=spatial/4,r=spatial%4;
    int y=(u/(gw/2))*28+(r/2)*14+pixel/14,x=(u%(gw/2))*28+(r%2)*14+pixel%14;
    int frame=still?0:2*(p/(gh*gw))+temporal;
    out[i]=lut[c*256+rgb[((size_t(frame)*gh*14+y)*gw*14+x)*3+c]];
  }
}
__global__ void minimax_round(float* x,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
    x[i]=round_bf16(x[i]);
}
__global__ void minimax_residual(float* x,const float* y,const float* bias,size_t count) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)
    x[i]=round_bf16(x[i]+round_bf16(y[i]+bias[i%MM_HIDDEN]));
}
__global__ void minimax_rope(const float* qkv,const float* bias,const int32_t* thw,
  const float* inv,int row0,uint16_t* q,uint16_t* k,uint16_t* v) {
  int row=blockIdx.x,global=row0+row;
  const float* src=qkv+size_t(row)*3840;
  for(int i=threadIdx.x;i<MM_HIDDEN;i+=blockDim.x) {
    int d=i%80;
    float a=round_bf16(src[i]+bias[i]),b=round_bf16(src[1280+i]+bias[1280+i]);
    if(d<78) {
      int pair=d%39,other=i+(d<39?39:-39);
      float angle=float(thw[global*3+pair/13])*inv[pair%13];
      float co=round_bf16(cosf(angle)),si=round_bf16(sinf(angle)),sign=d<39?-1.f:1.f;
      float aq=round_bf16(src[other]+bias[other]);
      float bk=round_bf16(src[1280+other]+bias[1280+other]);
      // The official BF16 rotary computes two rounded products, then their sum.
      a=round_bf16(__fmul_rn(a,co))+round_bf16(__fmul_rn(sign*aq,si));
      b=round_bf16(__fmul_rn(b,co))+round_bf16(__fmul_rn(sign*bk,si));
    }
    q[size_t(global)*1280+i]=bf16_bits(a);k[size_t(global)*1280+i]=bf16_bits(b);
    v[size_t(global)*1280+i]=bf16_bits(src[2560+i]+bias[2560+i]);
  }
}

int minimax_patch_gemm(Owner* o,int rows,const float* x,float* y) {
  const float one=1,zero=0;
  return blas_status(cublasGemmEx(o->blas,CUBLAS_OP_T,CUBLAS_OP_N,MM_HIDDEN,rows,MM_PATCH,&one,
    o->weights+o->spec.patch,CUDA_R_32F,MM_PATCH,x,CUDA_R_32F,MM_PATCH,&zero,y,CUDA_R_32F,MM_HIDDEN,
    CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
}

int encode_minimax(Owner* o,const uint8_t* rgb,uint64_t rgb_bytes,const float* lut,int gt,int gh,int gw,
  bool still,uint16_t* output,uint64_t output_bytes,cuteafd_vision_observer cb,void* ctx) {
  const auto& s=o->spec;
  if(!o->uploaded || !rgb || !lut || !output || gt<1 || gt>4 || gh<2 || gw<2 || gh%2 || gw%2 ||
     (still && gt!=1))return cudaErrorInvalidValue;
  int64_t count=int64_t(gt)*gh*gw;
  if(count>int64_t(s.max_tokens)*4 || rgb_bytes!=uint64_t(count)*14*14*3*(still?1:2) ||
     output_bytes!=uint64_t(count/4)*MM_OUTPUT*2)return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  int n=int(count),units=n/4;auto stream=o->stream;
#define MM_RUN(expr) do { int rc=int(expr);if(rc)return drain(o,rc); } while(0)
#define MM_LAUNCH(expr) do { expr;MM_RUN(cudaGetLastError()); } while(0)
  MM_RUN(cudaMemcpyAsync(o->rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  MM_RUN(cudaMemcpyAsync(o->lut,lut,768*4,cudaMemcpyHostToDevice,stream));
  MM_LAUNCH((minimax_grid<<<(n+255)/256,256,0,stream>>>(o->hw_row,n,gh,gw)));
  MM_LAUNCH((minimax_patches<<<grid_for(long(n)*MM_PATCH),256,0,stream>>>(o->rgb,o->lut,o->patches_f32,n,gh,gw,still)));
  for(int r=0;r<n;r+=MM_CHUNK)MM_RUN(minimax_patch_gemm(o,std::min(MM_CHUNK,n-r),
    o->patches_f32+size_t(r)*MM_PATCH,o->x+size_t(r)*MM_HIDDEN));
  auto w=[&](uint64_t off){return reinterpret_cast<const float*>(o->weights+off);};
  MM_LAUNCH((minimax_round<<<grid_for(long(n)*MM_HIDDEN),256,0,stream>>>(o->x,size_t(n)*MM_HIDDEN)));
  MM_LAUNCH((layernorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(s.merger_norm),w(s.merger_norm_bias),o->norm,MM_HIDDEN,s.norm_eps)));
  MM_LAUNCH((qwen_cast_float<<<grid_for(long(n)*MM_HIDDEN),256,0,stream>>>(o->norm,o->x,size_t(n)*MM_HIDDEN)));
  MM_RUN(observe(o,cb,ctx,0,o->x,n,MM_HIDDEN,false));
  for(int i=0;i<32;++i) {
    const auto& b=i<28?s.blocks[i]:s.blocks_extra[i-28];
    uint64_t n1=i<28?s.norm1_bias[i]:s.norm1_bias_extra[i-28];
    uint64_t n2=i<28?s.norm2_bias[i]:s.norm2_bias_extra[i-28];
    MM_LAUNCH((layernorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(b.norm1),w(n1),o->norm,MM_HIDDEN,s.norm_eps)));
    for(int r=0;r<n;r+=MM_CHUNK) {
      int rows=std::min(MM_CHUNK,n-r);
      MM_RUN(gemm(o,o->norm+size_t(r)*MM_HIDDEN,b.qkv,rows,3840,MM_HIDDEN,o->qkv));
      MM_LAUNCH((minimax_rope<<<rows,256,0,stream>>>(o->qkv,w(b.qkv_bias),o->hw_row,w(s.inv_freq),r,o->q,o->k,o->v)));
    }
    MM_RUN(mha_attention(o,n));
    for(int r=0;r<n;r+=MM_CHUNK) {
      int rows=std::min(MM_CHUNK,n-r);
      MM_RUN(gemm(o,o->attn+size_t(r)*MM_HIDDEN,b.proj,rows,MM_HIDDEN,MM_HIDDEN,o->proj));
      MM_LAUNCH((minimax_residual<<<grid_for(long(rows)*MM_HIDDEN),256,0,stream>>>(o->x+size_t(r)*MM_HIDDEN,o->proj,w(b.proj_bias),size_t(rows)*MM_HIDDEN)));
    }
    MM_LAUNCH((layernorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(b.norm2),w(n2),o->norm,MM_HIDDEN,s.norm_eps)));
    for(int r=0;r<n;r+=MM_CHUNK) {
      int rows=std::min(MM_CHUNK,n-r);
      MM_RUN(gemm(o,o->norm+size_t(r)*MM_HIDDEN,b.gate_up,rows,MM_INTER,MM_HIDDEN,o->gu));
      MM_LAUNCH((qwen_biased_gelu<<<grid_for(long(rows)*MM_INTER),256,0,stream>>>(o->gu,w(b.gate_up_bias),o->mlp,size_t(rows)*MM_INTER,MM_INTER,false)));
      MM_RUN(gemm(o,o->mlp,b.down,rows,MM_HIDDEN,MM_INTER,o->proj));
      MM_LAUNCH((minimax_residual<<<grid_for(long(rows)*MM_HIDDEN),256,0,stream>>>(o->x+size_t(r)*MM_HIDDEN,o->proj,w(b.down_bias),size_t(rows)*MM_HIDDEN)));
    }
    if((i+1)%4==0)MM_RUN(observe(o,cb,ctx,i+1,o->x,n,MM_HIDDEN,false));
  }
  MM_LAUNCH((cast_bf16<<<grid_for(long(n)*MM_HIDDEN),256,0,stream>>>(o->x,o->norm,size_t(n)*MM_HIDDEN)));
  for(int r=0;r<n;r+=MM_CHUNK) {
    int rows=std::min(MM_CHUNK,n-r);
    MM_RUN(gemm(o,o->norm+size_t(r)*MM_HIDDEN,s.merger_extra[0],rows,MM_OUTPUT,MM_HIDDEN,o->fc1));
    MM_LAUNCH((qwen_biased_gelu<<<grid_for(long(rows)*MM_OUTPUT),256,0,stream>>>(o->fc1,w(s.merger_extra[1]),o->gelu,size_t(rows)*MM_OUTPUT,MM_OUTPUT,false)));
    MM_RUN(gemm(o,o->gelu,s.merger_extra[2],rows,MM_OUTPUT,MM_OUTPUT,o->fc2));
    MM_LAUNCH((qwen_biased_cast<<<grid_for(long(rows)*MM_OUTPUT),256,0,stream>>>(o->fc2,w(s.merger_extra[3]),o->projected+size_t(r)*MM_OUTPUT,size_t(rows)*MM_OUTPUT,MM_OUTPUT)));
  }
  // The processor's merge-window patch order makes each four rows contiguous.
  for(int u=0;u<units;u+=MM_CHUNK) {
    int rows=std::min(MM_CHUNK,units-u);
    MM_RUN(gemm(o,o->projected+size_t(u)*MM_OUTPUT*4,s.merger_fc1,rows,MM_OUTPUT,MM_OUTPUT*4,o->fc1));
    MM_LAUNCH((qwen_biased_gelu<<<grid_for(long(rows)*MM_OUTPUT),256,0,stream>>>(o->fc1,w(s.merger_fc1_bias),o->gelu,size_t(rows)*MM_OUTPUT,MM_OUTPUT,false)));
    MM_RUN(gemm(o,o->gelu,s.merger_fc2,rows,MM_OUTPUT,MM_OUTPUT,o->fc2));
    MM_LAUNCH((qwen_biased_cast<<<grid_for(long(rows)*MM_OUTPUT),256,0,stream>>>(o->fc2,w(s.merger_fc2_bias),o->output+size_t(u)*MM_OUTPUT,size_t(rows)*MM_OUTPUT,MM_OUTPUT)));
  }
  MM_RUN(observe(o,cb,ctx,33,o->output,units,MM_OUTPUT,false));
  MM_RUN(cudaMemcpyAsync(output,o->output,output_bytes,cudaMemcpyDeviceToHost,stream));
  MM_RUN(cudaStreamSynchronize(stream));++o->ledger.encodes;
#undef MM_LAUNCH
#undef MM_RUN
  return 0;
}
