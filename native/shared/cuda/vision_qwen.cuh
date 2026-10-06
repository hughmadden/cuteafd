// Qwen's BF16 tower arithmetic; the legacy MiMo path is deliberately separate.
bool valid_qwen(const cuteafd_vision_spec& s) {
  if(s.abi_version!=2 || s.reserved!=2 || s.max_tokens==0 || s.max_tokens>4096 ||
     s.hidden!=1152 || s.depth!=27 || s.heads!=16 || s.kv_heads!=16 || s.head_dim!=72 ||
     s.intermediate!=4304 || s.patch_size!=16 || s.merger_width!=4608 || s.output_width!=2560 ||
     s.norm_eps!=1e-6f || s.geometry_reserved || !s.weight_bytes || s.weight_bytes>(2ULL<<30)) return false;
  const uint64_t offsets[]={s.patch,s.patch_bias,s.pos_embed,s.merger_norm,s.merger_norm_bias,
    s.merger_fc1,s.merger_fc1_bias,s.merger_fc2,s.merger_fc2_bias,s.inv_freq};
  const size_t bytes[]={1152ULL*1536*2,1152*4,2304ULL*1152*2,1152*4,1152*4,
    4608ULL*4608*2,4608*4,2560ULL*4608*2,2560*4,18*4};
  for(int i=0;i<10;++i)if(!extent(s,offsets[i],bytes[i]))return false;
  for(int i=0;i<27;++i) {
    const auto& b=s.blocks[i];
    const uint64_t off[]={b.qkv,b.qkv_bias,b.proj,b.proj_bias,b.gate_up,b.gate_up_bias,
      b.down,b.down_bias,b.norm1,b.norm2,s.norm1_bias[i],s.norm2_bias[i]};
    const size_t len[]={3456ULL*1152*2,3456*4,1152ULL*1152*2,1152*4,4304ULL*1152*2,
      4304*4,1152ULL*4304*2,1152*4,1152*4,1152*4,1152*4,1152*4};
    for(int j=0;j<12;++j)if(!extent(s,off[j],len[j]))return false;
    if(b.window || b.column_order || b.key0_bias!=UINT64_MAX)return false;
  }
  return true;
}
size_t scratch_qwen(Owner* o,size_t max_tokens,int output_width) {
  size_t cursor=0,n=max_tokens*4;
  auto take=[&](auto& ptr,size_t bytes) {
    using P=typename std::remove_reference<decltype(ptr)>::type;
    ptr=o->arena?reinterpret_cast<P>(o->arena+cursor):nullptr;
    cursor+=aligned(bytes);
  };
  take(o->x,n*1152*4);take(o->xt,n*1152*4);take(o->norm,n*1152*2);
  take(o->patches,n*1536*2);take(o->q,n*1152*2);take(o->k,n*1152*2);
  take(o->v,n*1152*2);take(o->attn,n*1152*2);take(o->output,max_tokens*output_width*2);
  take(o->qkv,CHUNK*3456ULL*4);take(o->proj,CHUNK*1152ULL*4);take(o->gu,CHUNK*4304ULL*4);
  take(o->mlp,CHUNK*4304ULL*2);take(o->fc1,CHUNK*4608ULL*4);take(o->gelu,CHUNK*4608ULL*2);
  take(o->fc2,CHUNK*output_width*4ULL);take(o->rgb,n*16*16*3);take(o->lut,3*256*4);
  take(o->hw_row,n*2*4);take(o->pos_indices,n*4*4);take(o->pos_weights,n*4*4);
  // AOT strides are capacity-based, even when the admission cap is smaller.
  take(o->lse,16384*16*4);take(o->cu_seqlens,2*4);take(o->workspace,BLAS_BYTES);
  return cursor;
}
__device__ float from_bf16(uint16_t x) {
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
}
__device__ float round_bf16(float x) { return from_bf16(bf16_bits(x)); }
__global__ void qwen_grid(int32_t* hw,int32_t* indices,float* weights,int gh,int gw) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p>=gh*gw)return;
  int unit=p/4,r=p%4,y=(unit/(gw/2))*2+r/2,x=(unit%(gw/2))*2+r%2;
  hw[p*2]=y;hw[p*2+1]=x;
  // Match HF's separate FP32 multiply then divide (not a precomputed ratio).
  float sy=__fdiv_rn(float(y)*47.f,float(gh-1)),sx=__fdiv_rn(float(x)*47.f,float(gw-1));
  int fy=int(floorf(sy)),fx=int(floorf(sx));
  for(int j=0;j<4;++j) {
    int dy=j/2,dx=j%2;
    indices[p*4+j]=min(47,fy+dy)*48+min(47,fx+dx);
    weights[p*4+j]=fmaxf(0.f,1.f-fabsf(sy-float(fy)-float(dy)))*
      fmaxf(0.f,1.f-fabsf(sx-float(fx)-float(dx)));
  }
}
__global__ void qwen_patch_position(float* x,const float* bias,const uint16_t* table,
  const int32_t* indices,const float* weights,size_t count) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  int p=i/1152,c=i%1152;
  float pos=0.f;
  for(int j=0;j<4;++j)pos+=from_bf16(table[size_t(indices[p*4+j])*1152+c])*weights[p*4+j];
  x[i]=round_bf16(round_bf16(x[i]+bias[c])+round_bf16(pos));
}
__global__ void qwen_rope(const float* qkv,const float* bias,const int32_t* hw,
  const float* inv,int row0,uint16_t* q,uint16_t* k,uint16_t* v,int rows) {
  int row=blockIdx.x;if(row>=rows)return;
  int global=row0+row;
  for(int i=threadIdx.x;i<1152;i+=blockDim.x) {
    int d=i%72,pair=d%36;
    float angle=float(hw[global*2+pair/18])*inv[pair%18];
    float co=cosf(angle),si=sinf(angle);
    int other=i+(d<36?36:-36);float sign=d<36?-1.f:1.f;
    const float* src=qkv+size_t(row)*3456;
    // Linear+bias is rounded before the official FP32 rotary multiply/add.
    q[size_t(global)*1152+i]=bf16_bits(round_bf16(src[i]+bias[i])*co+
      sign*round_bf16(src[other]+bias[other])*si);
    k[size_t(global)*1152+i]=bf16_bits(round_bf16(src[1152+i]+bias[1152+i])*co+
      sign*round_bf16(src[1152+other]+bias[1152+other])*si);
    v[size_t(global)*1152+i]=bf16_bits(src[2304+i]+bias[2304+i]);
  }
}
__global__ void qwen_residual(float* x,const float* y,const float* bias,size_t count) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count)x[i]=round_bf16(x[i]+round_bf16(y[i]+bias[i%1152]));
}
__global__ void qwen_biased_gelu(const float* x,const float* bias,uint16_t* out,size_t count,int width,bool tanh_mode) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  float a=round_bf16(x[i]+bias[i%width]);
  float y=tanh_mode ? 0.5f*a*(1.f+tanhf(0.7978845608028654f*(a+0.044715f*a*a*a))) :
    0.5f*a*(1.f+erff(a*0.7071067811865475f));
  out[i]=bf16_bits(y);
}
__global__ void qwen_biased_cast(const float* x,const float* bias,uint16_t* y,size_t count,int width) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count)y[i]=bf16_bits(x[i]+bias[i%width]);
}
int encode_qwen(Owner* o,const uint8_t* rgb,uint64_t rgb_bytes,const float* lut,int gh,int gw,
  uint16_t* output,uint64_t output_bytes,cuteafd_vision_observer cb,void* ctx) {
  const auto& s=o->spec;int64_t count=int64_t(gh)*gw;
  if(!o->uploaded || gh<2 || gw<2 || gh%2 || gw%2 || count>int64_t(s.max_tokens)*4 ||
     rgb_bytes!=uint64_t(count)*16*16*3 || output_bytes!=uint64_t(count/4)*2560*2)return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  int n=int(count),units=n/4;auto stream=o->stream;
#define Q_RUN(expr) do { int rc=int(expr);if(rc)return drain(o,rc); } while(0)
#define Q_LAUNCH(expr) do { expr;Q_RUN(cudaGetLastError()); } while(0)
  Q_RUN(cudaMemcpyAsync(o->rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  Q_RUN(cudaMemcpyAsync(o->lut,lut,768*4,cudaMemcpyHostToDevice,stream));
  Q_LAUNCH((qwen_grid<<<(n+255)/256,256,0,stream>>>(o->hw_row,o->pos_indices,o->pos_weights,gh,gw)));
  Q_LAUNCH((rgb_patches<<<(size_t(n)*1536+255)/256,256,0,stream>>>(o->rgb,o->lut,o->patches,gh,gw)));
  for(int r=0;r<n;r+=CHUNK)Q_RUN(gemm(o,o->patches+size_t(r)*1536,s.patch,std::min(CHUNK,n-r),1152,1536,o->x+size_t(r)*1152));
  auto w=[&](uint64_t off){return reinterpret_cast<const float*>(o->weights+off);};
  Q_LAUNCH((qwen_patch_position<<<grid_for(long(n)*1152),256,0,stream>>>(o->x,w(s.patch_bias),
    reinterpret_cast<const uint16_t*>(o->weights+s.pos_embed),o->pos_indices,o->pos_weights,size_t(n)*1152)));
  Q_RUN(observe(o,cb,ctx,0,o->x,n,1152,false));
  for(int i=0;i<27;++i) {
    const auto& b=s.blocks[i];
    Q_LAUNCH((layernorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(b.norm1),w(s.norm1_bias[i]),o->norm,1152,1e-6f)));
    for(int r=0;r<n;r+=CHUNK) {
      int rows=std::min(CHUNK,n-r);
      Q_RUN(gemm(o,o->norm+size_t(r)*1152,b.qkv,rows,3456,1152,o->qkv));
      Q_LAUNCH((qwen_rope<<<rows,256,0,stream>>>(o->qkv,w(b.qkv_bias),o->hw_row,w(s.inv_freq),r,o->q,o->k,o->v,rows)));
    }
    Q_RUN(mha_attention(o,n));
    for(int r=0;r<n;r+=CHUNK) {
      int rows=std::min(CHUNK,n-r);
      Q_RUN(gemm(o,o->attn+size_t(r)*1152,b.proj,rows,1152,1152,o->proj));
      Q_LAUNCH((qwen_residual<<<grid_for(long(rows)*1152),256,0,stream>>>(o->x+size_t(r)*1152,o->proj,w(b.proj_bias),size_t(rows)*1152)));
    }
    Q_LAUNCH((layernorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(b.norm2),w(s.norm2_bias[i]),o->norm,1152,1e-6f)));
    for(int r=0;r<n;r+=CHUNK) {
      int rows=std::min(CHUNK,n-r);
      Q_RUN(gemm(o,o->norm+size_t(r)*1152,b.gate_up,rows,4304,1152,o->gu));
      Q_LAUNCH((qwen_biased_gelu<<<grid_for(long(rows)*4304),256,0,stream>>>(o->gu,w(b.gate_up_bias),o->mlp,size_t(rows)*4304,4304,true)));
      Q_RUN(gemm(o,o->mlp,b.down,rows,1152,4304,o->proj));
      Q_LAUNCH((qwen_residual<<<grid_for(long(rows)*1152),256,0,stream>>>(o->x+size_t(r)*1152,o->proj,w(b.down_bias),size_t(rows)*1152)));
    }
    if((i+1)%4==0 || i==26)Q_RUN(observe(o,cb,ctx,i+1,o->x,n,1152,false));
  }
  Q_LAUNCH((layernorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(s.merger_norm),w(s.merger_norm_bias),o->norm,1152,1e-6f)));
  if(cb) {
    Q_LAUNCH((cast_float<<<grid_for(long(n)*1152),256,0,stream>>>(o->norm,o->xt,size_t(n)*1152)));
    Q_RUN(observe(o,cb,ctx,29,o->xt,n,1152,false));
  }
  for(int u=0;u<units;u+=CHUNK) {
    int rows=std::min(CHUNK,units-u);
    Q_RUN(gemm(o,o->norm+size_t(u)*4608,s.merger_fc1,rows,4608,4608,o->fc1));
    Q_LAUNCH((qwen_biased_gelu<<<grid_for(long(rows)*4608),256,0,stream>>>(o->fc1,w(s.merger_fc1_bias),o->gelu,size_t(rows)*4608,4608,false)));
    Q_RUN(gemm(o,o->gelu,s.merger_fc2,rows,2560,4608,o->fc2));
    Q_LAUNCH((qwen_biased_cast<<<grid_for(long(rows)*2560),256,0,stream>>>(o->fc2,w(s.merger_fc2_bias),o->output+size_t(u)*2560,size_t(rows)*2560,2560)));
  }
  Q_RUN(observe(o,cb,ctx,30,o->output,units,2560,false));
  Q_RUN(cudaMemcpyAsync(output,o->output,output_bytes,cudaMemcpyDeviceToHost,stream));
  Q_RUN(cudaStreamSynchronize(stream));++o->ledger.encodes;
#undef Q_LAUNCH
#undef Q_RUN
  return 0;
}
