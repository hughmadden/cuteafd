// GLM 5.3 Flash's official BF16 tower. Included after the shared Owner/toolkit.
// Coordinates are merge-block order; downsample gathers channel,kh,kw order.
constexpr int GLM_HIDDEN=1024, GLM_HEADS=16, GLM_INTER=4096;
constexpr int GLM_PATCH=1176, GLM_MERGED=4096, GLM_MERGER_INTER=10240;
constexpr int GLM_MERGER_CHUNK=1024;

size_t scratch_glm(Owner* o, size_t max_tokens, int output_width) {
  size_t cursor=0,n=max_tokens*4;
  auto take=[&](auto& ptr,size_t bytes) {
    using P=typename std::remove_reference<decltype(ptr)>::type;
    ptr=o->arena ? reinterpret_cast<P>(o->arena+cursor) : nullptr;
    cursor+=aligned(bytes);
  };
  take(o->x,n*GLM_HIDDEN*4);take(o->xt,n*GLM_HIDDEN*4);take(o->norm,n*GLM_HIDDEN*2);
  take(o->patches,n*GLM_PATCH*2);
  take(o->q,n*GLM_HIDDEN*2);take(o->k,n*GLM_HIDDEN*2);take(o->v,n*GLM_HIDDEN*2);
  take(o->attn,n*GLM_HIDDEN*2);take(o->output,max_tokens*output_width*2);
  take(o->qkv,CHUNK*GLM_HIDDEN*3*4);take(o->proj,CHUNK*GLM_HIDDEN*4);
  take(o->gu,std::max(CHUNK*2*GLM_INTER,GLM_MERGER_CHUNK*2*GLM_MERGER_INTER)*4);
  take(o->mlp,std::max(CHUNK*GLM_INTER,GLM_MERGER_CHUNK*GLM_MERGER_INTER)*2);
  take(o->fc1,CHUNK*GLM_MERGED*4);take(o->gelu,CHUNK*GLM_MERGED*2);
  take(o->fc2,CHUNK*output_width*4);
  take(o->rgb,n*14*14*3);take(o->lut,3*256*4);
  take(o->hw_row,n*2*4);take(o->hw_col,n*2*4);take(o->col,max_tokens*4);take(o->inverse,max_tokens*4);
  // AOT attention's LSE strides are fixed at its 16K patch capacity.
  take(o->lse,size_t(16384)*GLM_HEADS*4);take(o->cu_seqlens,2*4);
  take(o->workspace,BLAS_BYTES);return cursor;
}

bool valid_glm(const cuteafd_vision_spec& s) {
  if(s.abi_version!=2 || s.reserved!=3 || s.geometry_reserved || !s.max_tokens || s.max_tokens>4096 ||
     s.hidden!=1024 || s.depth!=24 || s.heads!=16 || s.kv_heads!=16 || s.head_dim!=64 ||
     s.intermediate!=4096 || s.patch_size!=14 || s.merger_width!=4096 || s.output_width!=4096 ||
     s.norm_eps!=1e-5f || !s.weight_bytes || s.weight_bytes>(2ULL<<30))return false;
  if(!extent(s,s.patch,1024*1176*2) || !extent(s,s.patch_bias,1024*4) ||
     !extent(s,s.inv_freq,16*4) || !extent(s,s.merger_norm,1024*4) ||
     !extent(s,s.merger_norm_bias,4096*4))return false;
  const size_t mb[]={4096*4096*2,4096*4,4096*4096*2,4096*4,2*10240*4096*2,0,4096*10240*2,0};
  for(int i=0;i<8;++i)if(mb[i] && !extent(s,s.merger_extra[i],mb[i]))return false;
  for(int i=0;i<24;++i) {
    const auto& b=s.blocks[i];
    const uint64_t offsets[]={b.qkv,b.qkv_bias,b.proj,b.proj_bias,b.gate_up,b.gate_up_bias,
      b.down,b.down_bias,b.norm1,b.norm2,s.q_norm[i],s.k_norm[i]};
    const size_t bytes[]={3072*1024*2,3072*4,1024*1024*2,1024*4,8192*1024*2,8192*4,
      1024*4096*2,1024*4,1024*4,1024*4,64*4,64*4};
    for(int j=0;j<12;++j)if(!extent(s,offsets[j],bytes[j]))return false;
    if(b.window || b.column_order || b.key0_bias!=UINT64_MAX)return false;
  }
  return true;
}

__global__ void glm_rgb_patches(const uint8_t* rgb,const float* lut,uint16_t* out,int gh,int gw) {
  size_t total=size_t(gh)*gw*GLM_PATCH;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x) {
    int p=i/GLM_PATCH,d=i%GLM_PATCH,c=d/(2*196),pixel=d%196,u=p/4,r=p%4;
    int y=(u/(gw/2))*28+(r/2)*14+pixel/14,x=(u%(gw/2))*28+(r%2)*14+pixel%14;
    out[i]=bf16_bits(lut[c*256+rgb[(size_t(y)*gw*14+x)*3+c]]);
  }
}
__global__ void glm_bias(float* x,const float* bias,size_t n,int width) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
    x[i]+=bias[i%width];
}
// One warp handles each q/k head's RMS reduction; head weights are shared.
__global__ void glm_rope_qkv(const float* qkv,const float* bias,const float* qw,const float* kw,
  const int32_t* hw,const float* freq,int r0,uint16_t* q,uint16_t* k,uint16_t* v) {
  int row=blockIdx.x,t=r0+row,head=threadIdx.x/32,lane=threadIdx.x%32;
  const float* source=qkv+size_t(row)*3072;
  for(int h=head;h<32;h+=blockDim.x/32) {
    int base=h*64;
    float a=source[base+lane]+bias[base+lane],b=source[base+lane+32]+bias[base+lane+32];
    float square=a*a+b*b;
    for(int d=16;d;d/=2)square+=__shfl_xor_sync(0xffffffff,square,d);
    float inv=rsqrtf(square/64.f+1e-5f);
    const float* weight=h<16?qw:kw;
    a=a*inv*weight[lane];b=b*inv*weight[lane+32];
    float angle=float(hw[2*t+(lane>=16)])*freq[lane%16];
    float c=cosf(angle),s=sinf(angle);
    uint16_t* dst=h<16?q+(size_t(t)*16+h)*64:k+(size_t(t)*16+h-16)*64;
    dst[lane]=bf16_bits(__fadd_rn(__fmul_rn(a,c),__fmul_rn(-b,s)));
    dst[lane+32]=bf16_bits(__fadd_rn(__fmul_rn(b,c),__fmul_rn(a,s)));
  }
  for(int i=threadIdx.x;i<1024;i+=blockDim.x)v[size_t(t)*1024+i]=bf16_bits(source[2048+i]+bias[2048+i]);
}
// The input norm rows are [unit,kh,kw,c], Conv2d weights use [out,c,kh,kw].
__global__ void glm_conv_gather(const uint16_t* norm,uint16_t* output,int units) {
  size_t total=size_t(units)*4096;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=size_t(gridDim.x)*blockDim.x) {
    int u=i/4096,d=i%4096;
    output[i]=norm[size_t(u)*4096+(d%4)*1024+d/4];
  }
}

int encode_glm(Owner* o,const uint8_t* rgb,uint64_t rgb_bytes,const float* lut,int gh,int gw,
  uint16_t* output,uint64_t output_bytes,cuteafd_vision_observer cb,void* ctx) {
  const auto& s=o->spec;int64_t count=int64_t(gh)*gw;
  if(!o->uploaded || !rgb || !lut || !output || gh<2 || gw<2 || gh%2 || gw%2 ||
     count>int64_t(s.max_tokens)*4 || rgb_bytes!=uint64_t(count)*14*14*3 ||
     output_bytes!=uint64_t(count/4)*4096*2)return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  int n=int(count),units=n/4;auto stream=o->stream;
#define GLM_RUN(expr) do { int rc=int(expr);if(rc)return drain(o,rc); } while(0)
#define GLM_LAUNCH(expr) do { expr;GLM_RUN(cudaGetLastError()); } while(0)
  GLM_RUN(cudaMemcpyAsync(o->rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  GLM_RUN(cudaMemcpyAsync(o->lut,lut,768*4,cudaMemcpyHostToDevice,stream));
  GLM_LAUNCH((prepare_grid<<<(units+255)/256,256,0,stream>>>(o->hw_row,o->hw_col,o->col,o->inverse,gh,gw)));
  GLM_LAUNCH((glm_rgb_patches<<<(size_t(n)*1176+255)/256,256,0,stream>>>(o->rgb,o->lut,o->patches,gh,gw)));
  auto w=[&](uint64_t offset){return reinterpret_cast<const float*>(o->weights+offset);};
  for(int r0=0;r0<n;r0+=CHUNK)GLM_RUN(gemm(o,o->patches+size_t(r0)*1176,s.patch,std::min(CHUNK,n-r0),1024,1176,o->x+size_t(r0)*1024));
  GLM_LAUNCH((glm_bias<<<(size_t(n)*1024+255)/256,256,0,stream>>>(o->x,w(s.patch_bias),size_t(n)*1024,1024)));
  GLM_RUN(observe(o,cb,ctx,0,o->x,n,1024,false));
  for(int i=0;i<24;++i) {
    const auto& b=s.blocks[i];
    GLM_LAUNCH((rmsnorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(b.norm1),o->norm,1024,1e-5f)));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      GLM_RUN(gemm(o,o->norm+size_t(r0)*1024,b.qkv,rows,3072,1024,o->qkv));
      GLM_LAUNCH((glm_rope_qkv<<<rows,256,0,stream>>>(o->qkv,w(b.qkv_bias),w(s.q_norm[i]),w(s.k_norm[i]),
        o->hw_row,w(s.inv_freq),r0,o->q,o->k,o->v)));
    }
    GLM_RUN(mha_attention(o,n));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      GLM_RUN(gemm(o,o->attn+size_t(r0)*1024,b.proj,rows,1024,1024,o->proj));
      GLM_LAUNCH((add_bias_residual_kernel<<<grid_for(long(rows)*1024),256,0,stream>>>(o->x+size_t(r0)*1024,o->proj,w(b.proj_bias),long(rows)*1024,1024)));
    }
    GLM_LAUNCH((rmsnorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(b.norm2),o->norm,1024,1e-5f)));
    for(int r0=0;r0<n;r0+=CHUNK) {
      int rows=std::min(CHUNK,n-r0);
      GLM_RUN(gemm(o,o->norm+size_t(r0)*1024,b.gate_up,rows,8192,1024,o->gu));
      GLM_LAUNCH((swiglu_bias_bf16_kernel<<<grid_for(long(rows)*4096),256,0,stream>>>(o->gu,w(b.gate_up_bias),o->mlp,rows,4096,10.f)));
      GLM_RUN(gemm(o,o->mlp,b.down,rows,1024,4096,o->proj));
      GLM_LAUNCH((add_bias_residual_kernel<<<grid_for(long(rows)*1024),256,0,stream>>>(o->x+size_t(r0)*1024,o->proj,w(b.down_bias),long(rows)*1024,1024)));
    }
    if((i+1)%4==0)GLM_RUN(observe(o,cb,ctx,i+1,o->x,n,1024,false));
  }
  GLM_LAUNCH((rmsnorm_bf16_kernel<<<n,256,0,stream>>>(o->x,w(s.merger_norm),o->norm,1024,1e-5f)));
  if(cb) {
    GLM_LAUNCH((cast_float<<<(size_t(n)*1024+255)/256,256,0,stream>>>(o->norm,o->xt,size_t(n)*1024)));
    GLM_RUN(observe(o,cb,ctx,25,o->xt,n,1024,false));
  }
  // Patches are no longer used: reuse their larger admitted extent for Conv2d gathering.
  GLM_LAUNCH((glm_conv_gather<<<(size_t(units)*4096+255)/256,256,0,stream>>>(o->norm,o->patches,units)));
  for(int u0=0;u0<units;u0+=GLM_MERGER_CHUNK) {
    int rows=std::min(GLM_MERGER_CHUNK,units-u0);
    GLM_RUN(gemm(o,o->patches+size_t(u0)*4096,s.merger_extra[0],rows,4096,4096,o->fc1));
    GLM_LAUNCH((glm_bias<<<(size_t(rows)*4096+255)/256,256,0,stream>>>(o->fc1,w(s.merger_extra[1]),size_t(rows)*4096,4096)));
    GLM_LAUNCH((cast_bf16<<<(size_t(rows)*4096+255)/256,256,0,stream>>>(o->fc1,o->gelu,size_t(rows)*4096)));
    GLM_RUN(gemm(o,o->gelu,s.merger_extra[2],rows,4096,4096,o->fc1));
    GLM_LAUNCH((layernorm_bf16_kernel<<<rows,256,0,stream>>>(o->fc1,w(s.merger_extra[3]),w(s.merger_norm_bias),o->gelu,4096,1e-5f)));
    GLM_LAUNCH((cast_float<<<(size_t(rows)*4096+255)/256,256,0,stream>>>(o->gelu,o->fc1,size_t(rows)*4096)));
    GLM_LAUNCH((gelu_bf16_kernel<<<grid_for(long(rows)*4096),256,0,stream>>>(o->fc1,o->gelu,long(rows)*4096,0)));
    GLM_RUN(gemm(o,o->gelu,s.merger_extra[4],rows,20480,4096,o->gu));
    GLM_LAUNCH((swiglu_bias_bf16_kernel<<<grid_for(long(rows)*10240),256,0,stream>>>(o->gu,nullptr,o->mlp,rows,10240,10.f)));
    GLM_RUN(gemm(o,o->mlp,s.merger_extra[6],rows,4096,10240,o->fc2));
    GLM_LAUNCH((cast_bf16<<<(size_t(rows)*4096+255)/256,256,0,stream>>>(o->fc2,o->output+size_t(u0)*4096,size_t(rows)*4096)));
  }
  GLM_RUN(observe(o,cb,ctx,26,o->output,units,4096,false));
  GLM_RUN(cudaMemcpyAsync(output,o->output,output_bytes,cudaMemcpyDeviceToHost,stream));
  GLM_RUN(cudaStreamSynchronize(stream));++o->ledger.encodes;
#undef GLM_LAUNCH
#undef GLM_RUN
  return 0;
}
