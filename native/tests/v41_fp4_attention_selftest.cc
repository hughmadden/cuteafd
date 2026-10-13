#include "cuteafd_v41_sparse_attention.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
  void* allocation = nullptr;
  uint8_t* data = nullptr;
  size_t bytes;
  explicit Buffer(size_t size, bool unaligned = false) : bytes(size) {
    Check(cudaMalloc(&allocation, size + size_t(unaligned)));
    data = static_cast<uint8_t*>(allocation) + size_t(unaligned);
  }
  ~Buffer() { cudaFree(allocation); }
  Buffer(const Buffer&) = delete;
  void Fill(int value) { Check(cudaMemset(data, value, bytes)); }
  template<class T> void Copy(const std::vector<T>& values) {
    if (values.size()*sizeof(T) != bytes) throw std::runtime_error("test buffer size mismatch");
    Check(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice));
  }
};
static uint16_t BFloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

// Nonuniform KV/query/sink catches arithmetic differences hidden by equal weights.
static void WidthInvariantChecks(cudaStream_t stream, int heads) {
  constexpr int rows=256, dim=512;
  Buffer query(size_t(rows)*heads*dim*2), output(query.bytes), serial(query.bytes), sink(heads*4);
  Buffer window(128*512), ws(128*16), proposal(rows*512), ps(rows*16);
  Buffer source(768*256), ss(768*32), priv(3*256), pss(3*32);
  Buffer alternate_window(window.bytes), alternate_source(source.bytes);
  Buffer end(8), se(8), pages(8), meta(rows*80), ids(rows*2048), bounds(rows*8);
  std::vector<uint16_t> q(query.bytes/2);
  std::vector<float> sk(heads);
  for(size_t i=0;i<q.size();++i)q[i]=BFloat16(float(int((i*17+i/512)%43)-21)/64.0f);
  for(int h=0;h<heads;++h)sk[h]=float(h-32)/19.0f;
  query.Copy(q);sink.Copy(sk);
  auto values=[](size_t n,bool fp4) {
    std::vector<uint8_t> v(n);
    for(size_t i=0;i<n;++i) {
      if(fp4)v[i]=uint8_t(((i*13+i/32)%16)|(((i*7+i/256)%16)<<4));
      else { const uint8_t choices[]={0x28,0x38,0xb0,0x40,0x00,0xbc};v[i]=choices[(i*11+i/512)%6]; }
    }
    return v;
  };
  window.Copy(values(window.bytes,false));proposal.Copy(values(proposal.bytes,false));
  source.Copy(values(source.bytes,true));priv.Copy(values(priv.bytes,true));
  auto alternate=values(window.bytes,false);std::reverse(alternate.begin(),alternate.end());alternate_window.Copy(alternate);
  alternate=values(source.bytes,true);std::reverse(alternate.begin(),alternate.end());alternate_source.Copy(alternate);
  ws.Fill(127);ps.Fill(127);ss.Fill(0x38);pss.Fill(0x38);
  end.Copy<uint64_t>({2048});se.Copy<uint64_t>({512});pages.Copy<uint32_t>({1,0});
  std::vector<uint64_t> metadata;
  std::vector<int32_t> selected(rows*512);
  for(int r=0;r<rows;++r) {
    const uint64_t fields[]={2048,0,rows,uint64_t(2048+r),0,514,512,2,0,1};
    metadata.insert(metadata.end(),fields,fields+10);
    for(int k=0;k<512;++k)selected[r*512+k]=(k%19==0)?-1:(k*37+r*11)%514;
  }
  meta.Copy(metadata);ids.Copy(selected);bounds.Copy(std::vector<uint64_t>(rows,0));
  cuteafd_v41_sparse_kv_t v{};
  v.values[0]=window.data;v.values[1]=proposal.data;v.values[2]=source.data;v.values[3]=priv.data;
  v.scales[0]=ws.data;v.scales[1]=ps.data;v.scales[2]=ss.data;v.scales[3]=pss.data;
  v.window_end=reinterpret_cast<const uint64_t*>(end.data);
  v.source_end=reinterpret_cast<const uint64_t*>(se.data);v.pages=reinterpret_cast<const uint32_t*>(pages.data);
  v.window_proposal_capacity=rows;v.source_capacity=768;v.source_proposal_capacity=3;v.page_stride=2;
  const auto* qp=reinterpret_cast<const uint16_t*>(query.data);
  const auto* sp=reinterpret_cast<const float*>(sink.data);
  const auto* mp=reinterpret_cast<const uint64_t*>(meta.data);
  const auto* ip=reinterpret_cast<const int32_t*>(ids.data);
  const auto* bp=reinterpret_cast<const uint64_t*>(bounds.data);
  auto* op=reinterpret_cast<uint16_t*>(output.data);
  auto* rp=reinterpret_cast<uint16_t*>(serial.data);
  for(int format : {0,2}) {
    v.compressed=format;
    const auto bounded=heads==64?cuteafd_v41_sparse_attention_bounded:cuteafd_v41_sparse_attention_heads32_bounded;
    for(int r=0;r<rows;++r)
      Check(static_cast<cudaError_t>(bounded(qp+size_t(r)*heads*dim,sp,
          mp+r*10,format?ip+r*512:nullptr,rp+size_t(r)*heads*dim,1,0,&v,stream,bp+r,nullptr,0,1)));
    Check(cudaStreamSynchronize(stream));
    std::vector<uint16_t> expected(serial.bytes/2),actual(output.bytes/2);
    Check(cudaMemcpy(expected.data(),serial.data,serial.bytes,cudaMemcpyDeviceToHost));
    for(int width : {64,128,256}) {
      Check(static_cast<cudaError_t>(bounded(qp,sp,mp,format?ip:nullptr,
          op,width,0,&v,stream,bp,nullptr,0,1)));
      Check(cudaStreamSynchronize(stream));
      Check(cudaMemcpy(actual.data(),output.data,size_t(width)*heads*dim*2,cudaMemcpyDeviceToHost));
      if(!std::equal(expected.begin(),expected.begin()+size_t(width)*heads*dim,actual.begin()))
        throw std::runtime_error("nonuniform head-group 1/2/4 output is not byte invariant");
    }
    const int parts=format?10:2;
    Buffer scratch(size_t(64)*parts*heads*514*4),descriptors(64*sizeof(v));
    std::vector<cuteafd_v41_sparse_kv_t> views(64,v);
    for(int r=0;r<64;++r) if((r/3)%2) {
      views[r].values[0]=alternate_window.data;views[r].values[2]=alternate_source.data;
    }
    descriptors.Copy(views);
    const auto* dp=reinterpret_cast<const cuteafd_v41_sparse_kv_t*>(descriptors.data);
    auto* partial=reinterpret_cast<float*>(scratch.data);
    auto launch=heads==64?cuteafd_v41_sparse_attention_batch:cuteafd_v41_sparse_attention_heads32_batch;
#ifdef CUTEAFD_TEST_ATTENTION_AOT
    if(format)launch=heads==64?cuteafd_v41_sparse_attention_batch_aot:cuteafd_v41_sparse_attention_heads32_batch_aot;
#endif
    const auto validate=heads==64?cuteafd_v41_sparse_attention_batch_validate:cuteafd_v41_sparse_attention_heads32_batch_validate;
    Check(static_cast<cudaError_t>(validate(qp,sp,mp,format?ip:nullptr,
        op,64,views.data(),dp,bp,partial,scratch.bytes,parts,format)));
    for(int r=0;r<64;++r)
      Check(static_cast<cudaError_t>(launch(qp+size_t(r)*heads*dim,sp,mp+r*10,format?ip+r*512:nullptr,
          rp+size_t(r)*heads*dim,1,dp+r,stream,bp+r,partial,parts,format)));
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(expected.data(),serial.data,64*heads*dim*2,cudaMemcpyDeviceToHost));
    for(int width=1;width<=64;++width) {
      Check(static_cast<cudaError_t>(launch(qp,sp,mp,format?ip:nullptr,op,width,dp,stream,bp,partial,parts,format)));
      Check(cudaStreamSynchronize(stream));
      Check(cudaMemcpy(actual.data(),output.data,size_t(width)*heads*dim*2,cudaMemcpyDeviceToHost));
      if(!std::equal(expected.begin(),expected.begin()+size_t(width)*heads*dim,actual.begin()))
        throw std::runtime_error("fixed descriptor decode differs from serial rows");
      if(width==1 || width==16 || width==64) {
        cudaGraph_t graph;cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        Check(static_cast<cudaError_t>(launch(qp,sp,mp,format?ip:nullptr,op,width,dp,stream,bp,partial,parts,format)));
        Check(cudaStreamEndCapture(stream,&graph));Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        output.Fill(0xcd);Check(cudaGraphLaunch(exec,stream));Check(cudaStreamSynchronize(stream));
        Check(cudaMemcpy(actual.data(),output.data,size_t(width)*heads*dim*2,cudaMemcpyDeviceToHost));
        if(!std::equal(expected.begin(),expected.begin()+size_t(width)*heads*dim,actual.begin()))
          throw std::runtime_error("fixed attention graph replay differs from serial rows");
        Check(cudaGraphExecDestroy(exec));Check(cudaGraphDestroy(graph));
      }
    }
  }
  std::cout << "PASS heads=" << heads << " nonuniform head groups, mixed descriptors widths 1..64 and graph replay" << std::endl;
}

int main() {
  Check(static_cast<cudaError_t>(cuteafd_v41_sparse_attention_initialize()));
  Check(static_cast<cudaError_t>(cuteafd_v41_sparse_attention_heads32_initialize()));
  cudaStream_t test_stream;
  Check(cudaStreamCreateWithFlags(&test_stream,cudaStreamNonBlocking));
  WidthInvariantChecks(test_stream,64);
  WidthInvariantChecks(test_stream,32);
  int checks = 0;
  for(int format : {0,1,2}) for (int rows : {1, 6, 128, 256}) {
    Buffer query(size_t(rows)*64*512*2), output(query.bytes), sink(64*4);
    Buffer window(128*512, !(rows==6 && format==2)), window_scales(128*16, !(rows==6 && format==2));
    Buffer proposal(size_t(rows)*512, !(rows==6 && format==2)), proposal_scales(size_t(rows)*16, !(rows==6 && format==2));
    Buffer source(768*(format==2?256:512), !(rows==6 && format==2)), source_scales(768*(format==2?32:16), !(rows==6 && format==2));
    Buffer private_source(3*(format==2?256:512), !(rows==6 && format==2)), private_scales(3*(format==2?32:16), !(rows==6 && format==2));
    Buffer window_end(8), source_end(8), pages(8), metadata(size_t(rows)*10*8);
    Buffer selected(size_t(rows)*512*4), bounds(size_t(rows)*8);
    Buffer scratch(size_t(rows)*10*64*514*4);
    query.Fill(0); sink.Fill(0); window.Fill(0); proposal.Fill(0);
    window_scales.Fill(127); proposal_scales.Fill(127);
    source.Fill(format==2?0x22:0x38); private_source.Fill(format==2?0x22:0x38); // Quantized ones.
    source_scales.Fill(format==2?0x38:127); private_scales.Fill(format==2?0x38:127); // Scale one.
    window_end.Copy<uint64_t>({2048}); source_end.Copy<uint64_t>({512});
    pages.Copy<uint32_t>({1, 0});
    cuteafd_v41_sparse_kv_t view{};
    view.values[0]=window.data; view.values[1]=proposal.data;
    view.values[2]=source.data; view.values[3]=private_source.data;
    view.scales[0]=window_scales.data; view.scales[1]=proposal_scales.data;
    view.scales[2]=source_scales.data; view.scales[3]=private_scales.data;
    view.window_end=reinterpret_cast<const uint64_t*>(window_end.data);
    view.source_end=reinterpret_cast<const uint64_t*>(source_end.data);
    view.pages=reinterpret_cast<const uint32_t*>(pages.data);
    view.window_proposal_capacity=rows; view.source_capacity=768;
    view.source_proposal_capacity=3; view.page_stride=2; view.compressed=format;
    for (int parts : {0, 3, 10}) for (int begin : {-1, 0, 2048, 2049}) for (int mode : {0, 1, 2}) {
      std::vector<uint64_t> meta;
      std::vector<int32_t> ids(size_t(rows)*512, -1);
      for (int row=0; row<rows; ++row) {
        const uint64_t fields[]={2048,0,uint64_t(rows),uint64_t(2048+row),0,
            uint64_t(mode?514:512),512,uint64_t(mode?2:0),uint64_t(mode==1?1:0),uint64_t(mode==2?2:1)};
        meta.insert(meta.end(), fields, fields+10);
        for (int key=0; key<(mode?2:512); ++key) ids[size_t(row)*512+key]=(mode?512:0)+key;
      }
      metadata.Copy(meta); selected.Copy(ids);
      bounds.Copy(std::vector<uint64_t>(rows, begin<0?0:uint64_t(begin)));
      output.Fill(0xcd);
      const auto* q=reinterpret_cast<const uint16_t*>(query.data);
      auto* out=reinterpret_cast<uint16_t*>(output.data);
      const auto* m=reinterpret_cast<const uint64_t*>(metadata.data);
      const auto* s=reinterpret_cast<const int32_t*>(selected.data);
      const auto* sk=reinterpret_cast<const float*>(sink.data);
      int status;
      if (begin>=0) status=cuteafd_v41_sparse_attention_bounded(q,sk,m,s,out,rows,0,&view,nullptr,
          reinterpret_cast<const uint64_t*>(bounds.data),parts?reinterpret_cast<float*>(scratch.data):nullptr,
          parts?scratch.bytes:0,parts?parts:1);
      else if (parts) status=cuteafd_v41_sparse_attention_split(q,sk,m,s,out,rows,0,&view,nullptr,
          reinterpret_cast<float*>(scratch.data),scratch.bytes,parts);
      else status=cuteafd_v41_sparse_attention(q,sk,m,s,out,rows,0,&view,nullptr);
      Check(static_cast<cudaError_t>(status));
      Check(cudaDeviceSynchronize());
      std::vector<uint16_t> actual(output.bytes/2);
      Check(cudaMemcpy(actual.data(), output.data, output.bytes, cudaMemcpyDeviceToHost));
      for (int row=0; row<rows; ++row) {
        // Zero queries/sink give equal attention weights. Window values are
        // zero, source values are one, and the sink contributes one to the denominator.
        int windows=begin==2048?std::min(row+1,128):128;
        int sources=format?(mode?2:512):0;
        uint16_t expected=begin==2049?0:BFloat16(float(sources)/float(sources+windows+1));
        for (size_t col=0; col<64*512; ++col)
          if (actual[size_t(row)*64*512+col]!=expected)
            throw std::runtime_error("attention disagrees with closed-form reference");
      }
      // Compare compact local heads with the full-head kernel using nonuniform
      // queries and sinks. This catches row/head strides and rank-one sink offsets.
      std::vector<uint16_t> varied_query(query.bytes/2);
      std::vector<float> varied_sink(64);
      for(size_t i=0;i<varied_query.size();++i)
        varied_query[i]=BFloat16(float(int((i*17+i/512)%31)-15)/64.0f);
      for(int head=0;head<64;++head)varied_sink[head]=float(head-32)/16.0f;
      query.Copy(varied_query);sink.Copy(varied_sink);
      const auto* wb=reinterpret_cast<const uint64_t*>(bounds.data);
      Check(static_cast<cudaError_t>(cuteafd_v41_sparse_attention_bounded(q,sk,m,s,out,rows,0,
          &view,nullptr,wb,parts?reinterpret_cast<float*>(scratch.data):nullptr,
          parts?scratch.bytes:0,parts?parts:1)));
      Check(cudaMemcpy(actual.data(),output.data,output.bytes,cudaMemcpyDeviceToHost));
      Buffer local_query(size_t(rows)*32*512*2),local_output(local_query.bytes),local_sink(32*4);
      Buffer local_scratch(size_t(rows)*(parts?parts:1)*32*514*4);
      for(int rank=0;rank<2;++rank) {
        std::vector<uint16_t> compact(local_query.bytes/2);
        for(int row=0;row<rows;++row)
          std::copy_n(varied_query.data()+(size_t(row)*64+rank*32)*512,32*512,
              compact.data()+size_t(row)*32*512);
        local_query.Copy(compact);
        local_sink.Copy(std::vector<float>(varied_sink.begin()+rank*32,varied_sink.begin()+(rank+1)*32));
        local_output.Fill(0xcd);local_scratch.Fill(0xcd);
        auto launch=[&](uint64_t bytes) {
          return cuteafd_v41_sparse_attention_heads32_bounded(
              reinterpret_cast<const uint16_t*>(local_query.data),reinterpret_cast<const float*>(local_sink.data),
              m,s,reinterpret_cast<uint16_t*>(local_output.data),rows,0,&view,test_stream,wb,
              parts?reinterpret_cast<float*>(local_scratch.data):nullptr,bytes,parts?parts:1);
        };
        if(parts && launch(local_scratch.bytes-1)!=cudaErrorInvalidValue)
          throw std::runtime_error("compact attention accepted undersized scratch");
        Check(static_cast<cudaError_t>(launch(parts?local_scratch.bytes:0)));
        if(rows==6 && begin==0 && mode==0) {
          Check(cudaStreamSynchronize(test_stream));
          cudaGraph_t graph;cudaGraphExec_t executable;
          Check(cudaStreamBeginCapture(test_stream,cudaStreamCaptureModeThreadLocal));
          Check(static_cast<cudaError_t>(launch(parts?local_scratch.bytes:0)));
          Check(cudaStreamEndCapture(test_stream,&graph));
          Check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
          bounds.Copy(std::vector<uint64_t>(rows,2049));
          Check(cudaGraphLaunch(executable,test_stream));Check(cudaStreamSynchronize(test_stream));
          Check(cudaMemcpy(compact.data(),local_output.data,local_output.bytes,cudaMemcpyDeviceToHost));
          if(std::any_of(compact.begin(),compact.end(),[](uint16_t v){return v!=0;}))
            throw std::runtime_error("compact replay ignored updated invalid bounds");
          bounds.Copy(std::vector<uint64_t>(rows,0));
          Check(cudaGraphLaunch(executable,test_stream));Check(cudaStreamSynchronize(test_stream));
          Check(cudaGraphExecDestroy(executable));Check(cudaGraphDestroy(graph));
        }
        Check(cudaStreamSynchronize(test_stream));
        Check(cudaMemcpy(compact.data(),local_output.data,local_output.bytes,cudaMemcpyDeviceToHost));
        for(int row=0;row<rows;++row)for(size_t col=0;col<32*512;++col)
          if(compact[size_t(row)*32*512+col]!=actual[(size_t(row)*64+rank*32)*512+col])
            throw std::runtime_error("compact TP2 attention disagrees with full-head output");
      }
      if(rows==6 && begin==0 && mode==0 && parts==3) {
        const int batch_parts=format?10:2;
        Buffer device_views(rows*sizeof(view)),bad_end(8);
        bad_end.Copy<uint64_t>({0});
        std::vector<cuteafd_v41_sparse_kv_t> views(rows,view);
        for(int row=1;row<rows;row+=2)
          views[row].window_end=reinterpret_cast<const uint64_t*>(bad_end.data);
        device_views.Copy(views);
        auto* dv=reinterpret_cast<const cuteafd_v41_sparse_kv_t*>(device_views.data);
        Buffer batch_scratch(size_t(rows)*batch_parts*64*514*4);
        auto* bs=reinterpret_cast<float*>(batch_scratch.data);
        Check(static_cast<cudaError_t>(cuteafd_v41_sparse_attention_batch_validate(
            q,sk,m,s,out,rows,views.data(),dv,wb,bs,batch_scratch.bytes,batch_parts,format)));
        auto batch_launch=cuteafd_v41_sparse_attention_batch;
        auto half_batch_launch=cuteafd_v41_sparse_attention_heads32_batch;
#ifdef CUTEAFD_TEST_ATTENTION_AOT
        if(format==2) {
          batch_launch=cuteafd_v41_sparse_attention_batch_aot;
          half_batch_launch=cuteafd_v41_sparse_attention_heads32_batch_aot;
        }
#endif
        Check(static_cast<cudaError_t>(batch_launch(
            q,sk,m,s,out,rows,dv,test_stream,wb,bs,batch_parts,format)));
        Check(cudaStreamSynchronize(test_stream));
        Check(cudaMemcpy(actual.data(),output.data,output.bytes,cudaMemcpyDeviceToHost));
        Buffer half_query(size_t(rows)*32*512*2),half_output(half_query.bytes),half_sink(128);
        Buffer half_scratch(batch_scratch.bytes/2);
        for(int rank=0;rank<2;++rank) {
          std::vector<uint16_t> compact(half_query.bytes/2);
          for(int row=0;row<rows;++row)
            std::copy_n(varied_query.data()+(size_t(row)*64+rank*32)*512,32*512,
                compact.data()+size_t(row)*32*512);
          half_query.Copy(compact);
          half_sink.Copy(std::vector<float>(varied_sink.begin()+rank*32,varied_sink.begin()+(rank+1)*32));
          auto* hq=reinterpret_cast<const uint16_t*>(half_query.data);
          auto* hs=reinterpret_cast<const float*>(half_sink.data);
          auto* ho=reinterpret_cast<uint16_t*>(half_output.data);
          auto* hp=reinterpret_cast<float*>(half_scratch.data);
          auto validate=[&](uint64_t bytes,const cuteafd_v41_sparse_kv_t* descriptors) {
            return cuteafd_v41_sparse_attention_heads32_batch_validate(hq,hs,m,s,ho,rows,
                views.data(),descriptors,wb,hp,bytes,batch_parts,format);
          };
          Check(static_cast<cudaError_t>(validate(half_scratch.bytes,dv)));
          if(validate(half_scratch.bytes-1,dv)!=cudaErrorInvalidValue ||
              validate(half_scratch.bytes,reinterpret_cast<const cuteafd_v41_sparse_kv_t*>(half_output.data))!=cudaErrorInvalidValue)
            throw std::runtime_error("compact batch validation missed scratch/descriptor overlap");
          cudaGraph_t graph;cudaGraphExec_t executable;
          Check(cudaStreamBeginCapture(test_stream,cudaStreamCaptureModeThreadLocal));
          Check(static_cast<cudaError_t>(half_batch_launch(
              hq,hs,m,s,ho,rows,dv,test_stream,wb,hp,batch_parts,format)));
          Check(cudaStreamEndCapture(test_stream,&graph));
          Check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
          for(int replay=0;replay<2;++replay) {
            Check(cudaGraphLaunch(executable,test_stream));Check(cudaStreamSynchronize(test_stream));
            Check(cudaMemcpy(compact.data(),half_output.data,half_output.bytes,cudaMemcpyDeviceToHost));
            for(int row=0;row<rows;++row)for(size_t col=0;col<32*512;++col) {
              const auto expected=replay?0:actual[(size_t(row)*64+rank*32)*512+col];
              if(compact[size_t(row)*32*512+col]!=expected)
                throw std::runtime_error("compact batch descriptor replay mismatch");
            }
            if(replay==0) {
              auto invalid=views;
              for(auto& v:invalid)v.window_end=reinterpret_cast<const uint64_t*>(bad_end.data);
              Check(static_cast<cudaError_t>(cuteafd_v41_sparse_attention_heads32_batch_validate(
                  hq,hs,m,s,ho,rows,invalid.data(),dv,wb,hp,half_scratch.bytes,batch_parts,format)));
              device_views.Copy(invalid);
            }
          }
          Check(cudaGraphExecDestroy(executable));Check(cudaGraphDestroy(graph));
          device_views.Copy(views);
        }
      }
      query.Fill(0);sink.Fill(0);
      ++checks;
    }
    std::cout << "PASS format=" << format << " rows=" << rows << std::endl;
  }
  Check(cudaStreamDestroy(test_stream));
  std::cout << "Attention closed-form and compact-head checks passed: " << checks << std::endl;
}
