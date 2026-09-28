// Component-only AOT harness; production initialization belongs in the native bridge.
#include "v41_hc_lagged.h"
static cuteafd_v41_hc_lagged_Kernel_Module_t module;
extern "C" int initialize() {
  cuteafd_v41_hc_lagged_Kernel_Module_Load(&module);
  return cudaGetLastError();
}
extern "C" int launch(void* residual, void* fn, void* scale, void* bias,
    void* incoming, void* norm, void* predicted, void* post, void* comb,
    void* normalized, void* scratch, int rows, cudaStream_t stream) {
  return cute_dsl_cuteafd_v41_hc_lagged_wrapper(&module,residual,fn,scale,bias,
      incoming,norm,predicted,post,comb,normalized,scratch,rows,stream);
}
