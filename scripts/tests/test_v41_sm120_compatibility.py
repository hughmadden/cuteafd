"""Execute the real native bridges against stubbed CUDA/AOT launch entries."""
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = Path.home() / ".cache/cuteafd/builds/plat1-sm120/native-host-tests"

CUDA = r"""
#pragma once
using cudaLibrary_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorInvalidDevice=101;
constexpr int cudaDevAttrComputeCapabilityMajor=1, cudaDevAttrComputeCapabilityMinor=2,
              cudaDevAttrMultiProcessorCount=3;
inline int device=2, major=12, minor=0, recorded_grid=0, recorded_cap=0;
inline int cudaGetDevice(int* out) { *out=device; return 0; }
inline int cudaDeviceGetAttribute(int* out, int attr, int dev) {
  *out=attr==1 ? major : attr==2 ? minor : dev==2 ? 170 : 188; return 0;
}
inline int cudaLibraryUnload(void*) { return 0; }
inline void init(void** args) { **static_cast<void***>(args[0])=reinterpret_cast<void*>(1); }
inline void load(void**) {}
inline void noop(void**, int) {}
inline void quant(void** args, int) { recorded_grid=*static_cast<int*>(args[5]); }
inline void grouped_quant(void** args, int) { recorded_grid=*static_cast<int*>(args[8]); }
inline void expert_launch(void** args, int) { recorded_cap=*static_cast<int*>(args[50]); }
"""

FP8_HEADER = r"""
#pragma once
#define CUTEAFD_V41_FP8_SMS 188
static const uint32_t grids[] = {752};
#define CUTEAFD_V41_FP8_VARIANTS \
 {{1,1,32,32,64,0,16,32,16},{init,load,quant},{init,load,noop},{init,load,noop},grids,0,1,1,0}, \
 {{1,1,256,32,64,0,16,32,16},{init,load,grouped_quant},{init,load,noop},{init,load,grouped_quant},grids,0,1,8,0}
#define CUTEAFD_V41_HC_PROJECT_MODULE {init,load,noop}
"""

FP8_MAIN = r"""
#include <cassert>
#include "SOURCE"
extern "C" int32_t cuteafd_v41_fp8_grouped_output(const uint16_t*,uint16_t*,int32_t,void*) { return 0; }
extern "C" int32_t cuteafd_v41_fp8_reduce_splits(const float*,uint16_t*,int32_t,int32_t,int32_t,void*) { return 0; }
extern "C" int32_t cuteafd_v41_fp8_initialize_storage(void*,uint64_t,float*,void*) { return 0; }
int main() {
 void* h=nullptr;
 minor=1;
 assert(cuteafd_v41_fp8_matrix_initialize(1,32,32,&h)==cudaErrorInvalidDevice && h==nullptr);
 minor=0;
 for (int dev : {2,5}) {
  device=dev;
  for (int k : {32,256}) {
   assert(cuteafd_v41_fp8_matrix_initialize(1,k,32,&h)==0);
   assert(handle(h)->device_sms==(dev==2 ? 170 : 188));
   cuteafd_v41_fp8_info_t info;
   assert(cuteafd_v41_fp8_matrix_info(1,k,32,&info)==0 && info.scratch_bytes==64);
   assert(cuteafd_v41_fp8_launch(h,reinterpret_cast<uint16_t*>(0x10000),
     reinterpret_cast<uint8_t*>(0x20000),reinterpret_cast<uint8_t*>(0x30000),
     reinterpret_cast<void*>(0x40000),64,reinterpret_cast<float*>(0x50000),
     reinterpret_cast<uint16_t*>(0x60000),1,nullptr)==0);
   assert(recorded_grid==(dev==2 ? 680 : 752));
  }
 }
 device=2;
 assert(cuteafd_v41_fp8_launch(h,reinterpret_cast<uint16_t*>(0x10000),
   reinterpret_cast<uint8_t*>(0x20000),reinterpret_cast<uint8_t*>(0x30000),
   reinterpret_cast<void*>(0x40000),64,reinterpret_cast<float*>(0x50000),
   reinterpret_cast<uint16_t*>(0x60000),1,nullptr)==cudaErrorInvalidDevice);
}
"""

EXPERT_HEADER = r"""
#pragma once
#define CUTEAFD_V41_SMS 188
#define CUTEAFD_V41_CC_MINOR 0
#define CUTEAFD_V41_VARIANTS {{2,0,1,5120,2304,2304,3,1,64,1,1,1,1,188,1},init,load,expert_launch,{}}
"""

INPUT_QUANT_HEADER = r"""
#pragma once
static const uint32_t cuteafd_v41_input_quant_grids[] = {752};
#define _mlir_cuteafd_v41_expert_input_quant_cuda_init ::init
#define _mlir_cuteafd_v41_expert_input_quant_cuda_load_to_device ::load
#define CUTEAFD_V41_INPUT_QUANT_ENTRY quant
"""

EXPERT_MAIN = r"""
#include <cassert>
#include "SOURCE"
extern "C" int32_t cuteafd_initialize_scratch_storage_async(void*,uint64_t,uint64_t,uint64_t,uint32_t,void*) { return 0; }
int main() {
 void* h=nullptr;
 minor=1;
 assert(cuteafd_v41_expert_initialize(1,&h)==cudaErrorInvalidDevice && h==nullptr);
 assert(cuteafd_v41_expert_input_quant_initialize(&h)==cudaErrorInvalidDevice && h==nullptr);
 minor=0;
 assert(cuteafd_v41_expert_initialize(1,&h)==0);
 assert(by_handle(h)->device_sms==170 && by_handle(h)->info.scratch_bytes==64);
 cuteafd_expert_launch_t args{};
 for (auto& p : args.tensors) p=reinterpret_cast<void*>(0x10000);
 args.num_tokens=1; args.scatter_rows=3; args.max_rows=1;
 args.rows_padded=1; args.max_tasks=1; args.max_phys_tiles=1;
 for (int cap : {188,170,94,1}) {
  args.max_active_clusters=cap;
  assert(cuteafd_v41_expert_launch(h,&args)==0);
  assert(recorded_cap==std::min(cap,170));
 }
 device=5;
 assert(cuteafd_v41_expert_launch(h,&args)==cudaErrorInvalidDevice);
 for (int dev : {2,5}) {
  device=dev;
  assert(cuteafd_v41_expert_input_quant_initialize(&h)==0);
  assert(cuteafd_v41_expert_input_quantize_async(h,reinterpret_cast<uint16_t*>(0x10000),
    reinterpret_cast<uint8_t*>(0x20000),1,nullptr)==0);
  assert(recorded_grid==(dev==2 ? 680 : 752));
 }
 device=2;
 assert(cuteafd_v41_expert_input_quantize_async(h,reinterpret_cast<uint16_t*>(0x10000),
   reinterpret_cast<uint8_t*>(0x20000),1,nullptr)==cudaErrorInvalidDevice);
}
"""


@pytest.mark.parametrize("family", ["fp8", "expert"])
def test_capability_only_admission_and_live_launch_caps(family):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("native bridge qualification needs a C++ compiler")
    subprocess.run(["python3", str(ROOT / "scripts/build/assert-build-filesystem.py"), str(BUILD_ROOT)], check=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUILD_ROOT) as temp:
        directory = Path(temp)
        (directory / "cuda_runtime.h").write_text(CUDA)
        if family == "fp8":
            (directory / "v41_fp8_variants.h").write_text(FP8_HEADER)
            source = ROOT / "native/families/deepseek_v41/src/v41_fp8.cc"
            main = FP8_MAIN
        else:
            (directory / "v41_expert_variants.h").write_text(EXPERT_HEADER)
            (directory / "v41_input_quant_dispatch.h").write_text(INPUT_QUANT_HEADER)
            source = ROOT / "native/shared/src/v41_experts.cc"
            main = EXPERT_MAIN
        (directory / "main.cc").write_text(main.replace("SOURCE", str(source)))
        subprocess.run(["flock", "-w", "600", str(Path.home() / ".cache/cuteafd/build.lock"),
                        compiler, "-std=c++17", "-pthread", "-I", str(directory),
                        "-I", str(ROOT / "native/shared/include"),
                        "-I", str(ROOT / "native/families/deepseek_v41/include"),
                        str(directory / "main.cc"), "-o", str(directory / "test")],
                       check=True, timeout=750)
        subprocess.run([str(directory / "test")], check=True, timeout=30)
