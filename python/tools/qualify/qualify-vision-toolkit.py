#!/usr/bin/env python3
"""Pointwise toolkit and graph-compatible BF16 feature scatter CUDA gate."""
import argparse
import ctypes as C
import json
import torch
import torch.nn.functional as F


def run(path):
    lib=C.CDLL(path)
    pointer=C.c_void_p
    lib.cuteafd_vision_rmsnorm_bf16.argtypes=[pointer,pointer,pointer,C.c_int32,C.c_int32,C.c_float,pointer]
    lib.cuteafd_vision_layernorm_bf16.argtypes=[pointer,pointer,pointer,pointer,C.c_int32,C.c_int32,C.c_float,pointer]
    lib.cuteafd_vision_swiglu_bf16.argtypes=[pointer,pointer,pointer,C.c_int64,C.c_int32,C.c_float,pointer]
    lib.cuteafd_vision_gelu_bf16.argtypes=[pointer,pointer,C.c_int64,C.c_int32,pointer]
    lib.cuteafd_embed_inject.argtypes=[pointer,pointer,pointer,C.c_int32,C.c_int32,C.c_int32,C.c_int32,pointer]
    torch.manual_seed(13)
    stream=torch.cuda.current_stream().cuda_stream
    checks=[]
    def check(rc):
        if rc:raise RuntimeError(f"CUDA status {rc}")
    def close(name,actual,expected):
        torch.cuda.synchronize()
        error=float((actual.float()-expected.float()).abs().max())
        # FP32 reductions may land either side of a BF16 rounding tie.
        torch.testing.assert_close(actual.float(),expected.float(),atol=0.03125,rtol=0.004)
        checks.append(dict(name=name,max_abs=error))
    for width in (72,1024,1152,1280,2816,4304,4608,5120,9216,10240):
        x=torch.randn(3,width,device="cuda")
        w=torch.randn(width,device="cuda")
        b=torch.randn(width,device="cuda")
        out=torch.empty_like(x,dtype=torch.bfloat16)
        check(lib.cuteafd_vision_rmsnorm_bf16(x.data_ptr(),w.data_ptr(),out.data_ptr(),3,width,1e-6,stream))
        close(f"rmsnorm-{width}",out,(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*w).bfloat16())
        for bias in (None,b):
            check(lib.cuteafd_vision_layernorm_bf16(x.data_ptr(),w.data_ptr(),None if bias is None else bias.data_ptr(),out.data_ptr(),3,width,1e-6,stream))
            close(f"layernorm-{width}-bias-{bias is not None}",out,F.layer_norm(x,(width,),w,bias,1e-6).bfloat16())
        gu=torch.randn(3,width*2,device="cuda")*12
        bias=torch.randn(width*2,device="cuda")
        for use_bias in (False,True):
            for clamp in (0.0,10.0):
                check(lib.cuteafd_vision_swiglu_bf16(gu.data_ptr(),bias.data_ptr() if use_bias else None,out.data_ptr(),3,width,clamp,stream))
                g,u=(gu+(bias if use_bias else 0)).chunk(2,-1)
                if clamp:g=g.clamp(max=clamp);u=u.clamp(-clamp,clamp)
                close(f"swiglu-{width}-bias-{use_bias}-clamp-{clamp}",out,(F.silu(g)*u).bfloat16())
        for tanh in (0,1):
            check(lib.cuteafd_vision_gelu_bf16(x.data_ptr(),out.data_ptr(),x.numel(),tanh,stream))
            close(f"gelu-{width}-{tanh}",out,F.gelu(x,approximate="tanh" if tanh else "none").bfloat16())
    # HC copies 1/4 (and supported 2/3), preserving ordinary text rows exactly.
    for copies in (1,2,3,4):
        features=torch.randn(3,1152,device="cuda").bfloat16()
        indices=torch.tensor([0,2,5],device="cuda",dtype=torch.int32)
        baseline=torch.randn(7,copies,1152,device="cuda").bfloat16()
        out=baseline.clone()
        def launch():
            check(lib.cuteafd_embed_inject(features.data_ptr(),indices.data_ptr(),out.data_ptr(),3,7,1152,copies,torch.cuda.current_stream().cuda_stream))
        launch();torch.cuda.synchronize()
        expected=baseline.clone();expected[indices.long()]=features[:,None,:]
        assert torch.equal(out,expected)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):launch()
        for _ in range(3):
            out.copy_(baseline);graph.replay();torch.cuda.synchronize()
            assert torch.equal(out,expected)
        checks.append(dict(name=f"injection-copies-{copies}-graph",byte_exact=True))
    print(json.dumps(dict(device=torch.cuda.get_device_name(0),sm=torch.cuda.get_device_capability(0),checks=checks,passed=True)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",required=True)
    run(parser.parse_args().library)
