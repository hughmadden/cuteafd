#!/usr/bin/env python3
"""Check native router initialization, rejected inputs, and AOT alignment contract."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-lib',type=Path,required=True)
    p.add_argument('--reference-native-lib',type=Path)
    p.add_argument('--output',type=Path,required=True)
    return p.parse_args(argv)


def assert_outputs_bitwise(actual, expected, experts, rows):
    import torch

    for name,value,reference in zip(('scores','ids','routing'),actual,expected,strict=True):
        assert torch.equal(value.view(torch.int32),reference[:rows].view(torch.int32)),(experts,rows,name)


def load_router(path):
    lib=C.CDLL(str(path.resolve()))
    call=lib.cuteafd_v41_router
    call.argtypes=[C.c_void_p]*8+[C.c_int32,C.c_int32,C.c_void_p]
    return lib,call


def main():
    import torch

    a=parse_args()
    torch.cuda.set_device(0);torch.manual_seed(4191)
    lib,call=load_router(a.native_lib)
    rows,experts=26,384
    x=torch.randn(rows,5120,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(experts,5120,device='cuda',dtype=torch.bfloat16)
    bias=torch.randn(experts,device='cuda');mask=torch.zeros(rows,device='cuda',dtype=torch.uint8)
    scores=torch.empty(rows,experts,device='cuda');ids=torch.empty(rows,6,device='cuda',dtype=torch.int32);routing=torch.empty(rows,6,device='cuda')
    args=[t.data_ptr() for t in (x,w,bias,bias,mask,scores,ids,routing)]+[rows,experts,torch.cuda.current_stream().cuda_stream]
    # A launch must not implicitly load modules during graph preparation/replay.
    assert call(*args)==600  # cudaErrorNotReady
    assert lib.cuteafd_v41_router_initialize()==0
    assert lib.cuteafd_v41_router_initialize()==0
    reference_lib=reference_call=None
    if a.reference_native_lib is not None:
        assert a.reference_native_lib.resolve()!=a.native_lib.resolve(), 'reference must be a separate library'
        reference_lib,reference_call=load_router(a.reference_native_lib)
        assert reference_lib.cuteafd_v41_router_initialize()==0
    width_checks=[]
    for e in (384,128):
        k=3 if e==128 else 6
        hidden=torch.randn(64,5120,device='cuda',dtype=torch.bfloat16)
        weight=torch.randn(e,5120,device='cuda',dtype=torch.bfloat16)
        correction=torch.randn(e,device='cuda')
        image_mask=torch.randint(0,2,(64,),device='cuda',dtype=torch.uint8)
        def run(row_x,row_mask,router=call):
            m=len(row_x)
            out=torch.full((m,e),float('nan'),device='cuda')
            top=torch.full((m,k),-1,device='cuda',dtype=torch.int32)
            routes=torch.full((m,k),float('nan'),device='cuda')
            assert router(*[t.data_ptr() for t in
                (row_x,weight,correction,correction,row_mask,out,top,routes)],
                m,e,torch.cuda.current_stream().cuda_stream)==0
            torch.cuda.synchronize()
            assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(routes).all())
            return out,top,routes
        single=[run(hidden[r:r+1],image_mask[r:r+1]) for r in range(64)]
        expected=[torch.cat([row[i] for row in single]) for i in range(3)]
        for m in range(1,65):
            actual=run(hidden[:m],image_mask[:m])
            assert_outputs_bitwise(actual,expected,e,m)
            if reference_call is not None:
                reference=run(hidden[:m],image_mask[:m],reference_call)
                assert_outputs_bitwise(actual,reference,e,m)
        width_checks.append(dict(experts=e,rows=list(range(1,65)),bitwise=True,nan_prefilled=True))
    checks=[]
    for slot,value in [(8,0),(8,-1),(8,4097),(9,127),(0,0),(1,0),(2,0),(3,0),(5,0),(6,0),(7,0),(5,x.data_ptr()),(5,(1<<64)-16)]:
        bad=args.copy();bad[slot]=value
        assert call(*bad)==1,(slot,value)  # cudaErrorInvalidValue
        checks.append(dict(argument=slot,value=value,status=1))
    assert call(*args)==0
    torch.cuda.synchronize()
    # Reject misaligned buffers instead of silently changing reduction order.
    for slot, tensor in ((0, x), (1, w), (5, scores)):
        storage=torch.empty(tensor.numel()+1,device='cuda',dtype=tensor.dtype)
        unaligned=storage[1:].view_as(tensor)
        test=args.copy();test[slot]=unaligned.data_ptr()
        assert call(*test)==1
        checks.append(dict(argument=slot,alignment=unaligned.data_ptr()%16,status=1))
    result=dict(passed=True,initialize_required=True,repeat_initialize=True,
        unaligned_rejected=True,rejected=checks,width_checks=width_checks,
        native_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest())
    if a.reference_native_lib is not None:
        result['reference']=dict(native_sha256=hashlib.sha256(a.reference_native_lib.read_bytes()).hexdigest(),
            width_checks=width_checks,bitwise=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS initialization and 16 input/alignment rejection cases',flush=True)


if __name__=='__main__':
    main()
