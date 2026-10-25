#!/usr/bin/env python3
"""Check native router initialization, rejected inputs, and AOT alignment contract."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-lib',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.cuda.set_device(0);torch.manual_seed(4191)
    lib=C.CDLL(str(a.native_lib.resolve()))
    call=lib.cuteafd_v41_router
    call.argtypes=[C.c_void_p]*8+[C.c_int32,C.c_int32,C.c_void_p]
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
    a.output.write_text(json.dumps(dict(passed=True,initialize_required=True,repeat_initialize=True,
        unaligned_rejected=True,rejected=checks,
        native_sha256=hashlib.sha256(a.native_lib.read_bytes()).hexdigest()),indent=2)+'\n')
    print('PASS initialization and 16 input/alignment rejection cases',flush=True)


if __name__=='__main__':
    main()
