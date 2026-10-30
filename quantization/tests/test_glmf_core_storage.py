"""Flash's routed down Hessian must see its native clamped gate/up."""
from pathlib import Path
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm5_flash.core_k6.storage import RouteSpool


def test_down_hessian_observes_flash_clamp_and_bf16_boundary():
    spool=object.__new__(RouteSpool)
    spool.width=2;spool.minimum=3;spool.swiglu_limit=10.
    spool.route_rows=[np.array([0,1],dtype=np.int64)]
    spool.recovery_rows=[np.array([],dtype=np.int64)]
    x=torch.tensor([[1.,1.],[-1.,-1.]],dtype=torch.bfloat16)
    spool.gather=lambda rows,device:x[rows].to(device)
    fused=torch.tensor([[100.,100.],[-100.,-100.]],dtype=torch.bfloat16)
    raw,evidence=spool.hessian(0,down=fused,device="cpu")
    # Native BF16 SwiGLU is -100 for the positive row, and zero after
    # underflow for the negative row; the third missing row contributes I.
    assert raw["H"].dtype==torch.float32
    assert raw["H"].item()==10001.
    assert raw["count"]==3 and raw["identity_rows"]==1
    assert evidence["natural_rows"]==2 and evidence["augmented_rows"]==0
    assert "clamped-swiglu" in evidence["down_inputs"]
    before,_=spool.hessian(0,device="cpu")
    assert torch.equal(before["H"],torch.tensor([[3.,2.],[2.,3.]]))


@pytest.mark.parametrize("limit",[float("nan"),float("inf"),0,-1,True])
def test_invalid_swiglu_limit_rejected(tmp_path,limit):
    with pytest.raises(ValueError,match="SwiGLU"):
        RouteSpool(tmp_path,{},1,swiglu_limit=limit)
