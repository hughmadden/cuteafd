"""Shared durable stores with Flash-native clamped routed-down calibration."""
from __future__ import annotations

import hashlib
import math
import torch

from glm5.core_k6.storage import ProjectionStore, RouteSpool as _RouteSpool, StageStore, load_tensors


class RouteSpool(_RouteSpool):
    def __init__(self, root, identity, tokens, width=4096, experts=288, minimum=1024,
                 swiglu_limit=10.0):
        if (not isinstance(swiglu_limit, (int, float)) or isinstance(swiglu_limit, bool)
                or not math.isfinite(swiglu_limit) or swiglu_limit <= 0):
            raise ValueError("Flash SwiGLU limit must be finite and positive")
        self.swiglu_limit = float(swiglu_limit)
        super().__init__(root, identity, tokens, width, experts, minimum)

    @classmethod
    def create(cls, root, identity, tokens, batches, router, progress, layer,
               width=4096, experts=288, minimum=1024):
        return super().create(root, identity, tokens, batches, router, progress, layer,
                              width=width, experts=experts, minimum=minimum)

    def hessian(self, expert, down=None, device="cuda:0", chunk=2048):
        fused=None if down is None else (torch.cat(down,dim=0) if isinstance(down,tuple) else down)
        if fused is not None and (fused.ndim!=2 or fused.shape[0]%2 or fused.shape[1]!=self.width):
            raise ValueError("down capture requires the native fused gate/up geometry")
        width=self.width if fused is None else fused.shape[0]//2
        H=torch.zeros((width,width),device=device,dtype=torch.float32)
        natural,recovery=self.route_rows[expert],self.recovery_rows[expert]
        for selection in (natural,recovery):
            for start in range(0,len(selection),chunk):
                x=self.gather(selection[start:start+chunk],device)
                if fused is not None:
                    gate,up=torch.nn.functional.linear(x,fused).chunk(2,dim=-1)
                    gate=gate.clamp(max=self.swiglu_limit)
                    up=up.clamp(min=-self.swiglu_limit,max=self.swiglu_limit)
                    x=torch.nn.functional.silu(gate)*up
                x=x.float()
                H.addmm_(x.T,x)
                del x
        residual=max(0,self.minimum-len(natural)-len(recovery))
        H.diagonal().add_(float(residual))
        evidence=dict(natural_rows=len(natural),augmented_rows=len(recovery),identity_rows=residual,
            effective_rows=len(natural)+len(recovery)+residual,ranks=[9,16],
            selection_policy="rank-major-then-original-corpus-row",
            recovery_selection_sha256=hashlib.sha256(recovery.tobytes()).hexdigest(),
            down_inputs=None if down is None else "serialized-reconstructed-fused-gate-up-native-clamped-swiglu-bf16",
            swiglu_limit=self.swiglu_limit if down is not None else None,
            identity_contract="raw-XTX-missing-times-I-equivalent-to-normalized-2I")
        return dict(H=H,count=evidence["effective_rows"],finalized=False,identity_rows=residual),evidence
