"""Causal replay and real disk-barrier faults using native Flash KDA/MLA and mHC blocks."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm5_flash.core_k6 import run
from glm5_flash.core_k6.storage import RouteSpool,StageStore
from gptqmodel.utils import glm5_flash_core as native


class Progress:
    def __init__(self):
        self.events=[]
    def emit(self,event,**fields):
        self.events.append((event,fields))


def fixture(tmp_path,kind="linear_attention"):
    import inspect
    from transformers.models.glm5_next import modeling_glm5_next as ref
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    config=Glm5NextTextConfig(hidden_size=128,intermediate_size=256,moe_intermediate_size=128,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,q_lora_rank=128,
        kv_lora_rank=128,qk_nope_head_dim=128,qk_rope_head_dim=0,v_head_dim=128,
        n_routed_experts=16,num_experts_per_tok=8,n_group=1,topk_group=1,
        first_k_dense_replace=1,index_n_heads=2,index_head_dim=128,index_topk=4,index_kpool=4,
        linear_num_heads=2,linear_head_dim=128,hc_mult=4,hc_sinkhorn_iters=3,
        indexer_types=["full"],layer_types=[kind],mlp_layer_types=["dense"])
    config._attn_implementation="sdpa"
    # Exercise checkpoint-native Torch math on CPU, without dispatching FLA's
    # GPU implementation merely because FLA is installed for production.
    raw=getattr(ref.chunk_kimi_delta_attention,"_gptqmodel_kda_original",ref.chunk_kimi_delta_attention)
    ref.chunk_kimi_delta_attention=native.bounded_kda(inspect.unwrap(raw))
    torch.manual_seed(787)
    block=ref.Glm5NextTextDecoderLayer(config,0).to(torch.bfloat16).eval()
    for parameter in block.parameters():
        parameter.data.normal_(0,.03)
    embedding=torch.randn(32,128).bfloat16()
    class Source:
        def __init__(self):
            self.config=config.to_dict()
            self.loads=0
        def load_block(self,*args,**kwargs):
            self.loads+=1
            return copy.deepcopy(block)
        def tensor(self,name,device="cpu"):
            assert name=="model.language_model.embed_tokens.weight"
            return embedding.clone().to(device)
        def rotary(self,device="cpu"):
            return None
        def close(self):
            pass
    calls=[]
    class Numerical:
        DenseRawHessian=native.DenseRawHessian
        attention_inputs=staticmethod(native.attention_inputs)
        prepare_context=staticmethod(native.prepare_context)
        capture_linear_input=staticmethod(native.capture_linear_input)
        attention_finish=staticmethod(native.attention_finish)
        shared_down_input=staticmethod(native.shared_down_input)
        attention_forward=staticmethod(native.attention_forward)
        finish_layer=staticmethod(native.finish_layer)
        module=staticmethod(native.module)
        @staticmethod
        def quantize_projection(weight,raw,**kwargs):
            # A deliberately visible weight perturbation isolates orchestration
            # from trellis accuracy (independently qualified on actual CUDA).
            changed=(weight.float()*.875+.005).bfloat16()
            calls.append(dict(weight=weight.clone(),changed=changed.clone(),H=raw["H"].clone(),count=raw["count"]))
            packed=dict(trellis=changed.view(torch.int16).clone(),suh=torch.ones(weight.shape[1]).half(),
                        svh=torch.ones(weight.shape[0]).half(),mcg=torch.tensor(0xCBAC1FED,dtype=torch.uint32).view(torch.int32))
            return packed,dict(hessian_weighted_relative_error=.01),dict(hessian_rows=raw["count"],identity_rows=0)
        @staticmethod
        def reconstruct_projection(packed,shape,**kwargs):
            return packed["trellis"].view(torch.bfloat16).clone().to(kwargs["device"])
    selection={}
    attention_parts=(["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]
        if kind=="linear_attention" else ["self_attn.q_a_proj","self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa","self_attn.o_proj"])
    for part in attention_parts+["mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        linear=block.get_submodule(part)
        selection["model.language_model.layers.0."+part]=dict(in_features=linear.in_features,out_features=linear.out_features,bits=6)
    records=[dict(id="a",input_ids=[1,2,3,4]),dict(id="b",input_ids=[5,6,7,8,9])]
    source=Source()
    engine=run.Engine(source,records,tmp_path/"run",tmp_path/"export",dict(run="test"),selection,
        Progress(),numerical=Numerical,device="cpu")
    return engine,source,block,embedding,records,calls


@pytest.mark.parametrize("kind",["linear_attention","deepseek_sparse_attention"])
def test_causal_phases_match_final_native_block_and_downstream_hessians(tmp_path,monkeypatch,kind):
    from glm5_flash.core_k6 import export
    engine,source,block,embedding,records,calls=fixture(tmp_path,kind)
    initial=engine.initial()
    class StopAfterF(Exception):
        pass
    def stop(*args,**kwargs):
        raise StopAfterF
    monkeypatch.setattr(export,"assemble_layer",stop)
    with torch.inference_mode(),pytest.raises(StopAfterF):
        engine.layer(0,initial)
    assert engine.boundary.current()["layer"]==-1
    store=StageStore(engine.root/"layers/000/stages",engine.identity)
    assert store.load("f",checksum=True) is not None
    assert store.load("a") is store.load("d") is None
    checkpoints=engine.checkpoints(0)
    for name in engine.selection:
        checkpoints.install(name,block.get_submodule(name.removeprefix("model.language_model.layers.0.")))
    captured_o=[];captured_down=[]
    handles=[block.self_attn.o_proj.register_forward_pre_hook(lambda m,a:captured_o.append(a[0].clone())),
             block.mlp.down_proj.register_forward_pre_hook(lambda m,a:captured_down.append(a[0].clone()))]
    for index,row,state in engine._read(store,"f"):
        ids=torch.tensor(row["input_ids"])
        x=torch.nn.functional.embedding(ids,embedding).unsqueeze(0)
        x=x.unsqueeze(2).expand(-1,-1,4,-1).contiguous()
        with torch.inference_mode():
            expected= native.native_forward(block,dict(hidden=x))["hidden"]
        assert torch.equal(state["hidden"],expected)
        assert "prev_topk_indices" not in state
    for handle in handles:
        handle.remove()
    assert len(calls)==7 and all(row["count"]==9 for row in calls)
    # Output and down inputs must observe the serialized predecessor weights.
    for call,values in ((calls[3],captured_o),(calls[6],captured_down)):
        x=torch.cat([x.reshape(-1,x.shape[-1]).double() for x in values])
        ref=x.T@x
        assert float(torch.linalg.norm(call["H"].double()-ref)/torch.linalg.norm(ref)) < 1e-6
    assert not torch.equal(calls[0]["weight"],calls[0]["changed"])
    # Recover after F but before assembly without reloading or requantizing.
    monkeypatch.setattr(export,"assemble_layer",lambda *a,**k:dict(payload_bytes=1))
    before=source.loads
    current=engine.layer(0,initial)
    assert current["layer"]==0 and source.loads==before and len(calls)==7
    assert not engine.boundary.directory(-1).exists()
    assert not list((engine.root/"layers/000/projections").glob("*.safetensors"))


def test_resume_after_frontier_commit_finishes_retirement(tmp_path,monkeypatch):
    from glm5_flash.core_k6 import export
    engine,source,_,_,_,_=fixture(tmp_path)
    current=engine.initial()
    monkeypatch.setattr(export,"assemble_layer",lambda *a,**k:dict(payload_bytes=1))
    original=StageStore.clear
    def fail_after_pointer(store):
        raise RuntimeError("power loss after frontier publication")
    monkeypatch.setattr(StageStore,"clear",fail_after_pointer)
    with torch.inference_mode(),pytest.raises(RuntimeError,match="power loss"):
        engine.layer(0,current)
    assert engine.boundary.current(checksum=True)["layer"]==0
    assert list((engine.root/"layers/000/projections").glob("*.safetensors"))
    monkeypatch.setattr(StageStore,"clear",original)
    before=source.loads
    recovered=engine.initial()
    assert recovered["layer"]==0 and source.loads==before
    assert not (engine.root/"layers/000/stages").exists()
    assert not list((engine.root/"layers/000/projections").glob("*.safetensors"))


def test_partial_projection_resume_never_requantizes_committed_predecessor(tmp_path,monkeypatch):
    from glm5_flash.core_k6 import export
    from glm5_flash.core_k6.storage import ProjectionStore
    engine,source,_,_,_,calls=fixture(tmp_path)
    current=engine.initial()
    original=ProjectionStore.ensure
    def stop_after_qa(store,name,*args,**kwargs):
        result=original(store,name,*args,**kwargs)
        if name.endswith("self_attn.q_proj"):
            raise RuntimeError("crash after q_a bytes and receipt")
        return result
    monkeypatch.setattr(ProjectionStore,"ensure",stop_after_qa)
    with torch.inference_mode(),pytest.raises(RuntimeError,match="q_a bytes"):
        engine.layer(0,current)
    assert len(calls)==1 and engine.boundary.current()["layer"]==-1
    monkeypatch.setattr(ProjectionStore,"ensure",original)
    monkeypatch.setattr(export,"assemble_layer",lambda *a,**k:dict(payload_bytes=1))
    with torch.inference_mode():
        final=engine.layer(0,current)
    assert final["layer"]==0 and source.loads==2 and len(calls)==7

