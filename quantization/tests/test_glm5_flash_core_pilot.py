"""Decisive contracts of the shortened Flash pilot, without GPUs or weights."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.core_k6.pilot import (ARMS, CHECKS, LAYERS, Capture, exact_state,
    modules, native_staged_boundary, select_records, tp2_projection_bytes, validate_qualification)


def records(prefix, lengths):
    return [dict(id=f"{prefix}-{i}", input_ids=list(range(n))) for i,n in enumerate(lengths)]


def report():
    return dict(status="passed", native_checks=dict.fromkeys(CHECKS, True), results=[
        dict(tensor=name, arm=arm, nmse={"mcg-k6":.00001,"fp8-block128":.001,"bf16":0}[arm])
        for layer in LAYERS for name in modules(layer) for arm in ARMS])


def test_pilot_has_fixed_representatives_and_only_decisive_arms():
    assert LAYERS == (1,4,7)
    assert ARMS == ("mcg-k6","fp8-block128","bf16")
    assert all(len(modules(layer))==7 for layer in LAYERS)
    assert "model.language_model.layers.1.self_attn.q_proj" in modules(1)
    assert "model.language_model.layers.4.mlp.shared_experts.down_proj" in modules(4)
    assert "model.language_model.layers.7.self_attn.q_a_proj" in modules(7)
    assert not any(".kv_b_proj" in name or ".indexer." in name or ".f_a_proj" in name for l in LAYERS for name in modules(l))
    with pytest.raises(ValueError,match="outside"):
        modules(3)


def test_whole_original_prompts_and_long_witness_not_duplicated():
    calibration=records("cal",[1200,1000,2300,600])
    heldout=records("held",[600,600,600])
    cal,held,witness=select_records(calibration,heldout,2048,1024)
    assert [r["id"] for r in cal] == ["cal-0","cal-1","cal-2"]
    assert [r["id"] for r in held] == ["held-0","held-1"]
    assert cal[-1] is witness is calibration[2]
    assert cal[0]["input_ids"]==calibration[0]["input_ids"]
    cal,_,witness=select_records(calibration,heldout,4096,1024)
    assert sum(r["id"]==witness["id"] for r in cal)==1


def test_short_panel_does_not_invent_long_context_or_allow_overlap():
    calibration=records("a",[1500,1000,1000])
    heldout=records("h",[600,600])
    _,_,witness=select_records(calibration,heldout,2048,1024)
    assert witness is None
    with pytest.raises(ValueError,match="overlap"):
        select_records(calibration,calibration,2048,1024)


def test_exact_boundary_includes_all_native_mhc_streams_and_dsa_indices():
    state=dict(hidden=torch.arange(4*8).reshape(1,1,4,8), prev_topk_indices=torch.tensor([[[3,-1]]]))
    exact_state(state,{k:v.clone() for k,v in state.items()})
    with pytest.raises(ValueError,match="boundary differs"):
        exact_state(state,dict(hidden=state["hidden"]))
    other={k:v.clone() for k,v in state.items()}
    other["hidden"][0,0,3,4]+=1
    with pytest.raises(ValueError,match="boundary differs"):
        exact_state(state,other)


def test_capture_detaches_staged_oracle_and_counts_only_observations():
    names=modules(1)
    block=torch.nn.Module()
    block.projections=torch.nn.ModuleDict({str(i):torch.nn.Linear(128,128,bias=False) for i in range(7)})
    lookup={name.removeprefix("model.language_model.layers.1."):block.projections[str(i)] for i,name in enumerate(names)}
    numerical=SimpleNamespace(module=lambda block,name:lookup[name])
    capture=Capture(numerical,block,1)
    x=torch.randn(1,17,128)
    for layer in block.projections.values(): layer(x)
    assert all(raw["count"]==17 for raw in capture.raw.values())
    for raw in capture.raw.values(): torch.testing.assert_close(raw["H"],x.reshape(17,128).T@x.reshape(17,128))
    capture.close()
    for layer in block.projections.values(): layer(x)
    assert all(raw["count"]==17 for raw in capture.raw.values())
    capture.attach()
    for layer in block.projections.values(): layer(x)
    assert all(raw["count"]==34 for raw in capture.raw.values())
    capture.close()


def test_tp2_storage_keeps_replicated_mla_and_split_kda():
    assert tp2_projection_bytes("model.language_model.layers.7.self_attn.q_a_proj",(1536,4096),"bf16")==[1536*4096*2]*2
    assert tp2_projection_bytes("model.language_model.layers.4.self_attn.q_proj",(8192,4096),"bf16")==[4096*4096*2]*2
    assert tp2_projection_bytes("model.language_model.layers.4.self_attn.o_proj",(4096,8192),"bf16")==[4096*4096*2]*2


def test_qualification_requires_every_measured_proof_and_exact_inventory():
    good=report()
    assert validate_qualification(good) is good
    good["native_checks"]["mtp_boundary_byte_exact"]=False
    with pytest.raises(ValueError,match="qualification failed"):
        validate_qualification(good)
    good=report(); good["results"].pop()
    with pytest.raises(ValueError,match="inventory"):
        validate_qualification(good)
    good=report(); good["results"][0]["nmse"]=float("nan")
    with pytest.raises(ValueError,match="metric"):
        validate_qualification(good)


def test_quality_gate_rejects_k6_worse_than_fp8_control():
    bad=report(); bad["results"][0]["nmse"]=.01
    with pytest.raises(ValueError,match="exceeds the FP8"):
        validate_qualification(bad)


def test_boundary_compares_hidden_indexer_output_and_observes_sparse_selection():
    class Indexer(torch.nn.Module):
        index_topk=8
        index_kpool=4
        def forward(self, hidden):
            return torch.arange(8).expand(1,hidden.shape[1],8).clone()
    block=SimpleNamespace(self_attn=SimpleNamespace(indexer=Indexer()))
    def run(block,state):
        block.self_attn.indexer(state["hidden"])
        return dict(hidden=state["hidden"])
    numerical=SimpleNamespace(native_forward=run,staged_forward=run)
    _,stats=native_staged_boundary(numerical,block,dict(hidden=torch.zeros(1,12,4,8)))
    assert stats==dict(byte_exact=True,tokens=12,valid_last_row=8,sparse_selection=True,
                       raw_topk=8,pool_size=4,pool_budget=2)
    assert not block.self_attn.indexer._forward_hooks
