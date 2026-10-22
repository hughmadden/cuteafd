"""Fault injection at the expensive replay/export/frontier retirement barriers."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kimi_k3.core_k6 import run
from kimi_k3.core_k6.storage import StageStore
from kimi_k3.uniform_k2.storage import atomic_json, commit_safetensors


class Progress:
    def emit(self, *args, **kwargs):
        pass


def completed_f(tmp_path, monkeypatch):
    records = [dict(id='a', input_ids=[1,2]), dict(id='b', input_ids=[3])]
    identity = dict(run='test')
    module = 'language_model.model.layers.0.self_attn.q_proj'
    engine = run.CoreEngine(SimpleNamespace(), records, tmp_path/'run', tmp_path/'export', identity,
                            {module:{}}, Progress(), device='cpu', numerical=SimpleNamespace())
    old, successor = [], []
    store = StageStore(engine.root/'layers/000/stages', identity)
    for index, row in enumerate(records):
        hidden = torch.full((1,len(row['input_ids']),4), index + .25)
        tensors = dict(prefix_sum=hidden, block_residual=hidden.new_empty((hidden.shape[1],0,4)))
        old.append(run.commit_batch(engine.boundary.directory(-1), index, row, tensors))
        tensors = dict(prefix_sum=hidden+1, block_residual=hidden.reshape(-1,1,4))
        successor.append(run.commit_batch(store.directory('f'), index, row, tensors))
    current = engine.boundary.commit(-1, old)
    store.commit('f', successor)
    packed = engine.root/'layers/000/projections/packed.safetensors'
    packed.parent.mkdir(parents=True)
    packed.write_bytes(b'packed-candidate')
    class Checkpoints:
        def replacements(self, modules):
            assert modules == [module]
            assert packed.exists()
            return {module:packed}
        def retire_payloads(self):
            packed.unlink()
    monkeypatch.setattr(engine, '_checkpoints', lambda layer:Checkpoints())
    return engine, current, store, packed


def fake_assembly(engine):
    def assemble(source, layer, replacements, export, identity):
        path = Path(export)/'model-00001-of-00094.safetensors'
        receipt = commit_safetensors(path, {'retained':torch.arange(4)})
        receipt.update(identity=identity, filename=path.name)
        atomic_json(Path(export)/'layer-000-receipt.json', receipt)
        return receipt
    return assemble


def test_resume_completed_replay_after_interrupted_shard_assembly(tmp_path, monkeypatch):
    from kimi_k3.core_k6 import export
    engine, current, store, packed = completed_f(tmp_path, monkeypatch)
    def interrupted(*args):
        raise OSError('power loss during assembly')
    monkeypatch.setattr(export, 'assemble_layer', interrupted)
    with pytest.raises(OSError, match='power loss'):
        engine.layer(0,current)
    assert engine.boundary.current()['layer'] == -1
    assert store.load('f',checksum=True) and packed.exists()
    monkeypatch.setattr(export, 'assemble_layer', fake_assembly(engine))
    # No source.load_block exists: committed F must resume without any model
    # materialization or another expensive K2 calibration forward.
    next_boundary = engine.layer(0,current)
    assert next_boundary['layer'] == 0
    assert not engine.boundary.directory(-1).exists()
    assert store.load('f') is None and not packed.exists()
    values = run.boundary_batch(engine.boundary,next_boundary,0,'cpu')
    assert torch.equal(values['prefix_sum'],torch.full((1,2,4),1.25))


def test_resume_after_boundary_commit_finishes_retirement(tmp_path, monkeypatch):
    from kimi_k3.core_k6 import export
    engine, current, store, packed = completed_f(tmp_path, monkeypatch)
    monkeypatch.setattr(export,'assemble_layer',fake_assembly(engine))
    original_clear = StageStore.clear
    def interrupted(self):
        raise OSError('power loss after boundary publication')
    monkeypatch.setattr(StageStore,'clear',interrupted)
    with pytest.raises(OSError,match='power loss'):
        engine.layer(0,current)
    successor = engine.boundary.current(checksum=True)
    assert successor['layer'] == 0 and packed.exists()
    assert store.load('f',checksum=True)
    monkeypatch.setattr(StageStore,'clear',original_clear)
    engine.retire_committed(successor)
    assert store.load('f') is None and not packed.exists()
    # The hardlinked successor still exists after deleting the phase names.
    assert torch.equal(run.boundary_batch(engine.boundary,successor,1,'cpu')['prefix_sum'],
                       torch.full((1,1,4),2.25))


def test_projection_orphan_is_regenerated_but_committed_bytes_are_reused(tmp_path, monkeypatch):
    module='language_model.lm_head'
    selection={module:dict(out_features=4,in_features=4)}
    calls=[]
    class Numerical:
        def quantize_core_projection(self,weight,raw,**kwargs):
            calls.append(raw['count'])
            return {'trellis':torch.arange(4,dtype=torch.int16)}, {'hessian_weighted_relative_error':.01}, {'hessian_rows':7}
    checkpoints=run.ProjectionCheckpoints(tmp_path/'packed',tmp_path/'evidence',{},selection,
                                         Progress(),device='cpu',numerical=Numerical())
    original=run.atomic_json
    def interrupted(path,record):
        if Path(path)==checkpoints.paths(module)[1]:
            raise OSError('power loss before receipt')
        original(path,record)
    monkeypatch.setattr(run,'atomic_json',interrupted)
    weight=torch.ones(4,4)
    raw=dict(H=torch.eye(4),count=7,finalized=False)
    with pytest.raises(OSError):
        checkpoints.ensure(module,weight,raw)
    assert checkpoints.paths(module)[0].exists() and checkpoints.load(module) is None
    monkeypatch.setattr(run,'atomic_json',original)
    path=checkpoints.ensure(module,weight,raw)
    bytes_before=path.read_bytes()
    assert checkpoints.ensure(module,weight)==path
    assert calls==[7,7] and path.read_bytes()==bytes_before
    assert torch.equal(load_file(str(path))[module+'.trellis'],torch.arange(4,dtype=torch.int16))
