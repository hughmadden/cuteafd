"""The shared block oracle retains GLM math and honors dense tap/Markov semantics."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "python/reference/families/glm5_flash/dspark/reference.py"
GOLDEN = ROOT / "python/reference/families/qwen3_5/golden.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_tap_argument_keeps_projector_order():
    golden = load(GOLDEN, "dense_taps_golden")
    assert golden.parse_tap_layers("52,4,16,28,40") == [52, 4, 16, 28, 40]
    assert golden.parse_tap_layers("") == []
    for value in ("-1", "4,4"):
        with pytest.raises(argparse.ArgumentTypeError):
            golden.parse_tap_layers(value)


def test_reference_does_not_use_inherited_spec_generate():
    source = ast.parse(ORACLE.read_text())
    assert not any(isinstance(node, ast.Attribute) and node.attr == "spec_generate"
                   for node in ast.walk(source))
    body = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "body")
    assert "repeat_interleave" in ast.unparse(body)
    assert 'drafter.window is not None' in ORACLE.read_text()


def test_markov_chain_confidence_and_dense_taps(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    oracle = load(ORACLE, "block_dspark_reference")
    drafter = object.__new__(oracle.Drafter)
    drafter.block, drafter.hidden = 2, 2
    drafter.head = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
    drafter.w1 = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
    drafter.w2 = torch.tensor([[0., 0.], [0., 0.], [4., 0.]])
    drafter.conf_w = torch.tensor([1., 0., 2., 3.])
    drafter.conf_b = torch.tensor(-1.)
    hidden = torch.tensor([[0., 2.], [3., 0.]])
    tokens, confidence = drafter.heads_(1, hidden)
    assert tokens == [2, 0]  # second row uses W1[2], not W1[anchor=1]
    assert confidence == pytest.approx([torch.sigmoid(torch.tensor(1.)).item(),
                                       torch.sigmoid(torch.tensor(5.)).item()])
    values = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    (tmp_path / "layer04.bin").write_bytes(values.view(torch.uint16).numpy().tobytes())
    assert torch.equal(oracle.golden_taps(tmp_path, [4], 2, "cpu", streams=1), values)
    streams = values[:, None].repeat(1, 4, 1)
    (tmp_path / "layer04.bin").write_bytes(streams.view(torch.uint16).numpy().tobytes())
    assert torch.equal(oracle.golden_taps(tmp_path, [4], 2, "cpu"), values)


def test_scored_logits_offset_is_not_a_target_position(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    oracle = load(ORACLE, "block_dspark_scoring")
    (tmp_path / "meta.json").write_text(json.dumps({"logits_start": 3}))
    np.asarray([[0., 2.], [3., 0.]], dtype="<f4").tofile(tmp_path / "logits.bin")
    assert oracle.greedy_picks(tmp_path, 6, 2) == [-1, -1, -1, 1, 0, -1]


def test_gqa_block_and_yarn_match_bundled_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    oracle = load(ORACLE, "block_dspark_yarn")
    drafter = object.__new__(oracle.Drafter)
    drafter.device = "cpu"
    drafter.cfg = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=1, head_dim=4, rope_parameters=dict(rope_type="yarn", rope_theta=10000.,
                       factor=32., original_max_position_embeddings=8192, beta_fast=32., beta_slow=1.))
    drafter.rope_config = drafter.cfg["rope_parameters"]
    x = torch.arange(24).reshape(3, 2, 4).to(torch.bfloat16) / 10
    positions = torch.tensor([0, 8192, 262143])
    rotary = Qwen3RotaryEmbedding(Qwen3Config(**drafter.cfg))
    cos, sin = rotary(x[None, :, 0], positions[None])
    expected = x * cos[0, :, None] + torch.cat([-x[..., 2:], x[..., :2]], -1) * sin[0, :, None]
    assert torch.equal(drafter.rotary(x, positions), expected)
