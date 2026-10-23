"""CPU arithmetic contracts for the MiniMax image and video tower."""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/reference"))
import minimax_m3_vision as reference


@pytest.fixture(scope="module")
def torch():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch
    torch.set_num_threads(previous)


def test_position_ids_merge_window_order():
    positions = reference.position_ids([(2, 2, 4), (1, 2, 2)])
    assert positions == [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (0, 0, 2), (0, 0, 3), (0, 1, 2), (0, 1, 3),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
        (1, 0, 2), (1, 0, 3), (1, 1, 2), (1, 1, 3),
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]


@pytest.mark.parametrize("grid", [[(0, 2, 2)], [(1, 3, 2)], [(1, 2, 3)], [(1, -2, 2)]])
def test_position_ids_reject_incomplete_grids(grid):
    with pytest.raises(ValueError, match="complete spatial merge windows"):
        reference.position_ids(grid)


@pytest.mark.parametrize("grids", [[(1, 48, 48)], [(2, 4, 6)], [(1, 24, 72), (2, 6, 4)]])
def test_positions_match_pinned_transformers(torch, grids):
    source = ROOT / "third_party/transformers/src/transformers/vision_utils.py"
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == reference.POSITION_SOURCE_SHA256
    node = next(n for n in ast.parse(data).body if isinstance(n, ast.FunctionDef)
                and n.name == "get_vision_position_ids")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    expected = scope["get_vision_position_ids"](torch.tensor(grids), 2, include_temporal=True)
    assert reference.position_ids(grids) == [tuple(row) for row in expected.tolist()]


def test_axial_rope_frequencies_and_passthrough(torch):
    positions = [(2, 3, 5), (7, 11, 13)]
    cos, sin = reference.rotary_factors(positions, torch.float32)
    assert cos.shape == sin.shape == (2, 78)
    for row, pos in enumerate(positions):
        for axis, position in enumerate(pos):
            for band in range(13):
                channel = axis * 13 + band
                phase = position * 10000 ** (-(2 * band) / 26)
                assert cos[row, channel].item() == pytest.approx(math.cos(phase), abs=1e-6)
                assert sin[row, channel].item() == pytest.approx(math.sin(phase), abs=1e-6)
    assert torch.equal(cos[:, :39], cos[:, 39:])
    values = torch.arange(2 * 16 * 80, dtype=torch.float32).reshape(2, 16, 80) / 500
    rotated = reference.apply_rope(values, cos, sin)
    assert torch.equal(rotated[..., 78:], values[..., 78:])
    expected = values.clone()
    for row in range(2):
        for i in range(39):
            expected[row, :, i] = values[row, :, i] * cos[row, i] - values[row, :, i + 39] * sin[row, i]
            expected[row, :, i + 39] = values[row, :, i + 39] * cos[row, i] + values[row, :, i] * sin[row, i]
    torch.testing.assert_close(rotated, expected, rtol=0, atol=0)


def test_bfloat16_rope_rounds_each_product(torch):
    x = torch.linspace(-2, 2, 80, dtype=torch.bfloat16).reshape(1, 1, 80)
    cos, sin = reference.rotary_factors([(1, 3, 7)], torch.bfloat16)
    actual = reference.apply_rope(x, cos, sin)
    a = (x[..., :39].float() * cos[:, None, :39].float()).bfloat16()
    b = (x[..., 39:78].float() * sin[:, None, :39].float()).bfloat16()
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual[..., :39], a - b)
    assert torch.equal(actual[..., 78:], x[..., 78:])


def test_merge_groups_consecutive_rows(torch):
    features = torch.arange(8 * 3).reshape(8, 3)
    assert reference.merge_patches(features).tolist() == [list(range(12)), list(range(12, 24))]
    with pytest.raises(ValueError, match="complete merge windows"):
        reference.merge_patches(features[:7])


def test_projector_applies_both_biased_exact_gelu_mlps(torch):
    F = torch.nn.functional
    generator = torch.Generator(device="cpu").manual_seed(2)
    shapes = {"multi_modal_projector.linear_1": (10, 8), "multi_modal_projector.linear_2": (6, 10),
              "patch_merge_mlp.linear_1": (10, 24), "patch_merge_mlp.linear_2": (6, 10)}
    weights = {}
    for name, (out_dim, in_dim) in shapes.items():
        weights[name + ".weight"] = torch.randn(out_dim, in_dim, generator=generator).bfloat16() / 8
        weights[name + ".bias"] = torch.randn(out_dim, generator=generator).bfloat16() / 8
    hidden = torch.randn(8, 8, generator=generator).bfloat16()
    expected = hidden
    for prefix in ("multi_modal_projector", "patch_merge_mlp"):
        if prefix == "patch_merge_mlp":
            expected = torch.stack([torch.cat(tuple(expected[i:i + 4])) for i in (0, 4)])
        for layer in (1, 2):
            name = prefix + f".linear_{layer}"
            expected = F.linear(expected, weights[name + ".weight"], weights[name + ".bias"])
            if layer == 1:
                expected = F.gelu(expected, approximate="none")
    actual = reference.project_and_merge(hidden, weights)
    assert actual.shape == (2, 6)
    assert torch.equal(actual, expected)


@pytest.fixture(scope="module")
def real_layer_weights(torch):
    generator = torch.Generator(device="cpu").manual_seed(31)
    weights = {}
    prefix = reference.VISION_PREFIX
    weights[prefix + "embeddings.patch_embedding.weight"] = torch.randn(1280, 3, 2, 14, 14, generator=generator) / 32
    for name in ("pre_layrnorm", "encoder.layers.0.layer_norm1", "encoder.layers.0.layer_norm2"):
        weights[prefix + name + ".weight"] = torch.ones(1280, dtype=torch.bfloat16)
        weights[prefix + name + ".bias"] = torch.randn(1280, generator=generator).bfloat16() / 32
    shapes = {"self_attn." + name + "_proj": (1280, 1280) for name in ("q", "k", "v", "out")}
    shapes.update({"mlp.fc1": (5120, 1280), "mlp.fc2": (1280, 5120)})
    for name, (out_dim, in_dim) in shapes.items():
        key = prefix + "encoder.layers.0." + name
        weights[key + ".weight"] = torch.randn(out_dim, in_dim, generator=generator, dtype=torch.bfloat16) / math.sqrt(in_dim)
        weights[key + ".bias"] = torch.randn(out_dim, generator=generator).bfloat16() / 32
    return weights


def test_one_real_geometry_layer_and_f32_convolution(torch, real_layer_weights):
    generator = torch.Generator(device="cpu").manual_seed(7)
    patches = torch.randn(4, 1176, generator=generator)
    result = reference.forward(patches, [(1, 2, 2)], real_layer_weights, layers=1, project=False)
    conv = torch.nn.functional.conv3d(patches.reshape(4, 3, 2, 14, 14),
                                     real_layer_weights[reference.VISION_PREFIX + "embeddings.patch_embedding.weight"],
                                     stride=(2, 14, 14)).reshape(4, 1280).bfloat16()
    assert torch.equal(result["embeddings"], conv)
    assert result["features"].shape == (4, 1280)
    assert result["features"].dtype == torch.bfloat16
    assert torch.isfinite(result["features"]).all()
    changed = patches.clone()
    changed[-1].neg_()
    other = reference.forward(changed, [(1, 2, 2)], real_layer_weights, layers=1, project=False)
    assert not torch.equal(result["features"][0], other["features"][0])


def test_images_do_not_attend_other_images(torch, real_layer_weights):
    generator = torch.Generator(device="cpu").manual_seed(9)
    patches = torch.randn(8, 1176, generator=generator)
    separate = reference.forward(patches[:4], [(1, 2, 2)], real_layer_weights, layers=1, project=False)
    packed = reference.forward(patches, [(1, 2, 2), (1, 2, 2)], real_layer_weights, layers=1, project=False)
    torch.testing.assert_close(separate["features"], packed["features"][:4], rtol=0, atol=0)


def test_forward_rejects_wrong_patch_shape_and_checkpoint_dtype(torch, real_layer_weights):
    with pytest.raises(ValueError, match="patch tensor shape"):
        reference.forward(torch.zeros(3, 1176), [(1, 2, 2)], real_layer_weights, layers=1, project=False)
    weights = dict(real_layer_weights)
    key = reference.VISION_PREFIX + "pre_layrnorm.weight"
    weights[key] = weights[key].float()
    with pytest.raises(ValueError, match="unexpected checkpoint dtype"):
        reference.forward(torch.zeros(4, 1176), [(1, 2, 2)], weights, layers=1, project=False)
