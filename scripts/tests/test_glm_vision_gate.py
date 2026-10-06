"""CPU contracts for GLM's ABI2 tower qualification; no CUDA imports."""
import ctypes
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_gate():
    path = ROOT / "python/tools/qualify/glm5_flash/qualify-vision.py"
    spec = importlib.util.spec_from_file_location("glm_vision_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glm_abi_preserves_mimo_prefix():
    gate = load_gate()
    assert ctypes.sizeof(gate.Block) == 96
    assert gate.Spec.blocks.offset == 64
    assert gate.Spec.hidden.offset == 2752
    assert gate.Spec.patch_bias.offset == 2792
    assert gate.Spec.norm1_bias.offset == 2896
    assert ctypes.sizeof(gate.Spec) == 3792


def test_glm_patch_order_duplicates_temporal_axis_and_keeps_merge_units():
    gate = load_gate()
    rgb = np.arange(28 * 56 * 3, dtype=np.uint16).astype(np.uint8).reshape(28, 56, 3)
    lut = gate.common.normalization_lut()
    values = gate.patches(rgb, lut)
    assert values.shape == (8, 1176)
    for patch in range(8):
        unit, within = divmod(patch, 4)
        y, x = within // 2 * 14, unit * 28 + within % 2 * 14
        for channel in range(3):
            expected = lut[channel, rgb[y:y + 14, x:x + 14, channel]].reshape(-1)
            np.testing.assert_array_equal(values[patch, channel * 392:channel * 392 + 196], expected)
            np.testing.assert_array_equal(values[patch, channel * 392 + 196:(channel + 1) * 392], expected)


def test_glm_qualifier_uses_snapshot_normalization_not_fixed_defaults(tmp_path):
    import json
    gate = load_gate()
    processor = dict(patch_size=14, temporal_patch_size=2, merge_size=2,
        do_rescale=True, image_mean=[0.1, 0.2, 0.3], image_std=[0.4, 0.5, 0.6])
    path = tmp_path / "processor_config.json"
    path.write_text(json.dumps({"image_processor": processor}))
    mean, std = np.asarray(processor["image_mean"], np.float32), np.asarray(processor["image_std"], np.float32)
    expected = (np.arange(256, dtype=np.float32)[None, :] * np.float32(1 / 255) - mean[:, None]) / std[:, None]
    np.testing.assert_array_equal(gate.normalization_lut(tmp_path), expected)
    assert not np.array_equal(expected, gate.common.normalization_lut())
    for changed in (dict(patch_size=16), dict(image_std=[0, 1, 1]), dict(do_rescale=False)):
        path.write_text(json.dumps({"image_processor": dict(processor, **changed)}))
        with pytest.raises(ValueError):
            gate.normalization_lut(tmp_path)


def test_glm_shapes_and_calibrated_floor_keep_strict_failure():
    gate = load_gate()
    for count in (256, 1024, 4096):
        gh, gw, rgb = gate.fixture(count)
        assert gh * gw == count * 4
        assert rgb.shape == (gh * 14, gw * 14, 3)
    floor = dict(relative_l2=0.035, mean_cosine=0.9996, worst_cosine=0.97, pass_=False)
    native = dict(relative_l2=0.036, mean_cosine=0.99957, worst_cosine=0.9695, **{"pass": False})
    measured = gate.common.calibrated_metrics(native, floor)
    assert measured["pass"] and not measured["strict_pass"]
    assert not gate.common.calibrated_metrics(dict(native, mean_cosine=0.99949), floor)["pass"]


def test_glm_bf16_relative_floor_preserves_strict_merger_miss():
    gate = load_gate()
    floor = dict(relative_l2=0.35, mean_cosine=0.9968, worst_cosine=0.13, **{"pass": False})
    native = dict(relative_l2=0.20, mean_cosine=0.9987, worst_cosine=0.38, **{"pass": False})
    calibrated = gate.calibrated_metrics(native, floor, "16")
    assert calibrated["pass"] and not calibrated["strict_pass"]
    merger = gate.calibrated_metrics(native, floor, "26")
    assert merger["pass"] and not merger["strict_pass"]
    for changed in (dict(relative_l2=0.353), dict(mean_cosine=0.9967), dict(worst_cosine=0.128)):
        assert not gate.calibrated_metrics(dict(native, **changed), floor, "16")["pass"]


def test_glm_pointwise_kernels_cover_capped_grid_tail():
    import re
    source = (ROOT / "native/shared/cuda/vision_glm.cuh").read_text()
    for name in ("glm_rgb_patches", "glm_bias", "glm_conv_gather"):
        body = re.search(r"__global__ void " + name + r"\(.*?(?=\n(?:__global__|//|int encode_glm))", source, re.S).group()
        assert "i+=size_t(gridDim.x)*blockDim.x" in body, name
    assert "namespace {" not in source, "included inside the shared owner's anonymous namespace"


def test_glm_g1_reads_actual_nested_processor(tmp_path):
    import json
    path = ROOT / "scripts/qualify/media/preprocess-goldens.py"
    spec = importlib.util.spec_from_file_location("preprocess_goldens_glm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.glm_processor_config()
    changed = dict(config, min_image_tokens=32, max_image_tokens=1024)
    (tmp_path / "processor_config.json").write_text(json.dumps({"image_processor": changed,
        "video_processor": {"patch_size": 99}}))
    assert module.glm_processor_config(tmp_path) == changed
    for invalid in ({}, {"image_processor": []}, {"image_processor": {"patch_size": 16}}):
        (tmp_path / "processor_config.json").write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            module.glm_processor_config(tmp_path)
