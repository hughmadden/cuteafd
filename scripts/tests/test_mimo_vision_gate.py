"""Qualification-floor checks; no CUDA/checkpoint needed."""
import importlib.util
from pathlib import Path


def _harness():
    path = Path(__file__).resolve().parents[2] / "python/tools/qualify/mimo_v2/qualify-vision.py"
    spec = importlib.util.spec_from_file_location("mimo_vision_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_keeps_strict_fail_and_demands_native_mean_floor():
    gate = _harness()
    native = {
        "relative_l2": 0.014, "mean_cosine": 0.99995,
        "worst_cosine": 0.962, "pass": False,
    }
    official = dict(relative_l2=0.054, mean_cosine=0.99918, worst_cosine=0.914)
    measured = gate.calibrated_metrics(native, official)
    assert measured["pass"] and not measured["strict_pass"]
    assert not gate.calibrated_metrics(dict(native, mean_cosine=0.99949), official)["pass"]
    assert native["pass"] is False


def test_calibration_is_no_worse_than_bf16_even_if_literal_bar_passes():
    gate = _harness()
    native = {
        "relative_l2": 0.011, "mean_cosine": 0.99996,
        "worst_cosine": 0.995, "pass": True,
    }
    official = dict(relative_l2=0.009, mean_cosine=0.99997, worst_cosine=0.999)
    assert not gate.calibrated_metrics(native, official)["pass"]
    assert not gate.calibrated_metrics(
        dict(native, relative_l2=0.02, worst_cosine=0.999), official
    )["pass"]


def test_allocation_gate_uses_ledger_not_cuda_free_memory():
    gate = _harness()
    initial = gate.Ledger(device_allocations=2)
    before = dict(free=1000, total=2000)
    after = dict(free=800, total=2000)
    for sm in ([12, 0], (12, 1)):
        result = gate.allocation_metrics(initial, gate.Ledger(device_allocations=2), before, after, sm)
        assert result["no_encode_device_allocation"]
        assert result["steady_cuda_memory_growth_bytes"] == 200
        assert result["unified_memory_observation"] == (tuple(sm) == (12, 1))
        assert not gate.allocation_metrics(initial, gate.Ledger(device_allocations=3), before, before, sm)["no_encode_device_allocation"]
    assert not gate.allocation_metrics(gate.Ledger(device_allocations=1), initial, before, before, (12, 1))["no_encode_device_allocation"]


def test_rectangular_rgb_patch_order_and_temporal_duplicate():
    import numpy as np

    gate = _harness()
    gh, gw, rgb = gate.fixture(256)
    lut = gate.normalization_lut()
    patches = gate.patches(rgb, lut).reshape(gh * gw, 3, 2, 16, 16)
    np.testing.assert_array_equal(patches[:, :, 0], patches[:, :, 1])
    for patch, y, x in [(0, 0, 0), (1, 0, 16), (2, 16, 0), (3, 16, 16), (4, 0, 32)]:
        expected = np.stack([lut[c, rgb[y:y + 16, x:x + 16, c]] for c in range(3)])
        np.testing.assert_array_equal(patches[patch, :, 0], expected)
