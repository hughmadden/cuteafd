"""Qwen tower ABI, interpolation order and head72 export contracts, CPU-only."""
import ctypes
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_head72_uses_the_b12x_le128_tile():
    export = module("python/tools/aot/export_b12x_vision_attention_aot.py", "vision_export")
    assert export.tile_for_head_dim(64) == (128,128)
    assert export.tile_for_head_dim(72) == (128,64)
    assert export.tile_for_head_dim(128) == (128,64)
    for dim in (0,73,136):
        with pytest.raises(ValueError): export.tile_for_head_dim(dim)


def test_qwen_abi_keeps_mimo_prefix_and_matches_c_rust_suffix():
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "qwen_vision_gate")
    assert ctypes.sizeof(gate.Block) == 96
    assert ctypes.sizeof(gate.common.Spec) == 2752
    assert ctypes.sizeof(gate.Spec) == 3792
    assert gate.Spec.hidden.offset == 2752
    assert gate.Spec.patch_bias.offset == 2792
    assert gate.Spec.norm1_bias.offset == 2896


def test_qwen_patch_lut_uses_host_f64_rescale_then_f32_normalize():
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "qwen_lut_gate")
    lut = gate.normalization_lut()
    expected = np.array([np.float32((np.float32(i*(1/255))-.5)/.5) for i in range(256)])
    np.testing.assert_array_equal(lut, np.broadcast_to(expected,(3,256)))
    gh,gw,rgb = gate.common.fixture(256)
    patches = gate.common.patches(rgb,lut).reshape(gh*gw,3,2,16,16)
    np.testing.assert_array_equal(patches[:,:,0],patches[:,:,1])


def test_qwen_calibration_keeps_strict_misses_and_final_output_mean_floor():
    gate = module("python/tools/qualify/qwen4/qualify-vision.py", "qwen_calibration_gate")
    floor = dict(relative_l2=.040262, mean_cosine=.997386, worst_cosine=.870111, **{"pass":False})
    measured = dict(relative_l2=.037299, mean_cosine=.997565, worst_cosine=.939182, **{"pass":False})
    intermediate = gate.calibrated_metrics(measured,floor,final_output=False)
    assert intermediate["pass"] and not intermediate["strict_pass"]
    assert not gate.calibrated_metrics(measured,floor,final_output=True)["pass"]
    # MiMo's original policy remains unchanged and still fails this intermediate.
    assert not gate.common.calibrated_metrics(measured,floor)["pass"]
    for field,value in (("relative_l2",.042263),("mean_cosine",.997335),("worst_cosine",.869110)):
        assert not gate.calibrated_metrics(dict(measured,**{field:value}),floor,final_output=False)["pass"]
    final = dict(relative_l2=.025771,mean_cosine=.999705,worst_cosine=.995945,**{"pass":True})
    final_floor = dict(relative_l2=.027005,mean_cosine=.999679,worst_cosine=.994904,**{"pass":True})
    assert gate.calibrated_metrics(final,final_floor,final_output=True)["pass"]


def test_qwen_pointwise_launches_cover_every_element():
    source = (ROOT / "native/shared/cuda/vision_qwen.cuh").read_text()
    # Every pointwise kernel using the unchanged capped launcher must grid-stride.
    assert source.count("i+=size_t(gridDim.x)*blockDim.x") == 5
    assert "(cast_float<<<grid_for(" not in source
    for count in (4096*256+1,256*4*1152,4096*4304,4096*2560):
        stride = min((count+255)//256,4096)*256
        first_thread = (count-1)%stride
        visited = range(first_thread,count,stride)
        assert visited[-1] == count-1
    test = (ROOT / "native/tests/vision_pointwise_test.cu").read_text()
    assert "4096*256+1" in test
    for kernel in ("qwen_patch_position","qwen_residual","qwen_biased_gelu","qwen_biased_cast","qwen_cast_float"):
        assert kernel in test


def test_qwen_interpolation_is_multiply_then_divide_not_ratio():
    # At rectangular sizes a precomputed ratio changes taps by an FP32 ULP.
    positions = np.arange(256,dtype=np.float32)
    official = positions*np.float32(47)/np.float32(255)
    ratio = positions*np.float32(np.float32(47)/np.float32(255))
    assert np.any(official != ratio)
    source = (ROOT/"native/shared/cuda/vision_qwen.cuh").read_text()
    assert "__fdiv_rn(float(y)*47.f,float(gh-1))" in source
    assert "__fdiv_rn(float(x)*47.f,float(gw-1))" in source
