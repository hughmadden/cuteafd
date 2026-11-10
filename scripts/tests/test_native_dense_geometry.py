"""Exercise native operand admission without importing the CUDA stack."""
import ast
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[2] / "third_party/sparkinfer/b12x/integration/cuteafd"


def geometry(module):
    tree = ast.parse((FORK / (module + ".py")).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "geometry"]
    namespace = {"ABI_VERSION": 1, "CTAS_PER_SM": 4}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), module, "exec"), namespace)
    return namespace["geometry"]


@pytest.mark.parametrize("module,k,n,kwargs,splits", [
    ("trellis_linear", 2048, 16384, {}, 2),
    ("trellis_linear", 16384, 6144, {"sharded_side": 2}, 16),
])
def test_real_dense_geometry(module, k, n, kwargs, splits):
    g = geometry(module)(k, n, **kwargs)
    assert g["input_features"] == k and g["output_features"] == n
    assert g["max_rows"] == 4096 and g["splits"] == splits
    assert g["ctas_per_sm"] == 4


@pytest.mark.parametrize("k,n,kwargs,segments", [
    (6144, 2048, {}, 8),
    (6144, 576, {}, 8),
    (2048, 4096, {}, 4),
    (6144, 128, {}, 8),
    (2048, 16384, {"encoding": "int4_g32"}, 1),
    (16384, 6144, {"encoding": "int4_g32", "sharded_side": 2}, 4),
    (6144, 154880, {"encoding": "fp8_row"}, 1),
])
def test_scaled_tile_geometry(k, n, kwargs, segments):
    # Tile core: shape-only K segments, no split-K partial scratch.
    g = geometry("scaled_linear")(k, n, **kwargs)
    assert g["core"] == "tile" and g["splits"] == 1 and g["segments"] == segments
    assert (k // 128) % segments == 0
    assert g["max_rows"] == 4096 and g["ctas_per_sm"] == 4


@pytest.mark.parametrize("module", ["trellis_linear", "scaled_linear"])
@pytest.mark.parametrize("k,n,kwargs", [
    (0, 128, {}), (128, -1, {}), (True, 128, {}), (128, 128, {"max_rows": 0}),
    (129, 128, {}), (128, 129, {}), (128, 128, {"max_rows": 2**31}),
    (2**30, 2**30, {}), (128, 128, {"sharded_side": 0}),
])
def test_reject_malformed_geometry(module, k, n, kwargs):
    with pytest.raises(ValueError):
        geometry(module)(k, n, **kwargs)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_trellis_supported_bit_widths(bits):
    g = geometry("trellis_linear")(128, 128, bits=bits)
    assert g["bits"] == bits
    assert g["codebook"] == "e4m3" and g["abi_version"] == 1


@pytest.mark.parametrize("kwargs", [{"bits": 1}, {"bits": 7}, {"codebook": "bogus"},
                                      {"rotation_count": 0}, {"rotation_count": True}])
def test_trellis_reject_unsupported_format(kwargs):
    with pytest.raises(ValueError):
        geometry("trellis_linear")(128, 128, **kwargs)


@pytest.mark.parametrize("kwargs", [{"encoding": "bogus"},
                                      {"encoding": "fp8_row", "sharded_side": 2}])
def test_scaled_reject_unsupported_format(kwargs):
    with pytest.raises(ValueError):
        geometry("scaled_linear")(128, 128, **kwargs)
