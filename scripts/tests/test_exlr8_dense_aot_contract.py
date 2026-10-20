"""Dense operand export catalog checks without importing CUDA."""
import ast
from functools import partial
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def catalog(monkeypatch):
    path = ROOT / "python/tools/aot/export_b12x_exlr8_dense_aot.py"
    tree = ast.parse(path.read_text())
    namespace = {"partial": partial}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    for name, compiler in (("trellis_linear", "compile_trellis_linear_aot"),
                           ("scaled_linear", "compile_scaled_linear_aot"),
                           ("fp8_row_embedding", "compile_fp8_row_embedding_aot")):
        fake = types.ModuleType("b12x.integration.cuteafd." + name)
        setattr(fake, compiler, lambda **kwargs: kwargs)
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["programs"]()


def test_dense_trellis_catalog_has_native_full_shapes(monkeypatch):
    entries = catalog(monkeypatch)[:4]
    assert [name for name, _, _, _ in entries] == [
        "trellis_q_b", "trellis_o", "trellis_gate_up", "trellis_down"]
    assert [(p["input_features"], p["output_features"], p["sharded_side"])
            for _, _, p, _ in entries] == [
        (2048, 16384, 1), (16384, 6144, 2), (6144, 12288, 1), (12288, 6144, 2)]
    for _, op, params, thunk in entries:
        assert op == "trellis_linear"
        assert params["max_rows"] == 4096
        assert params["bits"] == 4 and params["codebook"] == "e4m3"
        assert params["rotation_count"] == 4
        assert thunk() == params


def test_dense_tp2_catalog_halves_only_the_owned_axis(monkeypatch):
    entries = {name: (op, params, thunk)
               for name, op, params, thunk in catalog(monkeypatch)}
    assert len(entries) == 18
    for name in ("trellis_q_b", "trellis_o", "trellis_gate_up", "trellis_down",
                 "int4_q_b", "int4_o"):
        op, full, _ = entries[name]
        half_op, half, thunk = entries[name + "_tp2"]
        axis = "output_features" if full["sharded_side"] == 1 else "input_features"
        expected = dict(full, **{axis: full[axis] // 2})
        assert half_op == op
        assert half == expected
        assert half["max_rows"] == 4096
        assert thunk() == half
    assert not any(name.endswith("_tp2") and name.startswith(("mxfp8", "fp8_row"))
                   for name in entries)


def test_dense_has_opt_in_sm_comparison_group(monkeypatch):
    spec = importlib.util.spec_from_file_location("sm_exports", ROOT / "scripts/build/compare-sm-exports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name, exporter, args = next(j for j in module.jobs("all") if j[0] == "exlr8-dense-trellis")
    assert exporter == "export_b12x_dsv4_aot.py"
    assert args[:2] == ["--geometry", "exlr8_dense"]
    assert args[3].split(",") == [name for name, _, _, _ in catalog(monkeypatch)[:4]]
    groups = [j for j in module.jobs("all") if j[0].startswith("exlr8-dense-")]
    selected = [name for _, _, args in groups for name in args[3].split(",")]
    assert sorted(selected) == sorted(name for name, _, _, _ in catalog(monkeypatch))
    assert "exlr8_dense" not in module.GEOMETRIES
