"""Split Kimi AOT catalog is opt-in and independent of CUDA imports."""
import ast
from functools import partial
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]


def test_split_catalog_covers_whole_and_head_split(monkeypatch):
    path = ROOT / "python/tools/aot/export_b12x_kimi_split_aot.py"
    tree = ast.parse(path.read_text())
    for module, names in (
        ("_kda_contract", ("KDAGeometry",)),
        ("kimi_mla", ()),
        ("glmf", ("compile_glmf_kda_aot", "compile_glmf_kda_commit_aot")),
        ("pointwise", ("compile_bf16_projection_aot",)),
        ("kimi_split", ("compile_kimi_kda_core_aot", "compile_kimi_mla_rank_norm_aot",
                        "compile_kimi_mla_query_aot", "compile_kimi_mla_gate_output_aot"))):
        fake = types.ModuleType("b12x.integration.cuteafd." + module)
        for name in names:
            setattr(fake, name, lambda *args, **kw: (args, kw))
        if module == "kimi_mla":
            fake.KimiMlaGeometry = lambda **kw: kw
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    namespace = {"partial": partial}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)],
                           type_ignores=[]), str(path), "exec"), namespace)
    entries = namespace["programs"]()
    assert len(entries) == 56
    assert len({name for name, _, _ in entries}) == 56
    assert all(not name.startswith("kimik31_") for name, _, _ in entries[:32])
    assert all(name.startswith("kimik31_") for name, _, _ in entries[32:])
    for heads in (48, 96):
        for rows in (64, 4096):
            selected = {name: thunk for name, thunk, capacity in entries
                        if f"h{heads}_" in name and capacity == rows and not name.startswith("kimik31_")}
            assert len(selected) == (8 if heads == 96 else 7)
            if heads == 96:
                args, kw = selected[f"kda_fused_h96_m{rows}"]()
                assert args[0][1] == {"hidden": 7168, "kda_heads": 96}
                assert kw == {"max_rows": rows, "output_kind": "norm", "decode": rows == 64}
            assert selected[f"kda_core_h{heads}_m{rows}"]()[1] == {
                "max_rows": rows, "decode": rows == 64}
            for stride in (576, 640):
                for cache, suffix in (("bf16", ""), ("fp8", "_fp8")):
                    args, kw = selected[f"mla_rank_norm_h{heads}_kv{stride}{suffix}_m{rows}"]()
                    assert kw["kv_stride"] == stride and args[0] == {"heads": heads, "kv_cache": cache}

    catalogue = {name: thunk for name, thunk, _ in entries}
    for name, thunk, capacity in entries[32:]:
        args, kw = thunk()
        assert args[0]["norm_eps"] == 1e-5
        assert kw["max_rows"] == capacity
        if "rank_norm" in name:
            assert args[0]["kv_cache"] in ("bf16", "fp8")
            assert kw["kv_stride"] in (576, 640)
    args, kw = catalogue["kda_commit_h96"]()
    assert args[0][1] == {"hidden": 7168, "kda_heads": 96} and not kw
    assert catalogue["kda_output_h96"]() == ((), {"width": 12288, "outputs": 7168})


def test_split_has_dedicated_sm_comparison_group():
    spec = importlib.util.spec_from_file_location("sm_exports", ROOT / "scripts/build/compare-sm-exports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert next(j for j in module.jobs("all") if j[0] == "kimi-split") == (
        "kimi-split", "export_b12x_kimi_split_aot.py", [])
