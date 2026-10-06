"""GLM 5.3 Flash's second index top-k extent (--glmf-max-context, CMake CUTEAFD_GLMF_MAX_CONTEXT) in the
coordinator exporter (python/tools/aot/export_b12x_dsv4_aot.py), run end to end without SparkInfer or a
GPU: every compile function is a stub whose "object" is its call, so an object's SHA-256 changes exactly
when its compile call does, and the manifest lists what a real export would."""
from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "python" / "tools" / "aot" / "export_b12x_dsv4_aot.py"
EXTENT_STEMS = {"glmf_index_topk_decode_m64_ctx1048576", "glmf_index_topk_prefill_m4096_ctx1048576",
                "glmf_index_topk_decode_m128_ctx1048576"}


@dataclasses.dataclass(frozen=True)
class _Geometry:
    """Every field the exporter's head-split replacements and program lists read."""

    name: str
    heads: int = 64
    kda_heads: int = 64
    full_kv_heads: int = 8
    swa_kv_heads: int = 8
    o_groups: int = 8
    dense_inter: int = 12288
    moe_inter: int = 2048
    index_kpool: int = 4
    index_topk: int = 2048
    kda_width: int = 8192
    hidden: int = 4096


class _Program:
    """A compiled program stand-in: its object bytes are its compile call."""

    def __init__(self, call):
        self.call = call
        self.abi = {"pointers": [("x", "bfloat16", "[rows]", "in")], "scalars": [("rows", "int32")]}

    def export_to_c(self, directory, stem, symbol):
        Path(directory, f"{stem}.o").write_text(repr(self.call))
        Path(directory, f"{stem}.h").write_text(f"// {symbol}\n")

    def scratch_bytes(self, rows):
        return {"scratch": rows}


class _Compilers(types.ModuleType):
    """A SparkInfer module whose ``compile_*`` functions return a program of their call."""

    def __getattr__(self, name):
        if name.startswith("compile_"):
            module = self.__name__.rsplit(".", 1)[-1]
            return lambda *args, **kwargs: _Program((module, name, args, sorted(kwargs.items())))
        raise AttributeError(name)


def _load(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("B12X_COMPILE_DISK_CACHE", "B12X_COMPILE_MEMORY_CACHE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setitem(sys.modules, "_pinned_sparkinfer", types.SimpleNamespace(REVISION="test"))
    properties = types.SimpleNamespace(major=12, minor=0)
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_properties=lambda device: properties)
    monkeypatch.setitem(sys.modules, "torch", torch)
    package = types.ModuleType("b12x.integration.cuteafd")
    for name in ("FLASH", "PRO", "GLM53", "GLM53_FLASH", "MIMO_V2_FLASH", "MIMO_V26_FLASH", "MIMO_V26_PRO",
                 "QWEN38_FLASH_NEXT"):
        setattr(package, name, _Geometry(name.lower()))
    package.exportable_compilation = contextlib.nullcontext
    package.validate_exported_header = lambda program, header, symbol: {
        "argument_count": len(program.abi["pointers"]) + len(program.abi["scalars"]) + 1, "symbol": symbol}
    for name in ("glmf", "dsv4_mhc", "glm_sparse_mla", "dsv4_compressor", "dsv4_ffn", "weights", "dsv4_indexer",
                 "dsv4_producer", "dsv4_sparse_mla", "dsv4_wo"):
        module = _Compilers(f"b12x.integration.cuteafd.{name}")
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    package.glmf.mhc_geometry = lambda g: ("mhc geometry", g.name)
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", package)
    spec = importlib.util.spec_from_file_location("export_b12x_dsv4_aot_extent_under_test", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _topk_object(mode, rows, pages):
    """The stand-in object of GLM 5.3 Flash's index top-k at `rows` rows over `pages` pool pages."""
    return repr(("glmf", "compile_glmf_index_topk_aot", (_Geometry("glm53_flash"),),
                 [("max_pages", pages), ("max_rows", rows), ("mode", mode)]))


@pytest.fixture()
def export(monkeypatch, tmp_path):
    """Runs the exporter's main() with these arguments; returns its manifest."""
    exporter = _load(monkeypatch)

    def run(*arguments, geometry="glmf,glmf2"):
        output = tmp_path / f"out{len(list(tmp_path.iterdir()))}"
        monkeypatch.setattr(sys, "argv", ["export", "--output-dir", str(output), "--geometry", geometry, *arguments])
        exporter.main()
        manifest = json.loads((output / "dsv4_programs.json").read_text())
        manifest["objects"] = {p["name"]: (output / f"{p['name']}.o").read_text() for p in manifest["programs"]}
        return manifest

    return run


def test_default_build_records_the_extent_and_adds_no_program(export):
    manifest = export()
    assert manifest["capacities"]["max_context"] == 131072
    assert manifest["families"]["glmf"]["max_context"] == 131072
    assert "max_context" not in manifest["families"]["glmf2"]
    assert not any("_ctx" in p["name"] for p in manifest["programs"])
    # The same table as an explicit extent equal to --max-context.
    same = export("--glmf-max-context", "131072")
    assert same["programs"] == manifest["programs"]


def test_the_1m_extent_adds_only_top_k_stems_after_every_other_program(export):
    base = export()
    long = export("--glmf-max-context", "1048576")
    old = base["programs"]
    # Every existing program keeps its name, params, scratch, place and object.
    assert long["programs"][: len(old)] == old
    assert all(long["objects"][p["name"]] == base["objects"][p["name"]] for p in old)
    added = long["programs"][len(old):]
    assert {p["name"] for p in added} == EXTENT_STEMS
    for program in added:
        rows = program["params"]["max_rows"]
        assert program["op"] == "index_topk" and program["family"] == "glmf"
        assert program["params"] == {"mode": "prefill" if rows == 4096 else "decode", "max_rows": rows,
                                     "max_pages": 4096, "max_context": 1048576}
        assert program["capacity_rows"] == rows
        # 262,144 pools in 4,096 pages of 64: the plain program's compile call at that page count.
        mode = program["params"]["mode"]
        assert long["objects"][program["name"]] == _topk_object(mode, rows, 4096)
        assert base["objects"][program["name"].removesuffix("_ctx1048576")] == _topk_object(mode, rows, 512)
    assert long["families"]["glmf"]["max_context"] == 1048576
    assert long["capacities"]["max_context"] == 131072


def test_without_wide_decode_programs_the_extent_has_two_stems(export):
    manifest = export("--glmf-max-context", "1048576", "--glmf-wide-decode-rows", "0")
    assert {p["name"] for p in manifest["programs"] if "_ctx" in p["name"]} == {
        "glmf_index_topk_decode_m64_ctx1048576", "glmf_index_topk_prefill_m4096_ctx1048576"}
    assert not any("m128" in p["name"] for p in manifest["programs"])


def test_a_partial_unit_extent_rounds_up_to_whole_pool_pages(export):
    manifest = export("--glmf-max-context", "1000000")
    pages = {p["params"]["max_pages"] for p in manifest["programs"] if "_ctx1000000" in p["name"]}
    assert pages == {3907}  # ceil(1,000,000 / 256)
    assert manifest["families"]["glmf"]["max_context"] == 1000000


def test_an_extent_shorter_than_max_context_is_refused(export):
    with pytest.raises(SystemExit, match="shorter than --max-context"):
        export("--glmf-max-context", "65536")


def test_other_families_ignore_the_glmf_extent(export):
    manifest = export("--glmf-max-context", "1048576", geometry="flash")
    assert "glmf" not in manifest["families"]
    assert not any("_ctx" in p["name"] or p["name"].startswith("glmf") for p in manifest["programs"])
    assert manifest["programs"] == export(geometry="flash")["programs"]


def test_cmake_passes_the_extent_and_builds_forward_it():
    cmake = (ROOT / "native" / "cmake" / "shared" / "dsv4_programs.cmake").read_text()
    assert '--glmf-max-context "${glmf_max_context}"' in cmake
    assert 'set(CUTEAFD_GLMF_MAX_CONTEXT "" CACHE STRING' in (ROOT / "native" / "CMakeLists.txt").read_text()
    assert '-DCUTEAFD_GLMF_MAX_CONTEXT="${CUTEAFD_WIP_GLMF_MAX_CONTEXT:-}"' in \
        (ROOT / "scripts" / "build" / "build-wip-artifacts.sh").read_text()
    assert '-e "CUTEAFD_WIP_GLMF_MAX_CONTEXT=${CUTEAFD_WIP_GLMF_MAX_CONTEXT:-}"' in (ROOT / "wip.sh").read_text()
    assert '-DCUTEAFD_GLMF_MAX_CONTEXT="${CUTEAFD_RELEASE_GLMF_MAX_CONTEXT:-}"' in \
        (ROOT / "scripts" / "build" / "build-release-artifacts.sh").read_text()
    assert '-e "CUTEAFD_RELEASE_GLMF_MAX_CONTEXT=${CUTEAFD_RELEASE_GLMF_MAX_CONTEXT:-}"' in \
        (ROOT / "build.sh").read_text()
