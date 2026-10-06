"""``--glm-zero-masked-v`` (CMake CUTEAFD_GLM_ZERO_MASKED_V) in the coordinator exporter
(python/tools/aot/export_b12x_dsv4_aot.py), run end to end without SparkInfer or a GPU: every
compile function is a stub whose "object" is its call. The option reaches exactly the GLM 5.x and
GLM 5.3 Flash sparse MLA programs (b12x ``zero_masked_v``: a masked slot's staged V and FP32
scales are zeroed, so record slot 0 may hold any bytes); every other program, and the default
export, stay as they were."""
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
GLM = "glm,glm2,glmf,glmf2"
SPARSE_MLA = {f"{family}_sparse_mla_{route}_m{rows}" for family in ("glm", "glm2", "glmf", "glmf2")
              for route, rows in (("decode", 64), ("prefill", 4096))} | {"glmf_sparse_mla_decode_m128"}


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


@pytest.fixture()
def export(monkeypatch, tmp_path):
    """Runs the exporter's main() with these arguments; returns its manifest and objects."""
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
    for name in ("glmf", "dsv4_mhc", "glm_sparse_mla", "glm_attention", "glm_ffn", "glm_indexer",
                 "dsv4_compressor", "dsv4_ffn", "weights", "dsv4_indexer", "dsv4_producer", "dsv4_sparse_mla",
                 "dsv4_wo"):
        module = _Compilers(f"b12x.integration.cuteafd.{name}")
        setattr(package, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    package.glmf.mhc_geometry = lambda g: ("mhc geometry", g.name)
    monkeypatch.setitem(sys.modules, "b12x", types.ModuleType("b12x"))
    monkeypatch.setitem(sys.modules, "b12x.integration", types.ModuleType("b12x.integration"))
    monkeypatch.setitem(sys.modules, "b12x.integration.cuteafd", package)
    spec = importlib.util.spec_from_file_location("export_b12x_dsv4_aot_masked_v_under_test", EXPORTER)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    def run(*arguments, geometry=GLM):
        output = tmp_path / f"out{len(list(tmp_path.iterdir()))}"
        monkeypatch.setattr(sys, "argv", ["export", "--output-dir", str(output), "--geometry", geometry, *arguments])
        exporter.main()
        manifest = json.loads((output / "dsv4_programs.json").read_text())
        manifest["objects"] = {p["name"]: (output / f"{p['name']}.o").read_text() for p in manifest["programs"]}
        return manifest

    return run


def test_the_default_export_never_names_the_option(export):
    manifest = export()
    assert {p["name"] for p in manifest["programs"] if p["op"] == "sparse_mla"} == SPARSE_MLA
    assert not any("zero_masked_v" in p["params"] for p in manifest["programs"])
    assert not any("zero_masked_v" in obj for obj in manifest["objects"].values())


def test_the_option_reaches_only_the_glm_sparse_mla_programs(export):
    base, masked = export(), export("--glm-zero-masked-v")
    # Same stems in the same places: the serving engine loads them unchanged.
    assert [p["name"] for p in masked["programs"]] == [p["name"] for p in base["programs"]]
    for old, new in zip(base["programs"], masked["programs"]):
        before, after = base["objects"][old["name"]], masked["objects"][new["name"]]
        if old["name"] in SPARSE_MLA:
            assert new["params"] == {**old["params"], "zero_masked_v": True}, old["name"]
            assert "('zero_masked_v', True)" in after and after.replace(", ('zero_masked_v', True)", "") == before
            assert {k: v for k, v in new.items() if k not in ("params", "object_sha256")} == \
                {k: v for k, v in old.items() if k not in ("params", "object_sha256")}
        else:
            assert new == old and after == before, old["name"]


def test_other_families_ignore_the_option(export):
    assert export("--glm-zero-masked-v", geometry="flash")["programs"] == export(geometry="flash")["programs"]


def test_cmake_passes_the_option_and_builds_forward_it():
    cmake = (ROOT / "native" / "cmake" / "shared" / "dsv4_programs.cmake").read_text()
    option = "if(CUTEAFD_GLM_ZERO_MASKED_V)\n  list(APPEND CUTEAFD_DSV4_EXPORT_ARGS --glm-zero-masked-v)\nendif()"
    # Before the stamp, so toggling the option re-exports the programs.
    assert option in cmake and cmake.index(option) < cmake.index('file(GENERATE OUTPUT "${stamp}"')
    assert "option(CUTEAFD_GLM_ZERO_MASKED_V " in (ROOT / "native" / "CMakeLists.txt").read_text()
    assert '-DCUTEAFD_GLM_ZERO_MASKED_V="${CUTEAFD_WIP_GLM_ZERO_MASKED_V:-OFF}"' in \
        (ROOT / "scripts" / "build" / "build-wip-artifacts.sh").read_text()
    assert '-e "CUTEAFD_WIP_GLM_ZERO_MASKED_V=${CUTEAFD_WIP_GLM_ZERO_MASKED_V:-OFF}"' in (ROOT / "wip.sh").read_text()
    assert '-DCUTEAFD_GLM_ZERO_MASKED_V="${CUTEAFD_RELEASE_GLM_ZERO_MASKED_V:-OFF}"' in \
        (ROOT / "scripts" / "build" / "build-release-artifacts.sh").read_text()
    assert '-e "CUTEAFD_RELEASE_GLM_ZERO_MASKED_V=${CUTEAFD_RELEASE_GLM_ZERO_MASKED_V:-OFF}"' in \
        (ROOT / "build.sh").read_text()
