"""MXFP4 tail layout admission stays explicit; FP8/NVFP4 keep 128-row blocks."""
from dataclasses import dataclass, replace
import importlib.util
import sys
import subprocess
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fp8_tail_package", REPO / "python/tools/aot/package_fp8_moe_aot.py")
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)


@dataclass(frozen=True)
class Geometry:
    intermediate: int = 2048
    tp: int = 1
    weights: str = "mxfp4"

    @property
    def slice(self):
        return ((self.intermediate // 32 + self.tp - 1) // self.tp * 32 + 127) // 128 * 128

    def with_tp(self, tp):
        return replace(self, tp=tp)


def test_tail_widths_cover_every_checkpoint_block_without_padding():
    assert package.exact_widths(2048, 6, 32) == [352, 352, 352, 352, 320, 320]
    assert package.exact_widths(2048, 6) == [384, 384, 384, 384, 256, 256]
    assert sum(package.exact_widths(2048, 6, 32)) == 2048
    assert package.exact_widths(2048, 2, 32) == [1024, 1024]


@pytest.mark.parametrize("tp,width", [(6, 352), (6, 320), (6, 384), (6, 256), (2, 1024)])
def test_explicit_mxfp4_layout_uses_its_exact_width(tp, width):
    actual_tp, g = package.layout_geometry(Geometry(), f"tp{tp}-w{width}")
    assert actual_tp == tp and g.slice == width and g.tp == tp
    assert g.intermediate == 2048 and g.weights == "mxfp4"


@pytest.mark.parametrize("weights", ["fp8", "nvfp4"])
def test_other_formats_do_not_admit_mxfp4_tails(weights):
    with pytest.raises(SystemExit):
        package.layout_geometry(Geometry(weights=weights), "tp6-w352")
    _, g = package.layout_geometry(Geometry(weights=weights), "tp6-w256")
    assert g.slice == 256


@pytest.mark.parametrize("inter,tp,block", [(2048, 0, 32), (2047, 6, 32), (128, 6, 32), (2048, 6, 16)])
def test_invalid_partition_is_rejected(inter, tp, block):
    assert package.exact_widths(inter, tp, block) == []


def test_tail_packages_include_every_width_and_retain_qualified_layouts():
    assert package.package_layouts(Geometry(), ["tp6", "tp2"], exact_slices=True) == [
        "tp6", "tp2", "tp6-w384", "tp6-w256"]
    assert package.package_layouts(Geometry(), ["tp6", "tp2"], exact_slices=True, mxfp4_tails=True) == [
        "tp6", "tp2", "tp6-w384", "tp6-w352", "tp6-w320", "tp6-w256"]
    assert package.package_layouts(Geometry(), ["tp6-w352", "tp6-w320", "tp6-w352"]) == [
        "tp6-w352", "tp6-w320"]
    assert package.package_layouts(Geometry(), ["tp6"]) == ["tp6"]


@pytest.mark.parametrize("weights,exact", [("mxfp4", False), ("fp8", True), ("nvfp4", True)])
def test_tail_package_opt_in_rejects_incompatible_flags(weights, exact):
    with pytest.raises(SystemExit, match="requires MXFP4 geometry and --exact-slices"):
        package.package_layouts(Geometry(weights=weights), ["tp6"], exact_slices=exact, mxfp4_tails=True)


@pytest.mark.parametrize("architecture,opt_in,enabled", [("121", "ON", True), ("121", "OFF", False),
                                                       ("120", "ON", False)])
def test_cmake_exports_tail_layouts_only_for_mxfp4_spark_packages(tmp_path, architecture, opt_in, enabled):
    if not shutil.which("cmake") or not shutil.which("ninja"):
        pytest.skip("CMake/Ninja unavailable")
    source = tmp_path / "source/native"
    source.mkdir(parents=True)
    (source.parent / "python").symlink_to(REPO / "python", target_is_directory=True)
    (source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.20)
project(tail_packages LANGUAGES CXX)
add_library(CUDA::cudart SHARED IMPORTED GLOBAL)
set_target_properties(CUDA::cudart PROPERTIES IMPORTED_LOCATION /cuda/lib/libcudart.so)
set(CUDAToolkit_INCLUDE_DIRS /cuda/include)
set(Python3_EXECUTABLE "{sys.executable}")
set(CUTEAFD_CUDA_ARCHITECTURES "{architecture}")
set(CUTEAFD_EXPERT_FAMILIES "mimo:fp8;mimop:fp8;glm:nvfp4")
set(CUTEAFD_FP8_MOE_BF16_FAMILIES "mimop" CACHE STRING "" FORCE)
set(CUTEAFD_ENABLE_MXFP4_TAILS {opt_in} CACHE BOOL "" FORCE)
set(CUTEAFD_B12X_AOT_RUNTIME_LIBRARY /cuda/lib/libcute_dsl_runtime.so)
include("{REPO / 'native/cmake/shared/fp8_moe.cmake'}")
''')
    build = tmp_path / "cmake-build"
    configured = subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja"],
                                text=True, capture_output=True, timeout=60)
    assert configured.returncode == 0, configured.stderr
    commands = subprocess.run(["ninja", "-C", str(build), "-t", "commands", "cuteafd_fp8_moe_packages"],
                              text=True, capture_output=True, timeout=30)
    assert commands.returncode == 0, commands.stderr
    exporters = [line for line in commands.stdout.splitlines() if "package_fp8_moe_aot.py build" in line]
    mimop = [line for line in exporters if "--geometry mimop " in line]
    assert len(mimop) == (2 if architecture == "121" else 1)
    assert all(("--mxfp4-tails" in line) == enabled for line in mimop)
    assert all("--mxfp4-tails" not in line for line in exporters if "--geometry mimop " not in line)


@pytest.mark.parametrize("opt_in", ["ON", "OFF"])
def test_release_remote_decoder_preserves_tail_build_opt_in(opt_in):
    text = (REPO / "build.sh").read_text()
    decoder = 'set -euo pipefail\nremote_dir="$1"' + text.split(
        'set -euo pipefail\nremote_dir="$1"', 1)[1].split('\ncd "$remote_dir"', 1)[0]
    arguments = ["/source", "dev", "inference", "engine", "revision", "version", "OFF",
                 "__legacy__", "tp6", "__legacy__", "mimop:fp8", "mimop", opt_in]
    result = subprocess.run(["bash", "-c", decoder + '\nprintf "%s\\n" "$mxfp4_tails"',
                             "decoder", *arguments], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == opt_in
