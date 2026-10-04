"""Keep A8 opt-in packages separate from qualified decode/verify programs."""
import importlib.util
import json
from pathlib import Path
import os
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("package_exl3_aot", ROOT / "python/tools/aot/package_exl3_aot.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def package(root, *, capacity=256, activations="a8", variant_activations=None, suffix="-a8", direct=False, contract=False,
            warp_specialized=True, route_block=64):
    directory = f"rtx-tp1/m{capacity}{suffix}"
    meta = dict(capacity=capacity, hidden=6144, intermediate=2048, experts=256,
                top_k=8, output_dtype="bf16", bits=[3, 4], activations=activations,
                direct=direct, sparkinfer_revision="fixture", requires_route_preparation=False,
                warp_specialized=warp_specialized, route_block=route_block)
    variant = {k: meta[k] for k in ("capacity", "intermediate", "experts", "top_k", "output_dtype", "bits")}
    variant.update(directory=directory, activations=variant_activations or activations,
                   warp_specialized=warp_specialized, route_block=route_block)
    child = root / directory
    child.mkdir(parents=True)
    (child / "v41_exl3.json").write_text(json.dumps(meta))
    (child / "trellis_lut.bin").write_bytes(b"lut")
    (child / "libcuteafd_exl3.so").write_bytes(b"fixture")
    files = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=tool.digest(p))
             for p in root.rglob("*") if p.is_file()}
    manifest = dict(schema="cuteafd.exl3-package.v1", role="coordinator", geometry="glm",
                    sparkinfer_revision="fixture", variants=[variant], files=files)
    if contract:
        manifest["activation_variants"] = "a8"
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_a8_prefill_variant_verifies(tmp_path):
    package(tmp_path)
    assert tool.verify(tmp_path)["variants"][0]["activations"] == "a8"


@pytest.mark.parametrize("options", [dict(capacity=80), dict(suffix=""), dict(direct=True),
                                   dict(warp_specialized=False), dict(route_block=32)])
def test_a8_cannot_replace_decode_or_unpacked_programs(tmp_path, options):
    package(tmp_path, **options)
    with pytest.raises(ValueError, match="disjoint packed prefill"):
        tool.verify(tmp_path)


def test_a8_precision_claim_must_match_compiled_metadata(tmp_path):
    package(tmp_path, variant_activations="a16")
    with pytest.raises(ValueError, match="activation precision mismatch"):
        tool.verify(tmp_path)


def test_a8_build_contract_requires_prefill_siblings(tmp_path):
    package(tmp_path, activations="a16", suffix="", contract=True)
    with pytest.raises(ValueError, match="missing requested A8 prefill"):
        tool.verify(tmp_path)


def test_a8_build_rejects_more_than_two_tiers_before_cuda_import(tmp_path):
    args = SimpleNamespace(output=tmp_path / "package", activations="a8",
                           geometry="glmf", bits=[2, 3, 4])
    with pytest.raises(ValueError, match="requires two decoder tiers"):
        tool.build(args)


@pytest.mark.parametrize("entry,switch", [
    ("wip.sh", "CUTEAFD_WIP_EXL3_ACTIVATIONS"),
    ("build.sh", "CUTEAFD_RELEASE_EXL3_ACTIVATIONS"),
    ("scripts/launch/run-family.sh", "EXL3_ACTIVATIONS"),
])
def test_invalid_activation_switch_fails_before_build_or_restart(tmp_path, entry, switch):
    config = tmp_path / "config"
    config.write_text("MODEL_ID=fixture/model\n")
    env = dict(os.environ, **{switch: "bad"})
    result = subprocess.run(["bash", str(ROOT / entry), "--config", str(config)],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert f"{switch} must be a16 or a8" in result.stderr
