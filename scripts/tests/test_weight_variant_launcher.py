"""VARIANTS launcher keys (weight-variants contract §6): serve-glm gets one --variant per
named variant, and with no VARIANTS the coordinator command is byte-identical."""
import subprocess
from pathlib import Path

import pytest

from test_run_family_dispatch import _family_launch_result, _snapshot

GLM = {"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "first_k_dense_replace": 3}
ROOT = Path(__file__).resolve().parents[2]


def _coordinator(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, result.stderr
    return next(line for line in result.stderr.splitlines() if "docker run" in line and " serve-glm " in line)


def test_variants_reach_serve_glm(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hf", "drowzeys/keys-GLM-5.3-EXL3-Abliterated", GLM, "rev1")
    outside = tmp_path / "pack"
    outside.mkdir()
    keys = ("VARIANTS=abliterated,keys-2\n"
            "VARIANT_ABLITERATED_SOURCE=drowzeys/keys-GLM-5.3-EXL3-Abliterated\nVARIANT_ABLITERATED_REVISION=rev1\n"
            "VARIANT_ABLITERATED_ID=glm53-abliterated\nVARIANT_ABLITERATED_MODE=shared-only\n"
            f"VARIANT_KEYS_2_SOURCE={outside}\n")
    line = _coordinator(_family_launch_result(tmp_path, GLM, "zai-org/GLM-5.3", keys))
    hub = (tmp_path / "hf" / "hub").resolve()
    snapshot = hub / "models--drowzeys--keys-GLM-5.3-EXL3-Abliterated" / "snapshots" / "rev1"
    assert (f"--variant name=abliterated,source={snapshot},id=glm53-abliterated,mode=shared-only "
            f"--variant name=keys-2,source={outside}") in line
    # A path outside the namespace is mounted read-only at the same path.
    assert f"-v {outside}:{outside}:ro" in line


def test_no_variants_adds_nothing(tmp_path: Path) -> None:
    line = _coordinator(_family_launch_result(tmp_path, GLM, "zai-org/GLM-5.3", ""))
    assert "--variant" not in line and ":ro " not in line.split(" serve-glm ")[0].split("-v ")[-1]


def _byte_identical_against(revision: str, tmp_path: Path) -> None:
    """One-off check for a review: the coordinator command at `revision` and now, no VARIANTS."""
    before = tmp_path / "before" / "run-family.sh"
    before.parent.mkdir()
    before.write_text(subprocess.run(["git", "show", f"{revision}:scripts/launch/run-family.sh"], cwd=ROOT,
                                     capture_output=True, text=True, check=True).stdout)
    old = _coordinator(_family_launch_result(tmp_path / "old", GLM, "zai-org/GLM-5.3", "", launcher_source=before))
    new = _coordinator(_family_launch_result(tmp_path / "new", GLM, "zai-org/GLM-5.3", ""))
    assert new.replace(str(tmp_path / "new"), "T") == old.replace(str(tmp_path / "old"), "T")


@pytest.mark.parametrize("keys,diagnostic", [
    ("VARIANTS=a\n", "VARIANT_A_SOURCE is required"),
    ("VARIANTS=a b\nVARIANT_A_SOURCE=/x\n", "VARIANTS names must be"),
    ("VARIANTS=a\nVARIANT_A_SOURCE=/does/not/exist\n", "does not exist"),
    ("VARIANTS=a\nVARIANT_A_SOURCE=org/missing\n", "missing variant snapshot"),
    ("VARIANTS=a\nVARIANT_A_SOURCE=/tmp\nVARIANT_B_SOURCE=/tmp\n", "unused configuration key: VARIANT_B_SOURCE"),
])
def test_variant_keys_are_checked(tmp_path: Path, keys: str, diagnostic: str) -> None:
    result = _family_launch_result(tmp_path, GLM, "zai-org/GLM-5.3", keys)
    assert result.returncode != 0
    assert diagnostic in result.stderr, result.stderr


@pytest.mark.skipif(not __import__("os").environ.get("CUTEAFD_VARIANT_BASELINE_REV"),
                    reason="set CUTEAFD_VARIANT_BASELINE_REV to compare against a launcher revision")
def test_no_variants_is_byte_identical_to_baseline(tmp_path: Path) -> None:
    _byte_identical_against(__import__("os").environ["CUTEAFD_VARIANT_BASELINE_REV"], tmp_path)
