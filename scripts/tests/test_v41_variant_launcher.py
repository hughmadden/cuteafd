"""VARIANTS launcher keys on DeepSeek V4.1 (weight-variants contract §6): serve-native gets one
--variant per named variant; V4.1 has no variant modes; without VARIANTS nothing changes."""
import subprocess
from pathlib import Path

from test_run_family_dispatch import _family_launch_result, _snapshot

V41 = {"model_type": "deepseek_v41"}
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
CARD = "MODEL_REVISION=\nSPECULATOR=off\nVISION=off\nRTX_GPUS=1\nRTX_EXPERT_LAYERS=0\nSPARK_COUNT=2\nSPARK_1_HOST=h1\nSPARK_1_LANE_A=10.0.0.2\n"


def _coordinator(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, result.stderr
    return next(line for line in result.stderr.splitlines() if "serve-native" in line and "docker run -d" in line)


def test_variants_reach_serve_native(tmp_path: Path) -> None:
    _snapshot(tmp_path / "hf", "example-org/DeepSeek-V4.1-Flash-1017-alternate-weights", {"model_type": "x"}, "rev1")
    sidecar = tmp_path / "pack" / "wo_b_l10_35_1017.safetensors"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"")
    keys = (CARD + "VARIANTS=alternate,file\n"
            "VARIANT_ALTERNATE_SOURCE=example-org/DeepSeek-V4.1-Flash-1017-alternate-weights\nVARIANT_ALTERNATE_REVISION=rev1\n"
            "VARIANT_ALTERNATE_ID=dsf-alternate\n"
            f"VARIANT_FILE_SOURCE={sidecar}\n")
    line = _coordinator(_family_launch_result(tmp_path, V41, MODEL, keys))
    hub = (tmp_path / "hf" / "hub").resolve()
    snapshot = hub / "models--example-org--DeepSeek-V4.1-Flash-1017-alternate-weights" / "snapshots" / "rev1"
    assert (f"--variant name=alternate,source={snapshot},id=dsf-alternate "
            f"--variant name=file,source={sidecar}") in line
    assert f"-v {sidecar}:{sidecar}:ro" in line


def test_no_variants_adds_nothing(tmp_path: Path) -> None:
    line = _coordinator(_family_launch_result(tmp_path, V41, MODEL, CARD))
    # Byte-identity with the recorded V4.1 launches is test_v41_shared_launcher's.
    assert "--variant" not in line


def test_variant_mode_is_refused(tmp_path: Path) -> None:
    result = _family_launch_result(tmp_path, V41, MODEL,
        CARD + f"VARIANTS=a\nVARIANT_A_SOURCE={tmp_path}\nVARIANT_A_MODE=ffn\n")
    assert result.returncode == 2 and "VARIANT_A_MODE: this family has no variant modes" in result.stderr
    assert "docker run -d" not in result.stderr
