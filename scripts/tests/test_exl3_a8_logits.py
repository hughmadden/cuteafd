"""Semantic checks for the full-vocabulary A16/A8 quality evidence tool."""
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[2] / "python/tools/bench/compare_exl3_a8_logits.py"


def artifact(path, probs=(0.75, 0.25), token=0, rows=257):
    path.mkdir()
    np.full(rows, token, dtype="<u4").tofile(path / "tokens.bin")
    np.tile(np.log(probs), (rows, 1)).astype("<f4").tofile(path / "logits.bin")
    return path


def compare(tmp_path, base, candidate, golden):
    out = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(TOOL), "--a16", str(base), "--a8", str(candidate),
                             "--golden", str(golden), "--out", str(out)], text=True, capture_output=True)
    return result, json.loads(out.read_text()) if out.exists() else None


def test_full_vocab_kl_detects_confidence_change_without_top1_change(tmp_path):
    base = artifact(tmp_path / "base")
    candidate = artifact(tmp_path / "candidate", (0.5, 0.5))
    golden = artifact(tmp_path / "golden")
    result, report = compare(tmp_path, base, candidate, golden)
    assert result.returncode == 1
    metrics = report["metrics"]
    assert metrics["a16_kl_a8"] == pytest.approx(0.75 * math.log(1.5) + 0.25 * math.log(0.5), abs=1e-7)
    assert metrics["a16_nll"] == pytest.approx(-math.log(0.75), abs=1e-7)
    assert metrics["a8_nll"] == pytest.approx(math.log(2), abs=1e-7)
    assert metrics["a16_top1_a8"] == 1
    assert report["full_vocabulary"]
    assert not report["numerics_gate"]


def test_equal_models_pass_with_zero_drift(tmp_path):
    base = artifact(tmp_path / "base")
    candidate = artifact(tmp_path / "candidate")
    golden = artifact(tmp_path / "golden")
    result, report = compare(tmp_path, base, candidate, golden)
    assert result.returncode == 0, result.stderr
    assert report["numerics_gate"]
    assert report["metrics"]["a16_kl_a8"] == 0
    assert report["metrics"]["nll_delta"] == 0


@pytest.mark.parametrize("corrupt", ["tokens", "shape", "nonfinite", "decode_only"])
def test_invalid_comparison_cannot_emit_passing_evidence(tmp_path, corrupt):
    (tmp_path / "report.json").write_text('{"numerics_gate":true}')
    base = artifact(tmp_path / "base")
    candidate = artifact(tmp_path / "candidate", token=1 if corrupt == "tokens" else 0,
                         rows=80 if corrupt == "decode_only" else 257)
    golden = artifact(tmp_path / "golden")
    if corrupt == "shape":
        with (candidate / "logits.bin").open("ab") as stream:
            stream.write(b"x")
    if corrupt == "nonfinite":
        values = np.memmap(candidate / "logits.bin", dtype="<f4", mode="r+")
        values[0] = np.nan
        values.flush()
    result, report = compare(tmp_path, base, candidate, golden)
    assert result.returncode != 0
    assert report is None
