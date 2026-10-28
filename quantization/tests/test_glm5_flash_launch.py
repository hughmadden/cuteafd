"""MaxQ's remote hardware lease, durable supervision and pilot admission."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glm5_flash.core_k6 import launch, preflight


def test_gpu_lease_is_on_raptor_but_numerics_and_preflights_are_on_maxq():
    common = ["--source-snapshot", "/source", "--tokens", "/tokens", "--root", "/run", "--export", "/export"]
    command = launch.gpu_command(common, container="owned-glmf", lock_helper="/remote/hwlock.py",
                                 qualification_report="/proof.json")
    assert command[0] == "ssh" and command[-2] == "raptor"
    remote = shlex.split(command[-1])
    assert remote[:2] == ["python3", "/remote/hwlock.py"]
    assert remote[remote.index("--locks") + 1] == "maxq"
    precheck = shlex.split(remote[remote.index("--preflight") + 1])
    assert precheck[-2] == "maxq"
    assert "preflight.py" in precheck[-1] and "/proof.json" in precheck[-1]
    assert "30d" in remote and "run_guard.py" in remote[-1]
    assert "container.py" in remote[-1] and "run.py" in remote[-1]
    assert "--publish-only" not in remote[-1]
    assert "gpu0" not in command[-1] and "gpu1" not in command[-1]


def test_pilot_has_short_timeout_and_fresh_qualification_admission():
    command = launch.gpu_command(["--root", "/pilot"], container="pilot", lock_helper="/helper.py", pilot=True)
    remote = shlex.split(command[-1])
    assert "25m" in remote and "30d" not in remote
    assert "--qualification-only" in remote[remote.index("--preflight") + 1]
    assert "pilot.py" in remote[-1] and "run.py" not in remote[-1]


def test_qualification_rejects_failed_checks_changed_source_and_wrong_fork(tmp_path, monkeypatch):
    from glm5_flash.core_k6 import pilot
    monkeypatch.setattr(preflight, "REPO", tmp_path)
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda *a, **k: "fork\n")
    code = tmp_path / "run.py"
    code.write_text("producer version 1")
    proof = dict(status="passed", layers=[1, 4, 7],
        native_checks={k: True for k in preflight.CHECKS},
        results=[dict(tensor=name, arm=arm, nmse=0 if arm=="bf16" else 1e-4 if arm=="mcg-k6" else 5e-4)
                 for layer in pilot.LAYERS for name in pilot.modules(layer) for arm in pilot.ARMS],
        identity=dict(gptqmodel_commit="fork", code={"run.py": hashlib.sha256(code.read_bytes()).hexdigest()}))
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    assert preflight.qualified(path, "fork")["status"] == "passed"
    with pytest.raises(ValueError, match="incomplete"):
        preflight.qualified(path, "other-fork")
    proof["native_checks"]["mtp_boundary_byte_exact"] = False
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="qualification failed"):
        preflight.qualified(path, "fork")
    proof["native_checks"]["mtp_boundary_byte_exact"] = True
    path.write_text(json.dumps(proof))
    code.write_text("different producer")
    with pytest.raises(ValueError, match="producer changed"):
        preflight.qualified(path, "fork")


def test_source_admission_requires_the_maxq_copy(monkeypatch):
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: "maxq")
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda *a, **k: "  raptor ready\n  maxq missing 1 file\n")
    with pytest.raises(ValueError, match="local in Sparknest"):
        preflight.require_local_source()
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda *a, **k: "  maxq ready\n")
    preflight.require_local_source()
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: "raptor")
    with pytest.raises(ValueError, match="only on maxq"):
        preflight.require_local_source()


def test_publication_waits_for_gpu_lease_exit_and_uses_no_cuda(tmp_path, monkeypatch):
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(dict(command=["leased-gpu"], publication_command=["cpu-publish"],
        environment={"HOME": "/home/tj"}, log=str(tmp_path / "run.log"), container="owned")))
    events = []
    class Process:
        pid = 100
        def __init__(self, command, **kwargs):
            self.command = command
            events.append(("spawn", command, kwargs.get("env")))
        def wait(self):
            events.append(("exit", self.command))
            return 0
    monkeypatch.setattr(launch.subprocess, "Popen", Process)
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kwargs: events.append(("stop", command)))
    monkeypatch.setattr(launch.signal, "signal", lambda *args: None)
    assert launch.supervise(path) == 0
    assert [e[0] for e in events] == ["spawn", "exit", "stop", "spawn", "exit"]
    assert events[3][2]["CUDA_VISIBLE_DEVICES"] == ""
    assert json.loads((tmp_path / "launch-exit.json").read_text())["returncode"] == 0


def test_failed_gpu_job_cannot_publish(tmp_path, monkeypatch):
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(dict(command=["leased-gpu"], publication_command=["cpu-publish"],
        environment={}, log=str(tmp_path / "run.log"), container="owned")))
    calls = []
    def spawn(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=1, wait=lambda: 2)
    monkeypatch.setattr(launch.subprocess, "Popen", spawn)
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(launch.signal, "signal", lambda *args: None)
    assert launch.supervise(path) == 2
    assert calls == [["leased-gpu"]]


@pytest.mark.parametrize("lfs", [False, True])
def test_source_audit_checks_hub_content_identity_and_rejects_corruption(tmp_path, lfs):
    from glm5_flash.core_k6.source_audit import hash_file
    data = b"original checkpoint bytes"
    path = tmp_path / "weight-or-asset"
    path.write_bytes(data)
    expected = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest() if lfs else None,
        git_blob_sha1=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
    assert hash_file(path, expected)["sha256"] == hashlib.sha256(data).hexdigest()
    path.write_bytes(b"X" + data[1:])
    with pytest.raises(ValueError, match="checksum differs"):
        hash_file(path, expected)
