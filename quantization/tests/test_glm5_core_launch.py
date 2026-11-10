"""GPU ownership and publication separation in detached production launches."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from glm5.core_k6 import launch,preflight


@pytest.mark.parametrize("publication",[False,True])
def test_launch_records_bounded_guarded_quant_or_cpu_only_publish(tmp_path,monkeypatch,publication):
    root=tmp_path/"state"
    argv=["launch.py","--source-snapshot",str(tmp_path/"source"),"--tokens",str(tmp_path/"tokens"),
          "--root",str(root),"--export",str(tmp_path/"export")]
    if publication:
        argv.append("--publish-only")
    checks=[];spawns=[]
    monkeypatch.setattr(sys,"argv",argv)
    monkeypatch.setattr(launch.subprocess,"run",lambda command,**kwargs:checks.append(command))
    def spawn(command,**kwargs):
        spawns.append((command,kwargs));return SimpleNamespace(pid=123456789)
    monkeypatch.setattr(launch.subprocess,"Popen",spawn)
    launch.main()
    latest=json.loads((root/"latest-attempt.json").read_text())
    record=json.loads(Path(latest["launch"]).read_text())
    assert len(checks)==len(spawns)==1 and spawns[0][1]["start_new_session"] is True
    assert str(root) in record["command"]
    assert record["environment"]["HF_HUB_CACHE"].endswith("huggingface/cuteafd-local-hub")
    if publication:
        assert "--publish-only" in record["command"]
        assert record["environment"]["CUDA_VISIBLE_DEVICES"]==""
        assert "--publication-only" in checks[0]
        assert not any("hwlock.py" in arg or "container.py" in arg for arg in record["command"])
    else:
        command=record["command"]
        assert any(arg.endswith("hwlock.py") for arg in command)
        assert command[command.index("--wait")+1]=="60"
        assert command[command.index("--locks")+1]=="gpu1"
        assert "--preflight" in command and "30d" in command
        assert any(arg.endswith("run_guard.py") for arg in command)
        assert command.count(record["container"])==2
        assert command[command.index("--memory-gib")+1]=="60"
        assert "--publish-only" not in command


def test_live_prior_supervisor_prevents_duplicate_launch(tmp_path,monkeypatch):
    root=tmp_path/"state";(root/"attempts").mkdir(parents=True)
    (root/"attempts/old-process.json").write_text(json.dumps(dict(supervisor_pid=777,pid=778)))
    monkeypatch.setattr(sys,"argv",["launch.py","--source-snapshot",str(tmp_path/"source"),
        "--tokens",str(tmp_path/"tokens"),"--root",str(root),"--export",str(tmp_path/"export")])
    monkeypatch.setattr(launch.os,"kill",lambda *a:None)
    monkeypatch.setattr(launch.subprocess,"run",lambda *a,**k:pytest.fail("must not preflight a duplicate"))
    with pytest.raises(ValueError,match="remains active"):
        launch.main()


@pytest.mark.parametrize("host",["oliver","maxq"])
def test_quantization_host_requires_its_own_complete_source(monkeypatch,host):
    monkeypatch.setattr(preflight.socket,"gethostname",lambda:host)
    assert preflight.quantization_host()==host
    other="maxq" if host=="oliver" else "oliver"
    def where(command,**kwargs):
        assert command==["nest","where","hf:zai-org/GLM-5.3-BF16"]
        assert kwargs["timeout"]==60
        return f"  {other} ready\n  models ready\n  {host} missing 1 file\n"
    monkeypatch.setattr(preflight.subprocess,"check_output",where)
    with pytest.raises(ValueError,match="sealed locally on "+host):
        preflight.require_local_source(host)
    monkeypatch.setattr(preflight.subprocess,"check_output",lambda *a,**k:f"  {host} ready\n")
    preflight.require_local_source(host)


@pytest.mark.parametrize("host",["raptor","rhea","unrelated-maxq"])
def test_quantization_host_rejects_other_cluster_hosts(monkeypatch,host):
    monkeypatch.setattr(preflight.socket,"gethostname",lambda:host)
    with pytest.raises(ValueError,match="Oliver/MaxQ only"):
        preflight.quantization_host()


@pytest.mark.parametrize("qualification",[False,True])
@pytest.mark.parametrize("wsl_path",[False,True])
def test_preflight_bootstrap_still_checks_local_source_gpu_and_producer(tmp_path,monkeypatch,qualification,wsl_path):
    repo=tmp_path;fork=repo/"third_party/gptqmodel";fork.mkdir(parents=True)
    (fork.parent/"gptqmodel.lock.json").write_text(json.dumps(dict(revision="pinned",source_tree_sha256="tree")))
    monkeypatch.setattr(preflight,"REPO",repo)
    monkeypatch.setattr(preflight.socket,"gethostname",lambda:"maxq")
    monkeypatch.setattr(preflight.shutil,"which",lambda name:None if wsl_path else "/usr/bin/nvidia-smi")
    monkeypatch.setattr(preflight.runpy,"run_path",lambda path:{"source_tree_sha256":lambda source:"tree"})
    qualified=[];checks=[];runs=[]
    monkeypatch.setattr(preflight,"qualified",lambda *a,**k:qualified.append((a,k)))
    def output(command,**kwargs):
        checks.append(command)
        if "--porcelain" in command:return ""
        if "rev-parse" in command:return "pinned\n"
        if command[0]=="nest":return "maxq ready\n"
        if Path(command[0]).name=="nvidia-smi":
            assert command[0]==("/usr/lib/wsl/lib/nvidia-smi" if wsl_path else "/usr/bin/nvidia-smi")
            return "NVIDIA RTX PRO 6000 Blackwell Max-Q, 97887\n"
        pytest.fail(str(command))
    monkeypatch.setattr(preflight.subprocess,"check_output",output)
    monkeypatch.setattr(preflight.subprocess,"run",lambda command,**kwargs:runs.append(command))
    monkeypatch.setattr(sys,"argv",["preflight.py",*(["--qualification-only"] if qualification else []),
        "--","--root",str(tmp_path/"state")])
    preflight.main()
    assert bool(qualified) is not qualification
    assert any(command[0]=="nest" for command in checks)
    assert any(Path(command[0]).name=="nvidia-smi" for command in checks)
    assert runs[0]==["docker","image","inspect",preflight.IMAGE]
    assert runs[1][-1]=="--preflight"


def test_reservation_requires_a_real_held_flock(tmp_path,monkeypatch):
    monkeypatch.setenv("CUTEAFD_LOCK_DIR",str(tmp_path))
    monkeypatch.delenv("CUTEAFD_HW_LOCKS_HELD",raising=False)
    with pytest.raises(ValueError,match="enclosing hwlock"):
        launch.require_held_gpu_lock()
    monkeypatch.setenv("CUTEAFD_HW_LOCKS_HELD","gpu1")
    with pytest.raises(ValueError,match="not held"):
        launch.require_held_gpu_lock()
    with (tmp_path/"gpu1.lock").open("a") as guard:
        launch.fcntl.flock(guard,launch.fcntl.LOCK_EX|launch.fcntl.LOCK_NB)
        launch.require_held_gpu_lock()


@pytest.mark.parametrize("extra",[[],["--foreground","--publish-only"]])
def test_reservation_cannot_detach_or_publish(tmp_path,monkeypatch,extra):
    monkeypatch.setattr(sys,"argv",["launch.py","--source-snapshot",str(tmp_path/"source"),
        "--tokens",str(tmp_path/"tokens"),"--root",str(tmp_path/"state"),
        "--export",str(tmp_path/"export"),"--use-held-lock",*extra])
    monkeypatch.setattr(launch.subprocess,"run",lambda *a,**k:pytest.fail("must reject before admission"))
    with pytest.raises(ValueError,match="foreground quantization"):
        launch.main()


def test_reserved_foreground_job_retains_guard_without_requeueing(tmp_path,monkeypatch):
    root=tmp_path/"state";checks=[];execs=[]
    monkeypatch.setattr(sys,"argv",["launch.py","--source-snapshot",str(tmp_path/"source"),
        "--tokens",str(tmp_path/"tokens"),"--root",str(root),
        "--export",str(tmp_path/"export"),"--foreground","--use-held-lock"])
    monkeypatch.setattr(launch,"require_held_gpu_lock",lambda:None)
    monkeypatch.setattr(launch.subprocess,"run",lambda command,**kwargs:checks.append(command))
    monkeypatch.setattr(launch.subprocess,"Popen",lambda *a,**k:pytest.fail("must not detach"))
    class ReplacedProcess(Exception):pass
    def replace(python,command):
        execs.append(command);raise ReplacedProcess
    monkeypatch.setattr(launch.os,"execv",replace)
    with pytest.raises(ReplacedProcess):launch.main()
    latest=json.loads((root/"latest-attempt.json").read_text())
    record=json.loads(Path(latest["launch"]).read_text())
    assert len(checks)==len(execs)==1
    assert latest["pid"]==os.getpid()
    assert record["foreground"] and record["uses_held_lock"]
    assert not any(arg.endswith("hwlock.py") for arg in record["command"])
    assert any(arg.endswith("run_guard.py") for arg in record["command"])
    assert execs[0][1].endswith("run_guard.py")
    assert execs[0][execs[0].index("--container")+1]==record["container"]
    assert "--supervise" in execs[0] and str(latest["launch"]) in execs[0]


def test_hangup_stops_owned_container_before_guard_returns(tmp_path):
    started=tmp_path/"child-started";stopped=tmp_path/"container-stopped"
    docker=tmp_path/"docker"
    docker.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(stopped)!r}).write_text('stopped')\n")
    docker.chmod(0o755)
    guard=Path(launch.__file__).resolve().parents[2]/"kimi_k3/core_k6/run_guard.py"
    process=subprocess.Popen([sys.executable,str(guard),"--container","test-owned-container","--",
        sys.executable,"-c",f"from pathlib import Path; import time; Path({str(started)!r}).touch(); time.sleep(60)"],
        env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ["PATH"]))
    try:
        deadline=time.monotonic()+5
        while not started.exists():
            assert process.poll() is None and time.monotonic()<deadline
            time.sleep(.01)
        process.send_signal(signal.SIGHUP)
        process.wait(timeout=5)
        assert stopped.read_text()=="stopped"
    finally:
        if process.poll() is None:
            process.terminate();process.wait(timeout=5)
