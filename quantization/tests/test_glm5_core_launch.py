"""GPU ownership and publication separation in detached production launches."""
import json
from pathlib import Path
import sys
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
def test_preflight_bootstrap_still_checks_local_source_gpu_and_producer(tmp_path,monkeypatch,qualification):
    repo=tmp_path;fork=repo/"third_party/gptqmodel";fork.mkdir(parents=True)
    (fork.parent/"gptqmodel.lock.json").write_text(json.dumps(dict(revision="pinned",source_tree_sha256="tree")))
    monkeypatch.setattr(preflight,"REPO",repo)
    monkeypatch.setattr(preflight.socket,"gethostname",lambda:"maxq")
    monkeypatch.setattr(preflight.runpy,"run_path",lambda path:{"source_tree_sha256":lambda source:"tree"})
    qualified=[];checks=[];runs=[]
    monkeypatch.setattr(preflight,"qualified",lambda *a,**k:qualified.append((a,k)))
    def output(command,**kwargs):
        checks.append(command)
        if "--porcelain" in command:return ""
        if "rev-parse" in command:return "pinned\n"
        if command[0]=="nest":return "maxq ready\n"
        if command[0]=="nvidia-smi":return "NVIDIA RTX PRO 6000 Blackwell Max-Q, 97887\n"
        pytest.fail(str(command))
    monkeypatch.setattr(preflight.subprocess,"check_output",output)
    monkeypatch.setattr(preflight.subprocess,"run",lambda command,**kwargs:runs.append(command))
    monkeypatch.setattr(sys,"argv",["preflight.py",*(["--qualification-only"] if qualification else []),
        "--","--root",str(tmp_path/"state")])
    preflight.main()
    assert bool(qualified) is not qualification
    assert any(command[0]=="nest" for command in checks)
    assert any(command[0]=="nvidia-smi" for command in checks)
    assert runs[0]==["docker","image","inspect",preflight.IMAGE]
    assert runs[1][-1]=="--preflight"
