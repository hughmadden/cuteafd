"""Host policy fixtures: no host writes, privileges, or hardware required."""
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "host/tune.sh"


def put(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(value) + "\n")
    return target


def fixture(root, role):
    policy = "devices/system/cpu/cpufreq/policy0"
    put(root, f"{policy}/scaling_governor", "powersave")
    put(root, f"{policy}/energy_performance_preference", "balance_performance")
    # Names deliberately do not encode ordering or the latency policy.
    latencies = [0, 1, 49, 50, 100] if role == "coordinator" else [1, 165, 199, 200, 231, 433]
    for cpu in (0, 1):
        for index, latency in enumerate(latencies):
            state = f"devices/system/cpu/cpu{cpu}/cpuidle/state{index}"
            for name, value in {"name": f"arbitrary-{index}", "latency": latency,
                                "disable": 0, "usage": 123}.items():
                put(root, f"{state}/{name}", value)
    put(root, "proc/sys/vm/compaction_proactiveness", 20)
    put(root, "proc/sys/vm/swappiness", 60)
    put(root, "proc/sys/vm/zone_reclaim_mode", 0)
    put(root, "lscpu.csv", "# CPU,MAXMHZ\n0,2800.0000\n1,3900.0000")
    for name in ("enabled", "defrag"):
        put(root, f"kernel/mm/transparent_hugepage/{name}", "always [madvise] never")
    put(root, "class/infiniband/mlx5_0/ports/1/rate", "200 Gb/sec (4X HDR)")
    put(root, "class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/0", "roce0")
    put(root, "class/net/roce0/device/msi_irqs/42", "")
    put(root, "proc/irq/42/smp_affinity_list", "0-3")
    return latencies


def run(root, *args, unprivileged=False):
    env = dict(os.environ, TUNE_SYSFS_ROOT=str(root))
    # A caller must not be able to forge bash's readonly EUID through env.
    env["EUID"] = "0"
    kwargs = {}
    if unprivileged and os.geteuid() == 0:
        kwargs.update(user=65534, group=65534, extra_groups=[])
    return subprocess.run(["bash", str(SCRIPT), *args], env=env,
                          text=True, capture_output=True, timeout=15, **kwargs)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("role,threshold", [("coordinator", 50), ("spark", 200)])
def test_report_and_dry_run_select_by_latency(tmp_path, role, threshold):
    latencies = fixture(tmp_path, role)
    before = snapshot(tmp_path)
    report = run(tmp_path, "report", "--role", role)
    assert report.returncode == 0, report.stderr
    for cpu in (0, 1):
        for index, latency in enumerate(latencies):
            expected = int(latency >= threshold)
            marker = "differs" if expected else "pass"
            assert (f"[{marker}] cpu{cpu}/state{index} "
                    f"name=arbitrary-{index} latency_us={latency} usage=123 disable: 0 "
                    f"| target={expected}") in report.stdout
    assert "[pass] THP enabled (modes: always [madvise] never): madvise | target=madvise" in report.stdout
    assert "[pass] vm.zone_reclaim_mode: 0 | target=0" in report.stdout
    if role == "spark":
        assert "P=1 (3900MHz) E=0 (2800MHz)" in report.stdout
    assert "200 Gb/sec" in report.stdout
    assert "IRQs 42 affinity_list: 0-3" in report.stdout
    assert "[unknown] roce0 PFC" in report.stdout
    assert "[unknown] GPU performance / clocks-event reasons" in report.stdout

    dry = run(tmp_path, "apply", "--role", role, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    writes = {line.split(" <- ")[0].removeprefix("DRY-RUN write ")
              for line in dry.stdout.splitlines() if line.startswith("DRY-RUN write ")}
    expected = {str(tmp_path / "devices/system/cpu/cpufreq/policy0/scaling_governor"),
                str(tmp_path / "proc/sys/vm/swappiness")}
    if role == "coordinator":
        expected.add(str(tmp_path / "devices/system/cpu/cpufreq/policy0/energy_performance_preference"))
    else:
        expected.add(str(tmp_path / "proc/sys/vm/compaction_proactiveness"))
    for cpu in (0, 1):
        for index, latency in enumerate(latencies):
            if latency >= threshold:
                expected.add(str(tmp_path / f"devices/system/cpu/cpu{cpu}/cpuidle/state{index}/disable"))
    assert writes == expected
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("role", ["coordinator", "spark"])
def test_idempotent_preview_and_shallow_state_preserved(tmp_path, role):
    latencies = fixture(tmp_path, role)
    threshold = 50 if role == "coordinator" else 200
    put(tmp_path, "devices/system/cpu/cpufreq/policy0/scaling_governor", "performance")
    put(tmp_path, "devices/system/cpu/cpufreq/policy0/energy_performance_preference", "performance")
    put(tmp_path, "proc/sys/vm/compaction_proactiveness", 0)
    put(tmp_path, "proc/sys/vm/swappiness", 10)
    for cpu in (0, 1):
        for index in range(len(latencies)):
            put(tmp_path, f"devices/system/cpu/cpu{cpu}/cpuidle/state{index}/disable", 1)
    result = run(tmp_path, "apply", "--role", role, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "write " not in result.stdout
    assert f"idle_disable_latency_us>={threshold}" in result.stdout


def test_refuses_apply_without_root(tmp_path):
    result = run(tmp_path, "apply", "--role", "spark", unprivileged=True)
    assert result.returncode == 1
    assert "apply requires root (EUID 0)" in result.stderr
    assert not list(tmp_path.iterdir())


def test_fixture_report_never_runs_host_probes(tmp_path):
    fixture(tmp_path, "spark")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "host-probe-ran"
    for name in ("lscpu", "ps", "taskset", "ethtool", "dcb", "rdma", "nvidia-smi", "systemctl"):
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 1\n')
        stub.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               TUNE_SYSFS_ROOT=str(tmp_path))
    result = subprocess.run(["bash", str(SCRIPT), "report", "--role", "spark"],
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_unknown_latency_is_never_written(tmp_path):
    fixture(tmp_path, "coordinator")
    state = "devices/system/cpu/cpu0/cpuidle/state4"
    put(tmp_path, f"{state}/latency", "unavailable")
    result = run(tmp_path, "apply", "--role", "coordinator", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert f"{state}/disable" not in result.stdout


def test_missing_interfaces_report_unknown(tmp_path):
    result = run(tmp_path, "--role", "spark")
    assert result.returncode == 0, result.stderr
    for label in ("CPU governor", "CPU energy_performance_preference", "cpuidle",
                  "THP enabled", "rdma link", "RoCE port rate", "RoCE NIC diagnostics",
                  "vm.zone_reclaim_mode", "memlock", "irqbalance", "GB10 P/E cores"):
        assert f"[unknown] {label}" in result.stdout


def test_persist_prints_only_unit(tmp_path):
    result = run(tmp_path, "persist", "--role", "spark")
    assert result.returncode == 0, result.stderr
    assert "[Unit]" in result.stdout
    assert "ExecStart=/usr/local/sbin/cuteafd-host-tune apply --role spark" in result.stdout
    assert str(tmp_path) not in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("args", [("--role", "invalid"), ("--role",), ("report", "apply"),
                                   ("report", "--dry-run"), ("--unknown",)])
def test_invalid_arguments(tmp_path, args):
    assert run(tmp_path, *args).returncode == 2
