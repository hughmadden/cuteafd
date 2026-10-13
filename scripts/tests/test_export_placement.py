"""Coordinator export placement: admission from the host's nvidia-smi, the
memory cap, the SM-count check and the watchdog (scripts/build/export-placement.py)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/build/export-placement.py'
spec = importlib.util.spec_from_file_location('export_placement', SCRIPT)
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


def fake_smi(tmp_path, gpus, apps='', name='nvidia-smi'):
    """An nvidia-smi stub: GPU rows (index, uuid, used, total, free) and compute apps
    (uuid, pid, used). A `state` file, when present, replaces the GPU rows, so a
    test can change what the stub reports while a watch runs."""
    state = tmp_path / (name + '.state')
    state.write_text(gpus)
    app_rows = tmp_path / (name + '.apps')
    app_rows.write_text(apps)
    stub = tmp_path / name
    stub.write_text(f'''#!/usr/bin/env bash
case "$1" in
  --query-gpu=*) cat {state} ;;
  --query-compute-apps=*) cat {app_rows} ;;
esac
''')
    stub.chmod(0o755)
    return stub, state


def run(*args, timeout=60):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize('free,expected', [
    (97000, (placement.MAX_CAP_MIB, (placement.MAX_CAP_MIB - placement.SIZING_MIB) // placement.CONTEXT_MIB)),
    (32000, (placement.MAX_CAP_MIB, (placement.MAX_CAP_MIB - placement.SIZING_MIB) // placement.CONTEXT_MIB)),
    (10000, (10000 - placement.HEADROOM_MIB, (10000 - placement.HEADROOM_MIB - placement.SIZING_MIB) // placement.CONTEXT_MIB)),
    (placement.MIN_CAP_MIB + placement.HEADROOM_MIB - 1, None),
    (2000, None),
])
def test_cap_and_jobs_follow_the_free_memory_at_that_moment(free, expected):
    assert placement.admit(free) == expected


def test_admit_reads_the_host_smi_and_refuses_what_does_not_fit(tmp_path):
    smi, _ = fake_smi(tmp_path, '0, GPU-a, 65485, 97887, 31966\n1, GPU-b, 94127, 97887, 3760\n')
    ok = run('admit', '--smi', str(smi), '--gpu', '0')
    assert ok.returncode == 0, ok.stderr
    cap, jobs, used = map(int, ok.stdout.split())
    assert cap == placement.MAX_CAP_MIB and used == 65485 and jobs >= 1
    busy = run('admit', '--smi', str(smi), '--gpu', 'GPU-b')
    assert busy.returncode == placement.NO_TURN and 'MiB free' in busy.stderr


def test_wsl_admission_never_trusts_the_containers_view(tmp_path):
    # On oliver a container's cudaMemGetInfo reported 93.9 of 95.6 GiB free while
    # the host's WSL nvidia-smi showed TJ's job holding 65.5 GB. Admission takes
    # its reading from the HOST smi binary it is given; export-remote.sh passes
    # oliver's /usr/lib/wsl/lib/nvidia-smi and admits before any container starts.
    host, _ = fake_smi(tmp_path, '0, GPU-o, 65485, 97887, 4000\n', name='host-smi')
    container_view, _ = fake_smi(tmp_path, '0, GPU-o, 1700, 97887, 96187\n', name='container-smi')
    assert run('admit', '--smi', str(container_view), '--gpu', '0').returncode == 0
    assert run('admit', '--smi', str(host), '--gpu', '0').returncode == placement.NO_TURN
    remote = (ROOT / 'scripts/build/export-remote.sh').read_text()
    assert 'smi=/usr/lib/wsl/lib/nvidia-smi' in remote
    assert 'mem_get_info' not in SCRIPT.read_text().split('"""', 2)[2]


def test_sm_check_refuses_a_host_whose_sm_count_differs_from_the_target():
    assert run('sm-check', '--live', '188', '--target', '188').returncode == 0
    refused = run('sm-check', '--live', '188', '--target', '170')
    assert refused.returncode == placement.NO_TURN and '#122' in refused.stderr


def test_watch_passes_through_an_export_that_stays_under_its_cap(tmp_path):
    smi, _ = fake_smi(tmp_path, '0, GPU-o, 65600, 97887, 31000\n')
    done = run('watch', '--smi', str(smi), '--gpu', '0', '--cap-mib', '4000', '--baseline-mib', '65485',
               '--interval', '0.1', '--', 'bash', '-c', 'sleep 0.5; exit 3')
    assert done.returncode == 3 and 'export peak 115 MiB' in done.stderr


def test_watch_kills_an_export_over_its_cap_and_reports_no_turn(tmp_path):
    smi, state = fake_smi(tmp_path, '0, GPU-o, 65485, 97887, 31966\n')
    marker = tmp_path / 'survived'
    # The "export" grows past the cap after a moment; the watchdog must stop it.
    command = f'printf "0, GPU-o, 75000, 97887, 22000\\n" > {state}; sleep 30; touch {marker}'
    stopped = run('watch', '--smi', str(smi), '--gpu', '0', '--cap-mib', '4000', '--baseline-mib', '65485',
                  '--interval', '0.1', '--', 'bash', '-c', command, timeout=45)
    assert stopped.returncode == placement.NO_TURN and 'over its 4000 MiB cap' in stopped.stderr
    assert not marker.exists()


def test_watch_counts_only_the_export_containers_processes(tmp_path, monkeypatch):
    # Serving processes on the same GPU are not the export's: only pids in the
    # export container's cgroup count against its cap.
    smi, _ = fake_smi(tmp_path, '1, GPU-r, 88649, 97887, 9238\n',
                      apps='GPU-r, 100, 80000\nGPU-r, 200, 600\nGPU-r, 201, 1900\n')
    monkeypatch.setattr(placement, 'cgroup_pids', lambda cid: {200, 201})
    assert placement.export_mib(str(smi), '1', 'abc', None) == 2500


def test_watch_ignores_a_failed_reading(tmp_path, monkeypatch):
    calls = iter([subprocess.SubprocessError('smi hiccup'), 10, 10])
    def reading(*args):
        value = next(calls, 10)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(placement, 'export_mib', reading)
    args = placement.argparse.Namespace(smi='x', gpu='0', cgroup=None, baseline_mib=0, cap_mib=100,
                                        interval=0.05, command=['bash', '-c', 'sleep 0.3'])
    assert placement.watch(args) == 0
