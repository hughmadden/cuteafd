"""scripts/dev/cpu-gate: verdicts from parsed test summaries, never exit codes alone."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOADER = importlib.machinery.SourceFileLoader('cpu_gate', str(ROOT / 'scripts/dev/cpu-gate'))
SPEC = importlib.util.spec_from_loader('cpu_gate', LOADER)
gate = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(gate)

BUILD_OK = '{"reason":"build-finished","success":true}\n'

ARGS = SimpleNamespace(cargo_build_timeout=60, cargo_timeout=60, scripts_timeout=60, skip_live_planners=False, cargo_jobs=4, pytest_jobs=4)


def _fake(monkeypatch, outputs):
    """outputs: list of (exit, text) in call order for run_logged."""
    calls = []
    def run_logged(cmd, log_path, env, cwd, timeout):
        code, text = outputs[len(calls)]
        calls.append(cmd)
        Path(log_path).write_text(text)
        return code, 1.0
    monkeypatch.setattr(gate, 'nextest_for', lambda env: None)
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: '')
    return calls


def test_missing_toolchain_refuses_before_any_work(tmp_path):
    env = {'HOME': str(tmp_path), 'PATH': '/nonexistent'}
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/dev/cpu-gate'), '--task', 'path-probe'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'cargo, rustc not on PATH' in result.stderr
    assert not (tmp_path / '.cache/cuteafd/builds/path-probe').exists()


def test_cargo_passes_only_with_parsed_tests_and_no_failures(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    ok = 'test result: ok. 12 passed; 0 failed; 1 ignored; 0 measured\ntest result: ok. 3 passed; 0 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, BUILD_OK), (0, ok)])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'pass' and result['passed'] == 15 and result['binaries'] == 2
    # Exit 0 but nothing ran (e.g. a wrapper that masked a missing lock): failed.
    _fake(monkeypatch, [(0, BUILD_OK), (0, 'fd: unbound variable\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and 'no test summaries' in result['error']
    bad = ok + 'test a::b ... FAILED\ntest result: FAILED. 4 passed; 1 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, BUILD_OK), (101, bad)])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['a::b']
    _fake(monkeypatch, [(1, 'error[E0425]: cannot find value\nerror: could not compile\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and 'E0425' in result['error']


@pytest.mark.parametrize('build', ['', 'error: compile failed\n',
                                   '{"reason":"build-finished","success":false}\n',
                                   BUILD_OK + '{"reason":"compiler-message","message":{"level":"error"}}\n'])
def test_masked_compile_failure_never_runs_stale_test_binaries(tmp_path, monkeypatch, build):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, build), (0, 'test result: ok. 8 passed; 0 failed; 0 ignored\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and not result['build_complete']
    assert len(calls) == 1 and '--message-format=json-render-diagnostics' in calls[0]
    calls = _fake(monkeypatch, [(0, build), (0, 'test result: ok. 1 passed; 0 failed; 0 ignored\n')])
    assert gate.benches(ROOT, tmp_path, ARGS, {})['status'] == 'failed'
    assert len(calls) == len(gate.BENCHES)


def test_zero_exit_does_not_mask_test_failure_counts(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    _fake(monkeypatch, [(0, BUILD_OK), (0, 'test a::b ... FAILED\n'
                                             'test result: FAILED. 8 passed; 1 failed; 0 ignored\n')])
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['a::b']


@pytest.mark.parametrize('problem', ['timeout', 'missing-summary', 'moved', 'base-empty', 'shared-test'])
def test_base_comparison_does_not_mask_infrastructure_failures(tmp_path, monkeypatch, problem):
    suite = dict(suite='cargo', status='pass', passed=8, failed=0, failing=[], exit=0)
    head = dict(head='a', suites=[dict(suite)], status='pass', dirty=False, moved=False)
    base = dict(head='b', suites=[dict(suite)], status='pass', dirty=False, moved=False)
    if problem == 'timeout':
        head['suites'][0].update(status='failed', exit=124)
    elif problem == 'missing-summary':
        head['suites'][0].update(status='failed')
    elif problem == 'moved':
        head['moved'] = True
    elif problem == 'base-empty':
        base['suites'][0].update(status='failed', passed=0)
    else:
        for side in (head, base):
            side['suites'][0].update(status='failed', failed=1, failing=['a::b'], exit=101)
    monkeypatch.setattr(gate, 'gate', lambda source, *a, **k: head if source == ROOT else base)
    monkeypatch.setattr(gate, 'checked', lambda *a, **k: '')
    args = SimpleNamespace(task='test', source=ROOT, base='base', skip_live_planners=False)
    code = gate.run_gate(args, tmp_path)
    assert code == (0 if problem == 'shared-test' else 1)


def test_child_shell_preserves_errexit_inside_agent_and_list(tmp_path):
    script = tmp_path / 'gate.sh'
    script.write_text('set -euo pipefail\nfalse\nprintf false-green\\n\n')
    result = subprocess.run(['bash', '-c', 'true && eval \'bash "$1"\' && echo green', '_', str(script)],
                            text=True, capture_output=True)
    assert result.returncode != 0 and not result.stdout


def test_live_planner_skip_is_explicit_and_recorded(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, 'test result: ok. 1 passed; 0 failed; 0 ignored\n')])
    result = gate.cargo(ROOT, tmp_path, SimpleNamespace(**{**vars(ARGS), 'skip_live_planners': True}), {})
    test_cmd = calls[1]
    assert test_cmd[-1] == '--exact' and test_cmd.count('--skip') == 2
    assert result['skipped_live_planners'] == list(gate.LIVE_PLANNERS)
    record = {'task': 't', 'verdict': 'pass', 'head': 'a' * 40, 'dirty': False, 'suites': [result]}
    assert 'live V4 planner tests skipped' in gate.render(record)


def test_scripts_parse_failing_ids_and_empty_runs(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/usr/bin/python3'))
    _fake(monkeypatch, [(1, 'FAILED scripts/tests/test_x.py::test_y - assert 0\n=== 1 failed, 20 passed, 2 skipped in 3.10s ===\n')])
    result = gate.scripts(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['scripts/tests/test_x.py::test_y']
    assert (result['passed'], result['failed'], result['skipped']) == (20, 1, 2)
    _fake(monkeypatch, [(0, '808 passed in 100.00s\n')])
    assert gate.scripts(ROOT, tmp_path, ARGS, {})['status'] == 'pass'
    _fake(monkeypatch, [(5, 'no tests ran in 0.01s\n')])
    assert gate.scripts(ROOT, tmp_path, ARGS, {})['status'] == 'failed'


def test_build_slot_refuses_a_missing_lock_directory(tmp_path):
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'true'], capture_output=True, text=True,
                            env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path / 'absent')})
    assert result.returncode == 2 and 'lock directory' in result.stderr and 'missing' in result.stderr
    (tmp_path / 'locks').mkdir()
    result = subprocess.run(['bash', str(ROOT / 'scripts/build/build-slot.sh'), 'true'], capture_output=True, text=True,
                            env={**os.environ, 'CUTEAFD_LOCK_DIR': str(tmp_path / 'locks')})
    assert result.returncode == 0 and (tmp_path / 'locks/build.lock').exists()


def test_benches_record_medians_and_fail_on_a_bound_or_an_empty_run(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    ran = ('C16 V4.1 geometry, medians: select 937.1 us, choose_width 1603.7 us, observe 56.8 us, total 2597.6 us\n'
           'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n')
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, ran)])
    envs = []
    original = gate.run_logged
    def run_logged(cmd, log_path, env, cwd, timeout):
        envs.append(env)
        return original(cmd, log_path, env, cwd, timeout)
    monkeypatch.setattr(gate, 'run_logged', run_logged)
    result = gate.benches(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'pass' and result['passed'] == 1
    assert result['medians']['draft_policy_bench'] == {'select': 937.1, 'choose_width': 1603.7, 'observe': 56.8}
    assert '--release' in calls[0] and '--ignored' in calls[1]
    assert envs[1]['CUTEAFD_POLICY_BENCH_MAX_US'] == '1200'
    record = {'task': 't', 'verdict': 'pass', 'head': 'a' * 40, 'dirty': False, 'suites': [result]}
    assert 'select 937 us' in gate.render(record)
    over = 'medians: select 1500.0 us, choose_width 2500.0 us, observe 57.0 us\nselect median 1500.0 us exceeds 1200 us\n' \
           'test result: FAILED. 0 passed; 1 failed; 0 ignored\n'
    _fake(monkeypatch, [(0, BUILD_OK), (101, over)])
    result = gate.benches(ROOT, tmp_path, ARGS, {})
    assert result['status'] == 'failed' and result['failing'] == ['draft_policy_bench']
    assert result['medians']['draft_policy_bench']['select'] == 1500.
    _fake(monkeypatch, [(0, BUILD_OK), (0, 'running 0 tests\n')])
    assert gate.benches(ROOT, tmp_path, ARGS, {})['status'] == 'failed'


def test_benches_flag_runs_only_the_bench_suite(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(gate, 'prepare', lambda source, root, owned=False: None)
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: 'abc' if 'rev-parse' in cmd else '')
    for name in ('cargo', 'scripts', 'benches'):
        monkeypatch.setattr(gate, name, lambda s, r, a, e, n=name: seen.append(n) or {'suite': n, 'status': 'pass'})
    args = SimpleNamespace(**{**vars(ARGS), 'cargo': False, 'scripts': False, 'benches': True})
    assert gate.gate(ROOT, tmp_path, args)['status'] == 'pass' and seen == ['benches']


def _git(cwd, *args):
    subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True,
                   env={**os.environ, 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't',
                        'GIT_COMMITTER_EMAIL': 't@t', 'GIT_ALLOW_PROTOCOL': 'file'})


def test_pin_changes_refuses_an_uncommitted_submodule_bump(tmp_path):

    sub = tmp_path / 'sub'
    sub.mkdir()
    _git(sub, 'init', '-q')
    (sub / 'f').write_text('1')
    _git(sub, 'add', 'f')
    _git(sub, 'commit', '-qm', 'one')
    top = tmp_path / 'top'
    top.mkdir()
    _git(top, 'init', '-q')
    _git(top, '-c', 'protocol.file.allow=always', 'submodule', 'add', '-q', str(sub), 'third_party/sub')
    _git(top, 'commit', '-qm', 'pin')
    assert gate.pin_changes(top) == []
    # Edits inside the submodule with the same commit are not a pin change.
    (top / 'third_party/sub/f').write_text('dirty')
    assert gate.pin_changes(top) == []
    _git(top / 'third_party/sub', 'checkout', '-q', 'f')
    # A checked-out newer commit (unstaged) and a staged bump are both refused.
    (top / 'third_party/sub/f').write_text('2')
    _git(top / 'third_party/sub', 'commit', '-qam', 'two')
    assert gate.pin_changes(top) == ['third_party/sub']
    _git(top, 'add', 'third_party/sub')
    assert gate.pin_changes(top) == ['third_party/sub']
    _git(top, 'commit', '-qm', 'bump')
    assert gate.pin_changes(top) == []


def test_prepare_refuses_a_moved_pin_only_in_the_gated_source(tmp_path, monkeypatch):
    """The --base worktree is the gate's own: re-detached at each base, its submodule still
    sits at the previous base's pin, which `submodule update` then fixes. Refusing there broke
    every second --base run; the gated source is still checked."""
    monkeypatch.setattr(gate, 'pin_changes', lambda source: ['third_party/sparkinfer'])
    calls = []
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: calls.append(cmd) or '')
    with pytest.raises(SystemExit, match='submodule pins differ'):
        gate.prepare(tmp_path, tmp_path / 'head')
    gate.prepare(tmp_path, tmp_path / 'base', owned=True)
    assert any('submodule' in cmd for cmd in calls)


def test_gate_records_the_starting_head_and_fails_when_it_moves(tmp_path, monkeypatch):
    """A checkout that changes while the gate runs must not be credited with its tests."""
    monkeypatch.setattr(gate, 'prepare', lambda source, root, owned=False: None)
    heads = iter(['aaaa', 'bbbb'])
    def checked(cmd, **k):
        return next(heads) + '\n' if 'rev-parse' in cmd else ''
    monkeypatch.setattr(gate, 'checked', checked)
    monkeypatch.setattr(gate, 'scripts', lambda *a: {'suite': 'scripts', 'status': 'pass'})
    args = SimpleNamespace(cargo=False, scripts=True, benches=False)
    record = gate.gate(ROOT, tmp_path, args)
    assert record['head'] == 'aaaa' and record['moved'] and record['status'] == 'failed'


def test_nextest_machine_report_includes_crashes_and_skips(tmp_path):
    report = tmp_path / 'junit.xml'
    report.write_text('''<testsuites><testsuite name="crate">
      <testcase name="ok"/><testcase name="bad"><failure/></testcase>
      <testcase name="crash"><error/></testcase><testcase name="ignored"><skipped/></testcase>
    </testsuite></testsuites>''')
    assert gate.parse_nextest(report) == ([(1, 2, 1)], ['bad', 'crash'])
    assert gate.parse_nextest(tmp_path / 'missing') == ([], [])
    report.write_text('<testsuites>')
    assert gate.parse_nextest(report) == ([], [])


def test_worker_budget_reaches_both_runners(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, 'test result: ok. 1 passed; 0 failed; 0 ignored\n')])
    envs = []
    run_logged = gate.run_logged
    def capture_env(cmd, path, env, cwd, timeout):
        envs.append(env)
        return run_logged(cmd, path, env, cwd, timeout)
    monkeypatch.setattr(gate, 'run_logged', capture_env)
    gate.cargo(ROOT, tmp_path, ARGS, {})
    assert all(env['CARGO_BUILD_JOBS'] == env['RUST_TEST_THREADS'] == '4' for env in envs)
    monkeypatch.setattr(gate, 'python_for', lambda source: Path('/usr/bin/python3'))
    calls = _fake(monkeypatch, [(0, '1 passed in 1.00s\n')])
    gate.scripts(ROOT, tmp_path, ARGS, {})
    assert calls[0][calls[0].index('-n', 3) + 1] == '4'


def test_slow_runs_only_explicit_cpu_tests(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    ok = 'test result: ok. 1 passed; 0 failed; 0 ignored\n'
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, ok), *[(0, ok)] * len(gate.SLOW_TESTS)])
    args = SimpleNamespace(**vars(ARGS), slow=True)
    result = gate.cargo(ROOT, tmp_path, args, {})
    assert result['status'] == 'pass'
    assert len(calls) == 2 + len(gate.SLOW_TESTS)
    assert all('--ignored' in call and '--include-ignored' not in call for call in calls[2:])


def test_suites_overlap_without_changing_receipt_order(tmp_path, monkeypatch):
    import threading
    barrier = threading.Barrier(2, timeout=5)
    monkeypatch.setattr(gate, 'prepare', lambda *a, **k: None)
    monkeypatch.setattr(gate, 'checked', lambda cmd, **k: 'abc' if 'rev-parse' in cmd else '')
    def run(name):
        barrier.wait()
        return {'suite': name, 'status': 'pass'}
    monkeypatch.setattr(gate, 'cargo', lambda *a: run('cargo'))
    monkeypatch.setattr(gate, 'scripts', lambda *a: run('scripts'))
    args = SimpleNamespace(cargo=True, scripts=True, benches=False)
    assert [s['suite'] for s in gate.gate(ROOT, tmp_path, args)['suites']] == ['cargo', 'scripts']


def test_missing_nextest_report_cannot_be_masked_by_passing_doctests(tmp_path, monkeypatch):
    (tmp_path / 'logs').mkdir()
    calls = _fake(monkeypatch, [(0, BUILD_OK), (0, BUILD_OK), (0, 'test result: ok. 2 passed; 0 failed; 0 ignored\n')])
    monkeypatch.setattr(gate, 'nextest_for', lambda env: '/tools/cargo-nextest')
    result = gate.cargo(ROOT, tmp_path, ARGS, {})
    assert calls[1][4:6] == ['nextest', 'run']
    config = tmp_path / 'nextest.toml'
    assert str(tmp_path / 'logs/nextest') in config.read_text()
    assert calls[1][calls[1].index('--config-file') + 1] == str(config)
    assert result['status'] == 'failed' and result['exit'] != 0


def test_launcher_inputs_link_readonly_files_but_copy_mutable_tree_files(tmp_path, monkeypatch):
    import launcher_sandbox
    source, cache, clone = (tmp_path / name for name in ('source', 'cache', 'clone'))
    source.mkdir()
    cache.mkdir()
    (source / 'readonly.py').write_text('original')
    (source / 'mutable.sh').write_text('original')
    with monkeypatch.context() as patch:
        patch.setattr(launcher_sandbox, '_ROOT', cache)
        patch.setattr(launcher_sandbox, '_TEMPLATES', {})
        launcher_sandbox.copy_tree(source, clone)
        other = tmp_path / 'other'
        launcher_sandbox.copy_tree(source, other)
        assert (clone / 'readonly.py').stat().st_ino == (other / 'readonly.py').stat().st_ino
        (clone / 'mutable.sh').write_text('edited')
        assert (other / 'mutable.sh').read_text() == (source / 'mutable.sh').read_text() == 'original'
        launcher_sandbox.copy_input(source / 'readonly.py', tmp_path / 'input')
        assert (tmp_path / 'input').stat().st_ino != (source / 'readonly.py').stat().st_ino


def test_scratch_report_groups_by_test_and_stem(tmp_path):
    tmp = tmp_path / 'tmp'
    for worker, repeat in (('popen-gw0', 0), ('popen-gw3', 1)):
        path = tmp / 'pytest' / worker / f'test_big_fixture{repeat}' / 'blob'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'x' * (1 << 20))
    (tmp / 'pytest/popen-gw1/test_small0').mkdir(parents=True)
    (tmp / 'pytest/popen-gw1/test_small0/a').write_bytes(b'x' * 4096)
    for pid in (11, 12):
        (tmp / f'cuteafd-fabric-{pid}').mkdir(parents=True)
        (tmp / f'cuteafd-fabric-{pid}/f').write_bytes(b'x' * 65536)
    (tmp / 'tmpab12cd34').write_bytes(b'x' * 8192)
    rows = gate.scratch_usage(tmp, top=3)
    assert [row['name'] for row in rows] == ['pytest:test_big_fixture', 'cuteafd-fabric', 'tmp*']
    assert rows[0]['bytes'] >= 2 << 20
    assert gate.scratch_usage(tmp_path / 'missing') == []
    text = gate.render({'task': 't', 'verdict': 'pass', 'head': 'a' * 40, 'dirty': False, 'suites': [], 'scratch': rows})
    assert 'Largest scratch consumers' in text and 'pytest:test_big_fixture' in text


def _scratch_tree(root):
    for side in ('head', 'base'):
        for name in gate.SCRATCH:
            (root / side / name / 'deep').mkdir(parents=True)
            (root / side / name / 'deep/file').write_text('x')
        (root / side / 'logs').mkdir()
        (root / side / 'logs/pytest.log').write_text('kept')
    (root / 'head/tmp/deep').chmod(0o500)  # tests leave read-only directories behind


@pytest.mark.parametrize('keep, outcome', [(False, 'pass'), (False, 'raise'), (True, 'pass')])
def test_scratch_is_removed_on_every_exit_unless_kept(tmp_path, monkeypatch, keep, outcome):
    root = tmp_path / 'builds/task'
    _scratch_tree(root)
    monkeypatch.setattr(gate, 'BUILDS', tmp_path / 'builds')
    monkeypatch.setattr(gate.signal, 'signal', lambda *a: None)

    def run_gate(args, root):
        if outcome == 'raise':
            raise KeyboardInterrupt('signal 15')
        return 0
    monkeypatch.setattr(gate, 'run_gate', run_gate)
    monkeypatch.setattr(sys, 'argv', ['cpu-gate', '--task', 'task'] + (['--keep'] if keep else []))
    if outcome == 'raise':
        with pytest.raises(KeyboardInterrupt):
            gate.main()
    else:
        assert gate.main() == 0
    for side in ('head', 'base'):
        assert (root / side / 'logs/pytest.log').read_text() == 'kept'
        for name in gate.SCRATCH:
            assert (root / side / name).exists() == keep
    if keep:
        (root / 'head/tmp/deep').chmod(0o700)  # let pytest's basetemp cleanup remove it


def test_interrupt_stops_running_suites(tmp_path):
    log_path = tmp_path / 'log'
    import threading
    result = {}
    worker = threading.Thread(target=lambda: result.update(code=gate.run_logged(
        ['sh', '-c', 'sleep 60 & wait'], log_path, dict(os.environ), tmp_path, 120)))
    gate.STOPPING.clear()
    worker.start()
    try:
        deadline = time.time() + 10
        while not gate.CHILDREN and time.time() < deadline:
            time.sleep(0.01)
        with pytest.raises(KeyboardInterrupt):
            gate.interrupt(15, None)
        worker.join(10)
        assert not worker.is_alive() and result['code'][0] != 0
        assert gate.run_logged(['true'], log_path, dict(os.environ), tmp_path, 10)[0] == 130
    finally:
        gate.STOPPING.clear()
