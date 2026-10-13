import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('golden_run', ROOT / 'scripts/bench/golden-run.py')
golden = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(golden)
COMPARE = ROOT / 'scripts/bench/compare-logits.py'
CSPEC = importlib.util.spec_from_file_location('compare_logits', COMPARE)
compare = importlib.util.module_from_spec(CSPEC)
CSPEC.loader.exec_module(compare)
IMAGE = 'sha256:' + 'ab' * 32

STUB_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
state = Path(os.environ['STUB_DOCKER_STATE'])
names = json.loads(state.read_text()) if state.exists() else {}
with open(os.environ['STUB_DOCKER_LOG'], 'a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\n')
args = sys.argv[1:]
def save():
    state.write_text(json.dumps(names))
if args[:2] == ['image', 'inspect']:
    print(json.dumps([{'Id': args[2], 'RepoTags': ['cuteafd-coordinator-dev:test']}])); sys.exit(0)
if args[:1] == ['ps']:
    if '-q' in args[1] or '-aq' in args:
        wanted = args[args.index('--filter') + 1].split('=', 1)[1].strip('^$')
        print('\n'.join(ident for ident, n in names.items() if n['name'] == wanted))
    else:
        print('\n'.join(n['name'] for n in names.values()))
    sys.exit(0)
if args[:1] == ['inspect']:
    print(json.dumps(names[args[-1]]['labels'])); sys.exit(0)
if args[:2] == ['rm', '-f']:
    names.pop(args[2], None); save(); sys.exit(0)
if args[:1] == ['run']:
    name = args[args.index('--name') + 1]
    labels = {}
    if '--label' in args:
        key, _, value = args[args.index('--label') + 1].partition('=')
        labels[key] = value
    ident = 'id-' + name
    names[ident] = {'name': name, 'labels': labels}; save()
    out = next(v.split(':')[0] for v in (args[i + 1] for i, a in enumerate(args) if a == '-v') if v.endswith(':/out'))
    env = dict(args[i + 1].split('=', 1) for i, a in enumerate(args) if a == '-e')
    print('GLM admission placement=attention ' + env.get('CUTEAFD_ATTENTION_PLACEMENT', 'auto'))
    print('prefill logits: 4 rows | mean NLL engine 1.0 | logits digest 00ff')
    if os.environ.get('STUB_LOGITS'):
        Path(out, 'logits').mkdir(parents=True, exist_ok=True)
        Path(out, 'logits', 'logits.bin').write_bytes(Path(os.environ['STUB_LOGITS']).read_bytes())
    time.sleep(float(os.environ.get('STUB_RUN_SLEEP', '0')))
    code = int(os.environ.get('STUB_RUN_EXIT', '0'))
    if os.environ.get('STUB_LEAVE') != '1':
        names.pop(ident, None); save()
    sys.exit(code)
sys.exit(0)
'''


def write_slot(home, instance, slot='s1', tamper=False):
    root = home / '.cache/cuteafd/builds' / f'wip-{instance}' / 'slots' / slot / 'coordinator'
    art = root / 'workspace/.cuteafd-wip'
    (art / 'fp8/fp8-glm').mkdir(parents=True)
    files = {'cuteafd': b'bin', 'libcuteafd_native.so': b'lib', 'PROGRAMS.json': b'{}'}
    for name, data in files.items():
        (art / name).write_bytes(data)
    sums = ''.join(f'{hashlib.sha256(d).hexdigest()}  {n}\n' for n, d in files.items()).encode()
    (art / 'ARTIFACT_SHA256SUMS').write_bytes(sums)
    (root / 'workspace/docker').mkdir(parents=True)
    (root / 'workspace/docker/release-entrypoint.sh').write_text('#!/bin/sh\nexec "$@"\n')
    (root / 'workspace/third_party/sparkinfer').mkdir(parents=True)
    (root / 'workspace/third_party/sparkinfer.lock.json').write_text('{}')
    (root / 'workspace/scripts/build').mkdir(parents=True)
    (root / 'workspace/scripts/build/verify-sparkinfer-source.py').write_text('print("verified")\n')
    (root / 'workspace/BUILD_IDENTITY.json').write_text(json.dumps({'commit': 'c0ffee', 'dirty': False}))
    source = b'abc  ./x\n'
    (root / 'SOURCE_SHA256SUMS').write_bytes(source)
    meta = json.dumps({'slot': slot, 'role': 'coordinator', 'wip_instance': instance, 'base_image': 'dev:tc',
                       'base_image_id': IMAGE, 'source_manifest_sha256': hashlib.sha256(source).hexdigest(),
                       'artifact_manifest_sha256': hashlib.sha256(sums).hexdigest()}).encode()
    (root / 'META.json').write_bytes(meta)
    (root / 'FINGERPRINT').write_text(hashlib.sha256(meta + (b'x' if tamper else b'')).hexdigest() + '\n')
    return root


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'docker').write_text(STUB_DOCKER)
    (bindir / 'docker').chmod(0o755)
    hub = tmp_path / 'hub'
    snap = hub / 'models--org--model/snapshots/rev1'
    snap.mkdir(parents=True)
    (snap / 'config.json').write_text(json.dumps({'vocab_size': 8}))
    (hub / 'models--org--model/refs').mkdir()
    (hub / 'models--org--model/refs/main').write_text('rev1\n')
    locks = tmp_path / 'locks'
    locks.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('PATH', f'{bindir}:{os.environ["PATH"]}')
    monkeypatch.setenv('STUB_DOCKER_STATE', str(tmp_path / 'docker-state.json'))
    monkeypatch.setenv('STUB_DOCKER_LOG', str(tmp_path / 'docker.log'))
    monkeypatch.setenv('CUTEAFD_HF_HUB', str(hub))
    monkeypatch.delenv('CUTEAFD_HW_LOCKS_HELD', raising=False)
    monkeypatch.delenv('CUTEAFD_HW_GATE_TICKETS', raising=False)
    monkeypatch.setattr(golden.hwlock, 'LOCK_DIR', locks)
    monkeypatch.setattr(golden.PowerSampler, '_loop', lambda self: None)
    monkeypatch.setattr(golden, 'POWER_STATE', tmp_path / 'no-powerd.json')
    return dict(home=home, tmp=tmp_path, locks=locks, log=tmp_path / 'docker.log', state=tmp_path / 'docker-state.json')


def docker_calls(env):
    return [json.loads(line) for line in env['log'].read_text().splitlines()] if env['log'].exists() else []


def runs(env):
    return [c for c in docker_calls(env) if c[:1] == ['run']]


BASE = ['--task', 'gr', '--family', 'glm', '--model', 'org/model']
GOLDEN = ['--', '--snapshot', '{snapshot}', '--device', '0', '--split-device', '1', '--skip-routed-experts', '--nll']


@pytest.mark.parametrize('family,sub', sorted(golden.SUBCOMMANDS.items()))
def test_argv_per_family_passes_args_verbatim(family, sub):
    args = ['--snapshot', '{snapshot}', '--golden', '/g', '--resume-at', '100,256', '--save-logits', '{out}/l']
    built = golden.substitute(golden.golden_args(family, args), '/hf/hub/m')
    assert built == [sub, '--snapshot', '/hf/hub/m', '--golden', '/g', '--resume-at', '100,256', '--save-logits', '/out/l']


@pytest.mark.parametrize('args,match', [
    (['--native-lib', '/x.so'], 'comes from the slot'),
    (['--manifest=/p.json'], 'comes from the slot'),
    (['--peers', 'ostrich:1'], 'Spark peers'),
    (['--save-logits', '/tmp/x'], 'under the arm directory'),
    (['--save-logits=/tmp/x'], 'under the arm directory'),
])
def test_argv_refuses_slot_spark_and_unwritable_paths(args, match):
    with pytest.raises(ValueError, match=match):
        golden.golden_args('dsv4', args)


def test_lock_order_is_canonical_gpu0_then_gpu1():
    assert golden.locks_for(golden.parse_gpus('1,0')) == ['gpu0', 'gpu1']
    assert golden.locks_for(golden.parse_gpus('1')) == ['gpu1']
    with pytest.raises(ValueError):
        golden.parse_gpus('2')


def test_interleaved_schedule():
    arms = {'a': {}, 'b': {}}
    assert list(golden.schedule(arms, 3)) == [('a', 1), ('b', 1), ('b', 2), ('a', 2), ('a', 3), ('b', 3)]
    assert list(golden.schedule(arms, 2, interleave=False)) == [('a', 1), ('b', 1), ('a', 2), ('b', 2)]


def test_sparks_refused(env, capsys):
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--sparks', 'ostrich'] + GOLDEN) == 2
    assert 'not supported yet' in capsys.readouterr().err


def test_unsealed_slot_is_refused_before_any_lock(env, capsys, monkeypatch):
    write_slot(env['home'], 'i1', tamper=True)
    taken = []
    monkeypatch.setattr(golden.hwlock, 'hold', lambda *a, **k: taken.append(a))
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 2
    assert 'unsealed' in capsys.readouterr().err
    assert not taken and not runs(env)


def test_tampered_artifact_is_refused(env, capsys):
    root = write_slot(env['home'], 'i1')
    (root / 'workspace/.cuteafd-wip/libcuteafd_native.so').write_bytes(b'other')
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 2
    assert 'does not match ARTIFACT_SHA256SUMS' in capsys.readouterr().err


def test_env_per_arm_and_container_mounts(env, monkeypatch):
    write_slot(env['home'], 'i1')
    order = []
    real_hold = golden.hwlock.hold

    def hold(names, *args, **kwargs):
        order.append(list(names))
        return real_hold(names, *args, **kwargs)
    monkeypatch.setattr(golden.hwlock, 'hold', hold)
    argv = BASE + ['--arm', 'heads=i1:s1', '--arm', 'context=i1:s1', '--env', 'RUST_LOG=warn',
                   '--arm-env', 'heads:CUTEAFD_ATTENTION_PLACEMENT=heads',
                   '--arm-env', 'context:CUTEAFD_ATTENTION_PLACEMENT=context', '--gpus', '1,0'] + GOLDEN
    assert golden.main(argv) == 0
    assert order == [['gpu0', 'gpu1']]
    calls = runs(env)
    assert [c[c.index('--name') + 1] for c in calls] == ['golden-gr-heads', 'golden-gr-context']
    for call, mode in zip(calls, ['heads', 'context']):
        envs = [call[i + 1] for i, a in enumerate(call) if a == '-e']
        assert f'CUTEAFD_ATTENTION_PLACEMENT={mode}' in envs and 'RUST_LOG=warn' in envs
        assert call[call.index('--user') + 1] == f'{os.getuid()}:{os.getgid()}'
        assert call[call.index('--gpus') + 1] == '"device=0,1"'
        assert call[call.index('--ipc') + 1] == 'host' and 'memlock=-1:-1' in call
        mounts = [call[i + 1] for i, a in enumerate(call) if a == '-v']
        assert f"{env['tmp']}/hub:/hf/hub:ro" in mounts
        assert any(m.endswith(':/opt/cuteafd/lib/libcuteafd_native.so:ro') for m in mounts)
        assert any(m.endswith(':/opt/cuteafd/share:ro') for m in mounts)
        assert any(m.endswith(f'/gr/golden/{mode}/r1:/out') for m in mounts)
        assert IMAGE in call
        tail = call[call.index(IMAGE) + 1:]
        assert tail[-8:] == ['glm-golden', '--snapshot', '/hf/hub/models--org--model/snapshots/rev1', '--device', '0',
                             '--split-device', '1', '--skip-routed-experts', '--nll'][-8:]
    out = env['home'] / '.cache/cuteafd/builds/gr/golden'
    record = json.loads((out / 'heads/r1/argv.json').read_text())
    assert record['arm_env'] == {'RUST_LOG': 'warn', 'CUTEAFD_ATTENTION_PLACEMENT': 'heads'}
    assert record['image_id'] == IMAGE and len(record['seal_sha256']) == 64
    assert record['source_commit'] == 'c0ffee'
    assert 'power_before' in record and 'power_samples' in record
    assert (out / 'heads/r1/exit').read_text() == '0\n'
    summary = (out / 'summary.md').read_text()
    assert 'attention heads' in summary and 'logits digest 00ff' in summary
    # Locks are free again afterwards.
    for name in ('gpu0', 'gpu1'):
        with open(env['locks'] / f'{name}.lock', 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_failure_removes_container_before_releasing_locks(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_EXIT', '3')
    monkeypatch.setenv('STUB_LEAVE', '1')  # as if --rm had not cleaned up
    released = []
    real_hold = golden.hwlock.hold

    class Hold:
        def __init__(self, *a, **k):
            self.inner = real_hold(*a, **k)

        def __enter__(self):
            return self.inner.__enter__()

        def __exit__(self, *exc):
            released.append(json.loads(env['state'].read_text()))
            return self.inner.__exit__(*exc)
    monkeypatch.setattr(golden.hwlock, 'hold', Hold)
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 1
    assert released == [{}]
    assert any(c[:2] == ['rm', '-f'] for c in docker_calls(env))
    assert (env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/exit').read_text() == '3\n'


def test_timeout_kills_and_removes(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_SLEEP', '30')
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--timeout', '1'] + GOLDEN) == 1
    record = json.loads((env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert record['exit'] == 124 and 'timed out' in record['problem']
    assert json.loads(env['state'].read_text()) == {}


def test_gpu_access_failure_is_named(env, monkeypatch):
    write_slot(env['home'], 'i1')
    monkeypatch.setenv('STUB_RUN_EXIT', str(golden.GPU_CHECK_EXIT))
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 1
    record = json.loads((env['home'] / '.cache/cuteafd/builds/gr/golden/a/r1/argv.json').read_text())
    assert record['problem'] == 'GPU access check failed inside the container'


def test_foreign_container_is_refused(env, capsys):
    write_slot(env['home'], 'i1')
    env['state'].write_text(json.dumps({'id-x': {'name': 'golden-gr-a', 'labels': {}}}))
    assert golden.main(BASE + ['--arm', 'a=i1:s1'] + GOLDEN) == 2
    assert 'already exists' in capsys.readouterr().err
    assert not runs(env)


def test_failed_preflight_hook_runs_nothing(env):
    write_slot(env['home'], 'i1')
    assert golden.main(BASE + ['--arm', 'a=i1:s1', '--preflight', 'exit 3'] + GOLDEN) == 78
    assert not runs(env)


def test_compare_logits_byte_identical_arms(env, monkeypatch):
    write_slot(env['home'], 'i1')
    logits = env['tmp'] / 'logits.bin'
    np.arange(32, dtype='<f4').tofile(logits)
    monkeypatch.setenv('STUB_LOGITS', str(logits))
    argv = BASE + ['--arm', 'a=i1:s1', '--arm', 'b=i1:s1', '--compare-logits', 'a:b'] + GOLDEN + \
        ['--save-logits', '{out}/logits']
    assert golden.main(argv) == 0
    summary = (env['home'] / '.cache/cuteafd/builds/gr/golden/summary.md').read_text()
    assert '| a | b | 1 | logits/logits.bin | True | 4 | 0.0000e+00 |' in summary


def test_compare_tool_on_synthetic_logits(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.normal(size=(130, 50)).astype('<f4')
    b = a.copy()
    b[7] += 0.5 * rng.normal(size=50).astype('<f4')
    b[100, a[100].argmax()] -= 100
    pa, pb, tokens = tmp_path / 'a.bin', tmp_path / 'b.bin', tmp_path / 'tokens.bin'
    a.tofile(pa)
    b.tofile(pb)
    rng.integers(0, 50, size=131).astype('<u4').tofile(tokens)
    same = compare.compare(pa, pa, 50)
    assert same['byte_identical'] and same['max_abs_diff'] == 0 and same['mean_kl'] == 0 and same['top1_agreement'] == 1
    diff = compare.compare(pa, pb, 50, tokens)
    assert not diff['byte_identical'] and diff['rows'] == 130
    assert diff['max_abs_diff'] == pytest.approx(100, rel=1e-5)
    assert diff['top1_agreement'] < 1 and diff['top1_agreement'] >= 128 / 130
    assert diff['max_kl'] > diff['mean_kl'] > 0
    assert diff['nll_rows'] == 130 and diff['nll_a'] > 0
    with pytest.raises(ValueError):
        compare.compare(pa, pb, 7)
    cli = subprocess.run([sys.executable, str(COMPARE), str(pa), str(pa), '50'], capture_output=True, text=True)
    assert cli.returncode == 0 and 'byte-identical: True' in cli.stdout


def test_resolve_model_uses_hub_paths(env):
    host, container, extra = golden.resolve_model('org/model')
    assert container == '/hf/hub/models--org--model/snapshots/rev1' and not extra
    assert golden.vocab_size(host) == 8
