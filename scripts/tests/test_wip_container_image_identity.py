"""wip.sh reuses an existing WIP container whose image ID differs from the role image's
only when the images have the same content (scripts/lib/dev_image_content.py): docker
save | load into a containerd image store gives identical images other IDs. Stale
layers, config or toolchain, a missing image and a wrong GPU binding are still refused."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WIP = (ROOT / 'wip.sh').read_text()
A, B, C = ('sha256:' + c * 64 for c in 'abc')

DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
host = os.environ.get('STUB_HOST', 'local')
state = json.loads(open(os.environ['STUB_STATE']).read()).get(host, {})
with open(os.environ['STUB_LOG'], 'a') as log:
    log.write(json.dumps([host] + args) + '\n')
images, containers = state.get('images', {}), state.get('containers', {})
def image(ref):
    return images.get(ref) or next((r for r in images.values() if r['Id'] == ref), None)
if args[:2] == ['image', 'inspect']:
    if args[2] == '-f':
        record = image(args[4])
        if not record: sys.exit(1)
        print(record['Id']); sys.exit(0)
    record = image(args[2])
    if not record: print('[]'); sys.exit(1)
    print(json.dumps([record])); sys.exit(0)
if args[:2] == ['container', 'inspect']:
    sys.exit(0 if args[2] in containers else 1)
if args[0] == 'inspect' and args[1] == '-f':
    fmt, name = args[2], args[3]
    if name not in containers: sys.exit(1)
    c = containers[name]
    if fmt == '{{.Image}}': print(c['Image'])
    elif 'DeviceRequests' in fmt: print(c.get('device', 'nvidia.com/gpu=GPU-x'))
    elif '.Config.User' in fmt: print(f'{os.getuid()}:{os.getgid()}')
    elif 'build-caches' in fmt: print('on:hash')
    elif 'Mounts' in fmt: print(c.get('mount', ''))
    elif 'State.Running' in fmt: print('true')
    sys.exit(0)
sys.exit(0)
'''
SSH = '''#!/usr/bin/env bash
while [[ "$1" == -o ]]; do shift 2; done
host="$1"; shift
STUB_HOST="$host" exec bash -c "$*"
'''


def record(ident, **change):
    config = {'Env': ['PATH=/usr/bin'], 'WorkingDir': '/workspace',
              'Labels': {'io.cuteafd.toolchain.hash': 'aa74e55ca2130070'}}
    config.update({k: v for k, v in change.items() if k not in ('Layers', 'Architecture')})
    return {'Id': ident, 'Architecture': change.get('Architecture', 'arm64'), 'Os': 'linux', 'Config': config,
            'RootFS': {'Type': 'layers', 'Layers': change.get('Layers', ['sha256:' + '11' * 32])}}


STALE = {
    'toolchain': dict(Labels={'io.cuteafd.toolchain.hash': 'bb00000000000000'}),
    'layers': dict(Layers=['sha256:' + '11' * 32, 'sha256:' + '22' * 32]),
    'arch': dict(Architecture='amd64'),
    'env': dict(Env=['PATH=/usr/bin', 'LD_PRELOAD=/x.so']),
    'stop-timeout-0': dict(StopTimeout=0),
}


def run(tmp_path, state, script):
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    for name, body in (('docker', DOCKER), ('ssh', SSH)):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    (tmp_path / 'state.json').write_text(json.dumps(state))
    log = tmp_path / 'docker.log'
    env = {**os.environ, 'PATH': f'{bindir}:{os.environ["PATH"]}', 'STUB_STATE': str(tmp_path / 'state.json'),
           'STUB_LOG': str(log)}
    result = subprocess.run(['bash', '-c', 'set -euo pipefail\n' + script], capture_output=True, text=True,
                            timeout=60, env=env)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def function(name):
    start = WIP.index(name + '() {')
    return WIP[start:WIP.index('\n}\n', start) + 3]


HELPERS = function('wip_image_content') + function('wip_same_image_content')
PREFIX = (f'repo_root={ROOT}\nrelease_die() {{ echo "DIE $*" >&2; exit 9; }}\n'
          'COORDINATOR_DOCKER_DEV=coord:tc coordinator_container=c RELEASE_COORDINATOR_GPU_UUID=GPU-x\n'
          'SPARK_EXPERT_DOCKER_DEV=spark:tc spark_container=w seed_host=seed\n')


def spark_state(spark5_image, spark5_container=B):
    images = {'spark:tc': spark5_image} if spark5_image else {}
    return {'local': {'images': {'coord:tc': record(C)}},
            'seed': {'images': {'spark:tc': record(A)}, 'containers': {'w': {'Image': A}}},
            'spark5': {'images': images, 'containers': {'w': {'Image': spark5_container}}}}


def preflight(tmp_path, state):
    script = (PREFIX + 'wip_spark_leg=1 wip_container_hosts=(seed spark5)\n' + HELPERS +
              function('preflight_existing_container_images') + 'preflight_existing_container_images\n')
    return run(tmp_path, state, script)


def test_spark_container_on_a_reloaded_image_of_the_same_content_is_reused(tmp_path):
    result, calls = preflight(tmp_path, spark_state(record(B)))
    assert result.returncode == 0, result.stderr
    assert f"WIP: spark5 w runs {B}, the same content as seed's spark:tc" in result.stdout
    assert 'seed w runs' not in result.stdout  # the seed's own ID takes the fast path


@pytest.mark.parametrize('stale', list(STALE))
def test_spark_container_on_stale_content_is_refused(tmp_path, stale):
    result, _ = preflight(tmp_path, spark_state(record(B, **STALE[stale])))
    assert result.returncode == 9
    assert 'DIE spark5 w uses an old development image' in result.stderr


def test_spark_container_whose_image_is_gone_is_refused(tmp_path):
    result, _ = preflight(tmp_path, spark_state(None))
    assert result.returncode == 9 and 'DIE spark5 w uses an old development image' in result.stderr


def test_spark_hosts_without_a_container_or_with_the_seed_id_pass(tmp_path):
    state = spark_state(record(A), spark5_container=A)
    result, calls = preflight(tmp_path, state)
    assert result.returncode == 0, result.stderr
    assert not any(c[1:3] == ['image', 'inspect'] and c[3] != '-f' for c in calls)  # same ID: no content probe
    del state['spark5']['containers']['w']
    result, _ = preflight(tmp_path, state)
    assert result.returncode == 0, result.stderr


def coordinator_state(container_image, image=None, device='nvidia.com/gpu=GPU-x'):
    images = {'coord:tc': record(A)}
    if image:
        images[container_image] = image
    return {'local': {'images': images, 'containers': {'c': {'Image': container_image, 'device': device}}}}


def coordinator_preflight(tmp_path, state):
    script = (PREFIX + 'wip_spark_leg=0\n' + HELPERS + function('preflight_existing_container_images') +
              'preflight_existing_container_images\n')
    return run(tmp_path, state, script)


def test_coordinator_container_on_a_reloaded_image_of_the_same_content_is_reused(tmp_path):
    result, _ = coordinator_preflight(tmp_path, coordinator_state(B, record(B)))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('stale', ['toolchain', 'layers', 'env', 'missing'])
def test_coordinator_container_on_stale_or_missing_content_is_refused(tmp_path, stale):
    image = None if stale == 'missing' else record(B, **STALE[stale])
    result, _ = coordinator_preflight(tmp_path, coordinator_state(B, image))
    assert result.returncode == 9 and 'DIE c uses an old development image' in result.stderr


def test_coordinator_same_content_does_not_excuse_a_wrong_gpu(tmp_path):
    result, _ = coordinator_preflight(tmp_path, coordinator_state(B, record(B), device='nvidia.com/gpu=GPU-y'))
    assert result.returncode == 9 and 'not bound to the configured physical GPU' in result.stderr


def ensure_local(tmp_path, state):
    start = WIP.index('# wip_container_gpu_ok CONTAINER')
    body = WIP[start:WIP.index('\n}\n', WIP.index('ensure_local_container() {')) + 3]
    state['local']['containers']['c']['mount'] = str(tmp_path)
    script = (PREFIX + f'wip_toolchain_hash=hash wip_mount_root={tmp_path} hf_home={tmp_path} CUTEAFD_BUILD_CACHES=on\n'
              'cuteafd_build_cache_docker_args() { :; }\n' + HELPERS + body + 'ensure_local_container\n')
    return run(tmp_path, state, script)


@pytest.mark.parametrize('same', [True, False])
def test_local_container_reuse_follows_content(tmp_path, same):
    image = record(B) if same else record(B, **STALE['layers'])
    result, calls = ensure_local(tmp_path, coordinator_state(B, image))
    assert not any(c[1] in ('rm', 'run') for c in calls)  # never recreated or removed
    if same:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 9 and 'DIE c uses an old development image' in result.stderr


def remote_reuse_block():
    remote = WIP.split('"$wip_toolchain_hash" <<\'REMOTE\'\n', 1)[1].split('\nREMOTE', 1)[0]
    start = remote.index('  container_id="$(docker inspect -f \'{{.Image}}\' "$container")"')
    end = remote.index('\n  fi\n', start) + 6
    return remote[start:end]


@pytest.mark.parametrize('image,ok', [(record(B), True), (record(B, **STALE['toolchain']), False),
                                      (record(B, **STALE['stop-timeout-0']), False), (None, False)])
def test_remote_container_reuse_follows_content(tmp_path, image, ok):
    state = coordinator_state(B, image)
    state['local']['containers']['w'] = state['local']['containers'].pop('c')
    script = (f'container=w image_id={A} helper_lib={ROOT}/scripts/lib\n' + remote_reuse_block() + 'echo reused\n')
    result, _ = run(tmp_path, state, script)
    assert (result.returncode == 0) == ok, result.stderr
    if not ok:
        assert result.returncode == 2 and 'w uses an old development image' in result.stderr


def test_remote_helper_is_staged_with_the_cdi_helper():
    assert ('"$repo_root/scripts/lib/cdi-gpus.sh" "$repo_root/scripts/lib/dev_image_content.py" '
            '"$host:$cache_staging/scripts/lib/"') in WIP
    remote = WIP.split('"$wip_toolchain_hash" <<\'REMOTE\'\n', 1)[1].split('\nREMOTE', 1)[0]
    assert remote.index('helper_lib="$6/scripts/lib"') < remote.index('$helper_lib/dev_image_content.py')
