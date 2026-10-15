#!/usr/bin/env python3
"""Run a family golden subcommand from sealed WIP slots on raptor's RTX GPUs.

  scripts/bench/golden-run.py --task NAME --family dsv4|glm|glmf|mimo|qwen4|qwen35\\
      --model HF_ID_OR_SNAPSHOT_DIR --arm A=WIP_INSTANCE:SLOT [--arm B=...] \\
      [--arm-env A:KEY=VALUE] [--env KEY=VALUE] [--gpus 0,1] [--repeats N] \\
      [--timeout SECONDS] [--preflight CMD] [--compare-logits A:B] -- GOLDEN ARGS...

Coordinator-only: each arm runs its slot's `cuteafd`, `libcuteafd_native.so`,
`PROGRAMS.json` and expert packages, mounted read-only as /opt/cuteafd/{bin,lib,share}
in the slot's own dev image (META base_image_id), with CDI `--device nvidia.com/gpu=N`,
`--ipc host`, memlock unlimited, the host uid (release_serving_docker_args), the
hub read-only at /hf/hub (HF_HOME=/hf) and /mnt/sparknest read-only. GOLDEN ARGS
pass through verbatim except two placeholders: `{snapshot}` (the checkpoint's
container path) and `{out}` (the arm's writable output directory, e.g.
`--save-logits {out}/logits`). Other existing absolute host paths in ARGS (a
`--golden` directory) are mounted read-only at the same path. The slot supplies
--native-lib and --manifest; ARGS may not override them.

Hardware: the GPUs' locks (gpu0 before gpu1) are taken once through the fair
queue (scripts/hw/hwlock.py) and every arm and repeat runs back to back inside
that one turn, in interleaved order (B,A on even repeats), so A/B pairs share
conditions. Slots, image and container names are checked before enqueue and
again at grant; --preflight adds a shell check at both points, like hwlock's.
Each container (golden-TASK-ARM) first checks it sees its GPUs (exit 86 when
not), is removed on any exit before the locks are released, and runs under a
per-arm timeout. Sparks are refused: real-expert goldens are not supported yet.

Output, under ~/.cache/cuteafd/builds/TASK/golden/: ARM/rN/{log,exit,argv.json}
and whatever the golden wrote to {out}; argv.json holds the exact docker
command, image ID, seal SHA, env and the GPUs' power context (gpu-powerd state,
sampled power limit and draw). summary.md lists exit codes and the key log lines
(admission, logits digests, agreement/top-1, resume verdicts, errors), plus each
--compare-logits A:B result: sha256 byte-equality, then max |diff|, KL and top-1
over rows (vocab from the checkpoint's config.json, or --vocab).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBCOMMANDS = {'dsv4': 'dsv4-golden', 'v41': 'v41-golden', 'glm': 'glm-golden', 'glmf': 'glmf-golden',
               'mimo': 'mimo-golden', 'qwen4': 'qwen4-golden', 'qwen35': 'qwen35-golden'}
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
ENV_KEY = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
# Key result lines across the families' goldens.
KEY_LINE = re.compile(r'(?i)(admission|prefill logits|decode logits|logits digest|top-1|agreement|mean nll|mean kl|'
                      r'resume at|resume check|vs one straight|byte-identical|rows:|verify|'
                      r'\berror\b|panicked|failed|refus|out of memory|golden-run:)')
GPU_CHECK_EXIT = 86
# Golden flags that write files: only the arm's {out} directory is writable.
WRITERS = ('--save-logits', '--bench-dump', '--replay-dump', '--geometry-trace')
CONTAINER_HOME = '/tmp/cuteafd-home'
OUT = '/out'
SLOT_LIB, SLOT_BIN, SLOT_SHARE = '/opt/cuteafd/lib', '/opt/cuteafd/bin', '/opt/cuteafd/share'
ENTRYPOINT = '/opt/cuteafd/entry/release-entrypoint.sh'
POWER_STATE = Path('/run/gpu-powerd/state.json')
POWER_SAMPLE_S = 5.0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hwlock = _load('hwlock', REPO / 'scripts/hw/hwlock.py')
cards = _load('wip_cards', REPO / 'scripts/bench/wip-cards.py')


class Interrupted(Exception):
    pass


def _interrupt(signum, _frame):
    raise Interrupted(f'signal {signum}')


def run(cmd, timeout=120, **kwargs):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout, **kwargs)


# --------------------------------------------------------------------------- parsing

def parse_arms(values):
    arms = {}
    for value in values:
        name, sep, address = value.partition('=')
        instance, sep2, slot = address.partition(':')
        if not (sep and sep2) or not all(NAME.fullmatch(x) for x in (name, instance, slot)) \
                or len(instance) > 41 or len(slot) > 64:
            raise ValueError(f'invalid --arm {value!r} (NAME=WIP_INSTANCE:SLOT)')
        if name in arms:
            raise ValueError(f'duplicate arm {name}')
        arms[name] = {'instance': instance, 'slot': slot, 'env': {}}
    return arms


def parse_env(value):
    key, sep, val = value.partition('=')
    if not sep or not ENV_KEY.fullmatch(key) or '\n' in val:
        raise ValueError(f'invalid environment setting {value!r} (KEY=VALUE)')
    return key, val


def apply_env(arms, common, per_arm):
    for value in common:
        key, val = parse_env(value)
        for arm in arms.values():
            arm['env'][key] = val
    for value in per_arm:
        name, sep, setting = value.partition(':')
        if not sep or name not in arms:
            raise ValueError(f'--arm-env {value!r} names no arm (ARM:KEY=VALUE)')
        key, val = parse_env(setting)
        arms[name]['env'][key] = val


def parse_gpus(value):
    gpus = sorted({int(g) for g in value.split(',') if g.strip()})
    if not gpus or any(g not in (0, 1) for g in gpus):
        raise ValueError('--gpus takes raptor GPU indices 0 and/or 1')
    return gpus


def locks_for(gpus):
    return hwlock.canonical([f'gpu{g}' for g in gpus])


def schedule(arms, repeats, interleave=True):
    for repeat in range(1, repeats + 1):
        order = list(arms)
        if interleave and repeat % 2 == 0:
            order.reverse()
        for name in order:
            yield name, repeat


def golden_args(family, args):
    if family not in SUBCOMMANDS:
        raise ValueError(f'unknown family {family}')
    for arg in args:
        flag = arg.split('=', 1)[0]
        if flag in ('--native-lib', '--manifest'):
            raise ValueError(f'{flag} comes from the slot; drop it from the golden args')
        if flag in ('--peers',) or flag.startswith('--sparks'):
            raise ValueError('Spark peers are not supported: run coordinator-only (e.g. --skip-routed-experts)')
    for flag, value in zip(args, args[1:]):
        if flag in WRITERS and not value.startswith('{out}'):
            raise ValueError(f'{flag} {value}: write outputs under the arm directory ({flag} {{out}}/NAME)')
    for arg in args:
        flag, sep, value = arg.partition('=')
        if sep and flag in WRITERS and not value.startswith('{out}'):
            raise ValueError(f'{arg}: write outputs under the arm directory ({flag}={{out}}/NAME)')
    return [SUBCOMMANDS[family], *args]


def substitute(args, snapshot, out=OUT):
    return [a.replace('{snapshot}', snapshot).replace('{out}', out) for a in args]


def container_name(task, arm):
    return f'golden-{task}-{arm}'


# --------------------------------------------------------------------------- checkpoint

def hub_root():
    return Path(os.environ.get('CUTEAFD_HF_HUB', Path.home() / '.cache/huggingface/hub')).resolve()


def resolve_model(model, hub=None):
    """(host snapshot dir, container path, extra read-only mounts)."""
    hub = hub or hub_root()
    path = Path(model).expanduser()
    if not path.is_absolute():
        repo_id, _, revision = model.partition('@')
        if repo_id.count('/') != 1:
            raise ValueError(f'--model {model!r} is neither a snapshot directory nor ORG/NAME[@REVISION]')
        root = hub / ('models--' + repo_id.replace('/', '--'))
        if not revision:
            ref = root / 'refs/main'
            if not ref.is_file():
                raise ValueError(f'{repo_id}: no {ref}; pass the snapshot directory')
            revision = ref.read_text().strip()
        path = root / 'snapshots' / revision
    path = path.resolve()
    if not (path / 'config.json').is_file():
        raise ValueError(f'{path} has no config.json')
    if path.is_relative_to(hub):
        return path, '/hf/hub/' + str(path.relative_to(hub)), []
    if path.is_relative_to('/mnt/sparknest'):
        return path, str(path), []
    return path, str(path), [str(path)]


def vocab_size(snapshot):
    config = json.loads((Path(snapshot) / 'config.json').read_text())
    for scope in (config, config.get('text_config') or {}, config.get('language_config') or {}):
        if isinstance(scope.get('vocab_size'), int):
            return scope['vocab_size']
    raise ValueError(f'{snapshot}/config.json has no vocab_size; pass --vocab')


def path_mounts(args, covered):
    """Existing absolute host paths in ARGS, outside the paths already mounted."""
    mounts = []
    for arg in args:
        value = arg.split('=', 1)[1] if arg.startswith('--') and '=' in arg else arg
        if not value.startswith('/') or value.startswith(OUT) or '{' in value:
            continue
        path = Path(value)
        if not path.exists() or any(path.resolve().is_relative_to(c) for c in covered):
            continue
        if str(path) not in mounts:
            mounts.append(str(path))
    return mounts


# --------------------------------------------------------------------------- slot

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def verify_slot(arm):
    """Seal (META/FINGERPRINT, source and artifact manifests, as wip-cards'
    slot_check), then every listed artifact's own sha256. Returns the record."""
    probe = {'instance': arm['instance'], 'slot': arm['slot']}
    if arm.get('root'):
        probe['root'] = arm['root']
    meta = json.loads(cards.slot_check(probe, 'raptor'))
    source = meta.get('seal_source', '')
    if not source.startswith('host:'):
        raise ValueError(f"{arm['instance']}:{arm['slot']}: slot is readable only inside its container ({source}); "
                         'rebuild it with ./wip.sh as the host user')
    root = Path(source[5:])
    artifacts = root / 'workspace/.cuteafd-wip'
    listed = {}
    for line in (artifacts / 'ARTIFACT_SHA256SUMS').read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip().removeprefix('./')] = digest
    for name in ('cuteafd', 'libcuteafd_native.so', 'PROGRAMS.json'):
        if name not in listed:
            raise ValueError(f'{root}: ARTIFACT_SHA256SUMS does not list {name}')
    for name, digest in listed.items():
        if sha256_file(artifacts / name) != digest:
            raise ValueError(f'{root}: {name} does not match ARTIFACT_SHA256SUMS')
    for relative in ('workspace/docker/release-entrypoint.sh', 'workspace/third_party/sparkinfer'):
        if not (root / relative).exists():
            raise ValueError(f'{root}: slot lacks {relative}')
    identity = {}
    try:
        identity = json.loads((root / 'workspace/BUILD_IDENTITY.json').read_text())
    except (OSError, json.JSONDecodeError):
        pass
    image_id = meta.get('base_image_id')
    if not image_id:
        raise ValueError(f'{root}: META.json has no base_image_id')
    return dict(instance=arm['instance'], slot=arm['slot'], root=str(root), seal_sha256=meta['seal_sha256'],
                artifact_manifest_sha256=meta['artifact_manifest_sha256'], artifacts=listed,
                packages=(meta.get('contents') or {}).get('packages'), image=meta.get('base_image'),
                image_id=image_id, source_commit=identity.get('commit'), source_dirty=identity.get('dirty'),
                sparkinfer_revision=meta.get('sparkinfer_revision'))


def verify_sparkinfer(slot):
    workspace = Path(slot['root']) / 'workspace'
    run([sys.executable, str(workspace / 'scripts/build/verify-sparkinfer-source.py'),
         '--source', str(workspace / 'third_party/sparkinfer'),
         '--lock', str(workspace / 'third_party/sparkinfer.lock.json')])


def image_record(image_id):
    try:
        info = json.loads(run(['docker', 'image', 'inspect', image_id]).stdout)[0]
    except (subprocess.SubprocessError, IndexError, json.JSONDecodeError):
        raise ValueError(f'dev image {image_id} is missing (scripts/build/build-dev-images.sh, or rebuild the slot)') from None
    if info['Id'] != image_id:
        raise ValueError(f'dev image {image_id} resolved to {info["Id"]}')
    return dict(id=info['Id'], tags=info.get('RepoTags') or [])


def container_exists(name):
    names = run(['docker', 'ps', '-a', '--format', '{{.Names}}']).stdout.split()
    return name in names


def preflight(state, arms):
    """CPU-only checks, before enqueue and again at grant: every slot sealed and
    unchanged, its image present, and no container of ours left over."""
    records = {}
    for name, arm in arms.items():
        slot = verify_slot(arm)
        verify_sparkinfer(slot)
        slot['image_record'] = image_record(slot['image_id'])
        if container_exists(container_name(state['task'], name)):
            raise ValueError(f"container {container_name(state['task'], name)} already exists "
                             '(a previous run of this task left it; inspect and docker rm it)')
        records[name] = slot
    previous = state.get('slots')
    if previous:
        for name, slot in records.items():
            if (slot['seal_sha256'], slot['artifacts']) != (previous[name]['seal_sha256'], previous[name]['artifacts']):
                raise ValueError(f'arm {name}: sealed slot changed since enqueue')
    return records


# --------------------------------------------------------------------------- docker

def serving_args():
    """The shared serving identity (uid/gid, groups, USER, HOME, HF_HOME, caches)."""
    script = 'source scripts/lib/release-common.sh; release_serving_docker_args; printf "%s\\0" "${RELEASE_SERVING_ARGS[@]}"'
    out = subprocess.run(['bash', '-c', script], cwd=REPO, check=True, capture_output=True, timeout=60).stdout
    return out.decode().rstrip('\0').split('\0')


def docker_command(state, name, arm, slot, out_dir, golden):
    root = Path(slot['root'])
    art = root / 'workspace/.cuteafd-wip'
    gpus = state['gpus']
    mounts = [(str(art / 'cuteafd'), f'{SLOT_BIN}/cuteafd'),
              (str(art / 'libcuteafd_native.so'), f'{SLOT_LIB}/libcuteafd_native.so')]
    for source, target in (('bin-variants', SLOT_BIN), ('lib-variants', SLOT_LIB)):
        variants = art / source
        if variants.is_dir():
            for variant in sorted(variants.iterdir()):
                if variant.is_dir():
                    mounts.append((str(variant), f'{target}/{variant.name}'))
    for package in ('exl3', 'fp8'):
        if (art / package).is_dir():
            mounts.append((str(art / package), f'{SLOT_LIB}/{package}'))
    mounts += [(str(art), SLOT_SHARE),
               (str(root / 'workspace/docker/release-entrypoint.sh'), ENTRYPOINT),
               (str(root / 'workspace/third_party/sparkinfer'), '/source/third_party/sparkinfer'),
               (str(state['hub']), '/hf/hub')]
    if Path('/mnt/sparknest').is_dir():
        mounts.append(('/mnt/sparknest', '/mnt/sparknest'))
    mounts += [(p, p) for p in state['extra_mounts']]
    cmd = ['docker', 'run', '--rm', '--name', container_name(state['task'], name),
           '--label', f"cuteafd.golden-run={state['task']}/{name}",
           *state['serving_args'], *cards.cdi_gpu_args(gpus), '--ipc', 'host',
           '--ulimit', 'memlock=-1:-1']
    for source, target in mounts:
        cmd += ['-v', f'{source}:{target}:ro']
    cmd += ['-v', f'{out_dir}:{OUT}']
    env = {'PATH': f'{SLOT_BIN}:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
           'PYTHONPATH': '/source/third_party/sparkinfer', 'CUTEAFD_NATIVE_LIB': f'{SLOT_LIB}/libcuteafd_native.so',
           'RUST_LOG': 'info', **arm['env']}
    for key, value in env.items():
        cmd += ['-e', f'{key}={value}']
    # The container first proves it sees its GPUs (a systemd daemon-reload on the
    # host strips GPU access silently), then execs the golden through the slot's
    # release entrypoint (library path for the native lib and CuTe runtime).
    check = (f'mkdir -p "$HOME"; n=$(nvidia-smi -L 2>&1 | grep -c "^GPU "); nvidia-smi -L >&2; '
             f'if [ "$n" -ne {len(gpus)} ]; then echo "golden-run: GPU access check failed: '
             f'saw $n of {len(gpus)} GPUs" >&2; exit {GPU_CHECK_EXIT}; fi; exec "$@"')
    cmd += ['--entrypoint', 'bash', slot['image_id'], '-c', check, 'golden-run', ENTRYPOINT, 'cuteafd', *golden]
    return cmd, env


def remove_container(name):
    """Stop and remove our container; never one without our label."""
    try:
        found = run(['docker', 'ps', '-aq', '--filter', f'name=^{name}$']).stdout.split()
    except subprocess.SubprocessError as error:
        return f'could not list containers: {error}'
    for ident in found:
        labels = json.loads(run(['docker', 'inspect', '-f', '{{json .Config.Labels}}', ident]).stdout or 'null') or {}
        if 'cuteafd.golden-run' not in labels:
            return f'refusing to remove foreign container {name}'
        subprocess.run(['docker', 'rm', '-f', ident], capture_output=True, timeout=120)
    if container_exists(name):
        return f'container {name} is still present'
    return None


# --------------------------------------------------------------------------- power

def governor_summary():
    try:
        state = json.loads(POWER_STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return dict(source=str(POWER_STATE), updated=state.get('updated'), budget_w=state.get('budget_w'),
                reason=state.get('reason'),
                gpus=[dict(index=g.get('index'), limit_w=g.get('limit_w'), max_w=g.get('max_w'),
                           bound=(g.get('plan') or {}).get('bound'), reason=(g.get('plan') or {}).get('reason'))
                      for g in state.get('gpus', [])])


class PowerSampler:
    """nvidia-smi power limit/draw of the arm's GPUs every few seconds while it runs."""

    def __init__(self, gpus, period=POWER_SAMPLE_S):
        self.gpus, self.period, self.samples = gpus, period, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while True:
            try:
                out = run(['nvidia-smi', '-i', ','.join(map(str, self.gpus)), '--format=csv,noheader,nounits',
                           '--query-gpu=index,power.limit,power.draw'], timeout=20).stdout
                for line in out.splitlines():
                    index, limit, draw = (x.strip() for x in line.split(','))
                    self.samples.append((int(index), float(limit), float(draw)))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            if self.stop.wait(self.period):
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=30)

    def summary(self):
        result = {}
        for gpu in self.gpus:
            rows = [(limit, draw) for index, limit, draw in self.samples if index == gpu]
            if rows:
                result[str(gpu)] = dict(samples=len(rows), mean_limit_w=round(sum(r[0] for r in rows) / len(rows), 1),
                                        min_limit_w=min(r[0] for r in rows), max_limit_w=max(r[0] for r in rows),
                                        mean_draw_w=round(sum(r[1] for r in rows) / len(rows), 1),
                                        max_draw_w=max(r[1] for r in rows))
        return result


# --------------------------------------------------------------------------- running

def run_arm(state, name, repeat, arms):
    arm, slot = arms[name], state['slots'][name]
    out_dir = Path(state['root']) / name / f'r{repeat}'
    out_dir.mkdir(parents=True, exist_ok=False)
    golden = substitute(state['golden'], state['snapshot_container'])
    cmd, env = docker_command(state, name, arm, slot, out_dir, golden)
    record = dict(arm=name, repeat=repeat, argv=cmd, golden=golden, container=container_name(state['task'], name),
                  image=slot['image'], image_id=slot['image_id'], image_tags=slot['image_record']['tags'],
                  seal_sha256=slot['seal_sha256'], artifact_manifest_sha256=slot['artifact_manifest_sha256'],
                  slot=f"{slot['instance']}:{slot['slot']}", slot_root=slot['root'], source_commit=slot['source_commit'],
                  source_dirty=slot['source_dirty'], env=env, arm_env=arm['env'], gpus=state['gpus'],
                  locks=state['locks'], snapshot=state['snapshot'], timeout_s=state['timeout'],
                  power_before=governor_summary())
    print(f"golden-run: {name} r{repeat} -> {out_dir}", flush=True)
    started = time.time()
    code, problem = None, None
    with open(out_dir / 'log', 'w') as log, PowerSampler(state['gpus']) as sampler:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=state['timeout'])
        except subprocess.TimeoutExpired:
            problem = f"timed out after {state['timeout']} s"
        finally:
            removal = remove_container(record['container'])
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            if removal:
                problem = '; '.join(filter(None, [problem, removal]))
    if code == GPU_CHECK_EXIT:
        problem = 'GPU access check failed inside the container'
    record.update(started=started, seconds=round(time.time() - started, 1), exit=code if problem is None else
                  (code if code is not None else 124), problem=problem, power_after=governor_summary(),
                  power_samples=sampler.summary())
    (out_dir / 'exit').write_text(f"{record['exit']}\n")
    (out_dir / 'argv.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def key_lines(log, limit=16):
    try:
        text = ANSI.sub('', Path(log).read_text(errors='replace'))
    except OSError:
        return []
    lines = [line.strip() for line in text.splitlines() if KEY_LINE.search(line)]
    return [line[:400] for line in lines[-limit:]]


# --------------------------------------------------------------------------- comparison

def numpy_python():
    git_common = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                capture_output=True, text=True)
    candidates = [Path(sys.executable), REPO / '.venv/bin/python']
    if git_common.returncode == 0:
        candidates.append(Path(git_common.stdout.strip()).parent / '.venv/bin/python')
    for python in candidates:
        if python.exists() and subprocess.run([str(python), '-c', 'import numpy'], capture_output=True).returncode == 0:
            return [str(python)]
    return None


def logits_files(directory):
    return sorted(str(p.relative_to(directory)) for p in Path(directory).rglob('*logits*.bin') if p.is_file())


def compare_pairs(state, pairs, repeats):
    """Compare saved logits of arm pairs, per repeat, file by file."""
    results = []
    if not pairs:
        return results
    python = numpy_python()
    tool = REPO / 'scripts/bench/compare-logits.py'
    for a, b in pairs:
        for repeat in range(1, repeats + 1):
            da, db = Path(state['root']) / a / f'r{repeat}', Path(state['root']) / b / f'r{repeat}'
            files = [f for f in logits_files(da) if (db / f).is_file()]
            if not files:
                results.append(dict(a=a, b=b, repeat=repeat, file=None, error='no saved logits common to both arms'))
                continue
            for name in files:
                out = Path(state['root']) / f'compare-{a}-{b}-r{repeat}-{name.replace("/", "_")}.json'
                extra = []
                tokens = (da / name).parent / 'tokens.bin'
                if Path(name).name == 'logits.bin' and tokens.is_file():
                    extra = ['--tokens', str(tokens)]
                args = [str(da / name), str(db / name), str(state['vocab']), *extra, '--json', str(out)]
                if python:
                    cmd = [*python, str(tool), *args]
                else:  # the dev image has numpy; read-only views of the outputs
                    image = state['slots'][a]['image_id']
                    cmd = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
                           '-v', f'{tool}:{tool}:ro', '-v', f"{state['root']}:{state['root']}", '--entrypoint',
                           'python3', image, str(tool), *args]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if proc.returncode or not out.is_file():
                    results.append(dict(a=a, b=b, repeat=repeat, file=name,
                                        error=(proc.stderr.strip().splitlines() or ['compare failed'])[-1]))
                    continue
                measured = json.loads(out.read_text())
                results.append(dict(measured, a=a, b=b, repeat=repeat, file=name,
                                    path_a=measured['a'], path_b=measured['b']))
    return results


def write_summary(state, records, comparisons):
    lines = [f"# golden-run {state['task']}", '',
             f"family `{state['family']}`, checkpoint `{state['snapshot']}`, GPUs {state['gpus']}, locks "
             f"{','.join(state['locks'])}", '', f"golden args: `{shlex.join(state['golden'])}`", '',
             '| arm | repeat | exit | seconds | slot | seal | image | env | GPU limit W (mean) |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in records:
        env = ' '.join(f'{k}={v}' for k, v in r['arm_env'].items()) or '-'
        power = ', '.join(f"gpu{g} {p['mean_limit_w']:g}" for g, p in sorted(r['power_samples'].items())) or '-'
        exit_cell = str(r['exit']) + (f" ({r['problem']})" if r['problem'] else '')
        lines.append(f"| {r['arm']} | {r['repeat']} | {exit_cell} | {r['seconds']} | {r['slot']} | "
                     f"{r['seal_sha256'][:12]} | {r['image_id'][7:19]} | {env} | {power} |")
    for r in records:
        lines += ['', f"## {r['arm']} r{r['repeat']} (exit {r['exit']})", '', '```',
                  *(key_lines(Path(state['root']) / r['arm'] / f"r{r['repeat']}" / 'log') or ['(no key lines)']), '```']
    if comparisons:
        lines += ['', '## Logits comparisons', '', '| a | b | repeat | file | byte-identical | rows | max abs diff | mean KL | max KL | top-1 |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for c in comparisons:
            if c.get('error'):
                lines.append(f"| {c['a']} | {c['b']} | {c['repeat']} | {c['file'] or '-'} | error: {c['error']} | | | | | |")
                continue
            if not c['rows']:
                lines.append(f"| {c['a']} | {c['b']} | {c['repeat']} | {c['file']} | {c['byte_identical']} | 0 | | | | |")
                continue
            lines.append(f"| {c['a']} | {c['b']} | {c['repeat']} | {c['file']} | {c['byte_identical']} | {c['rows']} | "
                         f"{c['max_abs_diff']:.4e} | {c['mean_kl']:.6f} | {c['max_kl']:.6f} | {100 * c['top1_agreement']:.2f}% |")
    text = '\n'.join(lines) + '\n'
    (Path(state['root']) / 'summary.md').write_text(text)
    return text


# --------------------------------------------------------------------------- main

def build_state(args, arms):
    task_root = Path.home() / '.cache/cuteafd/builds' / args.task
    snapshot, container_path, extra = resolve_model(args.model)
    hub = hub_root()
    golden = golden_args(args.family, args.golden)
    covered = [hub, Path('/mnt/sparknest'), snapshot]
    state = dict(task=args.task, family=args.family, root=str(task_root / 'golden'), gpus=parse_gpus(args.gpus),
                 snapshot=str(snapshot), snapshot_container=container_path, hub=str(hub), timeout=args.timeout,
                 golden=golden, extra_mounts=extra + path_mounts(golden, covered + [Path(p) for p in extra]))
    state['locks'] = locks_for(state['gpus'])
    state['vocab'] = args.vocab or (vocab_size(snapshot) if args.compare_logits else None)
    return state


def parse_pairs(values, arms):
    pairs = []
    for value in values:
        a, sep, b = value.partition(':')
        if not sep or a not in arms or b not in arms or a == b:
            raise ValueError(f'--compare-logits {value!r} needs two different arms (A:B)')
        pairs.append((a, b))
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True, help='output under ~/.cache/cuteafd/builds/TASK/golden')
    parser.add_argument('--family', required=True, choices=sorted(SUBCOMMANDS))
    parser.add_argument('--model', required=True, help='HF id (ORG/NAME[@REV]) or snapshot directory; {snapshot} in ARGS')
    parser.add_argument('--arm', action='append', default=[], metavar='NAME=WIP_INSTANCE:SLOT')
    parser.add_argument('--arm-env', action='append', default=[], metavar='ARM:KEY=VALUE')
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE', help='environment for every arm')
    parser.add_argument('--gpus', default='0,1', help='raptor GPUs, locked gpu0 then gpu1 (default 0,1)')
    parser.add_argument('--sparks', help=argparse.SUPPRESS)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--no-interleave', action='store_true', help='keep arm order on every repeat')
    parser.add_argument('--timeout', type=int, default=3600, help='seconds per arm (default 3600)')
    parser.add_argument('--wait', type=float, default=float(os.environ.get('CUTEAFD_HW_LOCK_WAIT_S', 21600)))
    parser.add_argument('--preflight', default='', help='shell check before enqueue and at grant (as hwlock)')
    parser.add_argument('--preflight-timeout', type=float, default=300)
    parser.add_argument('--compare-logits', action='append', default=[], metavar='A:B')
    parser.add_argument('--vocab', type=int, help='logits row width (default: the checkpoint config.json vocab_size)')
    parser.add_argument('--overwrite', action='store_true', help='replace an earlier golden/ of this task')
    parser.add_argument('--dry-run', action='store_true', help='run the checks, print the commands, take no locks')
    parser.add_argument('--preflight-only', action='store_true', help='run the CPU checks and exit (0 = pass)')
    parser.add_argument('golden', nargs=argparse.REMAINDER, help='-- GOLDEN ARGS (verbatim, after the subcommand)')
    args = parser.parse_args(argv)
    if args.golden[:1] == ['--']:
        args.golden = args.golden[1:]
    try:
        if args.sparks:
            raise ValueError('--sparks: real-expert goldens are not supported yet; run coordinator-only '
                             '(--skip-routed-experts / --skip-experts)')
        if not NAME.fullmatch(args.task) or args.repeats < 1 or args.timeout < 1:
            raise ValueError('invalid --task, --repeats or --timeout')
        if not args.golden:
            raise ValueError('missing golden args after --')
        arms = parse_arms(args.arm)
        if not arms:
            raise ValueError('at least one --arm is required')
        apply_env(arms, args.env, args.arm_env)
        pairs = parse_pairs(args.compare_logits, arms)
        state = build_state(args, arms)
        for name in arms:
            if len(container_name(args.task, name)) > 128:
                raise ValueError('task and arm names make a container name over 128 characters')
        state['slots'] = preflight(state, arms)
        state['serving_args'] = serving_args()
        hwlock.check_preflight(args.preflight, args.preflight_timeout)
    except (ValueError, OSError, subprocess.SubprocessError, hwlock.PreflightFailed) as error:
        print(f'golden-run: {error}', file=sys.stderr)
        return 78 if isinstance(error, hwlock.PreflightFailed) else 2
    order = list(schedule(arms, args.repeats, not args.no_interleave))
    if args.preflight_only or args.dry_run:
        if args.dry_run:
            for name, repeat in order:
                out = Path(state['root']) / name / f'r{repeat}'
                cmd, _ = docker_command(state, name, arms[name], state['slots'][name], out,
                                        substitute(state['golden'], state['snapshot_container']))
                print(f'# {name} r{repeat}\n{shlex.join(cmd)}')
        print('golden-run: preflight pass')
        return 0
    root = Path(state['root'])
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            print(f'golden-run: {root} exists; pass --overwrite or use another --task', file=sys.stderr)
            return 2
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'plan.json').write_text(json.dumps(dict(state, order=order, arms=arms), indent=2, default=str) + '\n')

    held = set(filter(None, os.environ.get('CUTEAFD_HW_LOCKS_HELD', '').split(',')))
    records = []
    previous = {signum: signal.signal(signum, _interrupt) for signum in (signal.SIGTERM, signal.SIGHUP)}

    def body():
        # At grant: the slots are still what was checked at enqueue.
        state['slots'] = preflight(state, arms)
        print(f"golden-run: locks {','.join(state['locks'])} held; {len(order)} runs", flush=True)
        for name, repeat in order:
            records.append(run_arm(state, name, repeat, arms))

    code = 0
    try:
        if set(state['locks']) <= held:
            body()  # an enclosing hwlock already holds our locks
        else:
            with hwlock.hold(state['locks'], args.wait, f'golden-run {args.task}', preflight=args.preflight,
                             preflight_timeout=args.preflight_timeout):
                body()
    except hwlock.PreflightFailed as error:
        print(f'golden-run: {error}', file=sys.stderr)
        code = 78
    except TimeoutError as error:
        print(f'golden-run: {error}', file=sys.stderr)
        code = 75
    except (Interrupted, KeyboardInterrupt) as error:
        print(f'golden-run: interrupted ({error}); containers removed, locks released', file=sys.stderr)
        code = 130
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'golden-run: {error}', file=sys.stderr)
        code = 2
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    comparisons = compare_pairs(state, pairs, args.repeats) if records and len(records) == len(order) else []
    print(write_summary(state, records, comparisons))
    if code:
        return code
    if any(r['exit'] != 0 for r in records) or any(c.get('error') for c in comparisons):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
