#!/usr/bin/env python3
"""Admit an existing coordinator dev image using its build and live provenance."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid


class VerificationError(RuntimeError):
    pass


def run(argv, timeout=60):
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=10)
        raise
    if process.returncode:
        raise VerificationError(f"{' '.join(argv[:5])}: {stderr.strip() or stdout.strip()}")
    return stdout


def digest(data):
    return hashlib.sha256(data).hexdigest()


def provenance(image_id):
    # Older dev images do not embed Dockerfile hashes. Their BuildKit attestation
    # supplies the exact Dockerfile bytes and source revision; never guess from a tag.
    for line in run(['docker', 'buildx', 'history', 'ls', '--format', 'json']).splitlines():
        row = json.loads(line)
        if row['status'].lower() != 'completed':
            continue
        ref = row['ref'].rsplit('/', 1)[-1]
        history = json.loads(run(['docker', 'buildx', 'history', 'inspect', '--format', 'json', ref]))
        if not any(a['Digest'] == image_id for a in history.get('Attachments', [])):
            continue
        value = json.loads(run(['docker', 'buildx', 'history', 'inspect', 'attachment', '--type',
                               'https://slsa.dev/provenance/v0.2', ref]))
        metadata = value['metadata']['https://mobyproject.org/buildkit@v1#metadata']
        infos = metadata['source']['infos']
        documents = [base64.b64decode(info['data'], validate=True) for info in infos if info['filename'] == 'Dockerfile.dev']
        if len(documents) != 1:
            raise VerificationError('dev image attestation lacks a unique Dockerfile.dev')
        revision = metadata['vcs']['revision']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise VerificationError('dev image attestation lacks a clean source revision')
        return ref, revision, documents[0]
    raise VerificationError('dev image has no retained image-ID-bound BuildKit provenance')


PROBE = r'''
import hashlib, importlib.util, json, os, pathlib, subprocess
s = pathlib.Path('/checkout')
def require(ok, message):
    if not ok: raise RuntimeError(message)
def call(argv): return subprocess.check_output(argv, text=True).strip()
expected = json.loads(pathlib.Path('/expected.json').read_text())
require(call(['rustc', '--version']).split()[1] == expected['toolchain'], 'toolchain mismatch: rustc')
require(call(['cargo', '--version']).split()[1] == expected['toolchain'], 'toolchain mismatch: cargo')
require('rustfmt-' in call(['rustup', 'component', 'list', '--installed']), 'toolchain mismatch: rustfmt missing')
require(os.environ.get('CUTEAFD_SPARKINFER_COMMIT') == expected['sparkinfer_revision'], 'SparkInfer revision mismatch')
require(hashlib.sha256(pathlib.Path('/usr/local/bin/cuteafd-entrypoint').read_bytes()).hexdigest() == expected['entrypoint_sha256'], 'entrypoint mismatch')
require(pathlib.Path('/opt/cuteafd/third_party/sparkinfer.lock.json').read_bytes() == (s/'third_party/sparkinfer.lock.json').read_bytes(), 'SparkInfer lock mismatch')
call(['python3', str(s/'scripts/build/verify-sparkinfer-source.py'), '--source', '/opt/cuteafd/third_party/sparkinfer', '--lock', str(s/'third_party/sparkinfer.lock.json'), '--require-no-python-cache'])
# Worktree gitfiles refer outside the read-only mount. The host checks git identity;
# inside the image, the same official verifier computes the mounted source digest.
spec = importlib.util.spec_from_file_location('verify_transformers', s/'scripts/build/verify-transformers-source.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
lock = json.loads((s/'third_party/transformers.lock.json').read_text())
require(m.source_tree_sha256(s/'third_party/transformers') == lock['source_tree_sha256'], 'Transformers source-tree mismatch')
print(json.dumps(dict(passed=True, toolchain=expected['toolchain'], sparkinfer_revision=expected['sparkinfer_revision'], transformers_revision=lock['revision'])))
'''


def verify(source, image, output):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise VerificationError('CUTEAFD_RELEASE_DEV_IMAGE must be a full sha256 image ID')
    obj = json.loads(run(['docker', 'image', 'inspect', image]))[0]
    if obj['Id'] != image or obj['Architecture'] != 'amd64':
        raise VerificationError('coordinator dev image ID/architecture mismatch')
    env = dict(value.split('=', 1) for value in obj['Config']['Env'])
    if (env.get('CUTEAFD_ROLE') != 'coordinator' or env.get('CUTEAFD_CUDA_ARCH') != '120'
            or env.get('CUTEAFD_TARGET_PLATFORM') != 'linux/amd64'
            or obj['Config']['Entrypoint'] != ['/usr/local/bin/cuteafd-entrypoint']):
        raise VerificationError('coordinator dev image role/architecture/entrypoint mismatch')
    ref, revision, dockerfile = provenance(image)
    current = (source / 'docker/Dockerfile.dev').read_bytes()
    if dockerfile != current:
        raise VerificationError('Dockerfile.dev hash mismatch')
    transformers = (source / 'third_party/transformers.lock.json').read_bytes()
    historical = run(['git', '-C', str(source), 'show', f'{revision}:third_party/transformers.lock.json']).encode()
    if historical != transformers:
        raise VerificationError('Transformers lock mismatch against dev image build revision')
    run(['python3', str(source / 'scripts/build/verify-transformers-source.py'), '--source',
         str(source / 'third_party/transformers'), '--lock', str(source / 'third_party/transformers.lock.json')], timeout=180)
    matches = re.findall(rb'^ARG RUST_TOOLCHAIN=([0-9]+\.[0-9]+\.[0-9]+)$', current, re.MULTILINE)
    if len(matches) != 1:
        raise VerificationError('Dockerfile.dev must pin one Rust toolchain version')
    expected = dict(toolchain=matches[0].decode(),
                    sparkinfer_revision=json.loads((source / 'third_party/sparkinfer.lock.json').read_bytes())['revision'],
                    entrypoint_sha256=digest((source / 'docker/entrypoint.sh').read_bytes()))
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_path = output.with_suffix('.expected.json')
    with expected_path.open('x') as out:
        json.dump(expected, out)
    name = 'cuteafd-dev-image-check-' + uuid.uuid4().hex
    collision = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=30)
    if collision.returncode == 0 or 'no such' not in collision.stderr.lower():
        raise VerificationError(f'dev image verification container name unavailable: {name}')
    probe = None
    try:
        probe = json.loads(run(['docker', 'run', '--rm', '--name', name, '--runtime', 'runc', '--network', 'none',
            '-e', 'NVIDIA_VISIBLE_DEVICES=void', '--entrypoint', 'python3',
            '-v', f'{source}:/checkout:ro', '-v', f'{expected_path}:/expected.json:ro', image, '-c', PROBE], timeout=240))
    finally:
        found = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=30)
        if found.returncode == 0:
            container = json.loads(found.stdout)[0]
            if (container['Image'] != image or container['Config']['Cmd'] != ['-c', PROBE]
                    or container['Config']['Entrypoint'] != ['python3']
                    or container['HostConfig']['Runtime'] != 'runc'
                    or container['HostConfig']['NetworkMode'] != 'none'
                    or 'NVIDIA_VISIBLE_DEVICES=void' not in container['Config']['Env']):
                raise VerificationError('dev image verification container replaced; refusing cleanup')
            run(['docker', 'rm', '-f', container['Id']])
        absent = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=30)
        if absent.returncode == 0 or 'no such' not in absent.stderr.lower():
            raise VerificationError('cannot prove dev image verification container absent')
    if not probe or not probe['passed']:
        raise VerificationError('dev image verification probe failed')
    sparkinfer = json.loads((source / 'third_party/sparkinfer.lock.json').read_bytes())
    manifest = dict(schema='cuteafd.release-dev-image-reuse/1', image_id=image, buildkit_ref=ref,
        dev_source_revision=revision, dockerfile_sha256=digest(current),
        entrypoint_sha256=expected['entrypoint_sha256'], transformers_lock_sha256=digest(transformers),
        transformers_source_tree_sha256=json.loads(transformers)['source_tree_sha256'],
        transformers_source='verified read-only checkout mount (not installed in dev image)',
        sparkinfer_source_tree_sha256=sparkinfer['source_tree_sha256'], **probe)
    with output.open('x') as out:
        json.dump(manifest, out, indent=2); out.write('\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    def interrupted(signum, _frame):
        # Ignore repeated signals while exact-owned container teardown completes.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise VerificationError(f'dev image verification interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    verify(a.source.resolve(), a.image, a.output)
    print(a.image)


if __name__ == '__main__':
    try:
        main()
    except (VerificationError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'release dev image reuse refused: {error}', file=sys.stderr)
        raise SystemExit(1)
