from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('dev_reuse', ROOT / 'scripts/build/verify-release-dev-image.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
IMAGE = 'sha256:' + 'a' * 64
REVISION = 'b' * 40


class DevImageReuseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'src'
        for directory in ('docker', 'third_party', 'scripts/build'):
            (self.source / directory).mkdir(parents=True, exist_ok=True)
        self.dockerfile = b'ARG RUST_TOOLCHAIN=1.98.1\n'
        (self.source / 'docker/Dockerfile.dev').write_bytes(self.dockerfile)
        (self.source / 'docker/entrypoint.sh').write_bytes(b'entrypoint')
        self.lock = {'revision': REVISION, 'source_tree_sha256': 'c' * 64}
        (self.source / 'third_party/transformers.lock.json').write_text(json.dumps(self.lock))
        (self.source / 'third_party/sparkinfer.lock.json').write_text(json.dumps(self.lock))
        self.output = Path(self.temp.name) / 'admission.json'
        self.image = {'Id': IMAGE, 'Architecture': 'amd64', 'Config': {
            'Env': ['CUTEAFD_ROLE=coordinator', 'CUTEAFD_CUDA_ARCH=120', 'CUTEAFD_TARGET_PLATFORM=linux/amd64'],
            'Entrypoint': ['/usr/local/bin/cuteafd-entrypoint']}}
        self.commands = []

    def fake_run(self, argv, timeout=60):
        self.commands.append(argv)
        if argv[:3] == ['docker', 'image', 'inspect']:
            return json.dumps([self.image])
        if argv[0] == 'git':
            return (self.source / 'third_party/transformers.lock.json').read_text()
        if argv[0] == 'python3':
            return 'verified Transformers'
        if argv[:2] == ['docker', 'run']:
            self.assertIn('runc', argv)
            self.assertIn('none', argv)
            self.assertIn('NVIDIA_VISIBLE_DEVICES=void', argv)
            return json.dumps({'passed': True, 'toolchain': '1.98.1', 'sparkinfer_revision': REVISION})
        self.fail(f'unexpected command: {argv}')

    def verify(self, dockerfile=None):
        absent = subprocess.CompletedProcess([], 1, '', 'Error: No such object')
        with patch.object(MOD, 'run', self.fake_run), patch.object(MOD, 'provenance', return_value=('ref', REVISION, dockerfile or self.dockerfile)), patch.object(MOD.subprocess, 'run', return_value=absent):
            return MOD.verify(self.source, IMAGE, self.output)

    def test_match_reuses_immutable_id_and_writes_manifest(self):
        manifest = self.verify()
        self.assertEqual(manifest['image_id'], IMAGE)
        self.assertEqual(json.loads(self.output.read_text()), manifest)
        self.assertEqual(len([a for a in self.commands if a[:2] == ['docker', 'run']]), 1)

    def test_bad_dockerfile_refuses_before_probe(self):
        with self.assertRaisesRegex(MOD.VerificationError, 'Dockerfile.dev hash mismatch'):
            self.verify(b'ARG RUST_TOOLCHAIN=1.97.0\n')
        self.assertFalse(self.output.exists())

    def test_historical_transformers_lock_mismatch_refuses(self):
        original = self.fake_run
        def command(argv, timeout=60):
            return '{}' if argv[0] == 'git' else original(argv, timeout)
        with patch.object(self, 'fake_run', command), self.assertRaisesRegex(MOD.VerificationError, 'Transformers lock mismatch'):
            self.verify()

    def test_tag_is_not_an_immutable_id(self):
        with self.assertRaisesRegex(MOD.VerificationError, 'full sha256 image ID'):
            MOD.verify(self.source, 'latest', self.output)

    def test_wrong_architecture_refuses(self):
        self.image['Architecture'] = 'arm64'
        with self.assertRaisesRegex(MOD.VerificationError, 'architecture mismatch'):
            self.verify()

    def test_missing_provenance_refuses(self):
        with patch.object(MOD, 'run', return_value=''):
            with self.assertRaisesRegex(MOD.VerificationError, 'no retained'):
                MOD.provenance(IMAGE)

    def test_provenance_is_bound_to_image_digest(self):
        document = {'metadata': {'https://mobyproject.org/buildkit@v1#metadata': {
            'vcs': {'revision': REVISION}, 'source': {'infos': [{'filename': 'Dockerfile.dev',
            'data': base64.b64encode(self.dockerfile).decode()}]}}}}
        replies = [json.dumps({'status': 'Completed', 'ref': 'default/default/ref'}),
                   json.dumps({'Attachments': [{'Digest': IMAGE}]}), json.dumps(document)]
        with patch.object(MOD, 'run', side_effect=replies):
            self.assertEqual(MOD.provenance(IMAGE), ('ref', REVISION, self.dockerfile))
        replies[1] = json.dumps({'Attachments': [{'Digest': 'sha256:' + 'd' * 64}]})
        with patch.object(MOD, 'run', side_effect=replies[:2]):
            with self.assertRaisesRegex(MOD.VerificationError, 'no retained'):
                MOD.provenance(IMAGE)

    def test_foreign_probe_container_is_never_removed(self):
        container = {'Image': 'foreign', 'Config': {'Cmd': ['sleep', 'infinity']}}
        inspections = [subprocess.CompletedProcess([], 1, '', 'No such object'),
                       subprocess.CompletedProcess([], 0, json.dumps([container]), '')]
        with patch.object(MOD, 'run', self.fake_run), patch.object(MOD, 'provenance', return_value=('ref', REVISION, self.dockerfile)), patch.object(MOD.subprocess, 'run', side_effect=inspections):
            with self.assertRaisesRegex(MOD.VerificationError, 'replaced; refusing cleanup'):
                MOD.verify(self.source, IMAGE, self.output)
        self.assertFalse(any(a[:2] == ['docker', 'rm'] for a in self.commands))

    def test_probe_timeout_still_proves_absence(self):
        original = self.fake_run
        def command(argv, timeout=60):
            if argv[:2] == ['docker', 'run']:
                raise subprocess.TimeoutExpired(argv, timeout)
            return original(argv, timeout)
        absent = subprocess.CompletedProcess([], 1, '', 'No such object')
        with patch.object(MOD, 'run', command), patch.object(MOD, 'provenance', return_value=('ref', REVISION, self.dockerfile)), patch.object(MOD.subprocess, 'run', return_value=absent) as inspect:
            with self.assertRaises(subprocess.TimeoutExpired):
                MOD.verify(self.source, IMAGE, self.output)
            self.assertEqual(inspect.call_count, 3)
        self.assertFalse(self.output.exists())

    def probe(self, failure=None):
        # Execute the real probe arithmetic/checks with CPU-only file/command fixtures.
        checkout = self.source
        installed = Path(self.temp.name) / 'installed'
        installed.mkdir()
        (installed / 'entrypoint').write_bytes(b'entrypoint' if failure != 'entrypoint' else b'wrong')
        (installed / 'sparkinfer.lock.json').write_text(json.dumps(self.lock) if failure != 'lock' else '{}')
        verifier = checkout / 'scripts/build/verify-transformers-source.py'
        verifier.write_text("def source_tree_sha256(source): return '" + 'c' * 64 + "'\n")
        expected = Path(self.temp.name) / 'expected.json'
        expected.write_text(json.dumps({'toolchain': '1.98.1', 'sparkinfer_revision': REVISION,
                                       'entrypoint_sha256': MOD.digest(b'entrypoint')}))
        code = MOD.PROBE.replace('/checkout', str(checkout)).replace('/expected.json', str(expected)).replace('/usr/local/bin/cuteafd-entrypoint', str(installed / 'entrypoint')).replace('/opt/cuteafd/third_party/sparkinfer.lock.json', str(installed / 'sparkinfer.lock.json'))
        def call(argv, **kwargs):
            if argv[0] in ('rustc', 'cargo'):
                version = '1.97.0' if failure == 'toolchain' else '1.98.1'
                return argv[0] + ' ' + version
            if argv[0] == 'rustup':
                return 'rustfmt-x86_64-unknown-linux-gnu'
            if failure == 'source':
                raise subprocess.CalledProcessError(1, argv, stderr='SparkInfer source mismatch')
            return 'verified SparkInfer'
        env = {'CUTEAFD_SPARKINFER_COMMIT': 'wrong' if failure == 'revision' else REVISION}
        with patch.dict(os.environ, env), patch.object(subprocess, 'check_output', call), contextlib.redirect_stdout(io.StringIO()):
            exec(code, {})

    def test_live_probe_matches(self):
        self.probe()

    def test_live_probe_toolchain_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, 'toolchain mismatch'):
            self.probe('toolchain')

    def test_live_probe_sparkinfer_revision_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, 'SparkInfer revision mismatch'):
            self.probe('revision')

    def test_live_probe_sparkinfer_source_mismatch(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.probe('source')

    def test_live_probe_sparkinfer_lock_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, 'SparkInfer lock mismatch'):
            self.probe('lock')

    def test_live_probe_entrypoint_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, 'entrypoint mismatch'):
            self.probe('entrypoint')

    def test_build_branch_reuses_only_after_verification_and_unset_builds(self):
        text = (ROOT / 'build.sh').read_text()
        branch = text[text.index('release_dev_reuse_label_args=()'):text.index('echo "== compiling coordinator')]
        for image, refuses in (('', False), (IMAGE, False), (IMAGE, True)):
            commands = []
            script = 'release_die() { printf "%s\\n" "$*" >&2; exit 2; }\n' + branch + '\nprintf "%s\\n" "$COORDINATOR_DOCKER_DEV"\n'
            with tempfile.TemporaryDirectory() as temp:
                bin_dir = Path(temp)
                for command in ('python3', 'docker'):
                    fake = bin_dir / command
                    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n' +
                                    ('if [[ "$1" == *verify-release-dev-image.py ]]; then [[ "$REFUSE" == 0 ]] || exit 1; printf "%s\\n" "$CUTEAFD_RELEASE_DEV_IMAGE"; fi\nexit 0\n' if command == 'python3' else 'exit 0\n'))
                    fake.chmod(0o755)
                log = bin_dir / 'commands'
                env = dict(os.environ, PATH=str(bin_dir) + ':' + os.environ['PATH'], COMMAND_LOG=str(log),
                           CUTEAFD_RELEASE_DEV_IMAGE=image, COORDINATOR_DOCKER_DEV='normal-dev', repo_root=str(ROOT),
                           release_build_root=str(bin_dir), release_dev_reuse_manifest=str(bin_dir / 'DEV_IMAGE_REUSE.json'),
                           sparkinfer_commit=REVISION, REFUSE=str(int(refuses)))
                result = subprocess.run(['bash', '-ec', script], capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 2 if refuses else 0, result.stderr)
                commands = log.read_text()
                self.assertEqual('verify-release-dev-image.py' in commands, bool(image))
                self.assertEqual('--build-arg' in commands, not bool(image))
                if refuses:
                    self.assertIn('reuse verification failed', result.stderr)
                else:
                    self.assertEqual(result.stdout.splitlines()[-1], image or 'normal-dev')


if __name__ == '__main__':
    unittest.main()
