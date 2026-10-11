import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

DEV = Path(__file__).resolve().parents[1] / 'dev'
spec = importlib.util.spec_from_file_location('dev_common', DEV / '_common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
BODY = '\n'.join('## ' + section + '\nPending.\n' for section in common.SECTIONS)


def test_size_cap_is_utf8_bytes(tmp_path):
    common.validate_body(BODY, tmp_path)
    with pytest.raises(ValueError, match='4096'):
        common.validate_body(BODY + 'é' * 2048, tmp_path)


@pytest.mark.parametrize('section', common.SECTIONS)
def test_missing_section(tmp_path, section):
    with pytest.raises(ValueError, match='missing required'):
        common.validate_body(BODY.replace('## ' + section, 'Not a heading'), tmp_path)


def test_fenced_heading_does_not_count(tmp_path):
    with pytest.raises(ValueError, match='missing required'):
        common.validate_body('```md\n' + BODY + '\n```', tmp_path)


@pytest.mark.parametrize('secret', [
    'sk-example', 'ghp_example', 'github_pat_example', '-----BEGIN PRIVATE KEY-----',
    'AKIAEXAMPLE', 'xoxb-example', 'xoxp-example', 'Bearer example',
    'token: ' + 'a' * 40, 'secret = ' + 'QWxhZGRpbg' * 6,
    '"api_key": "' + 'a' * 48 + '"',
])
def test_secret_patterns_rejected(tmp_path, secret):
    with pytest.raises(ValueError, match='secret pattern'):
        common.validate_body(BODY + secret, tmp_path)
    with pytest.raises(ValueError):
        common.validate_public(secret, tmp_path)


@pytest.mark.parametrize('name', ['service-api-key.txt', 'api.key'])
def test_task_key_contents_are_rejected(tmp_path, name):
    nested = tmp_path / 'nested'
    nested.mkdir()
    (nested / name).write_text('unique-private-credential\n')
    with pytest.raises(ValueError, match='task API key'):
        common.validate_public('value unique-private-credential', tmp_path)


@pytest.fixture
def cli(tmp_path):
    gh = tmp_path / 'gh'
    gh.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ['repo', 'view']:
    print(json.dumps({'nameWithOwner': 'test/repo'}))
elif args[:2] == ['pr', 'list']:
    print(json.dumps([] if os.environ.get('NO_PR') else [{'number': 9, 'body': 'remote state', 'headRepositoryOwner': {'login': 'outside' if os.environ.get('FORK_PR') else 'test'}, 'headRepository': {'name': 'repo'}}]))
else:
    Path(os.environ['WRITE_LOG']).write_text(json.dumps({'args': args, 'body': sys.stdin.read()}))
    print('https://github.com/test/repo/pull/9')
''')
    gh.chmod(0o755)
    root = tmp_path / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-b', 'work/test-task', str(root)], check=True, capture_output=True)
    (root / '.github').mkdir()
    (root / '.github/pull_request_template.md').write_text(BODY)
    env = {**os.environ, 'HOME': str(tmp_path),
           'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'WRITE_LOG': str(tmp_path / 'write.json')}

    def call(*args, extra=None):
        return subprocess.run([sys.executable, str(DEV / 'pr-sync'), *args],
                              cwd=root, env={**env, **(extra or {})}, capture_output=True, text=True)
    return call, tmp_path


def test_pull_default_and_explicit_task_paths(cli):
    call, home = cli
    result = call('pull')
    assert result.returncode == 0, result.stderr
    assert (home / '.cache/cuteafd/builds/test-task/STATUS.md').read_text() == 'remote state'
    assert call('pull', '--task', 'nested/task').returncode == 0
    assert (home / '.cache/cuteafd/builds/nested/task/STATUS.md').exists()
    assert call('pull', '--task', '../escape').returncode == 1


def test_fork_pr_with_same_branch_name_is_not_current_branch_pr(cli):
    call, home = cli
    result = call('pull', extra={'FORK_PR': '1'})
    assert result.returncode == 1 and 'no open PR' in result.stderr
    assert not (home / '.cache/cuteafd/builds/test-task/STATUS.md').exists()


def test_push_and_comment_validate_before_write(cli):
    call, home = cli
    source = home / 'state.md'
    source.write_text(BODY)
    result = call('push', '--file', str(source))
    assert result.returncode == 0, result.stderr
    log = home / 'write.json'
    assert json.loads(log.read_text())['body'] == BODY
    log.unlink()
    source.write_text(BODY + 'ghp_example')
    assert call('push', '--file', str(source)).returncode == 1
    assert not log.exists()
    assert call('comment', '--body', 'Bearer private').returncode == 1
    assert not log.exists()
    assert call('comment', '--body', 'Verdict: green').returncode == 0
    assert 'comment' in json.loads(log.read_text())['args']


def test_init_creates_only_when_absent_and_honors_stack_base(cli):
    call, home = cli
    assert call('init', '--title', 'Feature').returncode == 0
    assert not (home / 'write.json').exists()
    result = call('init', '--title', 'Feature', '--base', 'work/parent', '--milestone', 'v3',
                  extra={'NO_PR': '1'})
    assert result.returncode == 0, result.stderr
    written = json.loads((home / 'write.json').read_text())
    assert '--draft' in written['args'] and 'work/parent' in written['args']
    assert written['body'] == BODY
