import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'dev' / 'prune-worktrees'


def git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def trees(tmp_path):
    root = tmp_path / 'main'
    root.mkdir()
    git(root, 'init', '-b', 'work/p0')
    git(root, 'config', 'user.email', 'test@example.invalid')
    git(root, 'config', 'user.name', 'Test')
    (root / 'source').write_text('base\n')
    git(root, 'add', 'source')
    git(root, 'commit', '-m', 'Base')
    remote = tmp_path / 'origin.git'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    git(root, 'remote', 'add', 'origin', str(remote))
    git(root, 'push', 'origin', 'work/p0')
    paths = {}
    for name in ('merged', 'unmerged', 'dirty', 'live', 'pid', 'deleted', 'target', 'locked', 'never-pushed'):
        path = tmp_path / name
        git(root, 'worktree', 'add', '-b', 'work/' + name, str(path))
        paths[name] = path
        if name in ('unmerged', 'deleted', 'never-pushed'):
            (path / 'source').write_text(name)
            git(path, 'add', 'source')
            git(path, 'commit', '-m', name)
        if name != 'never-pushed':
            git(root, 'push', '-u', 'origin', 'work/' + name)
    git(root, 'push', 'origin', '--delete', 'work/deleted')
    (paths['dirty'] / 'unknown').write_text('preserve me')
    (paths['target'] / 'target-debug').mkdir()
    (paths['target'] / 'target-debug/artifact').write_text('output')
    git(root, 'worktree', 'lock', str(paths['locked']))
    detached = tmp_path / 'detached'
    git(root, 'worktree', 'add', '--detach', str(detached))
    paths['detached'] = detached
    for path in paths.values():
        for item in path.rglob('*'):
            if item.is_file():
                os.utime(item, (time.time() - 3600, time.time() - 3600))
    recent = tmp_path / 'recent'
    git(root, 'worktree', 'add', '-b', 'work/recent', str(recent))
    paths['recent'] = recent
    home = tmp_path / 'home'
    pid_dir = home / '.cache/cuteafd/builds/pid'
    pid_dir.mkdir(parents=True)
    (pid_dir / 'worker.pid').write_text(str(os.getpid()))
    orphan = home / 'Developer/cuteafd-orphan'
    orphan.mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], cwd=paths['live'])
    try:
        yield root, paths, home, orphan
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def run(trees, *args):
    root, _, home, _ = trees
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=root,
                          env={**os.environ, 'HOME': str(home)}, capture_output=True, text=True)


def test_dry_run_keeps_dirty_live_unmerged_and_reports_orphans(trees):
    result = run(trees)
    assert result.returncode == 0, result.stderr
    removing, keeping = result.stdout.split('keep (review)')
    for name in ('merged', 'deleted', 'target', 'detached'):
        assert str(trees[1][name]) in removing
    for name in ('dirty', 'live', 'pid', 'unmerged', 'locked', 'never-pushed', 'recent'):
        assert str(trees[1][name]) in keeping
        assert str(trees[1][name]) not in removing
    assert str(trees[3]) in keeping
    assert all(path.exists() for path in trees[1].values())


def test_apply_uses_git_removal_and_preserves_review_and_orphan_dirs(trees):
    result = run(trees, '--apply')
    assert result.returncode == 0, result.stderr
    for name in ('merged', 'deleted', 'target', 'detached'):
        assert not trees[1][name].exists()
    for name in ('dirty', 'live', 'pid', 'unmerged', 'locked', 'never-pushed', 'recent'):
        assert trees[1][name].exists()
    assert trees[0].exists() and trees[3].exists()
