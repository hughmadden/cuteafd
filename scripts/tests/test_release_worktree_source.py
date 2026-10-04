"""Release compiler staging removes inaccessible worktree submodule metadata."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "scripts/build/stage-release-source.sh"


def test_stage_worktree_submodule_source(tmp_path):
    source = tmp_path / "worktree"
    nested = source / "third_party/xgrammar/3rdparty/dlpack"
    nested.mkdir(parents=True)
    (source / ".git").write_text("gitdir: /unmounted/main/.git/worktrees/task\n")
    for directory in (source / "third_party/xgrammar", nested):
        (directory / ".git").write_text("gitdir: /unmounted/submodule\n")
        (directory / "source.h").write_text("pinned source\n")
    cache = nested / "__pycache__"
    cache.mkdir()
    (cache / "old.pyc").write_bytes(b"stale")
    (source / "third_party/xgrammar.lock.json").write_text('{"source_tree_sha256":"pinned"}')
    stage = tmp_path / "staged"
    stage.mkdir()
    (stage / "old-source").write_text("remove me")
    result = subprocess.run([str(STAGE), str(source), str(stage)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not list(stage.rglob(".git"))
    assert not list(stage.rglob("*.pyc"))
    assert not (stage / "old-source").exists()
    for path in source.rglob("*"):
        if path.is_file() and path.name != ".git" and path.suffix != ".pyc":
            assert (stage / path.relative_to(source)).read_bytes() == path.read_bytes()
    build = (ROOT / "build.sh").read_text()
    assert '"$repo_root/scripts/build/stage-release-source.sh" "$repo_root" "$release_source_dir"' in build
    assert '-v "$release_source_dir:/source:ro"' in build


def test_stage_rejects_source_nested_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    result = subprocess.run([str(STAGE), str(source), str(source / "staged")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "outside the source tree" in result.stderr
    assert not (source / "staged").exists()
