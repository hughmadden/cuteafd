"""rebase-across-move.sh on a toy repository: a branch that forked before a
move lands on the moved tree with its edits, new files and path references
following the path map."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "build" / "rebase-across-move.sh"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts" / "build").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    shutil.copy(SCRIPT, repo / "scripts" / "build" / "rebase-across-move.sh")
    (repo / "scripts" / "old-tool.sh").write_text("echo one\n")
    (repo / "native" / "cuda" / "kernels").mkdir(parents=True)
    (repo / "native" / "cuda" / "kernels" / "v41_kv.cu").write_text("int a;\nint b;\n")
    (repo / "docs.txt").write_text("run scripts/old-tool.sh\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def test_branch_crosses_the_move(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    # A branch from before the move: edit a moved file, add a file beside it,
    # and reference a moved script by path.
    git(repo, "switch", "-q", "-c", "feature")
    kv = repo / "native" / "cuda" / "kernels" / "v41_kv.cu"
    kv.write_text("int a;\nint b;\nint c;\n")
    (repo / "native" / "cuda" / "kernels" / "v41_new.cu").write_text("int n;\n")
    (repo / "notes.txt").write_text("see scripts/old-tool.sh\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "feature work")
    # The move on main.
    git(repo, "switch", "-q", "main")
    (repo / "native" / "families" / "deepseek_v41" / "cuda").mkdir(parents=True)
    (repo / "scripts" / "launch").mkdir()
    git(repo, "mv", "native/cuda/kernels/v41_kv.cu", "native/families/deepseek_v41/cuda/v41_kv.cu")
    git(repo, "mv", "scripts/old-tool.sh", "scripts/launch/old-tool.sh")
    (repo / "docs.txt").write_text("run scripts/launch/old-tool.sh\n")
    (repo / "scripts" / "build" / "path-map.tsv").write_text(
        "# toy map\n"
        "native/cuda/kernels/v41_kv.cu\tnative/families/deepseek_v41/cuda/v41_kv.cu\n"
        "scripts/old-tool.sh\tscripts/launch/old-tool.sh\n"
        "native/cuda/kernels/v41_*\tnative/families/deepseek_v41/cuda/v41_*\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "M1: move daemon modules into shared/ and families/<id>/")

    result = subprocess.run(["bash", "scripts/build/rebase-across-move.sh", "feature"], cwd=repo,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native/cuda/kernels/v41_new.cu -> native/families/deepseek_v41/cuda/v41_new.cu" in result.stdout
    assert git(repo, "branch", "--show-current").strip() == "feature-moved"
    moved = repo / "native" / "families" / "deepseek_v41" / "cuda"
    assert (moved / "v41_kv.cu").read_text() == "int a;\nint b;\nint c;\n"
    assert (moved / "v41_new.cu").read_text() == "int n;\n"
    assert not [p for p in git(repo, "ls-files").split() if p.startswith("native/cuda/")]
    assert (repo / "notes.txt").read_text() == "see scripts/launch/old-tool.sh\n"
    # The original branch is untouched.
    assert "native/cuda/kernels/v41_kv.cu" in git(repo, "ls-tree", "-r", "--name-only", "feature")
