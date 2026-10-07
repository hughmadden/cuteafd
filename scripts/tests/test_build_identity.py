"""CPU-only build-script identity and relocatable fixture regression tests."""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def scripts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("identity-scripts")
    result = {}
    for crate in ("cuteafd-bench", "cuteafd-daemon"):
        binary = directory / crate
        subprocess.run(["rustc", str(ROOT / "rust/crates" / crate / "build.rs"),
                        "--edition=2024", "-o", str(binary)], check=True)
        result[crate] = binary
    return result


def run_script(binary, tree):
    manifest = tree / "rust/crates" / binary.name
    manifest.mkdir(parents=True, exist_ok=True)
    output = tree / "out"
    output.mkdir(exist_ok=True)
    env = {**os.environ, "CARGO_MANIFEST_DIR": str(manifest), "OUT_DIR": str(output),
           "CUTEAFD_BUILD_COMMIT": ""}
    # Avoid inheriting an unrelated git worktree from the test runner.
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(key, None)
    completed = subprocess.run([str(binary)], env=env, text=True,
                               capture_output=True, check=True)
    commit = next(line.split("=", 2)[2] for line in completed.stdout.splitlines()
                  if line.startswith("cargo:rustc-env=CUTEAFD_BUILD_COMMIT="))
    return commit, output


@pytest.mark.parametrize("crate", ["cuteafd-bench", "cuteafd-daemon"])
def test_empty_identity_falls_back_to_git(scripts, tmp_path, crate):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-q", "--allow-empty",
                    "-m", "Fixture"], check=True)
    expected = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    (tmp_path / "BUILD_IDENTITY.json").write_text("")
    (tmp_path / ".cuteafd-source-revision").write_text("")
    assert run_script(scripts[crate], tmp_path)[0] == expected


@pytest.mark.parametrize("crate", ["cuteafd-bench", "cuteafd-daemon"])
def test_empty_frozen_identity_is_explicitly_unknown(scripts, tmp_path, crate):
    (tmp_path / "BUILD_IDENTITY.json").write_text("")
    (tmp_path / ".cuteafd-source-revision").write_text("\n")
    assert run_script(scripts[crate], tmp_path)[0] == "unknown"


def test_generated_fixture_has_no_checkout_path(scripts, tmp_path):
    outputs = []
    for name in ("first", "second"):
        tree = tmp_path / name
        fixture = tree / "scripts/fixtures/agentic-repo"
        fixture.mkdir(parents=True)
        (fixture / "sample.py").write_text('print("escaped \\ content")\n')
        _, output = run_script(scripts["cuteafd-bench"], tree)
        text = (output / "agentic_repo.rs").read_text()
        assert str(tree) not in text
        assert "include_str!" not in text
        outputs.append(text)
    assert outputs[0] == outputs[1]
