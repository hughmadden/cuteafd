#!/usr/bin/env python3
"""Carry a branch that forked before the repo layout move (M1-M8) and the
naming pass (N1-N9) across them.

  rebase-across-move.sh [--onto REV] [--pre REV] [--squash] [--dry-run] BRANCH
  rebase-across-move.sh --continue

The branch's changes are replayed as three-way merges, file by file, on a new
branch BRANCH-moved at --onto (default HEAD), which is checked out:

  base   = the file at the commit the change was made on (the last pre-move
           commit or the branch's own parent commit)
  theirs = the file after the change
  ours   = the file at its new path in the checkout

base and theirs first get the same rewrite the move and the naming pass gave
the tree (scripts/build/path-map.tsv for paths, including repo paths written
inside files; scripts/build/rename-map.tsv for Rust paths), so the merge sees
only the branch's own edits. New files land where the map puts them (a
fallback rule places new files in split directories; relocated scripts get
the move's depth edits) and are listed for review.

BRANCH is not rewritten. If it does not contain the last pre-move commit
(--pre, default: the parent of "M1: move daemon modules" in --onto), a copy
(premove/BRANCH) is merged with it (branches with merge commits, or --squash)
or rebased onto it in a scratch worktree first; resolve any conflict there and
rerun with BRANCH=premove/BRANCH. Branches with merges are replayed as one
squashed commit; others commit by commit, keeping authors and messages. On a
merge conflict the markers are left in the checkout: resolve, commit, then run
--continue for the remaining commits.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rename_paths  # noqa: E402  (read now: the checkout changes branches later)

REPO = HERE.parents[1]
SPLIT_ROOTS = {"scripts", "python/tools", "python/reference", "rust/crates/cuteafd-daemon/src",
               "rust/crates/cuteafd-ffi/src", "rust/crates/cuteafd-loader/src"}


def git(*args: str, check: bool = True, input: bytes | None = None) -> str:
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, input=input)
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr.decode(errors='replace')}")
    return result.stdout.decode(errors="surrogateescape")


def blob(rev: str, path: str) -> bytes | None:
    result = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True)
    return result.stdout if result.returncode == 0 else None


class Map:
    def __init__(self, text: str, onto_files: set[str]):
        self.exact: dict[str, str] = {}
        self.dirs: list[tuple[str, str]] = []
        self.globs: list[tuple[re.Pattern[str], str]] = []
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            old, new = line.split("\t")[:2]
            if "*" in old:
                rx = "^" + "".join("([^/]*)" if c == "*" else re.escape(c) for c in old) + "$"
                self.globs.append((re.compile(rx), new))
            elif old.endswith("/"):
                self.dirs.append((old, new))
            else:
                self.exact[old] = new
        self.dirs.sort(key=lambda d: -len(d[0]))
        self.onto = onto_files
        self.added: dict[str, str] = {}
        self.review: set[str] = set()

    def path(self, path: str, new_file: bool = False) -> str:
        if path in self.exact:
            return self.exact[path]
        if path in self.added:
            return self.added[path]
        if path in self.onto and not new_file:
            return path
        for old, new in self.dirs:
            if path.startswith(old):
                return new + path[len(old):]
        if new_file:
            for rx, new in self.globs:
                m = rx.match(path)
                if m:
                    groups = list(m.groups())
                    out = re.sub(r"\*", lambda _: groups.pop(0), new)
                    self.review.add(f"{path} -> {out}")
                    return out
        return path

    def ref(self, m: re.Match[str]) -> str:
        p = m.group(1)
        if p in self.exact:
            return self.exact[p]
        if p in self.added:
            return self.added[p]
        for old, new in self.dirs:
            if p.startswith(old) and p not in self.onto:
                return new + p[len(old):]
        return p

    def moved(self, name: str, root: str) -> str | None:
        full = f"{root}/{name}"
        dest = self.exact.get(full) or self.added.get(full)
        return dest[len(root) + 1:] if dest and dest.startswith(root + "/") else None


REF = re.compile(r"(?<![\w.-])((?:rust/crates|native|python|scripts)/[A-Za-z0-9_./-]*[A-Za-z0-9_])")
TOOL_DIRS = ('("lib", "aot", "bench", "hf", "qualify/deepseek_v4", "qualify/deepseek_v41", '
             '"qualify/glm5_flash")')


def rewrite(text: str, path: str, pmap: Map, rules, relocated_from: str | None = None) -> str:
    """The move's and naming pass's edits for a file now at `path`."""
    text = REF.sub(pmap.ref, text)
    if path.startswith("scripts/tests/"):
        def up1(m):
            sub = pmap.moved(m.group(3), "scripts")
            return f"{m.group(1)}{m.group(2)}{sub}{m.group(2)}" if sub else m.group(0)
        text = re.sub(r"""(parents\[1\]\s*/\s*)(["'])([A-Za-z0-9_.-]+)\2""", up1, text)
    if path.startswith(("scripts/tests/", "python/tests/")):
        def tools(m):
            sub = pmap.moved(m.group(2), "python/tools")
            return ("TOOLS / " + " / ".join(m.group(1) + p + m.group(1) for p in sub.split("/"))) if sub else m.group(0)
        text = re.sub(r"""TOOLS / (["'])([A-Za-z0-9_]+\.(?:py|sh))\1""", tools, text)
        text = text.replace("sys.path.insert(0, str(TOOLS))", f"sys.path[:0] = [str(TOOLS / d) for d in {TOOL_DIRS}]")
    if path.endswith(".rs"):
        text = rename_paths.rewrite(path, text, rules)
    if relocated_from:
        text = relocate(text, relocated_from, path, pmap)
    return text


def relocate(text: str, src: str, dest: str, pmap: Map) -> str:
    """The depth edits M7/M8 gave the files around a new file a fallback moved down."""
    for root in ("scripts", "python/tools"):
        if os.path.dirname(src) == root and dest.startswith(root + "/"):
            break
    else:
        return text
    depth = dest[len(root) + 1:].count("/")
    if depth == 0:
        return text
    base = f"Path(__file__).resolve().parents[{depth}]"
    text = re.sub(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d)\]",
                  lambda m: f"Path(__file__).resolve().parents[{int(m.group(1)) + depth}]", text)

    def sibling(m):
        q, name = m.group(1), m.group(2)
        if name == "fixtures" and root == "scripts":
            return f"{base} / {q}fixtures{q}"
        sub = pmap.moved(name, root)
        return f"{base} / {q}{sub}{q}" if sub else m.group(0)
    text = re.sub(r"""Path\(__file__\)(?:\.resolve\(\))?\.with_name\((["'])([A-Za-z0-9_.-]+)\1\)""", sibling, text)
    return re.sub(r'\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/\.\.((?:/\.\.)*)',
                  lambda m: '$(dirname "${BASH_SOURCE[0]}")' + "/.." * (1 + m.group(1).count("/..") + depth), text)


def decode(data: bytes | None) -> str | None:
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def merge_file(ours: str, base: str, theirs: str, label: str) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, text in (("ours", ours), ("base", base), ("theirs", theirs)):
            path = Path(tmp) / name
            path.write_text(text, encoding="utf-8", errors="surrogateescape")
            files.append(str(path))
        result = subprocess.run(["git", "merge-file", "-p", "-L", f"{label} (moved tree)", "-L", "base",
                                 "-L", f"{label} (branch)", *files], capture_output=True)
        return result.stdout.decode("utf-8", errors="surrogateescape"), result.returncode != 0


def apply_change(base_rev: str, theirs_rev: str, pmap: Map, rules) -> tuple[list[str], list[str]]:
    """Replay base_rev..theirs_rev into the working tree. Returns (touched, conflicts)."""
    raw = git("diff", "--name-status", "--no-renames", base_rev, theirs_rev)
    changes = [line.split("\t") for line in raw.splitlines() if line]
    for status, path in changes:  # place new files first so references to them follow
        if status == "A":
            dest = pmap.path(path, new_file=True)
            if dest != path:
                pmap.added[path] = dest
    touched, conflicts = [], []
    for status, path in changes:
        if status == "A":
            dest = pmap.added.get(path, path)
            if dest == path and os.path.dirname(path) in SPLIT_ROOTS:
                pmap.review.add(f"{path} (left at the top of a split directory)")
            data = blob(theirs_rev, path)
            text = decode(data)
            out = REPO / dest
            out.parent.mkdir(parents=True, exist_ok=True)
            if text is None:
                out.write_bytes(data or b"")
            else:
                out.write_text(rewrite(text, dest, pmap, rules, relocated_from=path if dest != path else None),
                               encoding="utf-8", errors="surrogateescape")
            touched.append(dest)
            continue
        dest = pmap.path(path)
        target = REPO / dest
        base_text, theirs_text = decode(blob(base_rev, path)), decode(blob(theirs_rev, path))
        if status == "D":
            if target.exists():
                current = target.read_text(encoding="utf-8", errors="surrogateescape")
                if base_text is not None and current == rewrite(base_text, dest, pmap, rules):
                    target.unlink()
                    touched.append(dest)
                else:
                    conflicts.append(f"{dest}: deleted on the branch, changed by the move (kept)")
            continue
        if not target.exists():
            conflicts.append(f"{dest}: changed on the branch ({path}) but not in the moved tree")
            continue
        if base_text is None or theirs_text is None:
            target.write_bytes(blob(theirs_rev, path) or b"")
            touched.append(dest)
            continue
        ours = target.read_text(encoding="utf-8", errors="surrogateescape")
        merged, conflicted = merge_file(ours, rewrite(base_text, dest, pmap, rules),
                                        rewrite(theirs_text, dest, pmap, rules), dest)
        target.write_text(merged, encoding="utf-8", errors="surrogateescape")
        touched.append(dest)
        if conflicted:
            conflicts.append(f"{dest}: conflict markers")
    return touched, conflicts


def state_file() -> Path:
    return Path(git("rev-parse", "--absolute-git-dir").strip()) / "rebase-across-move.json"


def run_steps(steps: list[dict], pmap: Map, rules, squash_message: str | None) -> int:
    while steps:
        step = steps[0]
        touched, conflicts = apply_change(step["base"], step["theirs"], pmap, rules)
        if touched:
            git("add", "-A", "--", *touched)
        if conflicts:
            state_file().write_text(json.dumps({"steps": steps[1:], "added": pmap.added}))
            print("== conflicts (resolve, git add, commit" + (", then --continue" if len(steps) > 1 else "") + "):")
            for c in conflicts:
                print("   " + c)
            if not squash_message:
                print(f"   commit message: git commit -C {step['theirs']}")
            return 1
        if squash_message:
            git("commit", "-q", "-m", squash_message)
        else:
            git("commit", "-q", "--allow-empty", "-C", step["theirs"])
        steps.pop(0)
    state_file().unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("branch", nargs="?")
    parser.add_argument("--onto", default="HEAD")
    parser.add_argument("--pre")
    parser.add_argument("--squash", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue", dest="resume", action="store_true")
    args = parser.parse_args()
    map_text = (HERE / "path-map.tsv").read_text(encoding="utf-8")
    rules = rename_paths.load(HERE / "rename-map.tsv")

    if args.resume:
        state = json.loads(state_file().read_text())
        pmap = Map(map_text, set(git("ls-tree", "-r", "--name-only", "HEAD").split()))
        pmap.added.update(state["added"])
        return run_steps(state["steps"], pmap, rules, None)

    if not args.branch:
        parser.error("BRANCH is required")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("working tree is not clean")
    branch = args.branch
    onto = git("rev-parse", "--verify", f"{args.onto}^{{commit}}").strip()
    m1 = git("log", "--format=%H", "--grep=^M1: move daemon modules", onto).split()
    m1 = m1[-1] if m1 else ""
    pre = args.pre or (git("rev-parse", f"{m1}^").strip() if m1 else "")
    if not pre:
        raise SystemExit(f"cannot find the M1 move commit in {args.onto}; pass --pre")
    if m1 and subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", m1, branch]).returncode == 0:
        raise SystemExit(f"{branch} already contains the move; rebase it normally")
    fork = git("merge-base", branch, pre).strip()
    squash = args.squash or bool(git("rev-list", "--merges", f"{fork}..{branch}").strip())
    if squash and not args.squash:
        print(f"== {branch} contains merge commits: replaying its net change as one commit")

    source = branch
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", pre, branch]).returncode != 0:
        source = "premove/" + branch.removeprefix("premove/")
        if args.dry_run:
            print(f"(dry run: would bring {source} up to {pre[:12]} first)")
            return 0
        scratch = tempfile.mkdtemp(prefix="rebase-across-move-")
        git("worktree", "add", "-q", "-B", source, f"{scratch}/wt", branch)
        wt = f"{scratch}/wt"
        if squash:
            print(f"== merging the last pre-move commit {pre[:12]} into {source}")
            step = ["git", "-C", wt, "merge", "-q", "--no-edit", "-m", f"Merge the last pre-move commit into {branch}", pre]
            how = "commit"
        else:
            print(f"== rebasing {source} onto the last pre-move commit {pre[:12]}")
            step = ["git", "-C", wt, "rebase", "-q", pre]
            how = "git rebase --continue"
        if subprocess.run(step).returncode != 0:
            print(f"resolve the conflicts in {wt}, {how} there, then rerun with BRANCH={source}", file=sys.stderr)
            return 1
        git("worktree", "remove", "--force", wt)

    if squash:
        steps = [{"base": pre, "theirs": git("rev-parse", source).strip()}]
        log = git("log", "--format=- %h %s", "--no-merges", f"{fork}..{branch}")
        message = f"{branch.removeprefix('premove/')} across the repo layout move\n\nSquashed from:\n{log}"
    else:
        commits = git("rev-list", "--reverse", f"{pre}..{source}").split()
        steps = [{"base": f"{c}^", "theirs": c} for c in commits]
        message = None
    print(f"== {len(git('rev-list', '--no-merges', f'{fork}..{branch}').split())} commit(s) of {branch}")
    if args.dry_run:
        pmap = Map(map_text, set(git("ls-tree", "-r", "--name-only", onto).split()))
        for step in steps:
            for line in git("diff", "--name-status", "--no-renames", step["base"], step["theirs"]).splitlines():
                status, path = line.split("\t")
                print(f"   {status} {path} -> {pmap.path(path, new_file=status == 'A')}")
        return 0
    target = branch.removeprefix("premove/") + "-moved"
    if git("rev-parse", "--verify", "-q", f"refs/heads/{target}", check=False).strip():
        raise SystemExit(f"branch {target} already exists")
    git("switch", "-q", "-c", target, onto)
    print(f"== replaying onto {onto[:12]} as {target}")
    pmap = Map(map_text, set(git("ls-tree", "-r", "--name-only", onto).split()))
    rc = run_steps(steps, pmap, rules, message)
    if pmap.review:
        print("== new files to check (placed by a fallback, or left at the top of a split directory):")
        for r in sorted(pmap.review):
            print("   " + r)
    if rc == 0:
        print(f"== done: {target} = {git('rev-parse', '--short', 'HEAD').strip()}; "
              "check mod declarations for new Rust files, then build")
    return rc


if __name__ == "__main__":
    sys.exit(main())
