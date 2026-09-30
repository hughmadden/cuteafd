#!/usr/bin/env bash
# Carry a branch that forked before the repo layout move (M1-M8) across it.
#
#   scripts/build/rebase-across-move.sh [--onto REV] [--pre REV] [--map FILE]
#       [--squash] [--no-rewrite-refs] [--dry-run] BRANCH
#
# 1. A copy of BRANCH (premove/BRANCH) is brought up to the last pre-move
#    commit (--pre, default: the parent of the "M1: move daemon modules" commit
#    reachable from --onto) in a scratch worktree: rebased onto it, or, with
#    --squash (automatic when BRANCH contains merge commits), merged with it.
#    BRANCH and the current checkout are not touched. On a conflict the scratch
#    worktree is kept: resolve there and rerun with BRANCH=premove/BRANCH.
# 2. Its commits (with --squash, its whole diff against the pre-move commit)
#    are exported and every path goes through scripts/build/path-map.tsv: exact
#    moves, whole-directory moves, then fallbacks for new files in split
#    directories. Repo paths inside every hunk line follow the same map, as do
#    the joined forms the move rewrote in tests (parents[1] / "x.py" under
#    scripts/tests, TOOLS / "x.py" and the TOOLS sys.path entry) unless
#    --no-rewrite-refs.
# 3. The result is applied on a new branch BRANCH-moved at --onto (default
#    HEAD), which is checked out: `git am -3` per commit, or one `git apply -3`
#    commit with --squash. On a conflict, resolve and `git am --continue` (or
#    commit).
# New files placed by a fallback get the M7/M8 depth edits (parents[k],
# with_name siblings and fixtures, dirname BASH_SOURCE); they and new files left
# at the top of a split directory are listed for review. Paths built other ways
# are not rewritten: run the tests. Rust paths renamed by the naming pass
# (scripts/build/rename-map.tsv) are rewritten in .rs hunk lines; a multi-line
# `use crate::{...}` group is only rewritten where an item carries crate::. A Rust module
# added beside a moved one may need its `mod` line in the new parent
# (families/<id>/mod.rs, shared/mod.rs).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
map="$repo_root/scripts/build/path-map.tsv"
onto=HEAD
pre=""
rewrite_refs=1
dry_run=0
squash=""
branch=""
while (($#)); do
  case "$1" in
    --onto) onto="${2:?--onto needs a revision}"; shift 2 ;;
    --pre) pre="${2:?--pre needs a revision}"; shift 2 ;;
    --map) map="${2:?--map needs a file}"; shift 2 ;;
    --squash) squash=1; shift ;;
    --no-rewrite-refs) rewrite_refs=0; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) [[ -z "$branch" ]] || { echo "one BRANCH only" >&2; exit 2; }; branch="$1"; shift ;;
  esac
done
[[ -n "$branch" ]] || { echo "usage: $0 [--onto REV] [--pre REV] [--map FILE] [--squash] [--dry-run] BRANCH" >&2; exit 2; }
[[ -f "$map" ]] || { echo "path map not found: $map" >&2; exit 2; }
cd "$repo_root"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "working tree is not clean" >&2; exit 2; }
git rev-parse --verify -q "$branch^{commit}" >/dev/null || { echo "no such branch: $branch" >&2; exit 2; }
onto="$(git rev-parse --verify "$onto^{commit}")"
m1="$(git log --format=%H --grep='^M1: move daemon modules' "$onto" | tail -n 1)"
if [[ -z "$pre" ]]; then
  [[ -n "$m1" ]] || { echo "cannot find the M1 move commit in $onto; pass --pre" >&2; exit 2; }
  pre="$(git rev-parse "$m1^")"
fi
pre="$(git rev-parse --verify "$pre^{commit}")"
if [[ -n "$m1" ]] && git merge-base --is-ancestor "$m1" "$branch"; then
  echo "$branch already contains the move; rebase it normally" >&2; exit 2
fi
fork="$(git merge-base "$branch" "$pre")"
if [[ -z "$squash" && -n "$(git rev-list --merges "$fork..$branch")" ]]; then
  echo "== $branch contains merge commits: using --squash (one commit carrying its net change)"
  squash=1
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/rebase-across-move.XXXXXX")"
cleanup() { git worktree remove --force "$work/premove" >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT
cp "$map" "$work/map"   # the pre-move tree has no map; never read it from the checkout later
# Rust path renames of the naming pass (crate::v41_memory -> crate::shared::memory, ...).
for f in rename-map.tsv rename_paths.py; do
  [[ -f "$repo_root/scripts/build/$f" ]] && cp "$repo_root/scripts/build/$f" "$work/"
done

source_ref="$branch"
if ! git merge-base --is-ancestor "$pre" "$branch"; then
  # Bring a copy up to the last pre-move commit in a scratch worktree, so neither
  # BRANCH nor the current checkout is touched.
  source_ref="premove/${branch#premove/}"
  ((dry_run)) && { echo "(dry run: would bring $source_ref up to ${pre:0:12} first)"; exit 0; }
  git worktree add -q -B "$source_ref" "$work/premove" "$branch"
  if [[ -n "$squash" ]]; then
    echo "== merging the last pre-move commit ${pre:0:12} into $source_ref"
    step=(git -C "$work/premove" merge -q --no-edit -m "Merge the last pre-move commit into $branch" "$pre")
    how="commit"
  else
    echo "== rebasing $source_ref onto the last pre-move commit ${pre:0:12}"
    step=(git -C "$work/premove" rebase -q "$pre")
    how="git rebase --continue"
  fi
  if ! "${step[@]}"; then
    trap - EXIT
    echo "resolve the conflicts in $work/premove, $how there, then rerun with BRANCH=$source_ref" >&2
    exit 1
  fi
  git worktree remove --force "$work/premove"
fi

mkdir -p "$work/in"
if [[ -n "$squash" ]]; then
  git diff --binary -M "$pre" "$source_ref" > "$work/in/0001-squash.patch"
  [[ -s "$work/in/0001-squash.patch" ]] || { echo "== no changes on $source_ref"; exit 0; }
  git log --format='- %h %s' --no-merges "$fork..$branch" > "$work/log"
  echo "== squashing $(wc -l < "$work/log") commit(s) of $branch"
else
  git format-patch -q --binary --no-signature -M -o "$work/in" "$pre..$source_ref"
  count="$(find "$work/in" -name '*.patch' | wc -l)"
  echo "== $count commit(s) on $source_ref since ${pre:0:12}"
  ((count > 0)) || exit 0
fi
git ls-tree -r --name-only "$onto" > "$work/post-files"

python3 - "$work/map" "$work" "$rewrite_refs" <<'PY'
import os, re, sys
map_path, work, rewrite_refs = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
rename_rules = None
if os.path.exists(os.path.join(work, "rename_paths.py")):
    sys.path.insert(0, work)
    import rename_paths
    rename_rules = rename_paths.load(__import__("pathlib").Path(work) / "rename-map.tsv")
exact, dirs, globs = {}, [], []
for line in open(map_path, encoding="utf-8"):
    if not line.strip() or line.startswith("#"):
        continue
    old, new = line.rstrip("\n").split("\t")[:2]
    if "*" in old:
        rx = "^" + "".join("([^/]*)" if c == "*" else re.escape(c) for c in old) + "$"
        globs.append((re.compile(rx), new))
    elif old.endswith("/"):
        dirs.append((old, new))
    else:
        exact[old] = new
dirs.sort(key=lambda d: -len(d[0]))
post = set(open(os.path.join(work, "post-files"), encoding="utf-8").read().split())
review = set()

def remap(path):
    if path in exact:
        return exact[path]
    if path in post:
        return path
    for old, new in dirs:
        if path.startswith(old):
            return new + path[len(old):]
    for rx, new in globs:
        m = rx.match(path)
        if m:
            out, groups = new, list(m.groups())
            out = re.sub(r"\*", lambda _: groups.pop(0), out)
            review.add(f"{path} -> {out}")
            return out
    return path

# Files the branch adds: where they land, so references to them follow too.
added = {}
for name in sorted(os.listdir(os.path.join(work, "in"))):
    text = open(os.path.join(work, "in", name), encoding="utf-8", errors="surrogateescape").read()
    for m in re.finditer(r"^diff --git a/\S+ b/(\S+)\nnew file mode", text, re.M):
        dest = remap(m.group(1))
        if dest != m.group(1):
            added[m.group(1)] = dest

ref_rx = re.compile(r"(?<![\w.-])((?:rust/crates|native|python|scripts)/[A-Za-z0-9_./-]*[A-Za-z0-9_])")
def remap_ref(m):
    p = m.group(1)
    if p in exact:
        return exact[p]
    if p in added:
        return added[p]
    for old, new in dirs:
        if p.startswith(old) and p not in post:
            return new + p[len(old):]
    return p

def moved_script(name, root):
    """Where root/name went (exact move or a file this branch adds), relative to root."""
    full = f"{root}/{name}"
    dest = exact.get(full) or added.get(full)
    return dest[len(root) + 1:] if dest and dest.startswith(root + "/") else None

# Paths the move rewrote in joined form (the same rewrites M7/M8 made in tests).
def joined_rewrites(path, text):
    if path.startswith("scripts/tests/"):
        def up1(m):
            sub = moved_script(m.group(3), "scripts")
            return f"{m.group(1)}{m.group(2)}{sub}{m.group(2)}" if sub else m.group(0)
        text = re.sub(r"""(parents\[1\]\s*/\s*)(["'])([A-Za-z0-9_.-]+)\2""", up1, text)
    if path.startswith(("scripts/tests/", "python/tests/")):
        def tools(m):
            sub = moved_script(m.group(2), "python/tools")
            if not sub:
                return m.group(0)
            q = m.group(1)
            return "TOOLS / " + " / ".join(q + part + q for part in sub.split("/"))
        text = re.sub(r"""TOOLS / (["'])([A-Za-z0-9_]+\.(?:py|sh))\1""", tools, text)
        text = text.replace("sys.path.insert(0, str(TOOLS))",
                            'sys.path[:0] = [str(TOOLS / d) for d in ("lib", "aot", "bench", "hf", '
                            '"qualify/deepseek_v4", "qualify/deepseek_v41", "qualify/glm5_flash")]')
    return text

# A new file a fallback placed deeper than it was written for gets the edits
# the move made to the files around it (M7/M8): repo roots one level up per
# added directory, with_name() siblings and fixtures found from the new place.
def relocated_rewrites(src, dest, text):
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
        sub = moved_script(name, root)
        return f"{base} / {q}{sub}{q}" if sub else m.group(0)
    text = re.sub(r"""Path\(__file__\)(?:\.resolve\(\))?\.with_name\((["'])([A-Za-z0-9_.-]+)\1\)""", sibling, text)
    text = re.sub(r'\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/\.\.((?:/\.\.)*)',
                  lambda m: '$(dirname "${BASH_SOURCE[0]}")' + "/.." * (1 + m.group(1).count("/..") + depth), text)
    return text

header = re.compile(r"^(diff --git a/)(\S+)( b/)(\S+)$")
# Directories the move split by kind or family: a new file at their top level
# may belong in one of the new subdirectories.
SPLIT_ROOTS = {"scripts", "python/tools", "python/reference", "rust/crates/cuteafd-daemon/src",
               "rust/crates/cuteafd-ffi/src", "rust/crates/cuteafd-loader/src"}
unplaced, current, source_path = set(), "", ""
os.makedirs(os.path.join(work, "out"))
for name in sorted(os.listdir(os.path.join(work, "in"))):
    lines = open(os.path.join(work, "in", name), encoding="utf-8", errors="surrogateescape").read().split("\n")
    out, in_hunk = [], False
    for line in lines:
        m = header.match(line)
        if m:
            in_hunk = False
            source_path, current = m.group(4), remap(m.group(4))
            line = f"{m.group(1)}{remap(m.group(2))}{m.group(3)}{remap(m.group(4))}"
        elif line.startswith("new file mode") and source_path not in added \
                and os.path.dirname(source_path) in SPLIT_ROOTS:
            unplaced.add(source_path)
        elif line.startswith(("--- a/", "+++ b/")):
            line = line[:6] + remap(line[6:])
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            key, _, path = line.rpartition(" ")
            line = f"{key} {remap(path)}"
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and rewrite_refs and line[:1] in ("+", "-", " ") and not line.startswith(("+++", "---")):
            # The move rewrote repo paths in place, so context and removed lines
            # get the same rewrite as added ones or they would not match.
            text = joined_rewrites(current, ref_rx.sub(remap_ref, line[1:]))
            if rename_rules is not None and current.endswith(".rs"):
                text = rename_paths.rewrite(current, text, rename_rules)
            if source_path in added and line[0] == "+":
                text = relocated_rewrites(source_path, current, text)
            line = line[0] + text
        out.append(line)
    open(os.path.join(work, "out", name), "w", encoding="utf-8", errors="surrogateescape").write("\n".join(out))
if unplaced:
    print("== new files left at the top of a split directory (move them where they belong):")
    for u in sorted(unplaced):
        print("   " + u)
if review:
    print("== new files placed by a split-directory fallback (check the destination):")
    for r in sorted(review):
        print("   " + r)
PY

if ((dry_run)); then
  echo "== dry run: rewritten file headers"
  grep -h '^diff --git' "$work"/out/*.patch | sort -u
  exit 0
fi
target="${branch#premove/}-moved"
git rev-parse --verify -q "refs/heads/$target" >/dev/null && { echo "branch $target already exists" >&2; exit 2; }
git switch -q -c "$target" "$onto"
echo "== applying onto ${onto:0:12} as $target"
if [[ -n "$squash" ]]; then
  if ! git apply -3 --index "$work/out/0001-squash.patch"; then
    echo "git apply left conflicts on $target: resolve them, git add, and commit" >&2
    exit 1
  fi
  { echo "${branch#premove/} across the repo layout move"; echo; echo "Squashed from:"; cat "$work/log"; } | git commit -q -F -
else
  if ! git am -3 --keep-cr "$work"/out/*.patch; then
    echo "git am stopped on a conflict: resolve, then git am --continue" >&2
    exit 1
  fi
fi
echo "== done: $target = $(git rev-parse --short HEAD); check mod declarations for new Rust files, then build"
