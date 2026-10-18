#!/usr/bin/env bash
# wip-source-checkout.sh REPO REV DEST: a detached worktree of REV with only the
# submodules a WIP build reads (sparkinfer, xgrammar with its nested 3rdparty,
# transformers), each at the commit REV pins. Submodules clone from REPO's own
# module object stores when they hold the pinned commit (no network, ~1 s each),
# keep their real origin URL, and are verified against their tree locks; a
# missing commit falls back to the network. Publishing-only submodules
# (gptqmodel, tool-eval-bench) are left uninitialized.
set -euo pipefail

[[ $# -eq 3 ]] || { echo "usage: wip-source-checkout.sh REPO REV DEST" >&2; exit 2; }
repo="$(git -C "$1" rev-parse --show-toplevel)"
rev="$2"
dest="$3"
common="$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)"

git -C "$repo" worktree add --detach "$dest" "$rev" >/dev/null

# submodule_from_local SUPER PATH MODULES_DIR: initialize PATH inside SUPER from
# MODULES_DIR/PATH when that store holds the pinned commit, else from its URL.
submodule_from_local() {
  local super="$1" path="$2" modules="$3" pinned url reference
  pinned="$(git -C "$super" ls-tree HEAD -- "$path" | awk '$2 == "commit" {print $3}')"
  [[ -n "$pinned" ]] || { echo "$path is not a submodule of $super" >&2; return 2; }
  git -C "$super" submodule init -q -- "$path"
  url="$(git -C "$super" config "submodule.$path.url")"
  reference="$modules/$path"
  if [[ -d "$reference" ]] && git -C "$reference" cat-file -e "$pinned^{commit}" 2>/dev/null; then
    git -C "$super" -c protocol.file.allow=always -c "submodule.$path.url=$reference" \
      submodule update -q --reference "$reference" -- "$path"
    git -C "$super/$path" remote set-url origin "$url"
  elif [[ -d "$reference" ]]; then
    git -C "$super" submodule update -q --reference "$reference" -- "$path"
  else
    git -C "$super" submodule update -q -- "$path"
  fi
}

submodule_from_local "$dest" third_party/sparkinfer "$common/modules"
submodule_from_local "$dest" third_party/transformers "$common/modules"
submodule_from_local "$dest" third_party/xgrammar "$common/modules"
xgrammar_modules="$common/modules/third_party/xgrammar/modules"
while read -r nested; do
  submodule_from_local "$dest/third_party/xgrammar" "$nested" "$xgrammar_modules"
done < <(git -C "$dest/third_party/xgrammar" config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')

for name in sparkinfer xgrammar transformers; do
  python3 "$dest/scripts/build/verify-$name-source.py" \
    --source "$dest/third_party/$name" --lock "$dest/third_party/$name.lock.json" >/dev/null
done
echo "WIP source checkout: $(git -C "$dest" rev-parse HEAD) at $dest"
