#!/usr/bin/env bash
# write-build-identity.sh REPO OUT_JSON: the source identity (git remote,
# commit, dirty) of REPO for builds whose source tree has no .git (WIP slots,
# containers). rust/crates/cuteafd-bench/build.rs bakes it into the binary for
# benchmark report footers.
set -euo pipefail
repo="${1:?usage: write-build-identity.sh REPO OUT_JSON}"
out="${2:?usage: write-build-identity.sh REPO OUT_JSON}"
commit="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
remote="$(git -C "$repo" config --get remote.origin.url 2>/dev/null || true)"
dirty=false
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null)" ]] || dirty=true
[[ -n "$commit" ]] || dirty=null
printf '{"remote": "%s", "commit": "%s", "dirty": %s}\n' "$remote" "$commit" "$dirty" > "$out"
