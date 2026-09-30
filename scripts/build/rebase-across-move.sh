#!/usr/bin/env bash
# Carry a branch that forked before the repo layout move and the naming pass
# across them; see scripts/build/rebase_across_move.py (--help) for how.
#   scripts/build/rebase-across-move.sh [--onto REV] [--pre REV] [--squash] [--dry-run] BRANCH
#   scripts/build/rebase-across-move.sh --continue
exec python3 "$(dirname "${BASH_SOURCE[0]}")/rebase_across_move.py" "$@"
