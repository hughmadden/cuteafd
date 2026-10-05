Standing rules for every agent working on CuteAFD on this cluster (prepended by scripts/agents/codex-launch.sh; Agent-tool briefs tell the agent to read this file first).

You are working on CuteAFD at /home/tj/Developer/cuteafd (Rust engine + CuTe-DSL kernels in the fork /home/tj/Developer/sparkinfer-glmrt, pinned at third_party/sparkinfer with a tree lock). You have full shell, GPU, network and SSH access (raptor = this host with 2x RTX PRO 6000; Sparks ostrich dodo emu kiwi rhea moa by SSH). Read AGENTS.md, USING_AGENTS.md and the PLAN.md section named below first and follow AGENTS.md strictly:
- Work in your own git worktree OUTSIDE the repo directory (e.g. /home/tj/Developer/cuteafd-<task>), on the named branch off origin/work/p0; push the branch. Never edit the main checkout at /home/tj/Developer/cuteafd.
- Build only under ~/.cache/cuteafd/builds/<task> after scripts/build/assert-build-filesystem.py. Delete your Cargo target and staging directories when you finish.
- Hardware: take sparks.lock before gpu1.lock (~/.cache/cuteafd/{sparks,gpu1}.lock, see ~/.cache/cuteafd/builds/tp2/locked2.sh), only around runs; each run is one blocking command with a timeout; stop your servers/containers and Spark workers before releasing the locks. Other agents share the cluster.
- Root: run privileged commands directly as `agent-sudo -n --agent-context "<why>" <command>` with the whole command line visible; never sudo a wrapper script or interpreter.
- Kernel changes land on sparkinfer-glmrt master first, then bump the pin + lock here.
- Frugal measurement: one warm launch per arm unless a number is borderline. No attribution trailers in commits.
- Don't change defaults unless the gates pass. Don't merge into work/p0 or main, don't tag, don't push images — Claude reviews and merges.
Finish with a concise report: what changed, commits/branch, measurements (before -> after tables with conditions), gate results, open issues.
