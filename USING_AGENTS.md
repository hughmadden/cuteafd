# Using agents on cuteafd

How cuteafd work is split between an orchestrating Claude session, Claude
subagents and Codex, and the rules that keep many agents productive on one
shared cluster. `AGENTS.md` holds the engineering rules every agent follows;
this file is about running the agents themselves. It records what we learned
building v0 and v1 (2026-09-28 to 2026-10-05).

## Roles

| Who | Does | Doesn't |
|---|---|---|
| **Orchestrator** (one Claude session) | Plans, writes task briefs, makes default and policy calls, reviews every branch, resolves merge conflicts, merges into `work/p0`, cuts releases, talks to TJ | Long hardware runs, bulk implementation |
| **Codex** (`gpt-6.1-sol`) | Bounded engineering: kernels, ports, measurements, A/B campaigns, release builds and smoke matrices, housekeeping | Merging, tagging, pushing images, changing defaults without passing gates |
| **Claude subagents** | Judgment-heavy, cross-cutting work: engine design changes (e.g. the device-driven exchange), large merges, investigations that need many decisions | Work Codex can do from a clear brief |

Capacity drives the split. Claude agents are the scarce budget: eight
parallel Opus agents used ~60% of a week's capacity in under a day. Codex
had ample budget (under 1% of a weekly subscription after a full day of
work). Send most implementation and measurement to Codex; spend Claude on
orchestration, review and design.

## Codex

### Launching

Use the CLI wrapper, not the Claude Code Codex plugin. The plugin runs
Codex in a read-only or workspace-write sandbox: no writes to `~/.cache`, no
GPU, no SSH to the Sparks, no `git push`. Every cluster task sent through it
came back blocked with drafts only.

```sh
# 1. Write the brief:    ~/.cache/cuteafd/builds/codex-runs/<name>.md
# 2. Start it as a background command; its exit wakes the orchestrator:
scripts/agents/codex-launch.sh <name> [high|xhigh]
# 3. Read the result:    ~/.cache/cuteafd/builds/codex-runs/<name>.report.md
# Stop a run (whole process tree):
scripts/agents/codex-stop.sh <name>
```

`codex-launch.sh` prepends `scripts/agents/codex-preamble.md` (worktree,
build, lock, sudo, frugality and no-merge rules) to the brief and runs
`codex exec -s danger-full-access` under `setsid`, recording its PID. Override
the runs directory with `CODEX_RUNS`.

### Model and effort

- **Default:** `gpt-6.1-sol` at `high` (the CLI default in `~/.codex/config.toml`).
- **`xhigh`:** subtle numerics, kernels, root-cause investigations.
- **Astra (`medium` or `high`):** only for the hardest kernel optimizations; much
  stronger but burns budget fast. Escalate to it when `xhigh` stalls.
- Don't drop to lighter models for "mechanical" tasks; they cost review time.

### Writing a brief

A good brief is one bounded task with everything Codex needs to finish
without asking:

1. **Branch and base:** `work/<task>` off `origin/work/p0`, worktree outside the
   repo (the preamble covers this).
2. **Context:** the measured numbers and file paths that motivate the task,
   the `PLAN.md` item, and what was already tried. Point at earlier reports
   rather than restating them.
3. **Gates:** golden NLL/KL/top-1 bounds, byte-exactness or lossless-spec
   checks, the hardware configs (natural minimum / maximum) and how many
   launches per arm.
4. **Decision rule, in full.** Codex applies thresholds literally. It kept a
   BF16 default because dual-RTX C1 gained 1.35% against a 2% bar, despite
   C4 +15% and C16 +10%. Say which metrics matter most (C1 first for TJ, then
   C4 and the reasoning-on agent session; 32-token micro cases don't matter),
   or ask it to recommend and let the orchestrator decide.
5. **Defaults:** "don't change defaults unless the gates pass", plus what to
   leave opt-in.
6. **Report:** before → after tables with conditions, gate results, commits,
   open issues.

Restating TJ's standing rules where they apply avoids rework: single residency
(never two formats of one tensor resident), honor checkpoint numerics, never
read code whose licence covers re-implementations (e.g. b12x PR #342).

### Running

- **One run per task.** Never start a second run of a task that is still
  running; `codex-launch.sh` refuses to. A duplicate finds the first one's
  worktree, backs off, and reports an "ownership" question instead of working.
- **Stopping:** always `codex-stop.sh <name>`. Killing only the launcher
  leaves Codex alive as an orphan that keeps editing and collides with any
  relaunch.
- **Relaunching:** append a `RESUME NOTE` to the brief saying what the earlier
  run left (worktree, uncommitted edits, known bugs) and that no other run is
  active. Codex picks up from the worktree state.
- **Provider errors:** a run can end with "Selected model is at capacity".
  Its pushed commits survive; relaunch with a resume note.
- **Waiting:** the background launcher wakes the orchestrator on exit. Don't
  poll the log in between; peek only when TJ asks for status.

### Known behavior

- Follows gates and AGENTS.md rules carefully and reports missed gates
  honestly, including its own mistakes.
- Conservative on judgment calls; rarely pushes a borderline result over the
  line. That's acceptable because the orchestrator decides.
- Applies thresholds literally (see "Decision rule").
- Writes reusable tooling along the way (gate scripts, probes, profilers).
- Slower than a Claude agent per task; queue Codex work early and in parallel.
- Has made questionable policy calls when left to decide defaults (the
  codex/v1 "honor checkpoint precision" refusals); always review defaults.

## Claude subagents

- Launch with the Agent tool in the background; give the same kind of brief
  (branch, gates, decision rule, no defaults without gates).
- They wait on long runs with one blocking background command and wake on
  its completion; no polling.
- Resume a finished or interrupted agent with `SendMessage` to keep its
  context; a new Agent call starts from scratch.
- **Before a Claude Code restart or a usage-limit reset,** have every agent
  checkpoint: commit and push WIP, and write
  `~/.cache/cuteafd/builds/<task>/STATUS.md` (branch, head, what's measured,
  jobs in flight with their re-run commands, next steps). After the restart,
  resume each from its STATUS.md. Background shells and their completion
  watchers die with the session, so finished jobs must be collected by hand.
- A session limit stops every agent at once; resume each with a short note
  and tell it to check its own jobs before relaunching anything.

## Shared cluster discipline

These apply to every agent and are also in `AGENTS.md`.

- **Locks:** take `sparks.lock` before `gpu1.lock`, only around actual runs,
  each run one blocking command with a timeout. Reversed lock order
  deadlocked the cluster with both GPUs idle and ~30 jobs queued.
- **Teardown:** stop servers, containers and Spark workers before releasing
  the locks (lock scripts should trap exit). A stray server held 30 GB of GPU1
  for two hours and broke another agent's measurements.
- **Watchdogs:** device-side waits and long runs get timeouts. A device-mode
  hang held both locks for ~4 hours before a watchdog existed.
- **Disk:** builds fill raptor's root NVMe fast (`builds/` reached 1.4 TB and
  crashed runs with "No space left on device"). Delete Cargo `target*` and
  release staging when a task finishes; check `df -h /` before large builds.
- **Root:** run privileged commands directly as
  `agent-sudo -n --agent-context "<why>" <command>` with the whole command
  visible. A wrapper script under sudo (`sudo python profile.py`) is flagged by
  the approval engine; TJ pre-approved `ncu` itself.
- **Releases:** only the orchestrator tags, moves `main`, or pushes images, and
  only with TJ's approval. A release run fail-stops if any spot-smoke fails;
  check that no other agent is using the GPUs before release measurements
  (an overlapping kernel build once made V4.1 look 15% slower).

## Reviewing and merging

1. Read the report, then check the branch: `git merge-base --is-ancestor
   origin/work/p0 origin/<branch>` and `git diff --stat origin/work/p0...origin/<branch>`.
2. Merge in a temporary worktree. On conflicts, keep both sides' intent; if
   Codex wrote the branch and the conflict is substantial, send it back to
   Codex to merge `origin/work/p0` and re-verify.
3. Check launcher keys: every key `run-family.sh` reads must be in
   `release_known_key` (`scripts/lib/release-common.sh`).
4. Tests: `cargo test --workspace`, and compare failing script-test ids
   against `origin/work/p0` (add none).
5. Push to `work/p0` (fast-forward when possible).
6. Override a conservative Codex call when the measurements justify it, and
   record the reason in the commit (e.g. the V4.1 FP8 vocabulary head default).

## Lessons in brief

| What happened | Rule now |
|---|---|
| Codex plugin tasks all blocked (sandbox) | Launch with `scripts/agents/codex-launch.sh` |
| Killed launcher, orphaned Codex kept editing | Stop with `codex-stop.sh` |
| Duplicate runs deferred to each other | One run per task; resume notes |
| Literal 2% threshold kept a slower default | State the full decision rule |
| Lock-order deadlock | `sparks.lock`, then `gpu1.lock`, with timeouts |
| Stray 30 GB server | Teardown before releasing locks |
| 4-hour device-mode hang | Watchdogs and per-step timeouts |
| Disk full at 1.4 TB of builds | Clean build output at task end |
| `sudo python …` flagged | Sudo the real command directly |
| Restart lost agents' watchers | STATUS.md checkpoints before restarts |
| 8 Opus agents, 60% weekly in a day | Codex for bounded work; Claude orchestrates |
