# Brief: <family> smoke port, packet <Pn: name>

**Goal.** <one sentence: what exists when this packet is done>

**Read first.** `AGENTS.md`; `docs/architecture/shared-engine.md`; the smoke-port skill's
`SKILL.md`, `references/catalog.md` and `references/checklist.md`; the port's `SPEC.md` and
`PLAN.md`.

**Branch and worktree.** `<branch>` from `<base SHA>` in `<worktree path>`. Commit at every compiling
step. Rebase onto the base's tip before handing back.

**Scope.** <the files and symbols this packet owns>. Out of scope: <neighbouring packets>; anything
marked deferred in the spec.

**Rules.**
- Reuse first: compose or extend shared code; never copy a family file wholesale. When a shared
  file must change, put it on its own small branch with a test and name it in your report.
- No family name or per-family constant in shared code: add a descriptor or spec field.
- Every unsupported semantic is a typed refusal, not a silent default.
- Every per-layer property is data, asserted in tests.
- New or recompiled programs carry a GPU oracle against dense FP32 (`SM120` and `SM121`).
- Find symbols by grep; verify every citation you write.
- No hardware unless this brief grants it, and then only under the locks in `AGENTS.md`.

**Gate.** <exact commands and the passing result>

**Report.** Branch head, commands run with their results, shared files touched (and why), open
questions, and a "Resume here" paragraph.
