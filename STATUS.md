# WP-5 MiMo multimodal serving

Branch: work/mm-mimo-serve
Base: origin/work/p0 163c3a36

## Plan

1. Wire the resident encoder into MiMo readiness, admission and media stats.
2. Inject cached image features into target and drafter embeddings; preserve native token IDs and full media prefix identities.
3. First qualify local RTX Flash images using the standard root launcher and matching WP-4 native assets; add probes/features, then remote placement and Pro.
4. Commit and push each green step; report measurements and remaining gates.

## Progress

- d4261db5 (pushed): nonblocking local encoder adapter, canonical encoder identity and normalization helpers. Its gate was five passed/one ignored, correcting the commit-message count.
- dff38ddd: WP-4 explicit placement parser cherry-pick.
- 3e4b1244 (pushed) serving integration: host-only pending media, peek/encode/admit/reconcile with slot release before retry and bounded cold fallback, carved embedding-cache quota, readiness-gated API capability and media stats.
- Target serial/paired/pipelined prefills inject after token gather; MTP injects shifted known image rows. DFlash receives downstream target taps. Prefix-covered absent host features keep optional MTP cold until its reconstruction window clears.
- Local owner QueueFull after cancellation is backpressure, not a keyed encode failure; deterministic cancellation test passes.
- Planner accepts supported BF16 vision geometry; resident loading still validates the complete actual tower inventory.
- No defaults promoted; V4.1 serving untouched; main checkout untouched.
- No WP-5 hardware job queued or locks held. WP-4 full WIP pair in slot mm-wp4 is available for native reuse; no duplicate full build.
- Standard run.sh/run-family.sh/wip.sh UID behavior remains unchanged. UID1000 applies only to fidelity custom qualification containers.

## Gates

- Workspace check: pass (19.50 s).
- Full cargo test --workspace: exit0, zero failures. Latest joint gate bnfglj5rx is complete; ignored hardware tests remain ignored.
- Full scripts/tests after initializing pinned sparkinfer/xgrammar: 993 passed, 2 skipped, 193 subtests passed, 69.72 s, zero failing IDs.
- Focused planner vision gate: 1 passed, zero failures.
- Cancellation backpressure gate: 1 passed, zero failures; 58.40 s compilation.
- Initial script run failed from uninitialized pinned sources; resolved, not waived.
- Pending: first correct live Flash image answer, probe image support and strict feature swap, G4-G7, live text prefix qualification, 64-image history, C4 images, Flash text C1/C16 >=0.98 vs work/p0, Pro.

## Open Issues

- Imported WP-4 launcher 35b21708 and only its planner/report/CLI prerequisites from 21123512. Remote adapter, worker and expertd integration remain parked until the first correct local image answer.
- Launcher prerequisite gates: full cargo workspace zero failures (daemon 417 passed/123 ignored); scripts 998 passed, 2 skipped, 193 subtests passed in 74.85 s, zero failing IDs. First script phase used a nonexistent task-local venv; reran using the existing project Python environment.
- Root daemon build of 3e4b1244 completed in 12m26s; incremental planner CLI rebuild pending. Private derived images use standard release layout/entrypoint and exact mm-wp4 natives/PROGRAMS/fp8 assets; no launcher adaptation or UID change.
- WP-8 tooling 2990f8d1/39b2e21c needs live prepared keys and image probes; synthetic component keys are not G4 qualification.
- Live readiness time, text throughput and LM image numerical equivalence remain unmeasured.
- Final workspace gate b6r8dl13m after bounds/hint cleanup: exit0, zero failures (daemon 416 passed/123 ignored); diff check clean.
