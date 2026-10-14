# Agent scorecard

What each model is good and bad at on this project, from real tasks. The
orchestrator adds one line per evaluated task to the log below, then
periodically compacts the log into the summary table. Judge the work, not the
self-report: a claim counts only once a gate or a review confirmed it.

Rubric, each 1–5 (5 best), plus wall time:
- **Correct:** gates pass, no regressions, claims match evidence.
- **Judgement:** right scope, stops at the stop bar, asks where it should, no invented tooling.
- **Quality:** code and reasoning a reviewer would merge with few changes.
- **Creative:** found a better idea or root cause than the brief named.
- **Speed:** wall time to a merge-ready result, relative to the task.

## Summary (compacted)

| Model | Use for | Strengths | Weaknesses | Evidence |
|---|---|---|---|---|
| Sol 6.1 (Codex) | default engineering, numerics, A/B, lossless fixes | thorough, keeps gates honest, strong root-causing (#163, #190, #171) | over-literal on tooling rules; commits mid-gate (head-moved failures ×3 on 2026-10-14); verbose reports | ~15 tasks |
| Astra 6 (Codex) | hard kernel design | — | expensive; once spent a week on probe plumbing | few |
| DeepSeek Flash | structural, mechanical | — | — | none recently |
| Claude Opus 5.5 (orchestrator) | design, review, merges, decisions | fast cross-task diagnosis (#191 Spark regression, stale CMake cache) | pushed a red commit once (61256a22); held agents for an answer while GPUs idled; wrong guesses before evidence (masked-V, cold-start) | this session |
| Grok 4.7 (xAI, CLI `grok`) | bounded fixes, tests, races (early) | found a real race behind a "flaky test" and proved it deterministically | — (1 task) | 1 task |
| Gemini 4 Argon (Antigravity, CLI `agy`) | to be determined: kernels, deep research, vision | — | first call returned a transient 503 | trial starting 2026-10-14 |

## Log

| Date | Model | Task | Correct | Judgement | Quality | Creative | Wall | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-14 | Sol 6.1 | #163 MiMo Flash lossless | 5 | 5 | 5 | 5 | ~5 h | found the BF16 o_proj M9 switch from a verify-width trace; caught the W8A8 coupling and the #161 Qwen regression |
| 2026-10-14 | Sol 6.1 | #190 V4.1 compressor | 5 | 4 | 4 | 4 | ~6 h | correct cuBLAS diagnosis; first fixed plan was 40× slower (2-byte alignment); fixed after attribution |
| 2026-10-14 | Sol 6.1 | #144 V4 lossless | 5 | 5 | 5 | 4 | ~5 h | C128 boundary plus Spark width; honest about the eager-path cost |
| 2026-10-14 | Sol 6.1 | V4 layers stages 2–3 | 4 | 4 | 4 | 4 | ongoing | careful; one wrong "Spark suffix" claim, self-corrected |
| 2026-10-14 | Sol 6.1 | card renderer #203 | 5 | 5 | 5 | 4 | ~3 h | caught publisher drift and hand-edited caveats |
| 2026-10-14 | Opus (orchestrator) | merges | 4 | 3 | 4 | — | — | pushed 61256a22 with a failing test; later gated pushes on green |
| 2026-10-14 | Grok 4.7 | flaky prefill-cancel test (#208) | 5 | 5 | 5 | 5 | ~25 min | brief said test flake; found a real race in qualify-prefix-cache's cancel flag (timer.is_alive vs socket shutdown), fixed with an event + join, removed the sleep loop, proved it by forcing the window (40/40 before fails, after passes); full suite green. Merged as-is. |
| 2026-10-14 | Gemini 4 Argon | #175 kernel duel (plain -p) | — | — | — | — | ~1 h, no output | not scored: my -p launcher failed twice (prompt quoting, --effort with a model), then the run sat invisible behind a 503 retry for ~1 h and was stopped. Lesson: -p mode gives no visibility; use ext-agent with a driver. To rerun driven. |
