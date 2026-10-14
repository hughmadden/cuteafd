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
| Grok 4.7 (xAI, CLI `grok`) | bounded fixes, tests, races; design analysis with precise citations | found a real race behind a "flaky test"; challenged a wrong brief premise (#180 named V4.1/MiMo, but only GLM 5.3 uses the lane path); exact file:line citations (14/14 verified); caught a latent copies_pending bug | context fills fast (72% of 256K on a design task; needed a wrap-up nudge) | 3 tasks |
| Gemini 4 Argon (Antigravity, CLI `agy`) | to be determined: kernels, deep research, vision | — | first call returned a transient 503 | 1 duel (#195): strong design and exact numerics, slow but fine without steering |

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
| 2026-10-14 | Grok 4.7 (driven by DeepSeek Flash) | #180 Spark lane upload overlap: design + test-only prototype | 5 | 5 | 5 | 4 | ~21 min worker, ~27 min verified | correctly re-scoped from the brief's V4.1/MiMo to GLM 5.3 (the only SparkLane user) while still giving V4.1 numbers; 14 citations all accurate; ceiling arithmetic right; 7 real ownership/event tests (driver re-ran: 7 pass); flagged copies_pending flips only in join_uploads (verified :592). 1 steer (context wrap-up). |
| 2026-10-14 | DeepSeek Flash (driver) | drove grok-180 | 5 | 4 | 4 | — | +6 min overhead | verified claims itself (re-ran tests, grepped citations) instead of trusting the worker; good context-pressure nudge. Built an unapproved watcher (spurious idle returns from ext-agent were the cause); dropped it on request and fed back a good progress-event spec. |
| 2026-10-14 | Sol 6.1 | ext-agent driver + skill (#209) | 5 | 5 | 5 | 4 | ~54 min | real stdlib program + skill; took four rounds of field feedback (prune/resume, progress events, completion contract, idle debounce) quickly and cleanly; kept live transcripts out of git (account banner). |
| 2026-10-14 | Grok 4.7 (driven by DeepSeek Flash) | #177 MiMo 128-row verify: design + demand | 5 | 5 | 5 | 4 | ~19 min worker, ~22 min verified | every site of the 64 cap with file:line (driver spot-checked ~30, all exact); demand from a CPU replica of the shared policy (binds cold, and engaged only at ~0.9 acceptance); the 64/65 TMA second-tile cost and the capture-reach-to-0 consequence; recommended later and opt-in, matching the GLM Flash precedent. Labeled unverified inputs honestly. 1 steer (context 81%). |
| 2026-10-14 | DeepSeek Flash (driver) | drove grok-177 | 5 | 5 | 5 | — | +3 min overhead | used the completion template; reproduced the arithmetic and checked about 30 citations; one timely context nudge. |
| 2026-10-14 | Gemini 4 Argon (driven by DeepSeek Flash) | #195 duel: recover the width-128 M=1 Spark FC2 cost, keeping order | 4 | 5 | 4 | 4 | ~21 min | root cause with exact numbers (24 vs 48 CTAs on GB10, 2× per-CTA bytes and MMAs; smem sizes match code exactly); 6 options with order effects (correctly flagged split-K as order-breaking); picked cooperative sub-tiled width-128 with partner exchange, est. 53–54 µs, plus an invariance argument and a bitwise test. One misattributed citation. Unvalidated estimates (no GPU, as briefed). 0 steers, 0 503s, context reached 100%. Sol's measured split-2 (57.0 µs) is the comparable shipped result. |
| 2026-10-14 | DeepSeek Flash (driver) | drove gem-195 | 5 | 5 | 5 | — | — | checked about 6 citations and the bandwidth arithmetic; caught the misattributed citation; recovered the deliverable via the final.md backstop. |
