---
name: smoke-port
description: Bring a new model up on cuteafd as a smoke release, a well-designed implementation on the shared engine that serves coherent text on a named cell with a smoke card, in days rather than weeks; only qualification (fidelity sets, benchmark campaigns, tuning) waits. Use when asked to add, port or "smoke" a model or checkpoint on cuteafd for given hardware. Inputs are the checkpoint (HF id at a revision SHA, or a snapshot directory) and the target cell (RTX cards and Spark count). Not for qualification or release work (fidelity sets, lossless speculation, row curves, card A/Bs): those follow docs/architecture/shared-engine.md §6-7 after the smoke.
---

# Smoke port

A **smoke release** of model M on cell C is all of these, and nothing more:

1. A family branch on the **current upstream tip**, plus one small branch per shared-engine gap.
2. `cuteafd plan M --deployment` executable for C.
3. M served by `./run.sh --wip` on C: text only, no speculator, one lane, eager allowed.
4. Coherence shown three ways: the probe set; per-layer replay against the reference;
   teacher-forced windows inside the smoke bars (`references/contract.md`).
5. One `cuteafd bench --profile smoke` card labelled **SMOKE (unqualified)**.
6. A smoke section in `docs/models/<family>.md` that lists every deferral.
7. **Well designed** to the bar below.

## Design bar

A smoke port is fast because it defers measurement and tuning, never design. Its code is the
release's code: qualification later adds numbers and speed, not a rewrite.

- **Shared engine contracts only.** A `Family` descriptor, a placement adapter over the shared solver,
  the shared scheduler's `Backend` (or `LaneSet::run_one` + `lanes::commit`), `PrefillHooks`,
  `SparkLink`. No private serve loop, private scheduler, private pool policy or compat shim.
- **The checkpoint as shipped, at its own numerics.** Read through `Checkpoint`: a non-HF layout
  gets a `CheckpointFormat`, never an overlay or a repack. Use the calibrated numerics the checkpoint
  ships (W4A4 with its `input_scale`s). A converted interim path is allowed only where no native
  kernel exists, and then it is labelled and has an issue.
- **One geometry, from the config.** Every dimension comes from the family's one spec struct built
  from the checkpoint's config: no literal model dimensions in family code (Kimi's port carries about
  260). Per-layer properties are data; every unsupported semantic is a typed refusal; family names
  stay out of shared code (add a descriptor field).
- **The whole model.** Everything correct text needs at the planned context ships in the smoke. For
  Mistral Large 4 that means the sparse indexer, KV sharing, SSMax and the logit soft-cap, not a
  context cap that hides a missing indexer. Only optional subsystems (drafters, vision, audio) and
  performance work (graphs, packing, lanes, RTX expert onboarding, kernel tuning) wait.
- **Engine invariants hold.** Exact prefix-cache restores (`cache_exact` passes), weight and
  workspace admission before allocation, readiness after all fallible init, and workspace lifetimes
  that CUDA graphs can capture later without restructuring.
- **Programs from shared kernel classes.** Compose the model's maths from existing program classes;
  write new kernels only for new operators, each with a GPU oracle. Oracles and goldens are
  permanent tests, not scratch scripts.

Evidence for the budget (`references/evidence.md`): the first coherent tokens took 1 h 44 min for
GLM 5.3, 3 h 11 min for DeepSeek V4, 7 h after kernels for MiMo, 18.7 h for MiniMax, about 45 h
for Kimi K3 and about 71 h for Step-5. Weeks then went to re-hosting, optimisation, admission,
drafters, fidelity and fitting smaller cards. A smoke port does none of that.

## Inputs

| Input | Form | Rule |
| --- | --- | --- |
| `MODEL` | `org/name@<40-hex SHA>` or a snapshot dir | Never a branch name: vendors rewrite history (MiniMax lost a night to it). |
| `CELL` | coordinator card model × count, Spark count, the site config | Use the roomy card first (PRO 6000), then the minimum Spark world the census allows. |
| `REFERENCE` | the release's own modeling code: HF repo file, a transformers version, or a vendor serving image tag | Day-0 models often exist only in a vendor image: use `scripts/image_ref_extract.py`. |
| `SCOPE` | default text-only, speculator off | Turning anything else on needs the user's ask. |

**Hardware first.** Derive the layout from the census for the cell you were given. Existing software
caps are data to lift, not constraints to design around: Spark world lists, ABI depth limits,
geometry whitelists, family-name arms. If the checkpoint does not fit the cell, stop and report the
shortfall from `cuteafd plan`. Never quantize inside a smoke port.

## Phase 0: spec (CPU only, 1-2 h)

1. **Tip and contracts.** Fetch the upstream tip and branch from it. Read `AGENTS.md`,
   `docs/architecture/shared-engine.md` and `USING_AGENTS.md`. Check open PRs that touch shared
   prefill, exchange or the scheduler: MiniMax lost 1.5 days to two that landed mid-port. Find
   symbols by grep, not by the guide's line numbers, which drift.
2. **Census and stub** (seconds, no download):
   `python3 -I scripts/agents/skills/smoke-port/scripts/ckpt_census.py --hf ORG/NAME --rev SHA --out W/census --stub-dir W/stub`.
   The output gives bytes by component, the Spark fit for every TP world, and every per-layer
   shape variation; the stub is header-only and sparse, so `plan` can run on it.
3. **Reference.** Pin the official modeling code, or
   `python3 -I .../image_ref_extract.py vllm/vllm-openai:<tag> --out W/ref`. Record the digest.
4. **Spec.** Fill `templates/SPEC.template.md`: geometry, a per-layer pattern table taken from the
   census shapes, operator pseudo-code with its rounding points, numerics per module, tokenizer
   and template, drafter, vision. Source every number. Each open question is settled from the
   reference code, never guessed.
5. **Plan the checkpoint.** `cuteafd plan W/stub`. For a non-HF layout, first run
   `scripts/checkpoint_view.py ADAPTER W/stub W/view` and plan the view: it is a planning bridge only,
   replaced in P1 by a `CheckpointFormat` (`references/taxonomy.md` T5). Record the verdict and hints. "Family not recognized" is the
   expected answer for a new family. Any other refusal is a shared gap: give it its own branch and
   a test.
6. **Choose and cut.** Match the spec's features against `references/catalog.md` to pick a template
   per subsystem (attention, experts, scheduler skeleton, drafter). Tier each gap with
   `references/taxonomy.md`. Fill `templates/PLAN.template.md` with the layout, the fit, the packets
   and the gates.

**Exit:** SPEC and PLAN reviewed by the orchestrator, and the fit decision recorded. Worked example:
`examples/mistral_l4/`.

## Phase 1: build (CPU only, parallel packets, about 1 day)

Run each packet as its own branch and worktree from `templates/brief.template.md`, choosing the
model per `USING_AGENTS.md`. Every brief carries the reuse-first rule: compose or extend shared code
and never copy a family file wholesale (MiniMax's dedup work came from a swarm that copied).
`references/checklist.md` lists every registration point in order.

| Packet | Delivers | Gate |
| --- | --- | --- |
| P1 checkpoint + descriptor | format seam, config reader with a **typed refusal for every unsupported semantic**, `Family` descriptor (`decode_lanes` `One { reason }`, `prefill_limits`, `spark_decode`), `REGISTRY` entry | `cuteafd plan` on the stub names only package and program gaps |
| P2 experts | `ExpertGeometry` const with an **explicit activation and clamp** (unlisted shapes silently clamp SwiGLU at 10), SparkInfer `GEOMETRIES` row, packager choice, CMake regex, Spark worlds, slot census rows, catalog and name rules | unit tests; in phase 2, the family golden over real Spark experts on one MoE layer (`golden-run.py --spark-hosts H --expert-layers L`) |
| P3 programs | a geometry instance plus a program list **composed from existing classes**; new pointwise kernels only where nothing exists | CPU contract tests, plus **a GPU oracle per program against dense FP32** on SM120 and SM121 (both MiniMax bring-up bugs would have failed one) |
| P4 layer chain + golden | the chain over an `Ops` trait with a CPU oracle at tiny geometry (Kimi pattern); **every per-layer property as data, asserted** (Step-5's silently skipped layer 91); a golden subcommand with per-layer cosine and NLL | CPU oracle passes |
| P5 serving | shared `Backend` with no drafts, `PrefillHooks` with waves pinned to 1, a placement adapter over the shared solver, CLI, API encoding and parser (stop on the template's turn markers), launcher arms, one example config | `plan --deployment` executable for C; `run.sh --dry-run` |
| P6 reference golden | `python/reference/families/<fam>/golden.py` on the official code, dumping per-layer outputs for 2-8 windows that cover each layer kind | runs on the reference host |

Shared-engine gaps found along the way are separate small branches with tests, never edits buried
in the family branch. Rebase daily onto the tip. Commit at every compiling step and keep a
"Resume here" block in the plan: usage limits stop agents mid-packet.

**Exit:** compile; targeted tests; `scripts/dev/cpu-gate` default tier with no new failing ids
against the base; source gates; `compare-sm-exports.py --sms 188,170` if exports changed.

## Phase 2: bring-up (hardware, about 1 day)

1. Take locks per `AGENTS.md` and the site's own fleet rules; preflight hardware health. Stage the
   pinned weights once, on fast local disk or in SparkNest; never load from a network share.
2. `./wip.sh --slot S --role both --config CFG`, then `./run.sh --config CFG --wip S --dry-run`,
   then `--restart`. Use the **shipped launcher from the first run**: MiniMax's private harness hid
   seven launcher bugs for two days.
3. Seal check the slot (`cuteafd programs-check`), run the GPU oracles left over from P3, then the
   family golden on one MoE layer with real Spark experts
   (`scripts/bench/golden-run.py ... --spark-hosts H --expert-layers L`).
4. Probes: `scripts/qualify/api-smoke.sh`, then the probe set in `references/contract.md`.
5. **On any wrong output, run a layer replay before touching code**: per-layer dumps on real weights
   against the reference golden, the first failing layer, then the failing program's oracle. Fix
   at the root, with a test.
6. Teacher-forced windows against the reference (`compare-logits.py`), then
   `cuteafd bench --profile smoke`.
7. Stop every container you started, then release the locks. Leave nothing serving.

## Phase 3: release the smoke

Push the family branch and each shared-gap branch. Write the smoke section of
`docs/models/<family>.md`: cell, revision, bars met, card link, and the deferral list from
`references/contract.md`. Hand off with the plan's "Resume here" block. Publish only if asked.

## Stop rules

- **Doesn't fit the cell:** stop, and report the plan's shortfall and the layouts that would fit.
- **Wrong output:** layer replay, then the program oracle. Never guess-fix numerics.
- **A shared path doesn't fit:** a small shared branch with a test, or a reason written on the
  tracking issue. Never a private loop, a private scheduler or a compat shim.
- **A phase overruns its box by half again:** write status and "Resume here", then ask.
