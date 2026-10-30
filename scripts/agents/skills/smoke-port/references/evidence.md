# Evidence: where port time actually went

From the commit histories of every family on upstream `main` `1adc5bfb3` and the MiniMax M3 port on
the fork (reviewed 30 Oct 2026). Commit subjects quoted where they mark a milestone. Times are as
reported by each history; treat hour counts as calendar time.

## First coherent output was always fast

| Family | Milestone | Time from first family commit | What made it fast or slow |
| --- | --- | --- | --- |
| GLM 5.3 | `03173d478` "GLM 5.3 serves its first correct tokens end to end" | about 1 h 44 min, about 1,150 engine lines | every kernel existed (DeepSeek machinery); BF16 weights, no graphs |
| DeepSeek V4 | `9ce812439` port map → `eb288f334` "DeepSeek V4 Flash serves end to end" | 3 h 11 min (dSpark 2 h 45 min later; smoke cards on day 3.6) | order: torch golden, config reader, Python prototype on pinned fork ops compared layer by layer, expert geometry, program table, Rust engine, serve |
| MiMo V2 | `729473e54` "layer 0 matches the golden" → `96edf07b1` MoE layers | about 7 h after its kernels landed | engine patterned on GLM; BF16-dequantized interim weights |
| Qwen4 | golden and serving | about 1.3 h after its kernels | ported from an existing engine |
| Dense Qwen | KL 0.00058 | about 23 h | built on the shared scheduler: 4,675 daemon lines, 1.8 % duplicated |
| Kimi K3 | descriptor `c0084dd74` → whole-model fidelity top-1 0.926, KL 0.0626 | under 45 h (plus about 8 days of custom quantization, because the official experts do not fit six Sparks) | `Ops` chain with a CPU oracle; shared scheduler; launched through `run-family.sh` in `18472fb94` |
| Step-5 | → coherent full model, NLL 2.493 against reference 2.492 | about 71 h (first serving card about 89 h) | native CSA/SWA attention kernels; exact admission; launched in `312c1c520` |
| MiniMax M3 (fork) | `6754bccb8` config reader → `51f79cf68` coherent answers | 18.7 h, 46 commits: about 3.5 h of parallel CPU-only packets, a 5 h staging gap, about 10 h of bring-up | bring-up blockers: AOT scope vs CMake (`d6cbaf6a5`), a vendor history rewrite (`37d36bfd2`), no dma-buf on GeForce (`50ab9619a`), two attention kernel bugs found by layer replay |
| DeepSeek V4.1 | imported whole (about 60K daemon lines) | n/a | not a port; do not use as a template |

## Then weeks went elsewhere

- **GLM 5.3:** 31 days and 522 commits to a 12.6K-line family, mostly placement and admission, the
  Spark exchange, speculation, two-GPU executors and new quant formats.
- **DeepSeek V4:** about 30 % of 309 commits are planner, placement and admission.
- **Kimi / Step-5:** exact admission (plan == runtime; Kimi's `admission.rs` alone is 1,869 lines),
  performance kernels and kernel duels, and re-porting onto shared prefill changes. The adaptive
  prefill controller lived two days before `0b3d68886` deleted it.
- **MiniMax**, ranked by cost:

| # | Sink | Cost | Smoke-port rule |
| --- | --- | --- | --- |
| 1 | Re-host onto upstream's moving shared engine, then the PR split | 35.9 h calendar | start on the tip; shared gaps as separate small branches from day 1; rebase daily |
| 2 | "Fully working and optimised" campaign (perf, dedup, vision, host tier, graphs, lanes) | 27 h, 132 commits | defer all of it; the reuse-first brief avoids the dedup work |
| 3 | Fidelity set and publication | about 13 h, later dropped | defer |
| 4 | Drafter integration (DFlash2) | 6.75 h on the critical path | speculator off |
| 5 | Hardware bring-up debugging | about 10 h | GPU oracle per program and the layer replay harness **before** the first serve |
| 6 | RTX 5090 fit and the GeForce exchange | spread over 3 days, all discarded on v4 | roomy card first |
| 7 | Publication; a production candidate through `run.sh` found 7 launcher bugs | 13 commits | use the shipped launcher from the first run |
| 8 | Spark staging and loading | a separate session | stage once on fast local disk or SparkNest |
| 9 | Vision and video | about 30 commits | `VISION=off` |
| 10 | A new layout for an existing family (TP6, `a37cd328f`) | 3 h | cheap when Spark worlds derive from the shared policy |

## Gotchas to encode

| Area | Gotcha |
| --- | --- |
| Kernels | Check every emitted AOT scope against the CMake entries (a test, not a review) |
| Kernels | Push the SparkInfer branch before pinning it; a fresh worktree cannot fetch an unpushed pin |
| Kernels | CPU contracts passed while both MiniMax attention bugs were live: GPU oracles are not optional |
| Checkpoint | Pin by SHA and re-check it resolves before staging |
| Checkpoint | Scale direction differs by producer: compressed-tensors NVFP4 global scales divide, ModelOpt multiplies |
| Checkpoint | NVFP4 W4A4 needs bit-identical gate/up `input_scale`; a layer-max substitute failed fidelity |
| Hardware | GeForce cards have no dma-buf: start on the PRO 6000 |
| Hardware | Spark live free memory decides fit (a TP2 refusal by about 0.2 GB); drop caches before timed launches |
| Disk | `--save-logits` goldens are about 25 GB per 64 windows |
| Process | Wait loops: `pgrep -f '[p]attern'` (a self-matching waiter never exits) |
| Process | "Resume here" plus a commit at every compiling step survive usage-limit stops |
| Launcher | `golden-run.py` and parts of `wip.sh` assume one site's hosts and GPU indices: other cells need a shim |
| Gates | Compare logits on generated rows only: sparse top-k near-ties put prompt rows nats apart |
| Planner | Never hard-code Spark worlds; derive them from the shared range policy |
| Agents | The scorecard records agents fabricating code facts: verify every agent-written citation by grep |
