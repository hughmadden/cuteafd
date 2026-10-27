# MiniMax M3.1 Flash

A native multimodal MoE (428B total, 23B active) with MiniMax Sparse
Attention on every layer, quantization-aware-trained NVFP4 experts and
MXFP8 dense projections, a bundled vision tower and a bundled DSpark drafter.

Status: **under qualification**. MiniMax runs on the shared engine: prefill
units on the shared executor and Spark exchange, typed Spark owners, decode
as lane 0 of the shared Qwen scheduler and exact prefix-cache restores. No
release image ships `serve-minimax` yet: the launcher serves it only from a
`./wip.sh` slot (`--wip SLOT`), with its image tower, on one RTX PRO 6000
coordinator with 2, 3, 4 or 6 Sparks.
`cuteafd plan --layout` admits it from the slot's `PROGRAMS.json` with the
server's own allocation recipe. Its hardware qualification on the RTX PRO
6000 (golden fidelity, exact prefix restores, lossless speculation, N=1-vs-N
prefill identity, kernel row curves and the basic cards) is in progress; this
page publishes no measurement until it lands.

## Supported checkpoints / quants

- `MiniMaxAI/MiniMax-M3.1-Flash` (revision `54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e`),
  the official release, served in its own formats: routed experts
  compressed-tensors NVFP4 W4A4 (`weight_packed`, E4M3 per-16 scales, divisor
  `weight_global_scale` / `input_global_scale`), attention, indexer, dense and
  shared projections MXFP8 (E4M3, UE8M0 per 1x32), BF16 embedding and head,
  FP32 router. 233 GiB, plus the 2.1 GiB `dspark/` drafter.

**Starting configs** (one RTX PRO 6000 coordinator; replace the placeholder
Spark hosts and addresses): [TP2](../../examples/configs/minimax-m31-flash-nvfp4-pro6000-tp2.config) ·
[TP3](../../examples/configs/minimax-m31-flash-nvfp4-pro6000-tp3.config) ·
[TP4](../../examples/configs/minimax-m31-flash-nvfp4-pro6000-tp4.config) ·
[TP6](../../examples/configs/minimax-m31-flash-nvfp4-pro6000-tp6.config).
Each states only what differs from the launcher's defaults: the Spark world
and, from TP3 up, no routed layers on the RTX.

## Engineering summary

- 60 layers, hidden 6144; layers 0-2 dense (SwiGLU-OAI, 12288), 3-59 MoE:
  128 experts, sigmoid top-4 with a selection-only bias, renormalized, times
  2.0, plus one 3072-wide shared expert. SwiGLU-OAI everywhere:
  `g * sigmoid(1.702 g) * (u + 1)` with `g = min(gate, 7)`, `u = clamp(up, ±7)`
  (b12x `swigluoai_uninterleave`, gate rows first).
- Attention: GQA 64 query / 4 KV heads of 128, per-head Gemma RMSNorm on q
  and k (`x * (1 + w)`), NeoX RoPE on the first 64 channels (theta 1e7).
- MSA: a lightning indexer (4 query heads, one per KV head, over one shared
  128-wide key, both Gemma-normed and RoPE'd) scores every earlier token; each
  128-token block scores its max; the query's own block is forced; each query
  row and KV head attends its top 16 blocks (2,048 keys). Per-row selection by
  construction. Contract: b12x `attention/dsa_indexer/MSA.md`; CPU oracle
  `python/reference/families/minimax_m3/model.py` (`msa_select`).
- Cache: per layer, 64-row K and V pages for the rank's KV heads (E4M3) and
  the shared index key in the paged indexer layout (E4M3 row + FP32 scale);
  69,360 B/token. Purely paged: exact prefix restores share pages. BF16 K/V
  has no exported kernel and is refused.
- Layouts: one RTX PRO 6000 coordinator, whose memory the Sparks' replies land
  in. `cuteafd plan` refuses an RTX 5090-class coordinator budget by name, and
  two-GPU head splits are not implemented. The routed experts run on Spark
  TP2, TP3, TP4 or TP6, the worlds the `minimax:nvfp4` package ships, at exact
  1536/1024/768/512-wide slices; bottom routed layers onboard to the RTX (TP1)
  only when a Spark rank would exceed its budget.
- Prefill: a single-sequence call runs as row units on the shared executor,
  with one Spark link carrying decode and every prefill wave. The plan admits
  four slots of the 4096-row unit inside the shared 8256-row reserve, and the
  engine chooses each call's depth at runtime (#762): 2 x 4096, 3 x 2752 or
  4 x 2048, from the Sparks' measured unit pace and its own reply wait. It
  starts at two waves because MiniMax has no measured estimate yet, and
  calibrates every depth before readiness. `PREFILL_WAVES` pins the depth
  (`PREFILL_WAVES=3` runs 3 x 2752 throughout). Scoring calls that project
  every row keep them in one unit.
- Decode: lane 0 of the shared Qwen scheduler, one lane: a second lane would
  need its own decode link, decode workspaces and drafter scratch. Decode runs
  eagerly; the Spark exchange's stream waits are never captured.
- Spark ranks: each holds typed owners, one decode owner at the package's
  covering m80 export and one prefill owner whose m4096 scratch is allocated
  once. The prefill owner buffers a unit per admitted slot, each queued unit
  with its own pinned route staging.
- Vision: Conv3d (2x14x14, FP32) patch embed, 32 CLIP layers (1280, 16 heads
  of 80) with 3D axial RoPE on 78 channels, a GELU projector to 6144 and a 2x2
  patch-merge GELU MLP. Images only: each smart-resizes to at most 451,584 px
  (576 merged tokens; the `dynamic_res` pinpoints are inactive, so large images
  shrink and never tile). Video is not supported. See [Images](#images).
- Speculation: DFlash2 by default, the bundled DSpark with `SPECULATOR=dspark`,
  or none with `SPECULATOR=off`. See [Speculation](#speculation).

## Serving behaviour

- Thinking follows the vendor's documented controls on chat, Responses and
  Anthropic requests. `thinking.type: enabled` opens `<mm:think>` in the prompt;
  `disabled`, `enable_thinking: false` or effort `none`/`minimal` turn it off
  (the prompt ends in the close marker); absent is adaptive (the model decides).
  An effort (`low`..`max`, or an Anthropic budget: <=1024 low, <=4096 medium,
  <=16384 high, <=32768 xhigh, else max) is rendered only while thinking is on.
- Tool arguments: a `strict: true` tool's call must match its schema or it is
  returned as text; other tools are read best effort, as vLLM's `minimax_m3`
  parser does (values coerce to the declared type when they parse, a repeated
  parameter becomes a list).
- Prefix cache: prompt-end and turn-end snapshots restore byte for byte into
  fresh pages (see [Host prefix tier](#host-prefix-tier)); partial prefix
  reuse is refused.
- Fidelity gate: `minimax-golden` against `golden.py --activation served-cache`
  (an independent re-implementation of the release's forward pass with the
  served FP8 K/V and index cache emulated) on the standard 64-window set's
  generated-role rows: top-1 >= 0.94, KL <= 0.03 nat and NLL at most 0.01
  above the reference. Rows outside the chat template (raw text with no turn
  marker, text after an end of turn) are near-degenerate and swing between any
  two arithmetics, so the gate scores generated rows only. The bench fidelity
  panel needs the set published as a dataset with a `PUBLICATIONS` row.

## Speculation

- DFlash2 (the launcher's default) uses
  [RadixArk/Minimax-M3.1-Flash-Dflash2](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2/tree/923bf1d4b567d5ccb71b0488c6e61b5e4794e92a)
  at revision `923bf1d4b567d5ccb71b0488c6e61b5e4794e92a` (`--speculator dflash2 --draft SNAPSHOT`)
  on the shared DFlash2 drafter: 5 Gemma-norm sliding-4096 layers, 64 query / 4 KV heads, NeoX
  RoPE on 64 channels, SwiGLU-OAI, blocks of an anchor and 7 drafts, taps 3/17/31/45/59; it
  borrows the target's embedding and BF16 head. Its model card states no license, so cuteafd
  does not distribute these weights: download them from
  [that page](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2) (the launcher stops
  and says so when they are missing). `SPECULATOR_MODEL_ID` and `SPECULATOR_MODEL_REVISION`
  default to that draft at its pinned revision.
- DFlash2 taps the FP32 residual's last 4096 rows of every step into the shared drafter's ring,
  on the target's stream. Its block attends as SGLang serves the release: a per-row 4096 window
  and causal block rows (no `is_causal` in the config). The GEMMs load as one E4M3 copy by
  default (`SPECULATOR_FP8=off` keeps the checkpoint's BF16), the rings as FP8 K/V
  (`DRAFT_KV_CACHE=bf16`). The selector scores each top-16 candidate as its unary logit plus
  the BF16 edge in FP32, as SGLang's DFlash2 selector does. Every request drafts its whole
  room; the shared scheduler then prices each draft's selector features (margin, candidate
  probability, entropy, rank) through the shared generic `SelectorFit`. No MiniMax fit ships;
  `SPECULATION_TRACE` writes the `draft_confidence` records that
  `scripts/bench/calibrate-draft-confidence.py` fits from a fixed-7 run (`SPECULATOR_DRAFTS=7`).
  `minimax-golden --draft SNAPSHOT --draft-replay P` replays the drafter on a golden window's
  taps, and `--draft-oracle DIR` compares it with
  `python/reference/families/minimax_m3/dflash2/reference.py`.
- DSpark (`SPECULATOR=dspark`) is the drafter bundled with the release (`dspark/`): 5 dense
  sliding-4096 layers, blocks of 7, Markov rank 256, a confidence head and the same taps. Its
  draft length uses the shared `simple` policy (`DRAFT_POLICY`/`DRAFT_ROW_COST` select others,
  `SPECULATOR_DRAFTS=N` fixes the depth).
- Drafter context rings default to the requests that can draft together: concurrency, at most
  half the 64-row decode step (32 at the shipped shapes). A request admitted while every ring is
  busy decodes target-only.
- Speculation qualifies by strict lossless runs (`wip-cards.py --panels
  baseline,spec_lossless_set`, 10/10 ids and row hashes) for each drafter.

## Images

The bundled tower runs natively: BF16 cuBLAS GEMMs with FP32 accumulation,
the b12x D80 varlen attention export (build with
`CUTEAFD_WIP_VISION_ATTENTION_AOT=ON`) and q/k rotated in FP32 as MiniMax's
serving applier does. Preprocessing is the release's: smart resize on a
28-pixel factor to at most 451,584 px, so one image is at most 2,304 patches
and 576 merged tokens. The tower takes one image per encode and is sized for
it: about 201 MiB of scratch with its BLAS workspace.

`--vision auto` (the launcher's `VISION=auto`, the default) lets the plan
place the tower: a Spark rank with room beside its experts hosts it (rank 0
at TP3, TP4 and TP6), and otherwise the coordinator GPU keeps it whenever a
KV pool is left (TP2). `--vision-peers` with `expertd-native --encoder` on the
ranks hosts it on the Sparks explicitly; `VISION=off` serves text only.
Preprocessing parity on the CPU
(`python/tools/qualify/minimax_m3/processor-parity.py`, the release's own
processors): patches, grids and prompt expansion are bit-identical for 16
images. On hardware, `scripts/qualify/minimax_m3/qualify-native-vision.py`
on the `generate-media-fixtures.py --set semantics` fixtures checks the served
path (shape counts and colours, a four-quadrant image, a banner, a chart, a
terminal, the order of two images). An exact repeat of an image request
restores the whole prompt; leading media shared with an earlier, different
prompt is not a hit, since MiniMax restores prompt-end and turn-end snapshots
only.

## Planned pools

`cuteafd plan --layout` on the release headers with a production
`PROGRAMS.json`, at the example configs' shapes: one RTX PRO 6000, 64
sequences, the adaptive prefill (four 4096-row slots), the checkpoint's 1M
context, FP8 K/V, the default 114 GiB per Spark rank, automatic vision and the
default 32 drafter rings. No Spark package layout sits beside the manifest, so
each rank's prefill scratch is the 512 MiB estimate. Pools are offline
forecasts; serving re-admits measured free memory.

| Sparks | Vision tower | Routed layers on the RTX | KV pool, speculator off | KV pool, DFlash2 (E4M3) | Spark GiB/rank |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | GPU | 0 | 1,134,912 | 1,080,896 | 112.2 |
| 3 | Spark rank 0 | 0 | 1,161,536 | 1,107,456 | 78.0 |
| 4 | Spark rank 0 | 0 | 1,160,064 | 1,105,984 | 59.9 |
| 6 | Spark rank 0 | 0 | 1,157,120 | 1,103,104 | 41.9 |

The Spark figure is the busiest rank, rank 0 with the tower from TP3 up.

## Allocation ledger

`serve-minimax` and `cuteafd plan --layout` admit one coordinator through the
same recipe (`plan::families::minimax::admission::Recipe`): the server sizes it
against measured free memory and the rings its Spark link registers, the
planner against the card's budget with its estimated CUDA context and code.
Without the image's `PROGRAMS.json` the planner refuses, as Step-5's does.

- Logits are `logit_rows x 200,064 x 4` bytes: the 64-row decode tile, or seven
  rows per drafting ring if more; every prefill row only with
  `--full-prefill-logits`.
- The per-row buffers hold the prefill span: the 8256-row reserve that every
  admitted depth's units are cut from (a pinned shape holds its own waves x
  unit). Each unit views its own rows.
- Program scratch is one arena at the largest program's need (the prefill
  scorer's 134.8 MB). Prefill tiles each call through 1024-row index-prefill
  and attention-extend launches over index keys staged for the whole context
  (132 B/token: 138 MB at 1M).
- Decode runs eagerly in 64-row tiles: no startup decode graphs are reserved.
- RoPE tables are 64 MiB each for cos and sin at 1M context; `--max-context`
  shrinks them along with the supported context.
- DSpark: 2.30 GB of weights, 40 MiB of BF16 K/V rings per drafting ring, and
  taps holding a call's five target layers.
- DFlash2: the shared drafter's own admission (`families::minimax_m3::dflash::DflashMemory`,
  planner and server alike): 2.22 GB of E4M3 GEMMs and BF16 norms, kernels and selector
  codebooks (4.11 GB in BF16), 21 MiB of FP8 K/V rings per drafting ring (40 MiB in BF16),
  361 MB of taps and context-update buffers for a 4096-row tail, and its step buffers (205 MB of
  logits at 32 sequences). FP8 packing holds one 377 MB BF16 source matrix (`fc`) at load.
- Spark intake: one BF16 device range per rank holding the reserve, where the
  NIC lands every reply; each call's return slots are cut from it at the
  call's depth, and each wave's landing slice is placed per request.
- Each Spark rank holds its expert slice, transport, its typed owners and the
  runtime residual. The owners' scratch is the package's exported value when
  the package carries the world's layout beside the manifest, else 512 MiB at
  the m4096 export (scaled by export) until worker admission measures it.
  Bottom routed layers onboard to the RTX (TP1) only when a rank would exceed
  its budget; a Spark-hosted tower that alone would make onboarding necessary
  is refused by name instead.
- `--embedding-placement host` maps the 2.46 GB BF16 input table from host
  memory; the untied 2.46 GB output head stays on the GPU.

## Host prefix tier

Retained prompt and turn snapshots share the device page pool with live
requests; the pool evicts its least recently used snapshot when an admission
needs pages or a bank (`--prefix-cache-entries`, 20) is full. The shared host
tier keeps evicted snapshots in pinned RAM (`HOST_CACHE_BYTES`, launcher
default `auto`), and a returning conversation restores them into fresh device
pages byte for byte. MiniMax runs it `on-evict` (`HOST_CACHE_STORE`): nothing
is copied until the device evicts a snapshot, and that eviction waits for its
copy; `on-retain` copies each snapshot when it is kept. A page is 180 segments
(K, V and index planes of 60 layers, 4.44 MB per 64 tokens); snapshots under
512 tokens stay on the device.

## Known limits

- One RTX PRO 6000 coordinator only: the RTX 5090 is refused by name, and
  two-GPU head splits are not implemented.
- No video input, no partial prefix reuse and no BF16 K/V kernel.
- MiniMax has no measured prefill estimate: the depth starts at two waves and
  the runtime controller measures from there.
- A Spark rank's owner scratch is an estimate unless the Spark world's package
  layout is installed beside `PROGRAMS.json`.
- `--full-prefill-logits` charges logits for the whole span (about 6.6 GB at
  8256 rows).
- The DFlash2 draft is not distributed; download it from its model page.

## Changelog

| Version | Date | Change | Basic eval |
| --- | --- | --- | --- |

## Additional benchmarks

| Engine version | Date | Profile | Hardware | Report |
| --- | --- | --- | --- | --- |

_No additional benchmark reports yet._
