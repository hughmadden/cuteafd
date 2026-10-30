# New-build taxonomy: tier every gap before writing code

Each gap in the spec gets one tier. The tier decides who does it, how it is tested, and whether it
blocks the smoke. Costs are from the port histories in `evidence.md`.

| Tier | What | Typical cost | Test | Blocks the smoke? |
| --- | --- | --- | --- | --- |
| **T0 data** | a row in an existing table: `ExpertGeometry` const, SparkInfer `GEOMETRIES` row, packager choice, CMake regex, Spark worlds, slot census rows, launcher arms, `checkpoint-family.py`, example config, ModelOpt algo spelling | minutes each | existing tests, plus one asserting the new row | yes, but it is cheap |
| **T1 recompile** | an existing program class at a new geometry (new frozen dims, scale, capacity) | hours, with the export | GPU oracle against dense FP32 at the new geometry | yes |
| **T2 compose** | a new program variant chained from existing primitives (GLM's K6 path: stage programs around plain FP8 projections) | hours to a day | GPU oracle per program; layer replay | yes |
| **T3 pointwise kernel** | a genuinely new elementwise or reduction op (MiniMax's `_minimax_kernels.py`) | under a day with its oracle | GPU oracle on SM120 and SM121 | yes |
| **T4 engine semantics** | a new per-layer behaviour in family code: KV aliasing, NoPE layers, an indexer source map, a dense layer mix | hours to days | CPU `Ops` oracle at tiny geometry; per-layer kind asserted in the golden | yes |
| **T5 format, tokenizer, API** | a checkpoint layout, a tokenizer, a chat encoding or parser | about a day | contract tests; template goldens from the vendor's own renderer | the checkpoint and tokenizer do; full tool grammars don't |
| **T6 subsystem** | a drafter with a new backbone or head, a vision or audio tower, new expert numerics | days | the subsystem's own gates | **no: defer** |

## How to do each tier fast

**T0.** Grep a sibling family's name across the whole tree (`git grep -n step5 -- ':!*/step5/*'`),
copy its hunk, change the values. Never rely on a default: shared code fails open for unknown
families (`catalog.md` §3).

**T1.** Add a geometry instance next to the family's existing ones in SparkInfer, export at the
exact capacities, run its tests in the dev container, and land on `sparkinfer-glmrt` master (or
your fork branch) **before** bumping the pin and tree lock: a fresh worktree cannot fetch an
unpushed pin. Then add the exporter geometry, the CMake switch, the WIP scope, and a test that the
emitted scope is accepted by the CMake list (MiniMax lost its first hardware hour to a mismatch).

**T2.** Prefer composition over new kernels. The decomposition must keep the reference's rounding
points; write them into the spec before composing.

**T3.** Fold a pointwise op into an existing producer only if its rounding point allows. Example:
a per-head query scale can be applied to the absorbed query, but that moves rounding from FP32
logits to the BF16 query, so judge it by KL rather than bit equality. Otherwise write a standalone
kernel with a dense FP32 oracle.

**T4.** Encode every per-layer property as data in the family spec (kind, KV owner, index source,
RoPE or none, dense or MoE) and assert it from the golden's first run. Step-5's dense-FFN layer
silently never ran until the golden showed NLL 2.687 against 2.492 and a final-layer cosine of 0.785.

**T5.** Put the format behind `Checkpoint` (`L/plan/checkpoint.rs`) as a `CheckpointFormat` with an
in-memory config and an alias name map, following the EXLR8 inventory. Nothing is repacked: a
`CheckpointTensor` keeps its native shard and byte offset. Route the bypassing readers through it.
For **planning only**, before that exists, `scripts/checkpoint_view.py` presents the checkpoint as an
HF snapshot through hard links plus a translated `config.json`. That is enough for `cuteafd plan`;
the shared expert reader and MLA packer still see native tensor names, so it cannot serve.
Tokenizers follow `L/tokenizer/tiktoken.rs`. Chat encodings follow Kimi's native encoder, with goldens
rendered by the vendor's own library.

**T6.** Write the spec section, mark it deferred, and serve without it (`SPECULATOR=off`, `VISION=off`).
Turn a drafter on in a smoke only if the shared drafter already supports its backbone and head
unchanged.
