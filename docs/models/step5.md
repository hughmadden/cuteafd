# Step-5

Step-5-Preview serves text from its standard Hugging Face checkpoint headers.
The qualified WIP layouts are official FP8 on two RTX PRO 6000 GPUs plus six
Sparks, and official NVFP4 on one RTX PRO 6000 plus four Sparks. NVFP4 experts
use the checkpoint's calibrated W4A4 math, including decode.

## Native MTP

Native MTP is enabled by default at depth 3 for both layouts. The three
predictor stages are checkpoint layers 92-94; no separate drafter download is
needed. `SPECULATOR=off` disables MTP in the launcher; direct `serve-step5`
users can pass `--mtp 0`. Use `SPECULATOR=mtp` and `SPECULATOR_DEPTH=1`, `2`
or `3` to select another depth explicitly.

The planner defaults to the same three stages. `--native-mtp-layers` selects
the planned depth; `--no-speculator` or `--speculator off` excludes predictor
weights and state. The launcher forwards its effective depth to the planner,
including zero when disabled. Admission charges predictor weights, active KV
and hidden history, retained predictor marks, and the shared scratch arena
before sizing the target KV pool. Enabling MTP therefore reduces the available
target pool rather than allocating predictor state outside the budget.

Draft/verify/accept and rollback use the shared scheduler. Prompt-end and
turn-end snapshots retain all predictor stages and their canonical frontiers.
A drafting request reuses only predictor-complete ancestors; target-only
requests can still reuse target-only snapshots. Restoring a snapshot must
preserve target KV/CSA, predictor histories and cursors exactly.

## Launch

Start with `examples/configs/step5-fp8-max.config` or
`examples/configs/step5-nvfp4-min.config`. Replace the sample fabric endpoints
and select a sealed WIP slot containing `serve-step5`. Keep media disabled;
Step-5's bundled vision execution is not yet qualified. Pool sizing remains
automatic, and admission refuses configurations below the 256K context floor.
