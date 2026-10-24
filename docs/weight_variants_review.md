# Per-Request Weight Variants / Weight Overlays — Design & Implementation Review

- **Date:** 2026-10-24 AEDT (updated for post-PR#2 merge tips `4a90efa07`, `ae00d4733`, `7edde729a`)
- **Author:** Gemini 4 Argon
- **Repositories Inspected:**
  - **Design Specifications (private design notes, outside this repository):**
    - `weight-variants-contract.md` (v1 cross-branch interface contract, 2026-10-24)
    - `weight-variants-dsf1017-glm53.md` (DeepSeek V4.1 Flash 1017 & GLM-5.3 Full + GLM-5.3 Flash §4b, 2026-10-24)
    - `glm-flash-weight-variants.md` (GLM-5.3 Flash design, 2026-10-23)
  - **CUDA Kernel & AOT Export (local `sparkinfer-steer-rows` checkout):**
    - Branch `main` (`a1504a4e`): `b12x/integration/cuteafd/steer_rows.py`, `scripts/cuteafd_aot/csrc/cuteafd_steer_rows.h`, `scripts/cuteafd_aot/csrc/steer_rows_aot.cc`, `scripts/export_b12x_steer_rows_aot.py`, `tests/test_cuteafd_steer_rows.py`
  - **`cuteafd` Branches / PRs (this fork):**
    - Base branch `origin/hugh/weight-variants` (`8d6478143`, with PR #2 `claude/wv-kernel-87efde` @ `5961f2b82` merged): `cuteafd-core/src/steer_rows.rs`, `cuteafd-ffi/src/shared/steer_rows.rs`
    - PR #1 `claude/wv-core-glm5-60c2f0` (`4a90efa07`): shared core + GLM-5.3 full adapter
    - PR #4 `claude/wv-glmf-b1811f` (`ae00d4733`): GLM-5.3 Flash adapter + `SiteOp::Delta`
    - PR #3 `claude/wv-dsv41-fbe397` (`7edde729a`): DeepSeek V4.1 (`deepseek_v41`) adapter

---

## 1. Executive Summary

The weight-variants / per-request weight-overlays architecture is a high-leverage design: by factoring output-side projection edits $W' = (I - A B^\top) W$ into an in-place BF16 post-projection hook $y \leftarrow y - A (B^\top y)$ indexed by a per-token-row variant slot `vidx[row]`, a single `cuteafd` daemon serves the base checkpoint (`variant = 0`) and up to 7 derived variants (`variant = 1..=7`) concurrently in mixed prefill/decode batches without duplicating base weights, modifying Spark routed-expert weights, altering RoCE frames, or multiplying CUDA graph keys.

The 14:02–14:38 AEDT commits across PRs #1, #3, and #4 (`173b88c83`, `5bc027255`, `dbe3a3f22`, `38f92fe44`, `d7b5942d1`, `7edde729a`) cleanly resolved the initial kernel-binding seam:
- Replaced the temporary `steer_rows` / `add_rows` `bail!` stubs in all three families with `cuteafd_ffi::steer_rows::{SteerRows, AddRows}` bound at table upload time (prior to CUDA graph capture).
- Unified `cuteafd_core::variants::{steer_rows_ref, add_rows_ref, bf16_rne, bf16_to_f32}` as re-exports of the bit-exact 512-thread tree-reduction reference in `cuteafd_core::steer_rows`.
- Added the `set -u` guard and `VARIANTS` support for `glm5_flash` and `deepseek_v41` in `scripts/launch/run-family.sh`, and replaced exhaustive `f32_to_e4m3` search with a binary search over `E4M3_MAGNITUDES`.

Inspecting the current branch tips (`4a90efa07`, `ae00d4733`, `7edde729a`) reveals **1 `P0` cross-PR merge conflict (`PR #3` vs `PR #4`)**, **4 `P1` performance, memory-safety, and prefix-cache issues**, and **4 `P2` numerical, loader, and gateway edge cases**:

| ID | Severity | Component / Branch | Finding Summary | Impact |
|---|---|---|---|---|
| **P0-1** | **P0 (Cross-PR Conflict)** | `wv-glmf-b1811f` (PR #4) vs `wv-dsv41-fbe397` (PR #3) | (a) `SiteOp::Delta` is `Delta { rows, cols, weight }` in PR #4 (`c00ab1977`), whereas PR #3 matches on `SiteOp::Delta {}` (`v41_variant.rs:1462`, `v41_variants.rs:951`); (b) `scripts/launch/run-family.sh` diverged between PR #3 (`d7b5942d1`: `variant_launch_args`) and PR #4 (`38f92fe44`: `_variant_keys`); (c) `scripts/tests/test_variant_refusal_smoke.py` has different contents in PR #1/#4 vs PR #3 | Compile error (`E0027` pattern fields) and git merge conflicts in `run-family.sh` and `test_variant_refusal_smoke.py` when combining PR #3 and PR #4 |
| **P1-1** | **P1 (Performance Bug)** | `wv-glmf-b1811f` (`glm5_flash` L44 `SiteOp::Delta`) | (a) L44 `fp8_w8a16` delta GEMVs run unconditionally on **every decode step even when 100% of active sequences are base (`vidx == 0`)** (`43.25 MiB` wasted HBM read/step); (b) `slice_rows = args.decode_rows` (`64`) slices a 4096-row prefill into **64 separate GEMVs**, re-reading L44 $\Delta W$ 64 times (**`2.768 GiB` redundant HBM traffic and 256 kernel launches on L44 alone**) | Adds `~29–35 µs` to every base decode step and `~1.8–2.1 ms` (`2.768 GiB` HBM reads) to every 4096-token variant prefill chunk |
| **P1-2** | **P1 (GPU Memory Safety)** | `sparkinfer-steer-rows` (`steer_rows.py`) | `SteerRows.kernel` and `AddRows.kernel` check `v > 0` with **no upper-bound check `v <= variants`** (`variants` is validated on host in `SteerRowsShape` / `AddRowsShape` but never passed to the CUDA kernel) | Out-of-bounds global memory reads from `k[v]`, `a`, `b`, or `on[v]` if any `vidx[row] > variants` |
| **P1-3** | **P1 (Prefix Cache & Overhead)** | `wv-core-glm5-60c2f0` (`cuteafd-engine/src/prefix/namespace.rs`) | `VariantPlan.namespace: u64` (SHA-256 content hash) is unused by `PrefixCache`; `namespaced(tokens, tag)` keys on the process-local index `1..=7` and clones `tokens.to_vec()` (`Vec<u32>`) 6–7 times per request lifecycle | `1.5–3.0 MiB` of repeated `Vec<u32>` heap allocations per 64K–128K turn; requires re-keying if host prefix tier is shared or persisted across restarts |
| **P1-4** | **P1 (Multi-Wave / Multi-GPU Correctness)** | `wv-core-glm5-60c2f0` (`glm5/engine.rs`, `layers.rs`) | (a) Single `self.step_variant: Cell<Option<u8>>` shared across pipelined prefill waves forces base waves to launch `steer_rows` when paired with a variant wave; (b) `variants` buffer allocated unconditionally; (c) latent 2-GPU bugs behind `devices.len() == 1` (`rank != 0` early return on unreduced head-split `o_proj`/`shared_on`, missing `DenseFfn` steer in `PipelineKernels::ffn`) | Redundant `steer_rows` launches on mixed prefill waves; silent half-projection steering if 2-GPU guard is lifted |
| **P2-1** | **P2 (Loader Math)** | `wv-core-glm5-60c2f0` (`cuteafd-loader/src/variants.rs:525`) | `TensorSource::matrix` computes FP8 scale block size via `rows.div_ceil(srows)` instead of power-of-two block size (`128` / `32`) | Misaligned scale block boundaries when matrix dimensions are not exact multiples of block size |
| **P2-2** | **P2 (Loader Precision)** | `wv-core-glm5-60c2f0` (`cuteafd-loader/src/variants.rs:248-283`) | `d_times` and `dt_times` round `f64` dot-product accumulators to `f32` between power iterations before `orthonormalize` promotes back to `f64` | Avoidable `f32` rounding drift in subspace iteration when singular values are close |
| **P2-3** | **P2 (Diagnostics)** | `wv-core-glm5-60c2f0` (`cuteafd-loader/src/variants.rs:138`) | `VerifyReport.alphas` records only the diagonal `c[i * k + i]` of $C = -M_1 M_2^{-1}$, ignoring off-diagonal coupling for $k > 1$ | Under-reports effective projection energy when $k > 1$ basis vectors have off-diagonal coupling |
| **P2-4** | **P2 (Gateway Routing)** | `wv-core-glm5-60c2f0` (`cuteafd-api/src/gateway/models.rs:117`) | `ModelMap::resolve_variant` checks `self.accepts(requested)` before `self.variants`; `validate_specs` only checks `served_model_id`, not `--names-file` or official aliases | Variant `public_id` colliding with a base alias silently routes to base (`variant = 0`) |
| **P2-5** | **P2 (Missing Branch Docs)** | All `wv-*` branches (`docs/`, `PLAN.md`) | Zero files under `docs/` (`docs/client-apis.md`, `docs/models/{glm5,glm5_flash,deepseek_v41}.md`) or `PLAN.md` were updated in `cuteafd`; `cuteafd-core/src/variants.rs:7` references `designs/weight-variants-contract.md`, which lives in `recipes` (`cuteafd-upstream/designs/`), not in `cuteafd` | No in-repo operator/API documentation for `--variant`, `VARIANTS`, `/v1/models` `"variant"` metadata, or `/v1/stats` `"variants"` counters |

---

## 2. What Is Architecturally Sound and Well-Executed

### 2.1 Post-Projection Output-Side Steering ($y \leftarrow y - A (B^\top y)$)
1. **Zero Base-Weight Duplication & Zero RoCE Impact:**
   - Alternate checkpoints (`example-org`, `example-org`, `example-org`, `example-org`) apply output-row projections $W' = W - \sum_{r=1}^k \alpha_r u_r (u_r^\top W) = (I - A B^\top) W$ to coordinator-resident attention output projections (`o_proj`, `wo_b`) and shared/dense FFN `down_proj`.
   - Applying $y \leftarrow y - A (B^\top y)$ in-place to the BF16 sublayer output `[rows, H]` immediately prior to residual/mHC fusion (`glm_norm`, `mhc_post_pre`, `hc_post`) avoids touching quantized weights (`FP8` / `INT4` / `EXL3`), leaves Spark routed-expert weights and RoCE dispatch/combine packets untouched, and requires only `15–68 MiB` of device memory per variant.
2. **Zero Additional CUDA Graph Keys & Bit-Exact Base Preservation:**
   - Because `vidx: [rows] i32` is uploaded into a fixed-address workspace buffer alongside `positions` and `slots`, and each site's `a`, `b`, `k` tables live at static device pointers with kernel modules pre-loaded at table upload time (`library.steer_rows()`, `library.add_rows()`), captured decode and target-verify graphs require **zero additional graph keys** and never lazy-load modules during capture.
   - In `SteerRows.kernel` (`b12x/integration/cuteafd/steer_rows.py:953-960`) and `AddRows.kernel` (`steer_rows.py:1289-1296`), when `vidx[row] == 0` (`kv == 0` / `enabled == 0`), the CTA exits before issuing any global load or store on `x[row, :]`. Base rows (`variant = 0`) are therefore preserved bit-for-bit (including `-0.0` `0x8000` and arbitrary NaN payloads) in mixed batches.

### 2.2 Hybrid Low-Rank + Stored-Delta (`SiteOp::Delta`) for GLM-5.3 Flash (`example-org`)
- In `example-org/GLM-5.3-Flash-alternate-weights`, 56 of 58 modified tensors (`L12..=43` `o_proj` and `L20..=43` `shared_experts.down_proj`) are rank-1 (`next_share <= 1.2e-3`), while **L44 `self_attn.o_proj` (`[4096, 8192]`) and L44 `mlp.shared_experts.down_proj` (`[4096, 2048]`)** are full-rank deltas (`rank 48` still leaves 28% residual energy).
- Extending `SiteOp` with `SiteOp::Delta` (`cuteafd-loader/src/families/glm5_flash/variants.rs:265-351`) and quantizing those two L44 deltas into `Fp8Weight` (`43.25 MiB` total on device) while reading the sublayer's input activation directly from `w.scratch` at `kda_norm_offset(rows)` and `ffn_hidden_offset(rows)` (`cuteafd-daemon/src/families/glm5_flash/engine.rs:3482-3522`, verified exact against `glmf.py` `_KdaW8` and `_GlmfFfnW8` 1024-byte aligned layouts) is a clean, minimal-memory solution for non-low-rank tail layers.

### 2.3 DeepSeek V4.1 FP8 Block-32 `rebuild_alpha` Search & Shared-Direction Table Deduplication
- In `cuteafd-loader/src/families/deepseek_v41/v41_variant.rs:889-997` (`rebuild_alpha`), the loader accounts for the ~3% downward bias (`~3.35` vs true `3.50`) that standard least-squares $\alpha = -\|u^\top \Delta W\|_2^2 / (u^\top \Delta W W^\top u)$ suffers under block-32 E4M3 quantization noise by running a coarse-to-fine grid search (`step 0.01` then `0.001`) that maximizes exact FP8 byte agreement against the variant's UE8M0 scales (achieving 84.5–86.7% exact FP8 byte reproduction vs 61% stock).
- In `cuteafd-daemon/src/families/deepseek_v41/v41_variants.rs:42-81` (`V41SteerTables::new`), identical `(a, b, k)` site tables across layers are deduplicated by raw byte comparison on host before upload, collapsing 26 layers of shared-direction `wo_b` steering tables into a single `80 KiB` device allocation per GPU.

---

## 3. Detailed Findings & Recommendations

### P0 — Cross-PR Merge Conflicts (`PR #3` vs `PR #4`)

#### P0-1: `SiteOp::Delta` Enum Variant, `run-family.sh` Helpers, and `test_variant_refusal_smoke.py` Collisions
- **Locations:**
  - `cuteafd-core/src/variants.rs`:
    - PR #1 (`4a90efa07`) & PR #3 (`7edde729a`): `SiteOp::Delta {}`
    - PR #4 (`ae00d4733`, commit `c00ab1977`): `SiteOp::Delta { rows: u32, cols: u32, weight: Vec<u16> }`
  - PR #3 (`7edde729a`): `cuteafd-loader/src/families/deepseek_v41/v41_variant.rs:1462` and `cuteafd-daemon/src/families/deepseek_v41/v41_variants.rs:951` match `SiteOp::Delta {}`
  - `scripts/launch/run-family.sh`:
    - PR #3 (`d7b5942d1`, `7edde729a`): extracts top-level `variant_launch_args "$hf_home/hub" mode|nomode`
    - PR #4 (`38f92fe44`): extracts `_variant_keys` inside `_allowed_keys` and widens `[[ "$family" == glm5 || "$family" == glm5_flash ]]` in the coordinator block
  - `scripts/tests/test_variant_refusal_smoke.py`: added with different contents in PR #1/#4 (testing `scripts/bench/variant-refusal-smoke.py`) vs PR #3 (testing `scripts/bench/deepseek_v41/variant-refusal-smoke.py`)
- **Recommendation:**
  1. When rebasing/merging PR #3 and PR #4 onto the shared target branch, keep PR #4's `SiteOp::Delta { rows: u32, cols: u32, weight: Vec<u16> }` and update `v41_variant.rs:1462` and `v41_variants.rs:951` to match `SiteOp::Delta { .. }`.
  2. Combine PR #4's `_variant_keys` helper (used by `glm5`, `glm5_flash`, and `deepseek_v41`) with PR #3's `variant_launch_args "$hub_dir" mode|nomode` helper (used by `glm5`, `glm5_flash`, and `launch_deepseek_v41`).
  3. Rename PR #3's test file to `scripts/tests/test_v41_variant_refusal_smoke.py` (or merge both test files).

---

### P1 — Performance, Memory-Safety & Prefix-Cache Defects

#### P1-1: GLM-5.3 Flash L44 `SiteOp::Delta` Runs Unconditionally on 100%-Base Decode Batches and Slices 4096-Row Prefills into 64 Separate GEMVs (`2.768 GiB` Redundant HBM Traffic)
- **Locations (`wv-glmf-b1811f` @ `ae00d4733`):**
  - `cuteafd-daemon/src/families/glm5_flash/engine.rs:3683-3684`, `4481-4498` (`variant_site`)
  - `cuteafd-daemon/src/families/glm5_flash/variants.rs:208-226` (`GlmfVariants::delta`), `285-289` (`launches`)
  - `cuteafd-daemon/src/families/glm5_flash/mod.rs:1693` (`GlmfVariants::upload(..., args.decode_rows, ...)`)
- **Problem 1 — Unconditional `43.25 MiB` Weight Read on 100%-Base Decode Steps:**
  - In `variants.rs:287`, `launches(tables.decode, &tables.variants)` returns `true` for **every decode step** (`if decode { return true; }`) so that captured CUDA graphs keep a single graph topology regardless of batch composition.
  - On Layer 44 (`SiteKind::Attn` `[4096, 8192]` = `34.6 MiB` FP8 + scales, and `SiteKind::SharedExpert` `[4096, 2048]` = `8.65 MiB` FP8 + scales), `variant_site` calls `variants.delta(...)` (`variants.rs:208-226`):
    ```rust
    for first in (0..rows).step_by(self.slice_rows) {
        let n = self.slice_rows.min(rows - first);
        unsafe {
            site.weight.apply(library, x.byte_add(first * k * 2), out.buffer.ptr, true, n, 0, h, scratch, stream)?;
            add.launch_raw(y.byte_add(first * h * 2).cast(), out.buffer.ptr.cast_const().cast(),
                vidx.byte_add(first * 4).cast(), site.on.buffer.ptr.cast_const().cast(),
                cuteafd_ffi::steer_rows::AddRowsShape { rows: n, h, ld_y: h, ld_d: h, variants: self.slots }, stream)
        }.with_context(|| format!("delta at {key}"))?;
    }
    ```
  - While `add.launch_raw` early-exits in hardware when `on[vidx[row]] == 0`, `site.weight.apply(...)` (`fp8_w8a16`) has **no `vidx` or `on` check**! It unconditionally reads all `43.25 MiB` of L44 $\Delta W$ weights from HBM on **every decode step even when 100% of active sequences are base (`vidx == 0`)**, only for `add_rows` to multiply the resulting `out` buffer by `0` and discard it. On RTX 5090 (`~1.5 TB/s` effective HBM bandwidth), that wastes **`~29–35 µs` on every single base decode step** whenever a delta-bearing `--variant` is configured.
- **Problem 2 — 64x Redundant Weight Reads (`2.768 GiB`) on 4096-Token Variant Prefills:**
  - In `glm5_flash/mod.rs:1693`, `GlmfVariants::upload` sets `slice_rows = args.decode_rows` (`64`), allocating only `64 * 4096 * 4 = 1 MiB` (`2 MiB` for `decode_rows = 128`) for `self.out`.
  - When a `4096`-token prefill chunk containing any variant token executes Layer 44, `for first in (0..4096).step_by(64)` runs **64 iterations**!
  - Because `site.weight.apply` (`fp8_w8a16`) streams the entire `[4096, 8192]` (`34.6 MiB`) and `[4096, 2048]` (`8.65 MiB`) weight matrices from HBM on every iteration of the loop, a single 4096-row prefill chunk on Layer 44 reads $64 \times 43.25\text{ MiB} = \mathbf{2.768\text{ GiB}}$ of HBM traffic and issues **256 kernel launches** (`64 * 2` per site $\times 2$ sites), adding **`~1.8–2.1 ms` of pure memory-bandwidth stall to a single layer**!
- **Recommendation:**
  1. **Reuse `w.scratch` (or size `slice_rows` for prefill) to eliminate the 64-chunk prefill loop:**
     - During L44 `Attn`, `w.scratch` above `kda_norm_offset(rows) + rows * 8192 * 2` (`64 MiB` into a `240+ MiB` FFN scratch buffer) is completely unused during attention and can hold the `f32` `out` buffer (`4096 * 4096 * 4 = 64 MiB`) and `fp8_linear` scratch (`~36 MiB`) with **zero extra VRAM allocation** and **1 GEMM launch instead of 64**!
     - During L44 `SharedExpert`, `ffn_hidden_offset(rows)` uses only the first `rows * 2048 * 2 = 16 MiB` of `w.scratch`, leaving $>200\text{ MiB}$ of `w.scratch` free right above it with **zero extra VRAM allocation** and **1 GEMM launch instead of 64**!
     - Doing a single `rows`-wide `fp8_w8a16` + `add_rows` call per L44 site cuts L44 variant prefill HBM traffic by **64x** (`2.768 GiB` $\to$ `43.25 MiB`) and kernel launches from **256 to 4**.
  2. **Skip L44 `fp8_w8a16` when no row in the batch activates `site.on`:**
     - For eager prefill (`decode == false`), check on host whether any row in `tables.variants` has `site.on_host[v] != 0` before calling `variants.delta`.
     - For captured decode (`decode == true`), either pass `(vidx, on, rows)` to a CTA-head early-return check in `fp8_w8a16`, or gate decode graph capture on a 1-bit `has_variant_in_batch` key (only 2 decode graph topologies—pure-base vs has-variant—which eliminates both `SteerRows` across all 56 sites and the `43.25 MiB` L44 `Delta` GEMVs on 100%-base decode batches).

---

#### P1-2: Missing Upper-Bound Check on `v = vidx[row]` in `SteerRows` and `AddRows` CUDA Kernels
- **Locations:**
  - `sparkinfer-steer-rows`: `b12x/integration/cuteafd/steer_rows.py:953-960` (`SteerRows.kernel`) and `1289-1296` (`AddRows.kernel`)
  - `wv-core-glm5-60c2f0`: `cuteafd-ffi/src/shared/steer_rows.rs:195-210` (`launch_raw`), `261-276` (`AddRows::launch_raw`)
  - `wv-dsv41-fbe397`: `cuteafd-daemon/src/families/deepseek_v41/v41_variants.rs:159`
- **Problem:**
  - In `SteerRows.kernel` (`steer_rows.py:956-959`):
    ```python
    v = vidx[row]
    kv = Int32(0)
    if v > Int32(0):
        kv = k[v]
    ```
    and in `AddRows.kernel` (`steer_rows.py:1292-1295`):
    ```python
    v = vidx[row]
    enabled = Int32(0)
    if v > Int32(0):
        enabled = on[v]
    ```
    there is **no upper-bound check** on `v` (`v <= variants`), whereas the Rust host reference (`cuteafd-core/src/steer_rows.rs:106`) checks `if v <= 0 || (v as usize) >= k.len() { continue; }`.
  - Although `SteerRowsShape` and `AddRowsShape` include `variants: usize`, that field is only used on the CPU to check `shape.variants <= MAX_VARIANT_SLOTS` and buffer byte lengths in `launch(...)`; `variants` is **never passed to `cuteafd_steer_rows_launch` or `cuteafd_add_rows_launch`**.
  - Furthermore, in `deepseek_v41` (`v41_variants.rs:159`), `k` is uploaded with exact byte size `(V + 1) * 4` (`8 bytes` when `V = 1`). Any out-of-range `vidx[row] > V` on device will read out-of-bounds global memory from `k[v]`, `on[v]`, `a`, and `b`.
- **Recommendation:**
  1. Pad device allocations of `k` and `on` to 8 `i32` slots (`MAX_VARIANTS + 1 = 8`, `32 bytes`) on all families (including `v41_variants.rs`), and guard `if v > Int32(0) and v <= Int32(7):` in `SteerRows.kernel` and `AddRows.kernel` (or pass `variants: i32` into the kernel and guard `v <= variants`).

---

#### P1-3: Prefix Cache `namespaced(tokens, tag)` Clones Full Prompt `Vec<u32>` 6–7 Times per Request & Does Not Use `VariantPlan.namespace`
- **Locations:**
  - `cuteafd-upstream/designs/weight-variants-contract.md:113-127` (§5 Prefix-Cache Namespace Keying)
  - `wv-core-glm5-60c2f0`: `cuteafd-engine/src/prefix/namespace.rs:1-81` (`namespaced(tokens: &[u32], tag: u8)`)
  - `wv-core-glm5-60c2f0`: `cuteafd-daemon/src/families/glm5/serve.rs:789, 809, 1089, 1699, 1705, 1984, 2117`
  - `wv-glmf-b1811f`: `cuteafd-daemon/src/families/glm5_flash/serve.rs:582, 833, 882, 916, 1791, 1898`
  - `wv-dsv41-fbe397`: `cuteafd-daemon/src/families/deepseek_v41/v41_native_serve/admission.rs:64, 187`, `scheduler.rs:583, 622, 690, 991, 1188`
- **Problem:**
  1. Commit `173b88c83` documented in `namespace.rs:14-19` that `namespaced(tokens, tag)` uses the process-local CLI index `tag: u8` (`1..=7`) rather than `VariantPlan.namespace: u64`, noting that a persisted or shared host tier would need to key by `VariantPlan.namespace`. Currently `VariantPlan.namespace` is only logged at startup.
  2. `namespaced(tokens, tag)` allocates and copies `tokens.to_vec()` (`Cow::Owned(Vec<u32>)`) on every call when `tag != 0` (6–7 call sites per request lifecycle across `peek`, `admit`, `capture_points`, `retain_prompt`, `park`, and `Turn` capture). At 64K–128K context, cloning `Vec<u32>` 6–7 times per turn allocates and copies `1.5–3.5 MiB` on the scheduler/admission thread per request.
- **Recommendation:**
  - Either store the namespaced token vector once on the request/turn struct at admission time, or pass `namespace: u64` (`0` for base, `plan.namespace` for variant `v`) directly into `PrefixCache` and `page_chain` seed mixing (`SEED ^ namespace`), which eliminates all `Vec<u32>` cloning and satisfies Contract §5 even for persisted/shared host tiers.

---

#### P1-4: Multi-Wave & Multi-GPU Defects in `glm5` (`wv-core-glm5-60c2f0`)
- **Locations (`wv-core-glm5-60c2f0` @ `4a90efa07`):**
  - `cuteafd-daemon/src/families/glm5/engine.rs:684, 1594, 1805, 2036-2066, 2267`
  - `cuteafd-daemon/src/families/glm5/engine/layers.rs:274`
- **Problems:**
  1. **Single `self.step_variant` Shared Across Multi-Wave Pipelined Prefill (`glm5/engine.rs:1805`):**
     - `step_layers` computes a single `self.step_variant.set(uniform_variant(&all))` across all `lanes` in the step. When 2 pipelined prefill units run concurrently and unit 0 is base (`0`) while unit 1 is variant (`1`), `uniform_variant(&all)` becomes `None`, forcing unit 0 (100% base rows) to launch `steer_rows` on every layer.
     - **Fix:** Store `variant_active: Cell<bool>` per `Workspace` and set it inside `put_tables` / `put_lane_tables` (matching `glm5_flash`'s per-workspace `tables.variant_active`).
  2. **Unconditional `variants` Buffer Allocation (`glm5/engine.rs:684`):**
     - `GlmEngine::workspace` allocates `variants: Dev<'a>` (`t * 4` bytes per workspace) even when `args.variants.is_empty()`, while `planner_input` only accounts for variant bytes when `!args.variants.is_empty()`.
     - **Fix:** Allocate `variants: Option<Dev<'a>>` only when `self.variants.is_some()` (matching `glm5_flash` and `deepseek_v41`).
  3. **Latent 2-GPU Bugs Behind the `devices.len() == 1` Guard (`glm5/engine.rs:2063`, `layers.rs:274`):**
     - `GlmEngine::steer` has `if rank != 0 { return Ok(()); }`. However, in `o_proj` (`engine.rs:2267`) and `shared_on` (`engine.rs:1594`), `self.steer(rank, ...)` is called on each rank's partial buffer (`delta` / `shared`) **before** the cross-GPU all-reduce. Under 2-GPU head-split, skipping `rank == 1` steers only the rank-0 half of the dot product (`(I - A B^T) y_0 + y_1 != (I - A B^T)(y_0 + y_1)`). Either steer on both ranks before the reduce or steer on rank 0 after the reduce.
     - In `PipelineKernels::ffn` (`glm5/engine/layers.rs:274`), `e.steer(0, w, index, SiteKind::DenseFfn, w.shared.buffer.ptr, w.active_rows.get())` is missing after `e.dense_ffn(0, w, layer)`.

---

### P2 — Numerical, Loader & Gateway Polish

#### P2-1: `TensorSource::matrix` Block-Scale Divisor Uses `div_ceil` Instead of Block Size
- **Location:** `cuteafd-loader/src/variants.rs:532`
- **Problem:**
  ```rust
  let (br, bc) = (rows.div_ceil(srows), cols.div_ceil(scols));
  ```
  When `rows` is not an exact multiple of the scale block size $B$ (e.g., `rows = 129, B = 128`, so `srows = 2`), `rows.div_ceil(srows)` evaluates to `129.div_ceil(2) = 65` instead of `128`, shifting scale block boundaries across the entire tensor.
- **Recommendation:**
  - Infer the block size from the standard power-of-two block sizes (`128`, `32`, or `1`: e.g. `if srows == 1 { rows } else if srows == rows.div_ceil(32) { 32 } else { 128 }`, matching `v41_variant.rs:480-499`).

#### P2-2: Single-Precision (`f32`) Truncation Inside Subspace Power Iteration
- **Location:** `cuteafd-loader/src/variants.rs:248-283` (`d_times`, `dt_times`)
- **Problem:**
  - Although dot products inside `d_times` and `dt_times` accumulate in `f64` and `orthonormalize` runs in `f64`, both functions cast every output element back to `f32` (`acc as f32`) between power steps. Keeping the $m \times s$ and $n \times s$ subspace matrices in `Vec<f64>` costs `< 2 MiB` of host RAM and prevents `f32` rounding drift when extracting multi-rank directions ($k > 1$).

#### P2-3: `VerifyReport.alphas` Only Reports Diagonal of $C = -M_1 M_2^{-1}$ for $k > 1$
- **Location:** `cuteafd-loader/src/variants.rs:138`
- **Problem:**
  - `let alphas = (0..k).map(|i| c[i * k + i] as f32).collect();` only logs the diagonal entries of the $k \times k$ coupling matrix $C = -M_1 M_2^{-1}$. When $k > 1$ and $B = U^\top W$ is not orthogonal across rows, $C$ has off-diagonal entries. Reporting the singular values of $C$ (or diagonal + Frobenius norm of off-diagonal entries) gives a faithful measure of per-rank steering strength.

#### P2-4: `ModelMap::resolve_variant` Base-Alias Shadowing Check
- **Location:** `cuteafd-api/src/gateway/models.rs:117-125` and `cuteafd-core/src/variants.rs:248-261`
- **Problem:**
  - `ModelMap::resolve_variant` checks `if self.accepts(requested) { return Some(0); }` before checking `self.variants`. Because `validate_specs` only checks `spec.public_id != served_model_id` (and not the additional base aliases loaded from `ModelMap::official_names` or `--names-file`), a variant `public_id` that matches an alias of the base model is accepted at startup and then silently routed to `variant = 0`.
- **Recommendation:**
  - Either check `self.variants` before `self.accepts(requested)` in `resolve_variant`, or validate `with_variants` against `self.accepted` inside `ModelMap::with_variants` (`models.rs:88`).

#### P2-5: Feature Is Undocumented in `cuteafd` `docs/` and `PLAN.md` (and References an External `recipes` Path)
- **Locations:**
  - `cuteafd-core/src/variants.rs:7` (`//! Interface contract: designs/weight-variants-contract.md (v1, 2026-10-24).`)
  - `docs/client-apis.md`, `docs/models/glm5.md`, `docs/models/glm5_flash.md`, `docs/models/deepseek_v41.md`, `PLAN.md`
- **Problem:**
  - `git diff 0fc324ff2..HEAD -- docs/ PLAN.md README.md` is empty across all four `wv-*` branches (`wv-core-glm5-60c2f0`, `wv-glmf-b1811f`, `wv-dsv41-fbe397`, `wv-kernel-87efde`).
  - All three design documents (`weight-variants-contract.md`, `weight-variants-dsf1017-glm53.md`, `glm-flash-weight-variants.md`) live only in the private design notes, so the rustdoc citation `designs/weight-variants-contract.md` in `cuteafd-core/src/variants.rs:7` is a dangling path inside the `cuteafd` repository.
  - Neither `docs/client-apis.md` (which documents `/v1/models` and `/v1/stats`) nor `docs/models/{glm5,glm5_flash,deepseek_v41}.md` documents `--variant`, `VARIANTS` / `VARIANT_<NAME>_*` launcher keys, `/v1/models` `"variant"` objects, or `/v1/stats` `"variants"` counters.
- **Recommendation:**
  - Document `--variant` CLI syntax, `run-family.sh` `VARIANTS` / `VARIANT_<NAME>_{SOURCE,REVISION,BASE,ID,MODE}` keys, `/v1/models` `"variant"` metadata, and `/v1/stats` `"variants"` counters in `docs/client-apis.md` and `docs/models/{glm5,glm5_flash,deepseek_v41}.md`.

