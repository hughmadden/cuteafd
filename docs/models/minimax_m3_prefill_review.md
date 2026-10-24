# MiniMax M3.1 Flash (`minimax_m3`) Prefill Design & Implementation Review

**Author:** Gemini 4 Argon  
**Date:** 2026-10-24 AEDT  
**Branch / Commit:** `hugh/minimax-m3` at `51f79cf68` (merge-base with `work/p0`: `f7bb2da19`)  
**Scope:** Static design and implementation audit of `minimax_m3` prefill execution across `cuteafd-daemon`, `cuteafd-ffi`, `cuteafd-loader`, and `third_party/sparkinfer` (`b12x`).

---

## Executive Summary

There is **no dedicated prefill path** in the current `minimax_m3` executor. `MiniMaxBackend::prefill_device` calls the exact same `Execution::execute_through` method as single-token decode, slicing every prefill chunk into 64-row decode tiles while hitting six compounding design and implementation bottlenecks.

For a single **4,096-token prefill chunk across 60 layers**, the current implementation:

1. **Writes ~96.6 GiB of dead `-inf` padding to global memory inside the DSA indexer alone** (`3,840` tile launches × `25.17 MiB` of static 1M-context scratch zeroing and scanning per 64-row tile) and reads index K pages via 1-byte scalar `Uint8` loads (`use_scalar_k_load = 1`). **Crucially, the exported `minimax_index_prefill` (`_Prefill`) kernel has the exact same static `16,384`-page / `8,192`-block bug**, so merely wiring up `minimax_index_prefill` without fixing `minimax_indexer.py` will still write **96.6 GiB** of dead padding per 4,096-row chunk.
2. **Forces ~11,520 host `cudaEventSynchronize` stalls** (`3` uploads per 64-row tile × `64` tiles × `60` layers) because every tile overwrites the same three H2D device scratch buffers (`page_table`, `cache_lengths`, `cu_seqlens_q`) via `NativeLibrary::copy_h2d_ordered`, which synchronizes on an event recorded on `self.stream` *after* the previous tile's compute kernels.
3. **Treats every prefill token as an isolated `qo_len = 1` decode request (`cu_seqlens_q = 0..=64`)** and calls `minimax_attention_decode` (`msa_union_tile = False`, `cta_tile_q = 16`) instead of the already-exported `minimax_attention_extend` (`msa_union_tile = True`, `cta_tile_q = 128`, 8-query union tiles), streaming up to **~125 GiB** of redundant KV pages with TMA and native FP8 MMA disabled.
4. **Drains the CUDA stream (`commands.drain()`) *before* launching the shared expert (`ffn::dense`) on every MoE layer** on RTX 5090 (`PinnedSpark`), and **copies the `50.3 MiB` activation payload twice on the CPU per layer (`~5.74 GiB` of CPU `memcpy` across 57 MoE layers)** while the GPU sits idle; on both 5090 and PRO 6000, `prefill_lanes = 1` prevents any sub-wave overlap between coordinator compute and Spark RoCE exchange.
5. **Unconditionally runs the full `[rows, 200064]` FP32 `lm_head` GEMM (`3.28 GiB` written, `~10.07 TFLOPs`) on every prefill chunk**, even when `logit_rows` is `0` (non-final chunk) or `1` (final chunk).
6. **Pays avoidable secondary overheads** in router scoring (`SmallNGemvKernel` SIMT 8-row GEMV instead of Tensor Core GEMM), `Commands::widen` (`cudaMemsetAsync` + accumulate instead of direct cast), full-chunk DSpark context projection (`[rows, 30720] -> [rows, 6144]` + an extra `commands.drain()`), and 16× barrier multiplication under the 5090 C1 `--prefill-rows 256` profile.

---

## Summary Table (Ranked by Impact on a 4,096-Row Prefill Chunk)

| # | Bottleneck | Category | Cost per 4,096-Row Chunk (60 Layers) | Primary Locations |
|---|---|---|---|---|
| **1** | **Static 1M-context (`16,384`-page / `8,192`-block) zeroing & scanning in `minimax_indexer.py`** (affects **both** `_Decode` and `_Prefill`!) + scalar 1-byte K loads in `_Decode` | Design + Impl bug | **96.6 GiB** dead global memory writes (`64.4 GiB` in `clear_pages` + `32.2 GiB` in `finish_scores`) + **3,840** 8,192-block radix-topk launches | `third_party/sparkinfer/b12x/integration/cuteafd/minimax_indexer.py:25-40, 137-250, 253-401`, `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:243-267` |
| **2** | **Per-tile `copy_h2d_ordered` buffer reuse causes host `cudaEventSynchronize` stalls every 64 rows** + 3,840 Rust page-table rebuilds | Impl bug | **~11,520 host-GPU sync stalls** (destroys stream queuing) + **3,840** heap allocations of `[64 × stride]` page tables | `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:243-267`, `rust/crates/cuteafd-ffi/src/lib.rs:2584-2612` |
| **3** | **Prefill uses `minimax_index_decode` + `minimax_attention_decode` (`qo_len = 1`) instead of exported `minimax_index_prefill` + `minimax_attention_extend`** | Design | **Up to 8× redundant KV traffic (~125 GiB)**; `msa_union_tile=False`; TMA & native FP8 MMA disabled | `python/tools/aot/export_b12x_minimax_aot.py:19-28`, `rust/crates/cuteafd-loader/src/plan/families/minimax/admission.rs:62, 107-111`, `third_party/sparkinfer/b12x/integration/cuteafd/minimax_attention.py:49-61` |
| **4** | **`PinnedSpark::dispatch` drains the GPU stream *before* `ffn::dense` (shared expert)** + double-copies 50.3 MiB on CPU per layer + single-lane `prefill_lanes = 1` | Design + Impl | **57 synchronous GPU drains** while GPU is idle + **~5.74 GiB** CPU `memcpy` + zero shared-expert/RoCE overlap | `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:186-190`, `rust/crates/cuteafd-daemon/src/families/minimax_m3/routed.rs:80-101, 175-210`, `rust/crates/cuteafd-daemon/src/families/minimax_m3/serve.rs:97-109, 166` |
| **5** | **Unconditional all-row `lm_head` projection (`[rows, 200064]` FP32) on every prefill chunk** | Impl bug | **3.28 GiB FP32 global write (`3.05 GiB`) + ~10.07 TFLOPs GEMM** per chunk when only `0` or `1` row is needed | `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:199`, `rust/crates/cuteafd-daemon/src/families/minimax_m3/backend.rs:112, 146-155`, `rust/crates/cuteafd-daemon/src/families/minimax_m3/serve.rs:44` |
| **6** | **Secondary overheads:** SIMT `SmallNGemvKernel` for router, `widen` = `cudaMemsetAsync` + accumulate, full-chunk DSpark `update` + drain, 5090 `--prefill-rows 256` chunk multiplication | Impl | **~1.55 TFLOPs** DSpark `fc` + **>1 GiB** extra memset/copy traffic; **16×** more sync barriers at `--prefill-rows 256` | `third_party/sparkinfer/b12x/integration/cuteafd/_minimax_kernels.py:45`, `rust/crates/cuteafd-ffi/src/families/minimax.rs:310-315`, `rust/crates/cuteafd-daemon/src/families/minimax_m3/dspark.rs:107-137`, `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:121-128` |

---

## Detailed Technical Analysis

### 1. Catastrophic Static 1M-Context Zeroing & Scanning in `minimax_indexer.py` (Both `_Decode` and `_Prefill`)

In `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:243-267`, `Execution::attention` loops over `rows` in steps of `self.decode_rows` (`64`) and launches `minimax_index_decode` on each 64-row tile.

In `third_party/sparkinfer/b12x/integration/cuteafd/minimax_indexer.py:25-40`, both `_Prefill` and `_Decode` are compiled with static 1M-context upper bounds:
- `self.pages = max_pages = (1_048_576 + 63) // 64 = 16384` (64-token index pages)
- `self.blocks = max_pages // 2 = 8192` (128-token KV blocks)

On **every single 64-row tile**, regardless of the live `width` (even on the first 64 tokens of a prompt where `width == 1`), `_Decode` (`minimax_indexer.py:253-401`) executes:

1. **`clear_pages(pages, rows)` (`minimax_indexer.py:374-375, 396-401`):**
   Loops `while idx < rows * 4 * self.pages` (`64 * 4 * 16384 = 4,194,304` floats) and writes `-inf` to **16.78 MiB** (`16 MiB`) of global memory.
2. **`finish_scores(pages, scores, table, lengths, rows, width)` (`minimax_indexer.py:293-322, 345-346`):**
   Loops `while block < self.blocks` (`8192`) for all `rows * 4 = 256` query-head rows, reading `pages` at stride `16384` and writing `256 * 8192 * 4 B = 8.39 MiB` (`8 MiB`) of global memory.
3. **`self.selector` (`DSATiledTopkKernel`, `minimax_indexer.py:353-355`):**
   Launches `256` CTAs of 1,024 threads each to run radix top-k over a hardcoded stride of `self.blocks = 8192` elements per row-head.
4. **Scalar 1-byte K loads with no query tiling (`minimax_indexer.py:277-290`, `b12x/attention/dsa_indexer/kernel.py:524-536, 1834-1840`):**
   `init` sets `schedule[idx * 2] = idx` (1 query row per schedule interval) and `controls[1] = 1` (`use_scalar_k_load = 1`). Inside `DSAScheduledMultiRowLogitsKernel`, `use_scalar_k_load = 1` disables TMA and executes `_load_index_k_page_scalar`, reading every 8 KiB index K page via **1-byte `Uint8` scalar loads** independently for each query row.

Across `4096 / 64 = 64` tiles × `60` layers = **3,840 tile invocations per 4,096-token chunk**, `clear_pages` and `finish_scores` alone write:
- `clear_pages`: `3,840 × 16.78 MiB = 64.42 GiB` (`60.0 GiB`)
- `finish_scores`: `3,840 × 8.39 MiB = 32.21 GiB` (`30.0 GiB`)
- **Total dead padding writes:** **`96.64 GiB` (`90.0 GiB`) per 4,096-token chunk**.

> **Warning — `_Prefill` (`minimax_index_prefill`) has the exact same static-bound bug:**  
> Look at `_Prefill` in `minimax_indexer.py:137-250`:
> - `_Prefill.clear_pages(pages, rows)` (`lines 217, 242-246`) clears `rows * 4 * self.pages` (`4096 * 4 * 16384 * 4 B = 1.074 GiB` per layer = **64.42 GiB across 60 layers**).
> - `_Prefill.finish_scores(...)` (`lines 165-193, 219-227`) loops `while block < self.blocks` (`8192`) for all `4096 * 4 = 16,384` row-heads, writing `536.87 MiB` per layer = **32.21 GiB across 60 layers**.
> - `_Prefill.selector` (`lines 229-236`) scans `self.blocks = 8192` per row-head across `16,384` CTAs per layer.
>
> **Therefore, wiring up `minimax_index_prefill` without fixing `_Prefill` in `minimax_indexer.py` will NOT eliminate the 96.6 GiB of dead global memory writes.** Both `_Prefill` and `_Decode` must bound `clear_pages`, `finish_scores`, and `selector` by the live page/block width.

---

### 2. Per-Tile `copy_h2d_ordered` Buffer Reuse Forces ~11,520 Host `cudaEventSynchronize` Stalls per Chunk

In `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:243-267`, every iteration of `for start in (0..rows).step_by(self.decode_rows)` uploads the current tile's metadata into the **exact same three scratch buffers**:
```rust
self.commands.upload(s.get("page_table")?, &tables.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
self.commands.upload(s.get("cache_lengths")?, &lengths.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
self.commands.upload(s.get("cu_seqlens_q")?, &(0..=count as i32).flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
```
`Commands::upload` calls `NativeLibrary::copy_h2d_ordered` (`rust/crates/cuteafd-ffi/src/lib.rs:2584-2612`), which caches a single pinned staging buffer per `(device, stream, dst.ptr)`:
```rust
let key = (self.device, stream, dst.ptr);
...
if entry.pending {
    let sync_rc = unsafe { self.cuda_event_synchronize(entry.event) };
    ...
    entry.pending = false;
}
unsafe {
    ptr::copy_nonoverlapping(src.as_ptr(), entry.host_ptr as *mut u8, src.len());
}
self.copy_h2d_async(dst, entry.host_ptr as *const c_void, src.len(), stream)?;
let record_rc = unsafe { self.cuda_event_record(entry.event, stream) };
entry.pending = true;
```
Because all operations share `self.stream`:
1. Tile $t$ uploads `page_table` (`dst.ptr = P`), records `entry.event`, and launches `minimax_index_decode` + `minimax_attention_decode` on `self.stream`.
2. Tile $t+1$ calls `upload(page_table)` with the same `dst.ptr = P`. It synchronizes on tile $t$'s upload event, copies into `entry.host_ptr`, enqueues `copy_h2d_async` on `self.stream` **behind tile $t$'s compute kernels**, and records `entry.event` on `self.stream` **behind tile $t$'s compute kernels**.
3. Tile $t+2$ calls `upload(page_table)`. Its `self.cuda_event_synchronize(entry.event)` **blocks the CPU thread until tile $t$'s `minimax_index_decode` and `minimax_attention_decode` kernels have completely finished on the GPU**.

Instead of queuing kernels asynchronously ahead of the GPU, the host thread stalls on the GPU every 64 rows (**3,840 tile barriers / ~11,520 `copy_h2d_ordered` calls per 4,096-row step**). In addition, `execution.rs:247-258` allocates and populates `vec![-1i32; count * stride]` and three `Vec<u8>`s on the heap **3,840 times per chunk**, even though the page tables and cache lengths are identical across all 60 layers.

---

### 3. Prefill Runs Single-Query Decode Attention (`qo_len = 1`) Instead of Exported `minimax_attention_extend`

In `python/tools/aot/export_b12x_minimax_aot.py:19-28`, the AOT exporter already compiles and exports:
- `minimax_index_k_quant`
- `minimax_index_prefill` (`DSAContiguousLogitsPrefillKernelBlockScores`, which tiles 64 queries × 64 keys in Tensor Cores)
- `minimax_attention_extend` (`PagedForwardKernel` with `mode="extend"`, `msa_union_tile=True`, `cta_tile_q=128`)

However:
- `rust/crates/cuteafd-loader/src/plan/families/minimax/admission.rs:62, 107-111` only admits decode programs and explicitly filters out `minimax_index_prefill` (`539,197,952` B scratch) and `minimax_attention_extend` (`2,377,728` B scratch),
- `rust/crates/cuteafd-ffi/src/families/minimax.rs:85-100` omits them from FFI `definitions()`, and
- `execution.rs:258-266` passes `cu_seqlens_q = [0, 1, 2, ..., count]` to `minimax_attention_decode`.

In `third_party/sparkinfer/b12x/integration/cuteafd/minimax_attention.py:49-61` and `b12x/attention/paged/forward_extend_generic.py:4358-4375`:
- `minimax_attention_decode` uses `mode="decode"`, `cta_tile_q = 16`, `num_warps_q = 1`, `num_warps_kv = 4`, `msa_union_tile = False`. With `group_size = 64 / 4 = 16` Q heads per KV head and `qo_len = 1`, each CTA processes **only 1 token** (`16` Q heads) and loads its 16 selected 128-token KV blocks (`16 × 128 × 128 × 2 B = 512 KiB` of K+V per token per layer) from global memory with **zero KV reuse across adjacent prompt tokens**. Across 4,096 tokens and 60 layers, this streams up to **~125.8 GiB** of KV cache data.
- `minimax_attention_extend` uses `mode="extend"`, `cta_tile_q = 128`, `num_warps_q = 4`, `num_warps_kv = 1`, `msa_union_tile = True`. Each CTA processes **8 consecutive query tokens** (`8 × 16 = 128` Q heads), builds the deduplicated union of their selected sparse KV blocks in shared memory (`forward_extend_generic.py:4362-4375`), and loads each shared KV block once for all 8 queries—reducing KV global memory traffic by up to **8×**.
- Furthermore, `minimax_attention.py:59-61` hardcodes `use_native_fp8_qk=False`, `use_native_fp8_pv=False`, and `enable_paged_kv_tma=False` on both `decode` and `extend`.

---

### 4. `PinnedSpark::dispatch` Drains the GPU Stream *Before* the Shared Expert + Copies ~100.6 MiB/Layer on CPU

In `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:186-190`:
```rust
experts.dispatch(layer, rows, self.scratch.get("route_ids")?, self.scratch.get("route_weights")?, self.scratch.get("norm")?, &self.commands)?;
ffn::dense(&self.commands, &self.weights, &self.scratch, layer, rows, true)?;
...
experts.collect(layer, rows, self.scratch.get("routed")?, &self.commands)?;
```
On RTX 5090 (where `device_exchange_available` is `false` because GeForce lacks `dma-buf` peer RDMA landing, selecting `Spark::Pinned(PinnedSpark)` in `routed.rs:30-32`):
1. `PinnedSpark::dispatch` (`routed.rs:80-101`) enqueues D2H copies of `route_ids`, `route_weights`, and `norm` (`4096 × 6144 × 2 B = 50.33 MiB` at 4,096 rows) onto `self.stream` and **immediately calls `commands.drain()?` (`routed.rs:93`) *before* `ffn::dense` (the shared expert) is even queued on the GPU**.
2. While the GPU sits completely idle after `commands.drain()?`, the host CPU thread:
   - Parses `4096 × 4 = 16,384` route entries into a fresh `Vec<ExpertProtocolV2RouteEntry>` (`routed.rs:96`),
   - Allocates and copies `50.33 MiB` via `bytes::Bytes::copy_from_slice(layout.wire(staged)?)` (`routed.rs:96`),
   - Inside `serve::request` (`serve.rs:97-109`), allocates `4,096` `ExpertProtocolV2RowDescriptor` structs and calls `ExpertProtocolV2Request::new_bytes`, which allocates and copies the `50.33 MiB` wire payload **a second time**.
   - Across 57 MoE layers, this performs **`57 × 100.66 MiB = 5.74 GiB` of single-threaded CPU heap allocation and `memcpy`** per 4,096-row chunk while the GPU is idle.
3. Only after `dispatch` returns does `execution.rs:187` launch `ffn::dense` (the shared expert), followed immediately by `experts.collect` (`routed.rs:103-115`), which blocks the CPU in `runtime.block_on(link.receive(...))`.
4. Even on RTX PRO 6000 (`Spark::Device`), `Routed` (`routed.rs:133-210`) allows only a single `pending: Option<Pending>` wave and `admission.rs:57` / `serve.rs:166` hardcode `prefill_lanes: 1`, so there is **no two-lane sub-batch pipelining** to overlap sub-batch $B$'s attention/shared-MLP with sub-batch $A$'s Spark expert round-trip.

---

### 5. Unconditional All-Row `lm_head` Projection (`[rows, 200064]` FP32 = 3.28 GiB) on Every Prefill Chunk

In `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:199`:
```rust
if last_layer + 1 == self.cfg.layers {
    self.head.logits(&self.commands, &self.weights, &self.scratch, rows).map(Some)
}
```
`self.head.logits` (`head.rs:17-21`) projects **all `rows`** through `final_norm` and `lm_head` (`[4096, 6144] × [6144, 200064] -> [4096, 200064]` FP32 = **819.5M floats, 3.28 GiB (`3.05 GiB`) written to global memory, ~10.07 TFLOPs GEMM**).

Then `MiniMaxBackend::prefill_device` (`backend.rs:146-155`) immediately discards all but `logit_rows` (which is `0` on non-final chunks and `1` on the final chunk for normal generation):
```rust
if logit_rows == 0 {
    return Ok(None);
}
let start = rows - logit_rows;
```
Moreover, `ServeArgs::full_prefill_logits` (`serve.rs:44`) is parsed by Clap and ignored; `MiniMaxBackend::geometry` (`backend.rs:112`) hardcodes `full_prefill_logits: true`.

---

### 6. Secondary Kernel, Drafter, and Sizing Bottlenecks

1. **Router GEMV (`minimax_router_scores`) uses a SIMT 8-row GEMV kernel during prefill:**
   In `third_party/sparkinfer/b12x/integration/cuteafd/_minimax_kernels.py:45` and `b12x/gemm/bf16_gemv/_kernel.py:270-310`, `minimax_router_scores` is compiled as `SmallNGemvKernel(128, 6144, False, False)` with `rows_per_tile = 8`. At `rows = 4096`, it launches `128 × 512 = 65,536` warps per layer that re-read the `[128, 6144]` FP32 router weights `512` times via SIMT instructions instead of a Tensor Core GEMM.
2. **`Commands::widen` zeroes the destination buffer before accumulating:**
   In `rust/crates/cuteafd-ffi/src/families/minimax.rs:310-315`, `Commands::widen` (BF16 $\to$ FP32 conversion, called on `embedded` in `execution.rs:170` and in `dspark.rs:124, 164, 178, 185`) executes `self.zero(destination)` (`cudaMemsetAsync` of `100.66 MiB` at 4,096 rows) followed by `cuda_accumulate_bf16_to_f32_async` (which reads the zeroed destination and adds to it), turning a `150 MiB` cast into `300 MiB` of memory traffic.
3. **DSpark `update` (`prefill_draft`) projects all `rows` even when exceeding the 4,096-token ring window and adds a second synchronous `commands.drain()`:**
   When `--speculator dspark` is active (default in `serve.rs:33`), `execution.rs:193` taps 5 layers (`[3, 17, 31, 45, 59]`) via `narrow` + 2D strided `copy_rows` (`dspark.rs:107-113`). After `execute_through` already drained the stream (`execution.rs:203`), `MiniMaxBackend::prefill_device` calls `Execution::update_draft` (`execution.rs:121-128`, `dspark.rs:114-137`), which runs `bf16_linear` (`[n, 30720] × [30720, 6144]`, ~1.55 TFLOPs at 4,096 rows), `widen`, `norm`, and 5 layers of `kv_proj` + `kv_split` + `k_norm_rope` + per-span `copy`, followed by a **second synchronous `self.commands.drain()` (`execution.rs:125`)**.
4. **5090 C1 sizing profile (`--prefill-rows 256`) multiplies per-chunk barriers by 16×:**
   Under the 5090 KV-saving profile (`docs/models/minimax_m3.md:176-181`, `--prefill-rows 256`), a 4,096-token prompt is split into 16 sequential 256-row chunks, multiplying Spark round-trips (`16 × 57 = 912` sequential barriers), stream drains, and full-model weight streaming across all 60 layers by 16×.

---

## Recommendations & Implementation Order

1. **Fix `minimax_indexer.py` dynamic width bounds in *both* `_Prefill` and `_Decode` (P0):**
   - Pass dynamic `width` (`live_pages`) into `clear_pages(pages, rows, width)` (or only clear `width` pages per row-head instead of `self.pages = 16384`).
   - Bound `finish_scores` to `active_blocks = (width + 1) // 2` instead of `self.blocks = 8192` (or pass `active_blocks` to `DSATiledTopkKernel` so the selector only scans live blocks).
   - This single change removes **96.6 GiB of dead global memory writes per 4,096-row prefill chunk** and removes up to **960 MiB of dead writes per 64-row decode/verify step**.
2. **Wire `minimax_index_k_quant`, `minimax_index_prefill`, and `minimax_attention_extend` for prefill (P0):**
   - Admit and bind `minimax_index_k_quant`, `minimax_index_prefill`, and `minimax_attention_extend` in `cuteafd-loader` (`admission.rs`), `cuteafd-ffi` (`minimax.rs`), and `cuteafd-daemon` (`execution.rs`).
   - Pack contiguous request ranges (`cu_seqlens_q` per request, not per token) so `minimax_attention_extend` uses `msa_union_tile = True` (`cta_tile_q = 128`, 8-token query union tiles) and shares KV block loads across 8 queries per CTA.
   - Keep `minimax_index_decode` + `minimax_attention_decode` for decode and speculative target verification.
3. **Hoist page-table / length / `cu_seqlens_q` uploads out of the 60-layer loop (P0):**
   - Page tables, cache lengths, and query offsets are invariant across all 60 layers within a step. Build and upload them **once per `execute_through` step** before `for (layer, cache) in caches.iter().enumerate()`.
   - If tiled execution is ever used within a layer, slice distinct regions of a pre-uploaded step buffer rather than overwriting the same `dst.ptr` via `copy_h2d_ordered` inside the loop. This eliminates all 11,520 `copy_h2d_ordered` calls and their `cudaEventSynchronize` host stalls.
4. **Thread `logit_rows` into `Execution::execute_through` and `TargetHeadOwner::logits` (P0):**
   - Pass `logit_rows` (`0` on intermediate chunks, `1` on final generation chunk, `rows` only when `--full-prefill-logits` or scoring is requested) into `execute_through`.
   - Skip `self.head.logits` entirely when `logit_rows == 0`, and slice `residual` to the trailing `logit_rows` rows (`rows - logit_rows .. rows`) before `final_norm` and `lm_head`. Wire `ServeArgs::full_prefill_logits` into `MiniMaxBackend::geometry()`.
5. **Overlap `PinnedSpark::dispatch` with `ffn::dense` and remove the double 50.3 MiB CPU copy (P1):**
   - In `execution.rs:186-188`, launch `ffn::dense` (the shared expert) before draining the D2H staging event (or record an event after `RouteStaging::enqueue` on a D2H copy stream and synchronize only that event on the host while `ffn::dense` runs concurrently on the compute stream).
   - Avoid copying the `50.3 MiB` wire payload twice on the CPU between `routed.rs:96` (`Bytes::copy_from_slice`) and `serve.rs:103` (`ExpertProtocolV2Request::new_bytes`).
   - Consider 2-lane prefill sub-wave pipelining (`prefill_lanes = 2`) once single-lane prefill kernels are qualified.
6. **Clean up `Commands::widen`, router GEMM, and DSpark `update` windowing (P2):**
   - Replace `zero` + `cuda_accumulate_bf16_to_f32_async` in `Commands::widen` (`minimax.rs:310-315`) with a direct BF16 $\to$ FP32 cast kernel.
   - Use a Tensor Core GEMM for `minimax_router_scores` when `rows > 16`.
   - Cap `Dspark::update` (`dspark.rs:114-137`) to only the trailing `WINDOW = 4096` tokens of a prompt and merge its stream drain with the step completion barrier.
