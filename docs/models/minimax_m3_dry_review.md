# MiniMax M3 (`minimax_m3`) DRY & Shared-Layer Consolidation Review

**Author:** Gemini 4 Argon  
**Date:** 2026-10-24 AEDT  
**Scope:** `origin/hugh/minimax-m3` (`ae82fef80`) compared against `shared/`, `step5`, `glm5_flash`, `qwen4`, `mimo_v2`, `glm5`, and `kimi_k3` across `cuteafd-daemon`, `cuteafd-ffi`, `cuteafd-loader`, `cuteafd-api`, `native/`, and `python/`.

---

## Executive Summary

While `minimax_m3` properly reuses several shared building blocks (`shared::qwen::scheduler`, `shared::token_io::native`, `shared::experts::rtx::dense::DenseLoad`, `shared::experts::rtx::Fp8Experts`, `shared::golden_scores`, and `cuteafd_engine::prefix`), it introduces **a parallel mini-runtime in `cuteafd-ffi`** (`minimax::{Buffer, Commands, Head}`) and **copies significant subsystem boilerplate from `step5`, `glm5_flash`, and `qwen4`** instead of using or parameterizing existing shared modules.

Most notably:
1. **`cuteafd_ffi::families::minimax::{Buffer, Commands, Head}` (`rust/crates/cuteafd-ffi/src/families/minimax.rs:1873-2294`)** re-implements `shared::memory::device::{Device, Allocation, Stream}`, `cuteafd_ffi::programs::Programs`, and `shared::target_head::launch`.
2. Because `minimax_m3` uses its own `Buffer` and `Commands` types instead of `Allocation` / `Stream`, it had to add **`shared::minimax_ffi` (`rust/crates/cuteafd-daemon/src/shared/minimax_ffi.rs:1566-1611`)** solely as a type-erasure shim to call `Fp8Experts` and `SparkDeviceLink`!
3. **`minimax_m3/dspark.rs` + `cuteafd-ffi/src/families/minimax/dspark.rs` + `minimax_m3/backend.rs`** re-wrap and re-implement the `BlockDspark` / `DFlash2` context ring, tap staging, Markov head, and workspace sizing already in `shared::speculation::block_dspark`, `cuteafd-ffi/src/families/glm5_flash/glmf_dspark.rs`, and `cuteafd_loader::families::glm5::draft_representation`.
4. **Page-pool wave validation (`minimax_m3/transaction.rs`), coordinator weight slicing/fusion (`plan/families/minimax/coordinator.rs`), `run_serve` HTTP bootstrap (`minimax_m3/serve.rs`), vision service startup & tower packing (`minimax_m3/media.rs`, `shared/vision/minimax.rs`), golden harness (`minimax_m3/golden.rs`), and chat/tool XML parser helpers (`cuteafd-api/src/openai/chat/minimax_m3/*`)** are copied near-verbatim from `step5`, `glm5_flash`, and `qwen4`.

Consolidating these 10 areas eliminates ~2,400 lines of duplicated Rust/Python/CUDA glue and ensures bug fixes (such as stream quarantine, CUDA graph capture, prefix/page-pool invariants, and DSpark speculation fixes) automatically propagate to `minimax_m3`.

---

## Detailed Findings (Ordered by Architectural Impact)

### 1. [HIGH] Parallel Device Memory, Stream & Program-Launch Runtime (`cuteafd_ffi::minimax::{Buffer, Commands, Head}` vs `shared::memory::device` + `shared::target_head`)

- **MiniMax files:**
  - `rust/crates/cuteafd-ffi/src/families/minimax.rs:1873-2294` (`on_device`, `Storage`, `Buffer<'a>`, `Commands<'a>`, `Head<'a>`)
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/head.rs:2897-2921` (`TargetHeadOwner`)
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-daemon/src/shared/memory/device.rs:2697-2985` (`Device::run`, `Allocation`, `Stream`)
  - `rust/crates/cuteafd-daemon/src/shared/target_head.rs:2589-2600` (`shared::target_head::launch`)
  - `rust/crates/cuteafd-daemon/src/families/step5/head.rs:3038-3115` (`TargetHeadOwner`)
- **What is duplicated:**
  1. `minimax::on_device` (`minimax.rs:1873-1879`) duplicates `Device::run` (`device.rs:2697-2716`).
  2. `minimax::Storage` + `Buffer<'a>` (`minimax.rs:1881-1962`) duplicates `Allocation` (`device.rs:2773-2860`), including identical quarantine-on-drop semantics (`if !is_quarantined_after_failed_drain() { (self.library.free_device_buffer)(self.raw); }`).
  3. `minimax::Commands<'a>` (`minimax.rs:2066-2261`) duplicates `Stream` (`device.rs:2927-2985`), including identical `Drop` drain-or-abort handling (`if self.library.drain_stream(self.stream).is_err() { std::process::abort(); }`).
  4. `minimax::Head<'a>` (`minimax.rs:2269-2294`) duplicates `shared::target_head::launch` (`target_head.rs:2589-2600`) and `step5/head.rs:3062-3114` (`cuteafd_b12x_lm_head_bf16_tc_Rows_async` + `cuteafd_v41_argmax_bf16_rows_strided_async`).
  5. Notably, `shared/target_head.rs:2625` has an architectural guard test `every_family_target_head_takes_the_shared_route` that checks every family references `target_head::`. `minimax_m3/head.rs:2909` only passes that string check by calling `crate::shared::target_head::TargetHead::new(rows, VOCAB).tensor_cores()` and then immediately calling `cuteafd_ffi::minimax::Head::launch` instead of `crate::shared::target_head::launch`!
- **Recommendation:**
  - Adopt `shared::memory::device::{Device, Allocation, Stream}` in `minimax_m3` (just as `step5` does) and call `shared::target_head::launch(device.library, ...)` directly from `minimax_m3/head.rs`.
  - Keep only the typed MiniMax AOT program table (`Programs`) in `cuteafd-ffi/src/families/minimax.rs`, taking a raw `CuteafdStream` or `&Stream` like `cuteafd_ffi::programs::Programs`.

---

### 2. [HIGH] Redundant `shared::minimax_ffi` Type-Erasure Bridge & `routed.rs` Glue

- **MiniMax files:**
  - `rust/crates/cuteafd-daemon/src/shared/minimax_ffi.rs:1566-1611` (`local`, `dispatch`, `collect`)
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/routed.rs:4377-4515` (`Routed`, `PreparedLayer`)
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-daemon/src/families/step5/routed.rs:4753-4896`
  - `rust/crates/cuteafd-daemon/src/shared/experts/rtx/fp8moe.rs` (`Fp8Experts::run_bf16`)
  - `rust/crates/cuteafd-transport/src/expert/device.rs` (`SparkDeviceLink::{dispatch, collect, reduce}`)
- **What is duplicated:**
  - `shared/minimax_ffi.rs` exists solely because `minimax_m3` uses `cuteafd_ffi::minimax::{Buffer, Commands}` while `Fp8Experts` and `SparkDeviceLink` expect `CuteafdDeviceBuffer` and `shared::memory::device::Stream`.
  - Every routed MoE call in `minimax_m3` constructs a dummy `Device` (`Device { id: commands.device(), library: commands.library() }`), wraps raw pointers in `unsafe { CuteafdDeviceBuffer::from_raw_parts(...) }`, retains `Rc` owners, and runs inside `device.run(...)`.
- **Recommendation:**
  - Once `minimax_m3` uses `shared::memory::device::{Device, Allocation, Stream}` (Finding #1), delete `rust/crates/cuteafd-daemon/src/shared/minimax_ffi.rs` entirely and unify `minimax_m3/routed.rs` with `step5/routed.rs`.

---

### 3. [HIGH] Duplicate DSpark Speculation Runtime, FFI Wrappers & Workspace Math

- **MiniMax files:**
  - `rust/crates/cuteafd-ffi/src/families/minimax/dspark.rs:2471-2554` (`Commands::{dspark_attention_workspace, dspark_attention, dspark_markov_norms, dspark_markov_workspace, dspark_markov, dspark_confidence}`)
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/dspark.rs:208-390` (`State`, `context_spans`, `Dspark::{new, tap, update, propose}`)
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/backend.rs:488-520, 679-716` (`Rings`, `accept_context`, `prefill_draft`, `draft`)
  - `rust/crates/cuteafd-loader/src/families/minimax_m3/dspark.rs:607-656` (`DraftMemory::new`)
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-ffi/src/families/glm5_flash/glmf_dspark.rs:1429-1569`
  - `rust/crates/cuteafd-daemon/src/shared/speculation/block_dspark.rs:1-420`
  - `rust/crates/cuteafd-daemon/src/shared/speculation/dflash.rs`
  - `rust/crates/cuteafd-loader/src/families/glm5/draft_representation.rs` (`block_dspark_resident_bytes`)
- **What is duplicated:**
  1. `cuteafd-ffi/src/families/minimax/dspark.rs:2471-2554` re-wraps the exact same `NativeLibrary` C entrypoints (`block_dspark_attention_workspace`, `mimo_dflash_attention`, `glmf_dspark_markov_norms`, `glmf_dspark_markov_workspace`, `glmf_dspark_markov`, `glmf_dspark_confidence`) already wrapped in `glmf_dspark.rs:1429-1569`.
  2. `minimax_m3/dspark.rs:296-312` (`context_spans`) and `minimax_m3/backend.rs:488-520` (`Rings`) re-implement circular KV context ring allocation, wrap-around 2-span slicing (`first = total.min(capacity - start)`, `second = total - first`), and `tap` layer accumulation from `shared/speculation/block_dspark.rs`.
  3. `cuteafd-loader/src/families/minimax_m3/dspark.rs:627-637` (`DraftMemory::new`) hardcodes the C++ workspace formulas in Rust (`slices = (288 + 127) / 128`, `partitions = (max_kv + 127) / 128`, `sequences * 296 * 8 + sequences * cfg.block * 4`), and then `minimax_m3/dspark.rs:259-263` asserts at runtime that those hardcoded numbers match `library.glmf_dspark_markov_workspace(...)` and `library.block_dspark_attention_workspace(...)`.
- **Recommendation:**
  - Promote the `glmf_dspark_*` / `block_dspark_*` FFI wrappers in `cuteafd-ffi` to `cuteafd_ffi::shared::block_dspark`.
  - Parameterize `shared::speculation::block_dspark` (or extract a shared `ContextRing` + `MarkovHead` helper) so `glm5_flash`, `qwen3_5`, `deepseek_v41`, and `minimax_m3` share ring management, span splitting, and Markov/confidence launches.
  - Share the loader workspace calculation with `glm5::draft_representation::block_dspark_resident_bytes`.

---

### 4. [HIGH] Verbatim Page-Pool Wave Validation (`minimax_m3/transaction.rs` & `execution.rs` vs `step5/transaction.rs`)

- **MiniMax files:**
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/transaction.rs:5839-5892` (`SequenceWave`, `Step::begin`)
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/execution.rs:437-470` (`BatchView::validate`)
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-daemon/src/families/step5/transaction.rs:6191-6226` (`validate_wave`)
- **What is duplicated:**
  - `Step::begin` in `minimax_m3/transaction.rs:5854-5888` is a near line-for-line copy of `validate_wave` in `step5/transaction.rs:6191-6226`:
    - Checking `!sequences.is_empty() && !pool.is_parity() && pool.page_rows() == 64 && pool.capacity().checked_add(pool.reserved()) == Some(pages)`
    - Maintaining `BTreeSet`s `seen`, `referenced`, `writable`, `unique`
    - Verifying `committed.checked_add(query) == Some(total)`, `total.div_ceil(64) == s.pages.len()`
    - Iterating `(logical, &page)` to check `(page as usize) >= pool.reserved() && (page as usize) < pages`, `refs == 0`, `write = logical >= committed / 64`, and `!writable.contains(&page) && (!write || (!referenced.contains(&page) && refs == 1))`
  - Worse, `minimax_m3/execution.rs:437-470` (`BatchView::validate`) implements the **exact same `BTreeSet` page-ownership and COW validation loop a second time** inside `minimax_m3`!
- **Recommendation:**
  - Move `validate_wave` onto `cuteafd_engine::prefix::PagePool` (e.g. `PagePool::validate_wave(...)`) so `step5`, `minimax_m3`, and future paged families call a single authoritative verifier once per wave.

---

### 5. [MEDIUM] Duplicate Coordinator Weight Slicing, Gate/Up Fusion & Dense Loader (`plan/families/minimax/coordinator.rs` & `minimax_m3/weights.rs` vs `step5`)

- **MiniMax files:**
  - `rust/crates/cuteafd-loader/src/plan/families/minimax/coordinator.rs:1832-1925` (`WeightSlice`, `WeightOperand`, `RankWeightLayout`, `rank_weight_layouts`, `check_budget`)
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/weights.rs:7655-7705` (`load_operand`, `DsparkContract::plan`)
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-loader/src/plan/families/step5/coordinator.rs:141-355` (`WeightSlice`, `WeightOperand`, `RankWeightLayout`, `rank_weight_layouts`, `check_budget`)
  - `rust/crates/cuteafd-daemon/src/families/step5/weights.rs:7888-7944` (`load_records`)
- **What is duplicated:**
  1. `WeightSlice::{Replicated, Rows, Cols}`, `WeightOperand`, and `RankWeightLayout` in `plan/families/minimax/coordinator.rs:1832-1865` are structural clones of `plan/families/step5/coordinator.rs:141-190`.
  2. Both implementations duplicate `.gate_proj.` + `.up_proj.` -> `.gate_up_proj.` two-source row fusion, `ROW_SPLIT` / `.self_attn.o_proj.` column splitting across TP ranks, 256-byte allocation floor (`bytes.max(256)`), and `check_budget`.
  3. `DsparkContract::plan` in `minimax_m3/weights.rs:7681-7700` duplicates the `.gate_proj.` + `.up_proj.` -> `.gate_up_proj.` fusion logic a **third** time!
  4. `load_operand` in `minimax_m3/weights.rs:7655-7674` duplicates `load_records` in `step5/weights.rs:7888-7944` (converting `WeightOperand` sources + `WeightSlice::{Replicated, Rows, Cols}` into `DenseLoad` `Rect`s and calling `DenseLoad::mount(device.library).operand(...)`).
- **Recommendation:**
  - Lift `WeightSlice`, `WeightOperand`, `RankWeightLayout`, gate/up fusion, and TP row/col slicing into `cuteafd-loader/src/plan/layout/` (shared by `step5` and `minimax`).
  - Move `load_operand` (`WeightOperand` -> `DenseLoad` `Rect`) into `rust/crates/cuteafd-daemon/src/shared/experts/rtx/dense.rs`.

---

### 6. [MEDIUM] Verbatim `run_serve` Worker Thread + Axum Router Bootstrap (`minimax_m3/serve.rs` vs `step5/serve.rs`, `qwen4/serve.rs`, `mimo_v2/serve.rs`)

- **MiniMax file:**
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/serve.rs:5081-5122` (`run_serve`)
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-daemon/src/families/step5/serve.rs:5457-5502` (`run_serve`)
  - `rust/crates/cuteafd-daemon/src/families/qwen4/serve.rs`
  - `rust/crates/cuteafd-daemon/src/families/mimo_v2/serve.rs`
- **What is duplicated:**
  - Creating `mpsc::channel::<NativeRequest>(16)` and `oneshot::channel()`
  - Spawning the blocking scheduler thread `tokio::task::spawn_blocking` and awaiting `ready_rx`
  - Constructing `NativeLimits::new(...)` and `console::hub(&model, ...)`
  - Branching on `vision` (`router_for_model_with_media` vs `router_for_model`) with `.with_constraints(...)` and `.with_console(...)`
  - Binding `tokio::net::TcpListener::bind(&args.listen)`, emitting `cuteafd_bench::ready(...)`, and running `tokio::select! { result = axum::serve(...) => ..., result = scheduler::watch_scheduler(handle) => ... }`
- **Recommendation:**
  - Extract a shared `shared::qwen::scheduler::serve_http(...)` helper taking the common serve parameters (`listen`, `model`, `profile`, `encoding`, `limits`, `constraints`, `media`, and the `spawn_blocking` backend closure).

---

### 7. [MEDIUM] Duplicate Vision Service Readiness & Tower Tensor Packing (`minimax_m3/media.rs` & `shared/vision/minimax.rs` vs `glm5_flash` & `qwen4`)

- **MiniMax files:**
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/media.rs:1353-1436` (`RemoteVision`, `ReadyVision::load`)
  - `rust/crates/cuteafd-daemon/src/shared/vision/minimax.rs:1639-1720` (`TowerSpec::minimax`)
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-daemon/src/families/glm5_flash/media.rs:1550-1639`
  - `rust/crates/cuteafd-daemon/src/families/qwen4/media.rs:1955-2061`
  - `rust/crates/cuteafd-daemon/src/shared/vision/qwen.rs:2380-2437` (`TowerSpec::qwen`, `TowerSpec::glm_flash`)
- **What is duplicated:**
  1. `ReadyVision::load` in `minimax_m3/media.rs:1360-1436` duplicates the remote/local encoder setup from `glm5_flash/media.rs` and `qwen4/media.rs`: checking `--vision-peers` vs `--encoder-plan-hash`/`--encoder-revision`, computing `sm = compute_capability_major * 10 + compute_capability_minor`, building `EncoderHandshake`, calling `admit_device_reservations(95, ...)`, and starting `EncoderService::start(LocalEncoder::new(...))`.
  2. `TowerSpec::minimax` in `shared/vision/minimax.rs:1671-1689` duplicates the `put` tensor-packing closure from `shared/vision/qwen.rs:2383-2405` (256-byte offset alignment, `vector` BF16->FP32 2x byte expansion, `TensorRead` recording, `reads.len() != tensors.len()` verification, and trailing `inv_freq` alignment).
- **Recommendation:**
  - Extract a shared `TowerPacker` helper in `rust/crates/cuteafd-daemon/src/shared/vision.rs` for `TowerSpec::{qwen, glm_flash, minimax}`.
  - Move the common `ReadyVision::load` local/remote bootstrap into `rust/crates/cuteafd-daemon/src/shared/vision.rs`.

---

### 8. [MEDIUM] Duplicate Golden Qualification Harness Infrastructure (`minimax_m3/golden.rs` vs `step5/golden.rs`)

- **MiniMax file:**
  - `rust/crates/cuteafd-daemon/src/families/minimax_m3/golden.rs:848-1349`
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-daemon/src/families/step5/golden.rs:956-1520`
  - `rust/crates/cuteafd-daemon/src/shared/golden_scores.rs`
- **What is duplicated:**
  - `meta.json` / `windows.json` schema & SHA-256 verification (`load_panel`, `load_Partial`, `sha`, `file_sha`, `bytes`)
  - Binary little-endian `tokens.bin`, `logits.bin`, and `layerNN.bin` readers + `floats()` finiteness checks (`minimax_m3/golden.rs:1322-1349` vs `step5/golden.rs:1485-1520`)
  - Hidden-state cosine similarity and relative-L2 comparison (`compare_hidden` in `minimax_m3/golden.rs:1246-1259` vs `step5/golden.rs:1416-1429`)
  - `Scores` NLL/KL accumulator struct and `summary()` JSON formatting (`minimax_m3/golden.rs:943-990` vs `step5/golden.rs`)
  - Prefix-cache byte-exact restore/continuation test driver (`check_prefix` in `minimax_m3/golden.rs:1175-1213` vs `step5/golden.rs`)
  - Spark `ExpertProtocolV2Request` construction (`minimax_m3/golden.rs:1279-1287` vs `step5/golden.rs`)
- **Recommendation:**
  - Expand `rust/crates/cuteafd-daemon/src/shared/golden_scores.rs` (or a new `shared::golden` module) to hold `Scores`, `compare_hidden`, `tokens.bin`/`logits.bin`/`layerNN.bin` readers, and the generic `check_prefix` harness.

---

### 9. [LOW-MEDIUM] Duplicate Chat Encoding & XML Tool-Call Parser Helpers (`cuteafd-api/src/openai/chat/minimax_m3/*` vs `step5`, `glm5`, `qwen4`)

- **MiniMax files:**
  - `rust/crates/cuteafd-api/src/openai/chat/minimax_m3.rs:69-108` (`MiniMaxEncoding::from_snapshot`, `stop_token_ids`)
  - `rust/crates/cuteafd-api/src/openai/chat/minimax_m3/parser.rs:3230-3518` (`MiniMaxOutputParser`, `emit_content`, `held_prefix`, `typed`, `types`, `admits`, `field`)
- **Mirrored existing functionality:**
  - `rust/crates/cuteafd-api/src/openai/chat/step5.rs:180-230` (`Step5Encoding::from_snapshot`, `stop_token_ids`)
  - `rust/crates/cuteafd-api/src/openai/chat/qwen4/parser.rs:604-808` (`QwenOutputParser`)
  - `rust/crates/cuteafd-api/src/openai/chat/glm5/parser.rs:985-1210` (`GlmOutputParser`)
- **What is duplicated:**
  1. `MiniMaxEncoding::from_snapshot` and `stop_token_ids` (`minimax_m3.rs:69-108`) duplicate `Step5Encoding::from_snapshot` (`step5.rs:180-230`): loading `tokenizer_config.json`, falling back from `chat_template.jinja` to `config["chat_template"]`, looking up `added_tokens` in `tokenizer.json`, and parsing scalar-or-array `eos_token_id` from `generation_config.json`.
  2. `MiniMaxOutputParser` (`minimax_m3/parser.rs`) duplicates the `<think>` / `</think>` / stop-marker / stop-sequence scanner (`step_text`, `markers`, `held_prefix`), the `emit_content` leading/trailing whitespace state machine (`content_started`, `held`, `before`), and the JSON-schema-driven XML argument value coercion (`typed`, `types`, `admits`, `field` at lines 3449-3517) from `qwen4/parser.rs` and `glm5/parser.rs`.
- **Recommendation:**
  - Extract snapshot template/token-id/eos loading helpers into `rust/crates/cuteafd-api/src/openai/chat.rs`.
  - Extract the shared streaming XML tool-parser primitives (`held_prefix`, whitespace-holding `ContentBuffer`, `<think>`/`</think>` + stop-marker `step_text` scanner, and JSON-schema parameter coercion) into a shared submodule under `cuteafd-api/src/openai/chat/`.

---

### 10. [LOW] AOT Export CLI & Manifest Writer Boilerplate (`export_b12x_minimax_aot.py` vs `export_b12x_step5_aot.py`)

- **MiniMax file:**
  - `python/tools/aot/export_b12x_minimax_aot.py:120-155`
- **Mirrored existing functionality:**
  - `python/tools/aot/export_b12x_step5_aot.py:269-302`
  - `python/tools/aot/export_b12x_kimi_k6_dense_aot.py`
- **What is duplicated:**
  - Iterating `(name, build)` entries under `with exportable_compilation():`, calling `program.export_to_c(str(destination), label, "cuteafd_" + label)`, `validate_exported_header(...)`, hashing `.o` and `.h` artifacts with SHA-256, and writing `dict(schema=1, sparkinfer=_pinned_sparkinfer.LOCK_DATA, programs=records)`.
- **Recommendation:**
  - Move the manifest-export loop into `python/tools/lib/_export_jobs.py` so family AOT scripts only declare their `builders(...)` and `programs(...)` tables.

---

## Summary Priority Table

| Priority | Area | MiniMax Files | Shared / Existing Target | Est. Duplicated LOC |
|---|---|---|---|---|
| **P0** | Device memory, stream & LM head runtime | `cuteafd-ffi/src/families/minimax.rs`, `minimax_m3/head.rs`, `shared/minimax_ffi.rs`, `minimax_m3/routed.rs` | `shared::memory::device::{Device, Allocation, Stream}`, `shared::target_head::launch`, `step5/routed.rs` | ~550 |
| **P0** | DSpark speculation FFI, ring state & workspace math | `cuteafd-ffi/src/families/minimax/dspark.rs`, `minimax_m3/dspark.rs`, `minimax_m3/backend.rs`, `cuteafd-loader/.../minimax_m3/dspark.rs` | `shared::speculation::block_dspark`, `glmf_dspark.rs`, `glm5::draft_representation` | ~480 |
| **P1** | Page-pool wave & COW validation | `minimax_m3/transaction.rs`, `minimax_m3/execution.rs` | `cuteafd_engine::prefix::PagePool` (shared with `step5/transaction.rs`) | ~120 |
| **P1** | Coordinator weight slicing, gate/up fusion & `DenseLoad` | `plan/families/minimax/coordinator.rs`, `minimax_m3/weights.rs` | `cuteafd-loader/src/plan/layout/`, `shared::experts::rtx::dense` (shared with `step5`) | ~280 |
| **P1** | `run_serve` HTTP & blocking worker bootstrap | `minimax_m3/serve.rs` | `shared::qwen::scheduler` (shared with `step5`, `qwen4`, `mimo_v2`) | ~90 |
| **P2** | Vision readiness & `TowerSpec` tensor packing | `minimax_m3/media.rs`, `shared/vision/minimax.rs` | `shared::vision` (shared with `glm5_flash`, `qwen4`) | ~220 |
| **P2** | Golden qualification harness & hidden metrics | `minimax_m3/golden.rs` | `shared::golden_scores` (shared with `step5/golden.rs`) | ~350 |
| **P2** | Chat encoding snapshot loader & XML tool parser helpers | `cuteafd-api/src/openai/chat/minimax_m3*` | `cuteafd-api/src/openai/chat/` (shared with `step5`, `qwen4`, `glm5`) | ~260 |
| **P3** | Python AOT export manifest writer | `python/tools/aot/export_b12x_minimax_aot.py` | `python/tools/lib/_export_jobs.py` (shared with `step5`, `kimi_k3`) | ~50 |
