//! GLM 5.3 Flash as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! Paged state, one 256-row allocation unit per page index: on each of the 11 MLA layers the
//! unit's four 64-row MLA pages of FP8 latent records (528 B per row) and DSA token keys
//! (512 B per row), and the pool-key page of the same index (64 pools of 4 tokens: 64 x 128
//! E4M3 then 64 FP32 scales) = 3.02 MB per unit (11.8 KB per token). Units are refcounted:
//! full units are shared, nobody writes them again (every sequence appends past its own
//! length), and a partial tail unit is copied (its rows and its complete pools).
//!
//! Mark: the KDA layers' recurrent state is per sequence and overwritten by every step, so a
//! snapshot copies it whole: every KDA layer's FP32 state `[64, 128, 128]` and conv window
//! (the last three q/k/v inputs), 34 layers = 140.8 MiB. The arena is sized by decoding lanes,
//! with the host tier holding the rest. A restore copies the mark back into the new
//! sequence's own KDA slot and maps its pool pages; nothing else of the state is positional.
//! The capture point must be where the KDA state is: `kda_len` (a speculative verify leaves
//! the state behind the placement until its kept rows are committed), so `capture_reach` is 0.
//!
//! Restores are exact frontiers only (`ReuseRule::EXACT`): recurrent state exists at the
//! points it was captured and nowhere else. The DFlash2 drafter is not captured: a restored
//! sequence drafts cold with `valid_from` at the restore point.
//!
//! Every copy is enqueued on the engine stream, in order with the forward passes, and `drain`
//! synchronizes it. Under a head split both GPUs hold identical copies of the paged state (the
//! replicated MLA projection and indexer write them) and each its own KDA heads' state: page
//! copies run on each GPU's stream, a mark holds both GPUs' halves (an arena per GPU), and the
//! host tier is off.
use super::engine::{GlmfEngine, GlmfPlacement, KPOOL, PAGE_ROWS, RECORD_BYTES, UNIT_PAGES, UNIT_ROWS};
use crate::shared::memory::DeviceAllocation;
use crate::shared::prefix::view;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, PrefixFamily, ReuseRule, TailCopy};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::copy::DeviceRange;

/// Token-key bytes per row (BF16 keys | gates, 256 wide).
const KEY_BYTES: usize = 512;
/// Pool-key page: 64 pools x 128 E4M3, then 64 FP32 scales.
const POOL_KEY_BYTES: usize = 128;
const POOL_SCALES: usize = PAGE_ROWS * POOL_KEY_BYTES;
const POOL_PAGE_BYTES: usize = PAGE_ROWS * (POOL_KEY_BYTES + 4);

pub(crate) struct GlmfPrefix<'e, 'a> {
    engine: &'e GlmfEngine<'a>,
    /// Per rank and MLA layer: records, token keys, pool keys.
    paged: Vec<(usize, [CuteafdDeviceBuffer; 3])>,
    mark_bytes: usize,
    /// Per rank: its part of every mark (its KDA heads' state) and the arena of those parts.
    arenas: Vec<(usize, Option<DeviceAllocation<'a>>)>,
    slots: usize,
}

impl<'e, 'a> GlmfPrefix<'e, 'a> {
    /// The family over `engine`'s buffers with a device arena of `slots(mark_bytes)` marks.
    pub fn new(engine: &'e GlmfEngine<'a>, slots: impl FnOnce(usize) -> usize) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix");
        let paged: Vec<_> = (0..engine.ranks())
            .flat_map(|rank| engine.paged_buffers_on(rank).into_iter().map(move |buffers| (rank, buffers))).collect();
        for (_, [records, keys, pools]) in &paged {
            ensure!(records.bytes >= engine.pages * PAGE_ROWS * RECORD_BYTES && keys.bytes >= engine.pages * PAGE_ROWS * KEY_BYTES
                && pools.bytes >= engine.pool_pages * POOL_PAGE_BYTES, "MLA cache buffers smaller than the units");
        }
        let parts: Vec<usize> = (0..engine.ranks())
            .map(|rank| engine.slot_regions_on(rank, 0).iter().map(|r| r.bytes).sum()).collect();
        let mark_bytes = parts.iter().sum();
        let slots = slots(mark_bytes);
        let arenas = parts.into_iter().enumerate().map(|(rank, part)| -> Result<_> {
            let arena = if slots > 0 { Some(engine.on(rank, || DeviceAllocation::new(engine.library, slots * part))?) }
                else { None };
            Ok((part, arena))
        }).collect::<Result<_>>()?;
        Ok(Self { engine, paged, mark_bytes, arenas, slots })
    }

    pub fn mark_bytes(&self) -> usize {
        self.mark_bytes
    }

    pub fn page_bytes(&self) -> usize {
        self.paged.len() * (UNIT_ROWS * (RECORD_BYTES + KEY_BYTES) + POOL_PAGE_BYTES)
    }

    pub fn slots(&self) -> usize {
        if self.arenas.iter().all(|(_, arena)| arena.is_some()) { self.slots } else { 0 }
    }

    /// A copy on rank `rank`'s stream.
    fn copy(&self, rank: usize, dst: CuteafdDeviceBuffer, src: CuteafdDeviceBuffer) -> Result<()> {
        debug_assert_eq!(dst.bytes, src.bytes);
        // SAFETY: both views lie inside live engine allocations of that rank's GPU (checked by
        // `view`); the copy is ordered on its stream with every forward pass that reads or
        // writes them.
        self.engine.on(rank, || unsafe {
            self.engine.library.copy_d2d_async(dst, src, src.bytes, self.engine.stream_of(rank))
        })
    }

    /// The device ranges of one unit on rank `rank`, per MLA layer: records, token keys, pool keys.
    fn unit_ranges_on(&self, rank: usize, unit: u32) -> Result<Vec<CuteafdDeviceBuffer>> {
        let unit = unit as usize;
        let rows = UNIT_PAGES * PAGE_ROWS;
        let mut out = Vec::with_capacity(3 * self.paged.len());
        for &(_, [records, keys, pools]) in self.paged.iter().filter(|(r, _)| *r == rank) {
            out.push(view(records, unit * rows * RECORD_BYTES, rows * RECORD_BYTES)?);
            out.push(view(keys, unit * rows * KEY_BYTES, rows * KEY_BYTES)?);
            out.push(view(pools, unit * POOL_PAGE_BYTES, POOL_PAGE_BYTES)?);
        }
        Ok(out)
    }

    /// Mark `slot`'s part on each rank: (rank, its arena range).
    fn mark_parts(&self, slot: MarkSlot) -> Result<Vec<(usize, CuteafdDeviceBuffer)>> {
        ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
        self.arenas.iter().enumerate().map(|(rank, (part, arena))| {
            let arena = arena.as_ref().map(|a| a.buffer).ok_or_else(|| anyhow::anyhow!("no mark arena"))?;
            Ok((rank, view(arena, slot.0 as usize * part, *part)?))
        }).collect()
    }

    /// Copy every KDA layer's state of KDA slot `kda` to or from mark `slot` (each rank its heads).
    fn move_mark(&self, slot: MarkSlot, kda: i32, capture: bool) -> Result<()> {
        ensure!(kda >= 0 && (kda as usize) < self.engine.slots, "KDA slot {kda} of {}", self.engine.slots);
        for (rank, part) in self.mark_parts(slot)? {
            let mut offset = 0;
            for region in self.engine.slot_regions_on(rank, kda as usize) {
                let mark = view(part, offset, region.bytes)?;
                if capture {
                    self.copy(rank, mark, region)?;
                } else {
                    self.copy(rank, region, mark)?;
                }
                offset += region.bytes;
            }
        }
        Ok(())
    }

    /// Mark `slot`'s bytes, rank by rank (checks).
    fn mark_host(&self, slot: MarkSlot) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for (rank, part) in self.mark_parts(slot)? {
            out.extend(download(self.engine, rank, part)?);
        }
        Ok(out)
    }
}

impl PrefixFamily for GlmfPrefix<'_, '_> {
    type Placement = GlmfPlacement;

    fn layout(&self) -> FamilyLayout {
        FamilyLayout {
            page_rows: UNIT_ROWS,
            pages: self.engine.pool_pages,
            page_bytes: self.page_bytes(),
            mark_bytes: self.mark_bytes,
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
        }
    }

    fn pages<'p>(&self, placement: &'p GlmfPlacement) -> &'p [u32] {
        &placement.units
    }

    fn commit_point(&self, placement: &GlmfPlacement) -> usize {
        placement.kda_len.min(placement.len)
    }

    fn capture(&self, slot: MarkSlot, placement: &GlmfPlacement, len: usize) -> Result<(), BoxError> {
        if len != placement.len || len != placement.kda_len {
            return Err(format!("capture at {len}: the placement holds {} rows, its KDA state {}", placement.len,
                placement.kda_len).into());
        }
        Ok(self.move_mark(slot, placement.slot, true)?)
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut GlmfPlacement, len: usize) -> Result<(), BoxError> {
        let Some(slot) = mark else {
            return Err("GLM 5.3 Flash restores exact snapshots with their KDA state".into());
        };
        if len == 0 || len.div_ceil(UNIT_ROWS) > placement.units.len() {
            return Err(format!("restore of {len} rows into {} units", placement.units.len()).into());
        }
        self.engine.map_pools(placement)?;
        self.move_mark(slot, placement.slot, false)?;
        placement.len = len;
        placement.kda_len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let rows = UNIT_PAGES * PAGE_ROWS;
        let pools = copy.rows / KPOOL;
        for &(rank, [records, keys, pool_keys]) in &self.paged {
            for (buffer, row) in [(records, RECORD_BYTES), (keys, KEY_BYTES)] {
                self.copy(rank, view(buffer, copy.to as usize * rows * row, copy.rows * row)?,
                    view(buffer, copy.from as usize * rows * row, copy.rows * row)?)?;
            }
            // The complete pools: their E4M3 keys, then their scales.
            if pools > 0 {
                let (from, to) = (copy.from as usize * POOL_PAGE_BYTES, copy.to as usize * POOL_PAGE_BYTES);
                self.copy(rank, view(pool_keys, to, pools * POOL_KEY_BYTES)?,
                    view(pool_keys, from, pools * POOL_KEY_BYTES)?)?;
                self.copy(rank, view(pool_keys, to + POOL_SCALES, pools * 4)?,
                    view(pool_keys, from + POOL_SCALES, pools * 4)?)?;
            }
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> {
        Ok(self.engine.synchronize()?)
    }

    /// The host tier's ranges (rank 0's: a head split runs without the host tier).
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.unit_ranges_on(0, page).unwrap_or_default().into_iter()
            .map(|b| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.mark_parts(slot).unwrap_or_default().into_iter()
            .map(|(_, b)| DeviceRange { addr: b.ptr as u64, bytes: b.bytes }).collect()
    }
}

/// One continued prefill: every layer's output digest, every row's logits digest and argmax,
/// and the last row.
pub(crate) struct SuffixRun {
    pub layers: Vec<u64>,
    pub logits: u64,
    pub argmax: Vec<usize>,
    pub last: Vec<f32>,
}

/// Prefill `tokens` into `placement` in chunks of `chunk` rows, digesting each layer's streams
/// and (with `logits`) every row's logits.
pub(crate) fn prefill_digest(engine: &GlmfEngine<'_>, placement: &mut GlmfPlacement, tokens: &[u32], chunk: usize,
    logits: bool) -> Result<SuffixRun> {
    use std::hash::{Hash, Hasher};
    let vocab = engine.cfg.vocab_size;
    let mut hashers: Vec<std::collections::hash_map::DefaultHasher> = Vec::new();
    let mut logit_hash = std::collections::hash_map::DefaultHasher::new();
    let (mut argmax, mut last) = (Vec::new(), Vec::new());
    for part in tokens.chunks(chunk) {
        let mut on_layer = |layer: usize, rows: &[u8]| -> Result<()> {
            if hashers.len() <= layer {
                hashers.resize_with(layer + 1, Default::default);
            }
            rows.hash(&mut hashers[layer]);
            Ok(())
        };
        let out = engine.prefill_forced(placement, part, Some(&mut on_layer), None, logits)?
            .ok_or_else(|| anyhow::anyhow!("the resume check needs every layer"))?;
        if logits {
            for values in out.chunks_exact(vocab) {
                values.iter().for_each(|v| v.to_bits().hash(&mut logit_hash));
                argmax.push(values.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i));
            }
            last = out[out.len() - vocab..].to_vec();
        }
    }
    Ok(SuffixRun { layers: hashers.iter().map(Hasher::finish).collect(), logits: logit_hash.finish(), argmax, last })
}

/// `range` of rank `rank`'s GPU, after every stream drained (retiring every write to it).
fn download(engine: &GlmfEngine<'_>, rank: usize, range: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
    engine.synchronize()?;
    let mut bytes = vec![0u8; range.bytes];
    engine.on(rank, || engine.library.copy_d2h(&mut bytes, range))?;
    Ok(bytes)
}

/// Every paged byte of `placement`'s first `len` rows (records and token keys of each row,
/// pool keys and scales of each complete pool), MLA layer by layer, in position order.
pub(super) fn paged_rows(family: &GlmfPrefix<'_, '_>, placement: &GlmfPlacement, len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let pools = len / KPOOL;
    for (u, &unit) in placement.units.iter().enumerate().take(len.div_ceil(UNIT_ROWS)) {
        let rows = (len - u * UNIT_ROWS).min(UNIT_ROWS);
        let unit_pools = (pools.saturating_sub(u * PAGE_ROWS)).min(PAGE_ROWS);
        for rank in 0..family.engine.ranks() {
            for ranges in family.unit_ranges_on(rank, unit)?.chunks_exact(3) {
                out.extend_from_slice(&download(family.engine, rank, ranges[0])?[..rows * RECORD_BYTES]);
                out.extend_from_slice(&download(family.engine, rank, ranges[1])?[..rows * KEY_BYTES]);
                let page = download(family.engine, rank, ranges[2])?;
                out.extend_from_slice(&page[..unit_pools * POOL_KEY_BYTES]);
                out.extend_from_slice(&page[POOL_SCALES..POOL_SCALES + unit_pools * 4]);
            }
        }
    }
    Ok(out)
}

/// glmf-golden --resume-at P: prefill `tokens[..P]` into sequence A, capture its snapshot (shared
/// units, the copied tail unit, the KDA mark), restore it into sequence B (its own KDA slot and
/// pool-page mapping), continue both with the same chunks and `decode` greedy single-row steps,
/// and compare every layer's rows, every logit, every paged row, the KDA state and the mark
/// round trip byte for byte (a restore must be exact). A straight prefill without the boundary
/// at P is reported too (informational: chunking changes may round differently).
/// With `cold`, B is prefilled from scratch on its own units instead (no restore): the floor of
/// what the kernels themselves vary. The check runs `repeat` times on fresh sequences (every
/// attempt must be identical: the DSA top-k is deterministic, ties going to the lower index).
#[allow(clippy::too_many_arguments)]
pub(crate) fn resume_check(engine: &GlmfEngine<'_>, tokens: &[u32], at: usize, n: usize, chunk: usize, decode: usize, cold: bool, repeat: usize) -> Result<()> {
    use super::engine::Allocator;
    ensure!(engine.weights.layers.len() == engine.cfg.layers, "--resume-at needs every layer");
    ensure!(engine.full_prefill_logits, "--resume-at needs every prefill row's logits");
    ensure!(at > 0 && at < n && n <= tokens.len(), "--resume-at {at} must lie inside the {n} prefilled tokens");
    let chunk = chunk.clamp(1, engine.prefill_rows);
    let embed = &tokens[..n];
    let family = GlmfPrefix::new(engine, |_| 2)?;
    let mut allocator = Allocator::new(engine.pages, engine.slots);
    let err = |e: BoxError| anyhow::anyhow!("{e}");
    let (mut identical, mut marks_identical, mut last) = (0, 0, None);
    for _ in 0..repeat.max(1) {
    // A: prefill [0, P), capture, continue in place.
    let mut a = allocator.admit(n + decode)?;
    prefill_digest(engine, &mut a, &embed[..at], chunk, false)?;
    family.drain().map_err(err)?;
    let started = std::time::Instant::now();
    family.capture(MarkSlot(0), &a, at).map_err(err)?;
    // B: a second sequence restored from the snapshot (shared full units, its own tail and KDA slot).
    let mut b = if cold {
        // The floor: B prefilled cold on its own units (what placement alone changes).
        let mut b = allocator.admit(n + decode)?;
        prefill_digest(engine, &mut b, &embed[..at], chunk, false)?;
        family.capture(MarkSlot(1), &b, at).map_err(err)?;
        family.restore(Some(MarkSlot(1)), &mut b, at).map_err(err)?;
        b
    } else {
        let (mut b, copy) = allocator.fork(&a, at, n + decode)?;
        if let Some(copy) = copy {
            family.copy_rows(copy).map_err(err)?;
        }
        family.restore(Some(MarkSlot(0)), &mut b, at).map_err(err)?;
        b
    };
    family.drain().map_err(err)?;
    let restore_ms = started.elapsed().as_secs_f64() * 1e3;
    // The restored KDA state reads back exactly as the captured one.
    family.capture(MarkSlot(1), &b, at).map_err(err)?;
    let mark_equal = family.mark_host(MarkSlot(0))? == family.mark_host(MarkSlot(1))?;
    // The state at P (every paged row and the KDA state): what a restore must reproduce.
    let state_at = paged_rows(&family, &a, at)? == paged_rows(&family, &b, at)?
        && engine.slot_state(a.slot)? == engine.slot_state(b.slot)?;
    let straight = prefill_digest(engine, &mut a, &embed[at..], chunk, true)?;
    let restored = prefill_digest(engine, &mut b, &embed[at..], chunk, true)?;
    let first_layer = straight.layers.iter().zip(&restored.layers).position(|(x, y)| x != y);
    let logits_equal = straight.logits == restored.logits;
    let max_diff = straight.last.iter().zip(&restored.last).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    // Greedy single-row decode steps from both (decode graphs, the restored pool mapping).
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(i, _)| i as u32);
    let (mut next_a, mut next_b) = (argmax(&straight.last), argmax(&restored.last));
    let mut decode_equal = next_a == next_b;
    for _ in 0..decode {
        let la = engine.verify(&mut [(&mut a, 1)], &[next_a], None)?.context("decode logits")?;
        let lb = engine.verify(&mut [(&mut b, 1)], &[next_b], None)?.context("decode logits")?;
        decode_equal &= la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits());
        (next_a, next_b) = (argmax(&la), argmax(&lb));
    }
    let len = a.len;
    let paged_equal = paged_rows(&family, &a, len)? == paged_rows(&family, &b, len)?;
    let state_equal = engine.slot_state(a.slot)? == engine.slot_state(b.slot)?;
    println!("resume at {at} of {n} (chunks of {chunk}, {decode} decode steps): state at {at} {} | layers {} | logits \
        {} (last row max |diff| {max_diff:.3e}) | decode {} | paged rows 0..{len} {} | KDA state {} | mark round trip \
        {} ({} B), capture+restore {restore_ms:.1} ms", if state_at { "identical" } else { "DIFFERS" },
        first_layer.map_or("identical".to_string(), |l| format!("differ from layer {l}")),
        if logits_equal { "identical" } else { "DIFFER" }, if decode_equal { "identical" } else { "DIFFERS" },
        if paged_equal { "identical" } else { "DIFFER" }, if state_equal { "identical" } else { "DIFFERS" },
        if mark_equal { "identical" } else { "DIFFERS" }, family.mark_bytes());
    allocator.release(b);
    allocator.release(a);
    identical += usize::from(first_layer.is_none() && logits_equal && decode_equal && paged_equal && state_equal
        && mark_equal);
    marks_identical += usize::from(mark_equal && state_at);
    last = Some(restored);
    }
    let restored = last.context("no attempt")?;
    let repeat = repeat.max(1);
    // C: one prefill with no boundary at P (chunking changes may round differently; informational).
    let mut c = allocator.admit(n)?;
    let whole = prefill_digest(engine, &mut c, embed, chunk, true)?;
    let x = &whole.argmax[at..];
    let agree = x.iter().zip(&restored.argmax).filter(|(p, q)| p == q).count();
    let last_equal = whole.last.iter().zip(&restored.last).all(|(p, q)| p.to_bits() == q.to_bits());
    println!("vs one straight prefill without the boundary at {at}: suffix top-1 agreement {agree}/{}, last row logits {}",
        x.len(), if last_equal { "identical" } else { "differ" });
    allocator.release(c);
    println!("resume at {at} of {n} (chunks of {chunk}): {identical}/{repeat} attempts byte-identical, mark round trip \
        and state at {at} identical in {marks_identical}/{repeat}{}", if cold { " [cold floor: B prefilled, not restored]" } else { "" });
    ensure!(marks_identical == repeat, "the restored state differs from the captured one");
    ensure!(identical == repeat, "the restored sequence's continuation differs from the straight one");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::engine::{Allocator, GlmfPlacement};

    /// GLM-5.3-Flash's text config: 34 KDA layers (64 heads of 128) and 11 MLA layers.
    fn glm53_flash() -> cuteafd_loader::families::glm5_flash::GlmNextConfig {
        let types: Vec<&str> = (0..45).map(|l| if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" })
            .collect();
        let mlp: Vec<&str> = (0..45).map(|l| if l < 3 { "dense" } else { "sparse" }).collect();
        cuteafd_loader::families::glm5_flash::GlmNextConfig::from_hf(&serde_json::json!({
            "model_type": "glm5_next", "text_config": {
                "model_type": "glm5_next_text", "vocab_size": 154880, "hidden_size": 4096, "num_hidden_layers": 45,
                "layer_types": types, "mlp_layer_types": mlp, "intermediate_size": 12288, "n_routed_experts": 288,
                "num_experts_per_tok": 8, "moe_intermediate_size": 2048, "routed_scaling_factor": 2.5,
                "swiglu_limit": 10.0, "rms_norm_eps": 1e-5, "hc_mult": 4, "mla_use_nope": true,
                "qk_rope_head_dim": 0, "num_attention_heads": 64, "q_lora_rank": 1536, "kv_lora_rank": 512,
                "qk_nope_head_dim": 256, "v_head_dim": 256, "index_topk": 2048, "index_kpool": 4,
                "eos_token_id": [154820, 154827, 154829],
                "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4,
                                       "gate_lower_bound": -5.0}}})).unwrap()
    }

    /// The KV admission reserves the engine's own state: the planner's mark and replay bytes
    /// are the engine's slot regions and replay records (per GPU of a head split too), and the
    /// arena slots it reserves are the ones `prefix_cache` allocates (2C + 2 for 147.6 MB marks).
    #[test]
    fn the_planner_reserves_the_marks_and_replay_records_the_engine_allocates() {
        use super::super::engine::kda_layer_bytes;
        use crate::shared::prefix::PrefixArgs;
        use clap::Parser;
        use cuteafd_loader::families::glm5_flash::GlmNextAttention;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            prefix: PrefixArgs,
        }
        let cfg = glm53_flash();
        let kda = cfg.attention.iter().filter(|&&a| a == GlmNextAttention::Kda).count();
        for ranks in [1, 2] {
            let geometry = cuteafd_loader::serving_capacity::glm_flash_rank_cache_geometry(&cfg, cfg.layers, ranks)
                .unwrap();
            let (state, conv, replay) = kda_layer_bytes(&cfg, cfg.kda_heads / ranks);
            let rank = &geometry.ranks[0];
            assert_eq!((rank.retained_mark_bytes, rank.active_state_per_sequence_bytes),
                ((kda * (state + conv)) as u64, (kda * (state + conv)) as u64));
            assert_eq!(rank.speculative_replay_bytes, (kda * replay) as u64);
        }
        let mark = kda * (kda_layer_bytes(&cfg, cfg.kda_heads).0 + kda_layer_bytes(&cfg, cfg.kda_heads).1);
        assert_eq!((mark, kda * kda_layer_bytes(&cfg, cfg.kda_heads).2), (147_619_840, 321_421_312));
        let prefix = Cli::parse_from(["serve"]).prefix;
        for (lanes, slots) in [(4, 14), (8, 18), (16, 34), (64, 130)] {
            let planned = super::super::serve::arena_mark_slots(&prefix, &cfg, cfg.layers, lanes).unwrap();
            let allocated = cuteafd_engine::prefix::MarkArena::slots_for(lanes, prefix.prefix_cache_entries, mark,
                prefix.prefix_cache_mark_mib << 20);
            assert_eq!((planned, allocated), (slots, slots), "{lanes} lanes");
        }
        let off = PrefixArgs { prefix_cache_entries: 0, ..prefix };
        assert_eq!(super::super::serve::arena_mark_slots(&off, &cfg, cfg.layers, 16).unwrap(), 0);
    }

    #[test]
    fn units_expand_to_mla_and_pool_pages_and_forks_share_whole_units() {
        let p = GlmfPlacement::new(vec![2, 0], 1);
        assert_eq!(p.pages, vec![8, 9, 10, 11, 0, 1, 2, 3]);
        assert_eq!(p.pool_pages, vec![2, 0]);
        // Position 300: the second unit (index 0), its MLA page 0 (page 0), row 44; pool 74 (row 10
        // of pool page 0) completes at 299.
        assert_eq!(p.record(300).unwrap(), 44);
        assert_eq!(p.pool_slot(299).unwrap(), 10);
        assert_eq!(p.pool_slot(300).unwrap(), -1);
        assert_eq!(p.pool_slot(255).unwrap(), 2 * 64 + 63);
        let mut allocator = Allocator::new(4 * 8, 3);
        let a = allocator.admit(600).unwrap();
        assert_eq!(a.units.len(), 3);
        let (b, copy) = allocator.fork(&a, 300, 900).unwrap();
        assert_eq!((b.units.len(), b.units[0]), (4, a.units[0]));
        let copy = copy.unwrap();
        assert_eq!((copy.from, copy.to, copy.rows), (a.units[1], b.units[1], 44));
        assert_ne!(a.slot, b.slot);
        allocator.release(b);
        allocator.release(a);
        assert_eq!(allocator.admit(8 * 256).unwrap().units.len(), 8);
    }
}
