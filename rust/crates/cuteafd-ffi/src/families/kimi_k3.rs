//! Kimi NoPE MLA, residual-mix and Situ programs, resolved before capture.
//! The engine owns weights, caches, arenas, outputs and scratch through completion.
use crate::programs::{Program, ProgramSpec, Programs, Scalar};
use crate::CuteafdDeviceBuffer as Buffer;
use std::ffi::c_void;
use thiserror::Error;

pub const MLA_RECORD_BYTES: usize = 1152;
pub const MLA_PAGE_ROWS: usize = 64;
const PRODUCER: &[&str] = &["x", "kv_slots", "w_qkv_a", "q_a_norm", "kv_a_norm", "w_q_b", "w_uk", "kv_cache", "query", "q_resid", "scratch"];
const ATTENTION: &[&str] = &["query", "kv_cache", "page_table", "cache_seqlens", "cu_seqlens_q", "attn", "lse", "scratch"];
const OUTPUT: &[&str] = &["attn", "x", "w_uv", "w_g", "w_o", "out", "scratch"];

#[derive(Debug, Error)]
pub enum KimiMlaError {
    #[error("Kimi MLA invalid {0}")]
    Invalid(&'static str),
    #[error("Kimi MLA program: {0}")]
    Program(#[from] anyhow::Error),
}
type Result<T> = std::result::Result<T, KimiMlaError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KimiMlaMode { Decode, Verify, Extend }
impl KimiMlaMode {
    fn name(self) -> &'static str {
        match self { Self::Decode => "decode", Self::Verify => "verify", Self::Extend => "extend" }
    }
}

/// BF16 weights, packed once from the checkpoint. UK is [heads,512,128],
/// UV is [heads,128,512]; q_a and kv_a are concatenated by output rows.
#[derive(Clone, Copy, Debug)]
pub struct KimiMlaWeights {
    pub qkv_a: Buffer, pub q_a_norm: Buffer, pub kv_a_norm: Buffer,
    pub q_b: Buffer, pub uk: Buffer, pub uv: Buffer, pub gate: Buffer, pub output: Buffer,
}

/// All extents are bytes. Buffers are dense and on one CUDA device. The page
/// table and slots must address allocated pages; slots may be -1 to skip a
/// cache write. Each request has at least one query and sufficient KV history.
#[derive(Clone, Copy, Debug)]
pub struct KimiMlaBatch {
    pub rows: usize, pub requests: usize, pub page_table_width: usize,
    pub cache_pages: usize, pub active_splits: usize,
    pub x: Buffer, pub slots: Buffer, pub cache: Buffer,
    pub page_table: Buffer, pub lengths: Buffer, pub query_offsets: Buffer,
    pub query: Buffer, pub q_resid: Buffer, pub attention: Buffer, pub lse: Buffer,
    pub output: Buffer, pub producer_scratch: Buffer, pub attention_scratch: Buffer,
    pub output_scratch: Buffer,
}

/// Resolved producer -> full causal dense MLA -> UV/sigmoid gate/o_proj.
/// With 48 heads output is one TP2 partial; the caller performs the TP meet.
pub struct KimiMla<'a> {
    producer: Program<'a>, attention: Program<'a>, output: Program<'a>,
    heads: usize, rows: usize, context: usize, splits: usize,
    scratch: [usize; 3],
}

impl<'a> KimiMla<'a> {
    pub fn load(programs: &'a Programs<'a>, heads: usize, rows: usize, mode: KimiMlaMode) -> Result<Self> {
        if !matches!(heads, 48 | 96) || !(1..=4096).contains(&rows) {
            return Err(KimiMlaError::Invalid("head/row geometry"));
        }
        let suffix = if heads == 48 { "" } else { "_h96" };
        let producer = programs.program(&format!("kimik3_mla_producer{suffix}_m{rows}"), PRODUCER)?;
        let attention = programs.program(&format!("kimik3_dense_mla_{}{suffix}_m{rows}", mode.name()), ATTENTION)?;
        let output = programs.program(&format!("kimik3_mla_output{suffix}_m{rows}"), OUTPUT)?;
        let (context, splits, scratch) = validate_manifest(
            [producer.spec(), attention.spec(), output.spec()], heads, rows, mode)?;
        Ok(Self { producer, attention, output, heads, rows, context, splits, scratch })
    }

    pub fn scratch_bytes(&self) -> [usize; 3] { self.scratch }
    pub fn max_context(&self) -> usize { self.context }
    pub fn splits(&self) -> usize { self.splits }

    /// Enqueue without allocation of device storage, synchronization or lazy
    /// kernel loading. The owning engine serializes accesses and retains all
    /// buffers through completion and every replay of a captured graph.
    pub fn launch(&self, weights: KimiMlaWeights, b: KimiMlaBatch, stream: *mut c_void) -> Result<()> {
        if b.rows == 0 || b.rows > self.rows || b.requests == 0 || b.requests > b.rows
            || b.page_table_width == 0 || b.page_table_width > self.context.div_ceil(MLA_PAGE_ROWS)
            || b.cache_pages == 0 || b.cache_pages > i32::MAX as usize
            || b.active_splits != self.splits {
            return Err(KimiMlaError::Invalid("batch/context/split geometry"));
        }
        let r = b.rows;
        let h = self.heads;
        let device = b.x.device_id;
        let requirements = [
            (b.x, r * 7168 * 2), (b.slots, r * 8),
            (b.cache, b.cache_pages * MLA_PAGE_ROWS * MLA_RECORD_BYTES),
            (b.page_table, b.requests * b.page_table_width * 4),
            (b.lengths, b.requests * 4), (b.query_offsets, (b.requests + 1) * 4),
            (b.query, r * h * 576 * 2), (b.q_resid, r * 1536 * 2),
            (b.attention, r * h * 512 * 2), (b.lse, r * h * 4), (b.output, r * 7168 * 2),
            (b.producer_scratch, self.scratch[0]), (b.attention_scratch, self.scratch[1]),
            (b.output_scratch, self.scratch[2]),
            (weights.qkv_a, 2112 * 7168 * 2), (weights.q_a_norm, 1536 * 2),
            (weights.kv_a_norm, 512 * 2), (weights.q_b, h * 192 * 1536 * 2),
            (weights.uk, h * 512 * 128 * 2), (weights.uv, h * 128 * 512 * 2),
            (weights.gate, h * 128 * 7168 * 2), (weights.output, 7168 * h * 128 * 2),
        ];
        validate_buffers(&requirements, device)?;
        // SAFETY: extents, alignment and devices are checked; the engine owns
        // the live buffers and stream, ordered metadata and graph lifetimes.
        unsafe {
            self.producer.launch(&[b.x.ptr, b.slots.ptr, weights.qkv_a.ptr, weights.q_a_norm.ptr,
                weights.kv_a_norm.ptr, weights.q_b.ptr, weights.uk.ptr, b.cache.ptr, b.query.ptr,
                b.q_resid.ptr, b.producer_scratch.ptr], &[Scalar::I32(r as i32)], stream)?;
            self.attention.launch(&[b.query.ptr, b.cache.ptr, b.page_table.ptr, b.lengths.ptr,
                b.query_offsets.ptr, b.attention.ptr, b.lse.ptr, b.attention_scratch.ptr],
                &[Scalar::I32(r as i32), Scalar::I32(b.requests as i32), Scalar::I64(b.page_table_width as i64),
                  Scalar::I32(b.active_splits as i32)], stream)?;
            self.output.launch(&[b.attention.ptr, b.x.ptr, weights.uv.ptr, weights.gate.ptr,
                weights.output.ptr, b.output.ptr, b.output_scratch.ptr], &[Scalar::I32(r as i32)], stream)?;
        }
        Ok(())
    }
}

fn validate_manifest(specs: [&ProgramSpec; 3], heads: usize, rows: usize, mode: KimiMlaMode)
    -> Result<(usize, usize, [usize; 3])> {
    for (i, spec) in specs.iter().enumerate() {
        let geometry = &spec.geometry;
        if spec.capacity_rows as usize != rows || geometry["heads"].as_u64() != Some(heads as u64)
            || [("hidden", 7168), ("q_lora_rank", 1536), ("kv_lora_rank", 512),
                ("qk_nope_dim", 128), ("qk_rope_dim", 64), ("v_head_dim", 128), ("page_rows", 64)]
                .iter().any(|&(key, n)| geometry[key].as_u64() != Some(n))
            || geometry["norm_eps"].as_f64() != Some(1e-6)
            || spec.scalar_kinds.as_slice() != if i == 1 { &b"iili"[..] } else { &b"i"[..] } {
            return Err(KimiMlaError::Invalid("manifest geometry/capacity/scalars"));
        }
    }
    let attention = &specs[1].geometry;
    let context = attention["max_context"].as_u64().filter(|&n| n > 0 && n <= 1048576)
        .ok_or(KimiMlaError::Invalid("manifest context"))? as usize;
    let splits = if mode == KimiMlaMode::Extend { 1 } else { context.div_ceil(64).min(16) };
    let chunks = context.div_ceil(64).div_ceil(splits);
    if attention["mode"].as_str() != Some(mode.name()) || attention["num_splits"].as_u64() != Some(splits as u64)
        || attention["chunks_per_split"].as_u64() != Some(chunks as u64) {
        return Err(KimiMlaError::Invalid("manifest traversal"));
    }
    let align = |n: usize| n.div_ceil(1024) * 1024;
    let expected = [align(rows * 2112 * 2) + align(rows * heads * 192 * 2),
        if splits > 1 { align(rows * heads * splits * 512 * 2) + align(rows * heads * splits * 4) } else { 0 },
        2 * align(rows * heads * 128 * 2)];
    for (spec, bytes) in specs.iter().zip(expected) {
        if spec.scratch.get("scratch").copied() != Some(bytes as u64) {
            return Err(KimiMlaError::Invalid("manifest scratch"));
        }
    }
    Ok((context, splits, expected))
}

fn validate_buffers(requirements: &[(Buffer, usize)], device: i32) -> Result<()> {
    if device < 0 { return Err(KimiMlaError::Invalid("device")); }
    for (buffer, bytes) in requirements {
        if buffer.ptr.is_null() || buffer.bytes < *bytes || buffer.device_id != device
            || buffer.ptr as usize % 16 != 0 {
            return Err(KimiMlaError::Invalid("buffer extent/alignment/device"));
        }
    }
    Ok(())
}

/// The prefix update is stored in BF16 before candidate scores are computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KimiPrefixMode { Keep, Add, Replace }
impl KimiPrefixMode {
    fn name(self) -> &'static str {
        match self { Self::Keep => "keep", Self::Add => "add", Self::Replace => "replace" }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KimiResidualSite {
    pub candidates: usize,
    pub prefix: KimiPrefixMode,
    pub append: bool,
}
impl KimiResidualSite {
    fn valid(self) -> bool {
        if self.candidates == 1 {
            return self.prefix == KimiPrefixMode::Keep && self.append;
        }
        (2..=9).contains(&self.candidates)
            && (!self.append || (self.candidates <= 8 && self.prefix == KimiPrefixMode::Add))
    }
    fn name(self) -> String {
        format!("kimik3_attn_res_c{}_{}{}", self.candidates, self.prefix.name(),
            if self.append { "_append" } else { "" })
    }
}

#[derive(Debug, Error)]
pub enum KimiW8Error {
    #[error("Kimi residual/Situ invalid {0}")]
    Invalid(&'static str),
    #[error("Kimi residual/Situ program: {0}")]
    Program(#[from] anyhow::Error),
}
type W8Result<T> = std::result::Result<T, KimiW8Error>;

/// Arena is [9, arena_rows, 7168] BF16, including immutable blocks 0..7
/// and the mutable prefix at slot 8. Padding rows retain their contents.
#[derive(Clone, Copy, Debug)]
pub struct KimiResidualBatch {
    pub rows: usize,
    pub arena_rows: usize,
    pub arena: Buffer,
    pub delta: Buffer,
    /// F32 checkpoint norm.float() * projection.float(), prepared once.
    pub score_weight: Buffer,
    pub norm_weight: Buffer,
    pub output: Buffer,
}

/// Resolved before capture; no allocation, lazy loading or synchronization
/// occurs on launch. The caller retains buffers through queued work/replays.
pub struct KimiResidual<'a> {
    program: Program<'a>,
    rows: usize,
}
impl<'a> KimiResidual<'a> {
    pub fn load(programs: &'a Programs<'a>, site: KimiResidualSite) -> W8Result<Self> {
        if !site.valid() { return Err(KimiW8Error::Invalid("residual site")); }
        let program = programs.program(&site.name(),
            &["arena", "delta", "score_weight", "norm_weight", "out"])?;
        validate_residual_manifest(program.spec(), site)?;
        let rows = program.spec().capacity_rows as usize;
        Ok(Self { program, rows })
    }

    pub fn arena_bytes(arena_rows: usize) -> W8Result<usize> {
        if arena_rows == 0 || arena_rows > i32::MAX as usize {
            return Err(KimiW8Error::Invalid("arena rows"));
        }
        arena_rows.checked_mul(9 * 7168 * 2).ok_or(KimiW8Error::Invalid("arena extent overflow"))
    }

    pub fn launch(&self, b: KimiResidualBatch, stream: *mut c_void) -> W8Result<()> {
        if b.rows == 0 || b.rows > self.rows || b.rows > b.arena_rows {
            return Err(KimiW8Error::Invalid("residual rows/capacity"));
        }
        let arena_bytes = Self::arena_bytes(b.arena_rows)?;
        let row_bytes = b.rows.checked_mul(7168 * 2).ok_or(KimiW8Error::Invalid("row extent overflow"))?;
        let buffers = [(b.arena, arena_bytes), (b.delta, row_bytes),
            (b.score_weight, 7168 * 4), (b.norm_weight, 7168 * 2), (b.output, row_bytes)];
        validate_w8_buffers(&buffers, b.arena.device_id)?;
        // Every mutable range is disjoint from all other operands, including
        // reads. Arena slots themselves are intentionally one inout operand.
        for writer in [0, 4] {
            for other in 0..buffers.len() {
                if writer != other { reject_overlap(buffers[writer], buffers[other])?; }
            }
        }
        // SAFETY: checked geometry, integer bounds, extents, alignment, device
        // and write aliases; the engine owns buffer/stream/graph lifetimes.
        unsafe {
            self.program.launch(&[b.arena.ptr, b.delta.ptr, b.score_weight.ptr,
                b.norm_weight.ptr, b.output.ptr], &[Scalar::I32(b.rows as i32),
                Scalar::I32(b.arena_rows as i32), Scalar::F32(1e-5)], stream)?;
        }
        Ok(())
    }
}

pub struct KimiSitu<'a> {
    program: Program<'a>,
    width: usize,
    rows: usize,
}
impl<'a> KimiSitu<'a> {
    pub fn load(programs: &'a Programs<'a>, intermediate: usize) -> W8Result<Self> {
        if !matches!(intermediate, 3072 | 6144 | 16896 | 33792) {
            return Err(KimiW8Error::Invalid("Situ width"));
        }
        let program = programs.program(&format!("kimik3_situ_i{intermediate}"), &["gate_up", "out"])?;
        validate_w8_manifest(program.spec(), b"i")?;
        if program.spec().geometry["intermediate"].as_u64() != Some(intermediate as u64) {
            return Err(KimiW8Error::Invalid("Situ manifest width"));
        }
        let rows = program.spec().capacity_rows as usize;
        Ok(Self { program, width: intermediate, rows })
    }

    /// Gate-first input [rows,2*I], output [rows,I], both dense BF16.
    pub fn launch(&self, rows: usize, gate_up: Buffer, output: Buffer, stream: *mut c_void) -> W8Result<()> {
        if rows == 0 || rows > self.rows { return Err(KimiW8Error::Invalid("Situ rows/capacity")); }
        let bytes = rows.checked_mul(self.width * 2).ok_or(KimiW8Error::Invalid("Situ extent overflow"))?;
        let input_bytes = bytes.checked_mul(2).ok_or(KimiW8Error::Invalid("Situ extent overflow"))?;
        validate_w8_buffers(&[(gate_up, input_bytes), (output, bytes)], gate_up.device_id)?;
        reject_overlap((gate_up, input_bytes), (output, bytes))?;
        // SAFETY: checked rows, extents, alignment, device and aliases; the
        // engine retains buffers and the stream through completion/replays.
        unsafe { self.program.launch(&[gate_up.ptr, output.ptr], &[Scalar::I32(rows as i32)], stream)?; }
        Ok(())
    }
}

/// Dense first-layer or shared-expert MLP. The TP2 caller packs intermediate
/// slices gate-first and owns the partial-output meet after this chain.
pub struct KimiFfn<'a> {
    gate_up: Program<'a>,
    situ: KimiSitu<'a>,
    down: Program<'a>,
    rows: usize,
    width: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct KimiFfnBatch {
    pub rows: usize,
    pub input: Buffer,
    pub gate_up_weight: Buffer,
    pub down_weight: Buffer,
    pub gate_up: Buffer,
    pub activated: Buffer,
    pub output: Buffer,
}
impl<'a> KimiFfn<'a> {
    pub fn load(programs: &'a Programs<'a>, intermediate: usize) -> W8Result<Self> {
        let situ = KimiSitu::load(programs, intermediate)?;
        let gate_up = programs.program(&format!("kimik3_ffn_gate_up_i{intermediate}"), &["x", "w", "out"])?;
        let down = programs.program(&format!("kimik3_ffn_down_i{intermediate}"), &["x", "w", "out"])?;
        validate_projection_manifest(gate_up.spec(), 7168, 2 * intermediate)?;
        validate_projection_manifest(down.spec(), intermediate, 7168)?;
        let rows = situ.rows.min(gate_up.spec().capacity_rows as usize).min(down.spec().capacity_rows as usize);
        Ok(Self { gate_up, situ, down, rows, width: intermediate })
    }

    pub fn launch(&self, b: KimiFfnBatch, stream: *mut c_void) -> W8Result<()> {
        if b.rows == 0 || b.rows > self.rows { return Err(KimiW8Error::Invalid("FFN rows/capacity")); }
        let row_bytes = b.rows.checked_mul(7168 * 2).ok_or(KimiW8Error::Invalid("FFN extent overflow"))?;
        let intermediate_bytes = b.rows.checked_mul(self.width * 2).ok_or(KimiW8Error::Invalid("FFN extent overflow"))?;
        let buffers = [(b.input, row_bytes), (b.gate_up_weight, 2 * self.width * 7168 * 2),
            (b.down_weight, 7168 * self.width * 2), (b.gate_up, 2 * intermediate_bytes),
            (b.activated, intermediate_bytes), (b.output, row_bytes)];
        validate_w8_buffers(&buffers, b.input.device_id)?;
        for writer in 3..6 {
            for other in 0..buffers.len() {
                if writer != other { reject_overlap(buffers[writer], buffers[other])?; }
            }
        }
        // SAFETY: all three programs are resolved, operands are validated and
        // disjoint, and the caller retains storage/stream through graph replay.
        unsafe {
            self.gate_up.launch(&[b.input.ptr, b.gate_up_weight.ptr, b.gate_up.ptr],
                &[Scalar::I32(b.rows as i32)], stream)?;
        }
        self.situ.launch(b.rows, b.gate_up, b.activated, stream)?;
        // SAFETY: validated down-projection extents/device/aliases above; the
        // preceding stages on the same stream establish the input dependency.
        unsafe {
            self.down.launch(&[b.activated.ptr, b.down_weight.ptr, b.output.ptr],
                &[Scalar::I32(b.rows as i32)], stream)?;
        }
        Ok(())
    }
}

fn validate_projection_manifest(spec: &ProgramSpec, width: usize, outputs: usize) -> W8Result<()> {
    validate_w8_manifest(spec, b"i")?;
    let g = &spec.geometry;
    if g["width"].as_u64() != Some(width as u64) || g["outputs"].as_u64() != Some(outputs as u64)
        || g["output_dtype"].as_str() != Some("torch.bfloat16") || g["skinny_rows"].as_u64() != Some(0) {
        return Err(KimiW8Error::Invalid("BF16 projection manifest geometry"));
    }
    Ok(())
}

fn validate_w8_manifest(spec: &ProgramSpec, scalars: &[u8]) -> W8Result<()> {
    if spec.capacity_rows == 0 || spec.capacity_rows > i32::MAX as u32
        || spec.scalar_kinds != scalars || !spec.scratch.is_empty() {
        return Err(KimiW8Error::Invalid("manifest capacity/scalars/scratch"));
    }
    Ok(())
}
fn validate_residual_manifest(spec: &ProgramSpec, site: KimiResidualSite) -> W8Result<()> {
    validate_w8_manifest(spec, b"iif")?;
    let g = &spec.geometry;
    if !site.valid() || g["hidden"].as_u64() != Some(7168)
        || g["candidates"].as_u64() != Some(site.candidates as u64)
        || g["prefix_mode"].as_str() != Some(site.prefix.name())
        || g["append"].as_bool() != Some(site.append)
        || g["has_mix"].as_bool() != Some(site.candidates != 1) {
        return Err(KimiW8Error::Invalid("residual manifest geometry"));
    }
    Ok(())
}
fn validate_w8_buffers(buffers: &[(Buffer, usize)], device: i32) -> W8Result<()> {
    if device < 0 { return Err(KimiW8Error::Invalid("device")); }
    for &(buffer, bytes) in buffers {
        if buffer.ptr.is_null() || buffer.bytes < bytes || buffer.device_id != device
            || buffer.ptr as usize % 16 != 0 || (buffer.ptr as usize).checked_add(bytes).is_none() {
            return Err(KimiW8Error::Invalid("buffer extent/alignment/device"));
        }
    }
    Ok(())
}
fn reject_overlap(a: (Buffer, usize), b: (Buffer, usize)) -> W8Result<()> {
    let start_a = a.0.ptr as usize;
    let start_b = b.0.ptr as usize;
    let end_a = start_a.checked_add(a.1).ok_or(KimiW8Error::Invalid("address overflow"))?;
    let end_b = start_b.checked_add(b.1).ok_or(KimiW8Error::Invalid("address overflow"))?;
    if start_a < end_b && start_b < end_a { return Err(KimiW8Error::Invalid("overlapping operands")); }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn specs(heads: usize, rows: usize, mode: KimiMlaMode) -> [ProgramSpec; 3] {
        let context: usize = 1048576;
        let splits = if mode == KimiMlaMode::Extend { 1 } else { 16 };
        let align = |n: usize| n.div_ceil(1024) * 1024;
        let scratch = [align(rows * 2112 * 2) + align(rows * heads * 192 * 2),
            if splits > 1 { align(rows * heads * splits * 512 * 2) + align(rows * heads * splits * 4) } else { 0 },
            2 * align(rows * heads * 128 * 2)];
        std::array::from_fn(|i| ProgramSpec {
            name: format!("test{i}"), pointers: Vec::new(),
            scalar_kinds: if i == 1 { b"iili".to_vec() } else { b"i".to_vec() },
            scratch: [("scratch".to_string(), scratch[i] as u64)].into_iter().collect(),
            capacity_rows: rows as u32,
            geometry: serde_json::json!({"heads": heads, "hidden": 7168, "q_lora_rank": 1536,
                "kv_lora_rank": 512, "qk_nope_dim": 128, "qk_rope_dim": 64, "v_head_dim": 128,
                "page_rows": 64, "norm_eps": 1e-6, "max_context": context,
                "num_splits": splits, "chunks_per_split": context.div_ceil(64).div_ceil(splits), "mode": mode.name()}),
        })
    }

    #[test]
    fn manifest_rejects_stale_geometry_traversal_and_scratch_before_launch() {
        for heads in [48, 96] {
            for mode in [KimiMlaMode::Decode, KimiMlaMode::Verify, KimiMlaMode::Extend] {
                let entries = specs(heads, 64, mode);
                assert!(validate_manifest(entries.each_ref(), heads, 64, mode).is_ok());
                for index in 0..3 {
                    for (key, bad) in [("heads", serde_json::json!(32)), ("hidden", serde_json::json!(4096)),
                        ("page_rows", serde_json::json!(128)), ("norm_eps", serde_json::json!(1e-5))] {
                        let mut wrong = entries.clone(); wrong[index].geometry[key] = bad;
                        assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                    }
                    let mut wrong = entries.clone(); wrong[index].scratch.clear();
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                    let mut wrong = entries.clone(); wrong[index].scalar_kinds = b"iii".to_vec();
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                    let mut wrong = entries.clone(); wrong[index].capacity_rows = 4096;
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                }
                for (key, bad) in [("max_context", serde_json::json!(0)), ("num_splits", serde_json::json!(0)),
                    ("chunks_per_split", serde_json::json!(1)), ("mode", serde_json::json!("wrong"))] {
                    let mut wrong = entries.clone(); wrong[1].geometry[key] = bad;
                    assert!(validate_manifest(wrong.each_ref(), heads, 64, mode).is_err());
                }
            }
        }
    }

    fn residual_spec(site: KimiResidualSite) -> ProgramSpec {
        ProgramSpec { name: site.name(), pointers: vec![], scalar_kinds: b"iif".to_vec(),
            scratch: Default::default(), capacity_rows: 4096,
            geometry: serde_json::json!({"hidden": 7168, "candidates": site.candidates,
                "prefix_mode": site.prefix.name(), "append": site.append,
                "has_mix": site.candidates != 1}) }
    }

    #[test]
    fn residual_sites_and_manifests_match_only_exported_transitions() {
        let mut count = 0;
        for candidates in 0..=10 {
            for prefix in [KimiPrefixMode::Keep, KimiPrefixMode::Add, KimiPrefixMode::Replace] {
                for append in [false, true] {
                    let site = KimiResidualSite { candidates, prefix, append };
                    let spec = residual_spec(site);
                    assert_eq!(validate_residual_manifest(&spec, site).is_ok(), site.valid());
                    if !site.valid() { continue; }
                    count += 1;
                    for (key, value) in [("hidden", serde_json::json!(3584)),
                        ("candidates", serde_json::json!(0)), ("prefix_mode", serde_json::json!("unknown")),
                        ("append", serde_json::json!(!append)), ("has_mix", serde_json::json!(candidates == 1))] {
                        let mut wrong = spec.clone(); wrong.geometry[key] = value;
                        assert!(validate_residual_manifest(&wrong, site).is_err());
                    }
                    let mut wrong = spec.clone(); wrong.scalar_kinds = b"iii".to_vec();
                    assert!(validate_residual_manifest(&wrong, site).is_err());
                    let mut wrong = spec.clone(); wrong.capacity_rows = 0;
                    assert!(validate_residual_manifest(&wrong, site).is_err());
                    let mut wrong = spec.clone(); wrong.scratch.insert("scratch".into(), 1);
                    assert!(validate_residual_manifest(&wrong, site).is_err());
                }
            }
        }
        assert_eq!(count, 32);
    }

    #[test]
    fn ffn_projection_manifests_preserve_bf16_materialization() {
        for inter in [3072, 6144, 16896, 33792] {
            for (width, outputs) in [(7168, 2 * inter), (inter, 7168)] {
                let site = KimiResidualSite { candidates: 1, prefix: KimiPrefixMode::Keep, append: true };
                let mut spec = residual_spec(site);
                spec.scalar_kinds = b"i".to_vec();
                spec.geometry = serde_json::json!({"width": width, "outputs": outputs,
                    "output_dtype": "torch.bfloat16", "skinny_rows": 0});
                validate_projection_manifest(&spec, width, outputs).unwrap();
                for (key, value) in [("width", serde_json::json!(1)), ("outputs", serde_json::json!(1)),
                    ("output_dtype", serde_json::json!("torch.float32")), ("skinny_rows", serde_json::json!(16))] {
                    let mut wrong = spec.clone(); wrong.geometry[key] = value;
                    assert!(validate_projection_manifest(&wrong, width, outputs).is_err());
                }
            }
        }
    }

    #[test]
    fn w8_extents_aliases_and_graph_stable_arena_stride() {
        assert_eq!(KimiResidual::arena_bytes(4099).unwrap(), 9 * 4099 * 7168 * 2);
        assert!(KimiResidual::arena_bytes(0).is_err());
        assert!(KimiResidual::arena_bytes(i32::MAX as usize + 1).is_err());
        let a = Buffer { ptr: 0x1000usize as *mut c_void, bytes: 1024, device_id: 0, flags: 0 };
        let b = Buffer { ptr: 0x1400usize as *mut c_void, ..a };
        validate_w8_buffers(&[(a, 1024), (b, 1024)], 0).unwrap();
        reject_overlap((a, 1024), (b, 1024)).unwrap();
        assert!(reject_overlap((a, 1025), (b, 1024)).is_err());
        assert!(reject_overlap((a, 16), (a, 16)).is_err());
        assert!(validate_w8_buffers(&[(a, 1025)], 0).is_err());
        assert!(validate_w8_buffers(&[(a, 16)], 1).is_err());
        assert!(validate_w8_buffers(&[(Buffer { ptr: 0x1001usize as *mut c_void, ..a }, 16)], 0).is_err());
        let wrapping = Buffer { ptr: (usize::MAX - 15) as *mut c_void, ..a };
        assert!(validate_w8_buffers(&[(wrapping, 16)], 0).is_err());
        assert!(reject_overlap((wrapping, 16), (a, 16)).is_err());
    }

    #[test]
    fn native_bf16_record_geometry_and_buffer_validation() {
        assert_eq!(MLA_RECORD_BYTES, 576 * 2);
        assert_eq!(MLA_RECORD_BYTES * MLA_PAGE_ROWS, 73728);
        let b = Buffer { ptr: 16usize as *mut c_void, bytes: 73728, device_id: 0, flags: 0 };
        validate_buffers(&[(b, 73728)], 0).unwrap();
        assert!(validate_buffers(&[(b, 73729)], 0).is_err());
        assert!(validate_buffers(&[(b, 73728)], 1).is_err());
        assert!(validate_buffers(&[(Buffer { ptr: std::ptr::null_mut(), ..b }, 0)], 0).is_err());
        assert!(validate_buffers(&[(Buffer { ptr: 17usize as *mut c_void, ..b }, 1)], 0).is_err());
    }
}
