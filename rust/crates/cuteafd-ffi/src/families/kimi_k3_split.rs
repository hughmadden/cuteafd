//! Already-projected Kimi seams. The original fused BF16 bindings are unchanged.
use super::kimi_k3::{KimiKda, MLA_RECORD_BYTES, MLA_PAGE_ROWS};
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use std::ffi::c_void;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KimiSplitError {
    #[error("Kimi projected core invalid {0}")]
    Invalid(&'static str),
    #[error("Kimi projected core program: {0}")]
    Program(#[from] anyhow::Error),
}
type Result<T> = std::result::Result<T, KimiSplitError>;
const RANK: &[&str] = &["q_a", "kv_a", "q_a_norm", "kv_a_norm", "kv_slots", "q_resid", "kv_cache", "scratch"];
const QUERY: &[&str] = &["q", "w_uk", "query"];
const GATE: &[&str] = &["attn", "w_uv", "gates", "out"];
const KDA: &[&str] = &["x", "q", "k", "v", "gate", "w_fab", "w_f", "conv_w", "a_log", "dt_bias", "o_norm",
    "conv_state", "state", "slots", "seq_first", "out", "replay", "scratch"];

pub struct KimiMlaSplit<'a> {
    rank: Program<'a>, query: Program<'a>, gate: Program<'a>,
    heads: usize, rows: usize, kv_stride: usize, scratch: usize,
}
impl<'a> KimiMlaSplit<'a> {
    pub fn load(programs: &'a Programs<'a>, names: [&str; 3], heads: usize, rows: usize,
        kv_stride: usize) -> Result<Self> {
        if !matches!(heads, 48 | 96) || !(1..=4096).contains(&rows) || !matches!(kv_stride, 576 | 640) {
            return Err(KimiSplitError::Invalid("MLA geometry"));
        }
        let specs = [programs.spec(names[0])?, programs.spec(names[1])?, programs.spec(names[2])?];
        let scratch = validate_mla(specs, heads, rows, kv_stride)?;
        let rank = programs.program(names[0], RANK)?;
        let query = programs.program(names[1], QUERY)?;
        let gate = programs.program(names[2], GATE)?;
        Ok(Self { rank, query, gate, heads, rows, kv_stride, scratch })
    }
    pub fn scratch_bytes(&self) -> usize { self.scratch }

    /// The rank-norm stage packs only the logical kv_a prefix (576 channels),
    /// preserving the quantizer's complete stored H640 transform beforehand.
    pub fn rank_norm(&self, b: KimiMlaRankBatch, stream: *mut c_void) -> Result<()> {
        self.rows(b.rows)?;
        if b.cache_pages == 0 || b.cache_pages > i32::MAX as usize {
            return Err(KimiSplitError::Invalid("MLA cache pages"));
        }
        let cache = b.cache_pages.checked_mul(MLA_PAGE_ROWS*MLA_RECORD_BYTES)
            .ok_or(KimiSplitError::Invalid("MLA cache overflow"))?;
        checked(&[(b.q_a,b.rows*1536*2),(b.kv_a,b.rows*self.kv_stride*2),
            (b.q_norm,1536*2),(b.kv_norm,512*2),(b.slots,b.rows*8),
            (b.q_resid,b.rows*1536*2),(b.cache,cache),(b.scratch,self.scratch)], &[5,6,7])?;
        // SAFETY: capacities/devices and disjoint writers checked; the engine
        // owns valid slots and retains cache and all operands through replay.
        unsafe { self.rank.launch(&[b.q_a.ptr,b.kv_a.ptr,b.q_norm.ptr,b.kv_norm.ptr,b.slots.ptr,
            b.q_resid.ptr,b.cache.ptr,b.scratch.ptr], &[Scalar::I32(b.rows as i32)], stream)?; }
        Ok(())
    }

    pub fn query(&self, rows: usize, q_b: Buffer, uk: Buffer, query: Buffer,
        stream: *mut c_void) -> Result<()> {
        self.rows(rows)?;
        let h = self.heads;
        checked(&[(q_b,rows*h*192*2),(uk,h*512*128*2),(query,rows*h*576*2)], &[2])?;
        // SAFETY: complete projected q_b rows and UK/output extents checked;
        // caller keeps allocations and stream live through queued work/replay.
        unsafe { self.query.launch(&[q_b.ptr,uk.ptr,query.ptr], &[Scalar::I32(rows as i32)], stream)?; }
        Ok(())
    }

    pub fn gate_output(&self, rows: usize, attention: Buffer, uv: Buffer, gates: Buffer,
        output: Buffer, stream: *mut c_void) -> Result<()> {
        self.rows(rows)?;
        let h = self.heads;
        checked(&[(attention,rows*h*512*2),(uv,h*128*512*2),(gates,rows*h*128*2),
            (output,rows*h*128*2)], &[3])?;
        // SAFETY: projected gates, UV and writable output extents/devices
        // checked; caller retains live storage through completion/replay.
        unsafe { self.gate.launch(&[attention.ptr,uv.ptr,gates.ptr,output.ptr],
            &[Scalar::I32(rows as i32)], stream)?; }
        Ok(())
    }
    fn rows(&self, rows: usize) -> Result<()> {
        if rows == 0 || rows > self.rows { return Err(KimiSplitError::Invalid("MLA live rows")); }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KimiMlaRankBatch {
    pub rows: usize, pub cache_pages: usize,
    pub q_a: Buffer, pub kv_a: Buffer, pub q_norm: Buffer, pub kv_norm: Buffer,
    pub slots: Buffer, pub q_resid: Buffer, pub cache: Buffer, pub scratch: Buffer,
}

/// The small f_a/b projections, f_b and recurrence parameters remain BF16/F32;
/// q/k/v/g are separately calibrated BF16 projected rows. Output precedes o.
#[derive(Debug, Clone, Copy)]
pub struct KimiKdaCoreWeights {
    pub fab: Buffer, pub forget: Buffer, pub convolution: Buffer,
    pub a_log: Buffer, pub dt_bias: Buffer, pub norm: Buffer,
}
#[derive(Debug, Clone, Copy)]
pub struct KimiKdaCoreBatch {
    pub rows: usize, pub sequences: usize, pub state_slots: usize, pub speculative: bool,
    pub input: Buffer, pub q: Buffer, pub k: Buffer, pub v: Buffer, pub gate: Buffer,
    pub conv_state: Buffer, pub state: Buffer, pub slots: Buffer, pub first: Buffer,
    pub output: Buffer, pub replay: Buffer, pub scratch: Buffer,
}
pub struct KimiKdaCore<'a> { program: Program<'a>, rows: usize, decode: bool, scratch: usize }
impl<'a> KimiKdaCore<'a> {
    pub fn load(programs: &'a Programs<'a>, name: &str, rows: usize, decode: bool) -> Result<Self> {
        let scratch = validate_kda(programs.spec(name)?, rows, decode)?;
        let program = programs.program(name, KDA)?;
        Ok(Self { program, rows, decode, scratch })
    }
    pub fn scratch_bytes(&self) -> usize { self.scratch }
    pub fn launch(&self, w: KimiKdaCoreWeights, b: KimiKdaCoreBatch, stream: *mut c_void) -> Result<()> {
        if b.rows == 0 || b.rows > self.rows || b.sequences == 0 || b.sequences > b.rows
            || b.sequences > b.state_slots || (!self.decode && (b.speculative || (b.rows > 64 && b.sequences != 1))) {
            return Err(KimiSplitError::Invalid("KDA batch/replay geometry"));
        }
        checked(&[(b.input,b.rows*7168*2),(b.q,b.rows*6144*2),(b.k,b.rows*6144*2),
            (b.v,b.rows*6144*2),(b.gate,b.rows*6144*2),(w.fab,176*7168*2),(w.forget,6144*128*2),
            (w.convolution,3*6144*4*4),(w.a_log,48*4),(w.dt_bias,6144*4),(w.norm,128*4),
            (b.conv_state,KimiKda::conv_state_bytes(b.state_slots).map_err(|_|KimiSplitError::Invalid("KDA conv slots"))?),
            (b.state,KimiKda::state_bytes(b.state_slots).map_err(|_|KimiSplitError::Invalid("KDA state slots"))?),
            (b.slots,b.rows*4),(b.first,b.rows*4),(b.output,b.rows*6144*2),
            (b.replay,KimiKda::replay_bytes()),(b.scratch,self.scratch)], &[11,12,15,16,17])?;
        // SAFETY: all operand/state/replay extents checked with disjoint writes;
        // the engine owns valid sequence metadata and graph-safe lifetimes.
        unsafe { self.program.launch(&[b.input.ptr,b.q.ptr,b.k.ptr,b.v.ptr,b.gate.ptr,w.fab.ptr,w.forget.ptr,
            w.convolution.ptr,w.a_log.ptr,w.dt_bias.ptr,w.norm.ptr,b.conv_state.ptr,b.state.ptr,
            b.slots.ptr,b.first.ptr,b.output.ptr,b.replay.ptr,b.scratch.ptr],
            &[Scalar::I32(b.rows as i32), Scalar::I32(i32::from(b.speculative))], stream)?; }
        Ok(())
    }
}

fn schema(spec: &ProgramSpec, pointers: &[&str], scalars: &[u8], rows: usize) -> Result<()> {
    if spec.capacity_rows as usize != rows || spec.scalar_kinds != scalars
        || spec.pointers.iter().map(String::as_str).ne(pointers.iter().copied())
        || spec.geometry["max_rows"].as_u64() != Some(rows as u64) {
        return Err(KimiSplitError::Invalid("manifest capacity/pointers/scalars"));
    }
    Ok(())
}
fn validate_mla(specs: [&ProgramSpec;3], heads: usize, rows: usize, kv_stride: usize) -> Result<usize> {
    for (spec, pointers) in specs.iter().zip([RANK,QUERY,GATE]) {
        schema(spec,pointers,b"i",rows)?;
        if [("heads",heads),("hidden",7168),("q_lora_rank",1536),("kv_lora_rank",512),
            ("qk_nope_dim",128),("qk_rope_dim",64),("v_head_dim",128),("page_rows",64)]
            .iter().any(|&(key,value)|spec.geometry[key].as_u64()!=Some(value as u64))
            || spec.geometry["norm_eps"].as_f64()!=Some(1e-6) {
            return Err(KimiSplitError::Invalid("MLA manifest geometry"));
        }
    }
    let bytes = (rows*2112*2).div_ceil(1024)*1024;
    if specs[0].geometry["kv_stride"].as_u64()!=Some(kv_stride as u64)
        || specs[0].scratch.get("scratch").copied()!=Some(bytes as u64)
        || specs[0].scratch.len()!=1 || !specs[1].scratch.is_empty() || !specs[2].scratch.is_empty() {
        return Err(KimiSplitError::Invalid("MLA manifest stride/scratch"));
    }
    Ok(bytes)
}
fn validate_kda(spec: &ProgramSpec, rows: usize, decode: bool) -> Result<usize> {
    schema(spec,KDA,b"ii",rows)?;
    let g = &spec.geometry;
    if rows != if decode {64} else {4096}
        || [("hidden",7168),("kda_heads",48),("kda_head_dim",128),("conv_kernel",4)]
            .iter().any(|&(key,value)|g[key].as_u64()!=Some(value as u64))
        || g["gate_lower_bound"].as_f64()!=Some(-5.) || g["norm_eps"].as_f64()!=Some(1e-5)
        || g["output_gate"].as_str()!=Some("full_rank") || g["o_norm_dtype"].as_str()!=Some("float32")
        || g["output_kind"].as_str()!=Some("norm") || g["decode"].as_bool()!=Some(decode)
        || !g["state_dtype"].is_null() || !g["state_rounding"].is_null() || !g["replay_rows"].is_null() {
        return Err(KimiSplitError::Invalid("KDA manifest arithmetic/state"));
    }
    let base = spec.scratch.get("scratch").copied().and_then(|v|usize::try_from(v).ok())
        .ok_or(KimiSplitError::Invalid("KDA manifest scratch"))?;
    let fab = g["fab_offset"].as_u64().and_then(|v|usize::try_from(v).ok())
        .ok_or(KimiSplitError::Invalid("KDA manifest fab offset"))?;
    let align = |v: usize|v.div_ceil(1024)*1024;
    let min = align(rows*24752*2)+align(rows*2*6144*2)+align(rows*3*6144*2)+2*align(rows*6144*2);
    if fab < min || fab%1024!=0 || (decode && fab!=min) || (!decode && fab==min)
        || fab.checked_add(align(rows*176*2))!=Some(base) || spec.scratch.len()!=1 {
        return Err(KimiSplitError::Invalid("KDA manifest scratch layout"));
    }
    Ok(base)
}
fn checked(items: &[(Buffer,usize)], writers: &[usize]) -> Result<()> {
    let device = items[0].0.device_id;
    if device<0 || items.iter().any(|(b,n)|b.ptr.is_null() || b.bytes<*n || b.device_id!=device
        || b.flags!=0 || b.ptr as usize%16!=0 || (b.ptr as usize).checked_add(*n).is_none()) {
        return Err(KimiSplitError::Invalid("buffer extent/alignment/device"));
    }
    for &i in writers {
        let (a,na) = items[i];
        for (j,&(b,nb)) in items.iter().enumerate() {
            if i!=j && (a.ptr as usize)<b.ptr as usize+nb && (b.ptr as usize)<a.ptr as usize+na {
                return Err(KimiSplitError::Invalid("overlapping writable operands"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    fn spec(pointers: &[&str], rows: usize) -> ProgramSpec {
        ProgramSpec {name:"fixture".into(),pointers:pointers.iter().map(|s|s.to_string()).collect(),
            scalar_kinds:b"i".to_vec(),scratch:HashMap::new(),capacity_rows:rows as u32,
            geometry:json!({"hidden":7168,"heads":48,"q_lora_rank":1536,"kv_lora_rank":512,
                "qk_nope_dim":128,"qk_rope_dim":64,"v_head_dim":128,"page_rows":64,"norm_eps":1e-6,"max_rows":rows})}
    }
    #[test]
    fn mla_accepts_stored_h640_without_changing_logical_geometry() {
        let mut a = spec(RANK,64);
        a.geometry["kv_stride"]=json!(640);
        a.scratch.insert("scratch".into(),(64usize*2112*2).div_ceil(1024) as u64*1024);
        let b=spec(QUERY,64); let c=spec(GATE,64);
        assert!(validate_mla([&a,&b,&c],48,64,640).is_ok());
        assert!(validate_mla([&a,&b,&c],48,64,576).is_err());
        a.pointers.swap(0,1);
        assert!(validate_mla([&a,&b,&c],48,64,640).is_err());
    }
    #[test]
    fn kda_rejects_different_state_and_scratch_arithmetic() {
        let mut a=spec(KDA,64);
        a.scalar_kinds=b"ii".to_vec();
        let align=|v:usize|v.div_ceil(1024)*1024;
        let fab=align(64*24752*2)+align(64*2*6144*2)+align(64*3*6144*2)+2*align(64*6144*2);
        a.geometry=json!({"hidden":7168,"kda_heads":48,"kda_head_dim":128,"conv_kernel":4,
            "gate_lower_bound":-5.,"norm_eps":1e-5,"output_gate":"full_rank","o_norm_dtype":"float32",
            "output_kind":"norm","decode":true,"max_rows":64,"fab_offset":fab});
        a.scratch.insert("scratch".into(),(fab+align(64*176*2))as u64);
        assert!(validate_kda(&a,64,true).is_ok());
        a.geometry["state_dtype"]=json!("bfloat16");
        assert!(validate_kda(&a,64,true).is_err());
        a.geometry["state_dtype"]=serde_json::Value::Null;
        a.scratch.insert("scratch".into(),1);
        assert!(validate_kda(&a,64,true).is_err());
    }
}
