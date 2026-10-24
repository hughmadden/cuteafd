//! The MiniMax coordinator program table: the family ABI, the geometry and
//! scratch its numerics need, and each program's operand extents, checked on
//! every launch. The daemon's device owners hold the operands and the stream.
use crate::{CuteafdDeviceBuffer, NativeLibrary, programs::{Program, ProgramSpec, Programs, Scalar}};
use anyhow::{ensure, Context, Result};
use std::{collections::BTreeMap, ffi::c_void, path::Path};

#[derive(Debug, thiserror::Error)]
pub enum MiniMaxNativeError {
    #[error("MiniMax unsupported program {program}: {reason}")]
    Program { program:String, reason:&'static str },
    #[error("MiniMax invalid native operand: {0}")]
    Operand(&'static str),
    #[error("MiniMax native {operation} failed: {source}")]
    Runtime { operation:String, #[source] source:anyhow::Error },
}

fn native<T>(operation:&str, result:Result<T>)->Result<T> {
    result.map_err(|source|MiniMaxNativeError::Runtime{operation:operation.into(),source}.into())
}

/// Trusted generated native library, checked for the MiniMax coordinator inventory.
pub fn load_library(path: &Path) -> Result<NativeLibrary> {
    // SAFETY: the caller chooses the installed engine module; resolving the
    // inventory below rejects libraries that cannot execute this family.
    let library = native("library load",unsafe { NativeLibrary::load(path) })?;
    { let programs = native("program inventory",library.programs())?;
        for (name, _) in definitions() { programs.spec(name).map_err(|_|MiniMaxNativeError::Program {
            program:name.into(),reason:"missing coordinator export" })?; } }
    for name in ["cuteafd_cuda_accumulate_bf16_to_f32_async", "cuteafd_add_tp2_shared_async", INDEX_STAGE,
        "cuteafd_vocabulary_head_create_vocab", "cuteafd_vocabulary_head_launch_width",
        "cuteafd_vocabulary_head_launch_tensor_invariant", "cuteafd_v41_markov_destroy"] {
        // SAFETY: lookup verifies symbol presence only; this uncalled address
        // is never cast to or invoked through a mismatched function signature.
        unsafe { library.lib.get::<*const c_void>(name.as_bytes()) }.map_err(|_|MiniMaxNativeError::Program {
            program:name.into(),reason:"missing shared execution export" })?;
    }
    Ok(library)
}

const INDEX_STAGE: &str = "cuteafd_minimax_index_stage_async";

pub fn definitions() -> Vec<(&'static str, &'static [&'static str])> {
    let norm: &[&str] = &["x", "weight", "bias", "cos", "sin", "positions", "out"];
    let linear: &[&str] = &["x", "payload", "scales", "partial", "output"];
    let mut v = vec![("minimax_norm", norm), ("minimax_q_norm_rope", norm),
        ("minimax_k_norm_rope", norm), ("minimax_index_q_norm_rope", norm), ("minimax_index_k_norm_rope", norm),
        ("minimax_rope", &["cos", "sin"][..]), ("minimax_scale_prep", &["scale", "scale_mma"]),
        ("minimax_kv_split", &["kv", "k", "v"]), ("minimax_swiglu_i3072", &["gate_up", "out"]),
        ("minimax_swiglu_i12288", &["gate_up", "out"]), ("minimax_router_scores", &["x", "weight", "logits"]),
        ("minimax_route", &["logits", "bias", "ids", "weights"]),
        ("minimax_residual", &["residual", "delta", "out"]),
        ("minimax_index_q_quant", &["x", "slots", "values", "scales"]),
        ("minimax_index_k_pack", &["x", "slots", "values", "scales"]),
        ("minimax_kv_pack", &["k", "v", "slots", "batch_ids", "k_descale", "v_descale", "key_cache", "value_cache"]),
        ("minimax_index_decode", &["q_fp8", "q_scale", "index_k_cache", "page_table", "cache_lengths", "q2k_indices", "scratch"]),
        ("minimax_attention_decode", &["query", "key_cache", "value_cache", "k_descale", "v_descale", "page_table", "cache_lengths", "cu_seqlens_q", "q2k_indices", "output", "lse", "scratch"]),
        ("minimax_index_prefill", &["q_fp8", "q_scale", "k_fp8", "k_scale", "k_start", "k_end", "q2k_indices", "scratch"]),
        ("minimax_attention_extend", &["query", "key_cache", "value_cache", "k_descale", "v_descale", "page_table", "cache_lengths", "cu_seqlens_q", "q2k_indices", "output", "lse", "scratch"])];
    for name in ["minimax_mxfp8_q", "minimax_mxfp8_kv", "minimax_mxfp8_o", "minimax_mxfp8_index_q",
        "minimax_mxfp8_index_k", "minimax_mxfp8_shared_gate_up", "minimax_mxfp8_shared_down",
        "minimax_mxfp8_dense_gate_up", "minimax_mxfp8_dense_down",
        // Decode rows fill one M tile; these split K across FP32 partial planes
        // where an unsplit grid leaves the projection latency-bound.
        "minimax_mxfp8_q_decode", "minimax_mxfp8_kv_decode", "minimax_mxfp8_index_q_decode",
        "minimax_mxfp8_index_k_decode"] {
        v.push((name, linear));
    }
    v
}

/// A projection program's shape: `minimax_mxfp8_{projection}[_decode]`.
fn projection_shape(name: &str) -> Option<(usize, usize)> {
    let projection = name.strip_prefix("minimax_mxfp8_")?;
    linear_shape(projection.strip_suffix("_decode").unwrap_or(projection))
}

fn scalar_kinds(name:&str)->&'static [u8] {
    if name=="minimax_rope" { b"ifffff" }
    else if name=="minimax_scale_prep" || name.starts_with("minimax_mxfp8_") { b"ii" }
    else if matches!(name,"minimax_index_decode"|"minimax_attention_decode"|"minimax_attention_extend") { b"iiil" }
    else if name=="minimax_index_prefill" { b"ii" }
    else { b"i" }
}

fn geometry(name:&str)->serde_json::Value {
    use serde_json::json;
    if name.starts_with("minimax_mxfp8_") {
        let (n,k)=projection_shape(name).expect("fixed MiniMax projection inventory");
        return json!({"abi_version":2,"input_features":k,"output_features":n,"encoding":"mxfp8",
            "activation_encoding":"mxfp8","sharded_side":1,"core":"mxfp8_mma","scale_layout":"mma",
            "scale_block_size":32,"output_dtype":"bfloat16","grid":"runtime_num_sm"});
    }
    if name=="minimax_norm" || name.ends_with("norm_rope") {
        let (width,heads,rotary)=match name {"minimax_norm"=>(6144,1,0),"minimax_q_norm_rope"=>(128,64,64),
            "minimax_index_k_norm_rope"=>(128,1,64),_=>(128,4,64)};
        return json!({"width":width,"heads":heads,"rotary":rotary,"eps":1e-6,"zero_centered":true,"subtract_mean":false,"bias":false});
    }
    match name {
        "minimax_rope"=>json!({"rotary_span":64,"llama3":false}),
        "minimax_scale_prep"=>json!({"scale_block":32,"rows_per_tile":128,"groups_per_tile":4}),
        "minimax_kv_split"=>json!({"kv_heads":4,"head_dim":128}),
        "minimax_swiglu_i3072"=>json!({"width":3072,"alpha":1.702,"limit":7.0}),
        "minimax_swiglu_i12288"=>json!({"width":12288,"alpha":1.702,"limit":7.0}),
        "minimax_router_scores"=>json!({"hidden":6144,"experts":128}),
        "minimax_route"=>json!({"experts":128,"topk":4,"factor":2.0}),
        "minimax_residual"=>json!({"width":6144}),
        "minimax_index_q_quant"=>json!({"heads":4,"head_dim":128,"paged":false,"page_rows":0}),
        "minimax_index_k_pack"=>json!({"heads":1,"head_dim":128,"paged":true,"page_rows":64}),
        "minimax_kv_pack"=>json!({"heads":4,"head_dim":128,"page_rows":64}),
        "minimax_index_decode"=>json!({"heads":4,"head_dim":128,"block_tokens":128,"topk":16,"mode":"decode"}),
        "minimax_index_prefill"=>json!({"heads":4,"head_dim":128,"block_tokens":128,"topk":16,"mode":"prefill"}),
        "minimax_attention_decode"=>json!({"page_rows":64,"query_heads":64,"kv_heads":4,"head_dim":128,
            "block_tokens":128,"topk":16,"mode":"decode","softmax_scale":128f64.powf(-0.5)}),
        "minimax_attention_extend"=>json!({"page_rows":64,"query_heads":64,"kv_heads":4,"head_dim":128,
            "block_tokens":128,"topk":16,"mode":"extend","softmax_scale":128f64.powf(-0.5)}),
        _=>unreachable!("fixed MiniMax program inventory"),
    }
}

fn scratch_bytes(spec:&ProgramSpec)->Option<(&'static str,u64)> {
    let c=u64::from(spec.capacity_rows);
    let align=|n:u64,unit:u64|n.div_ceil(unit)*unit;
    if spec.name.starts_with("minimax_mxfp8_") {
        let (n,k)=projection_shape(&spec.name)?; let (n,k)=(n as u64,k as u64);
        // Split-K programs add their FP32 partial planes after the stage scratch.
        let splits=spec.geometry["splits"].as_u64().filter(|&s|s>0 && (k/128)%s==0)?;
        let planes=if splits>1 {c*n*4*splits} else {0};
        return Some(("partial",align(c*k,1024)+align(c*k/32,1024)+align(c.div_ceil(128)*(k/128)*512,1024)+1024+planes));
    }
    match spec.name.as_str() {
        "minimax_index_decode"=>{
            let pages=spec.geometry["max_pages"].as_u64().filter(|&p|p>0 && p<=16384)?;
            Some(("scratch",[4*c*4,4*c*pages.div_ceil(2)*4,4*c*pages*4,4*c*4,4*c*4,
                4*c*16*4,4*c*16*4,(c+1)*8,256].into_iter().map(|n|align(n,256)).sum()))
        }
        "minimax_index_prefill"=>{
            let pages=spec.geometry["max_pages"].as_u64().filter(|&p|p>0 && p<=16384)?;
            Some(("scratch",[4*c*4,4*c*pages.div_ceil(2)*4,4*c*4,4*c*4,
                4*c*16*4,4*c*16*4,(c+1)*8,256].into_iter().map(|n|align(n,256)).sum()))
        }
        "minimax_attention_decode"=>{
            let batch=spec.geometry["max_batch"].as_u64().filter(|&b|b>0 && b<=64)?;
            Some(("scratch",[c*4,c*4,c.max(batch+1)*4,c*4].into_iter().map(|n|align(n,256)).sum()))
        }
        "minimax_attention_extend"=>{
            let batch=spec.geometry["max_batch"].as_u64().filter(|&b|b>0 && b<=64)?;
            let work=c.div_ceil(8)+batch;
            Some(("scratch",[work*4,work*4,work.max(batch+1)*4,work*4,work*2048,work*2048,work*16]
                .into_iter().map(|n|align(n,256)).sum()))
        }
        _=>None,
    }
}

fn validate_geometry(spec:&ProgramSpec)->Result<()> {
    let mismatch=||MiniMaxNativeError::Program{program:spec.name.clone(),reason:"incompatible native geometry or scratch"};
    ensure!(spec.capacity_rows>0 && spec.capacity_rows<=4096,mismatch());
    ensure!(geometry(&spec.name).as_object().unwrap().iter().all(|(key,value)|spec.geometry.get(key)==Some(value)),mismatch());
    let attention=matches!(spec.name.as_str(),"minimax_index_prefill"|"minimax_attention_extend") || spec.name.ends_with("_decode");
    if spec.name.starts_with("minimax_mxfp8_") || attention {
        ensure!(spec.geometry["max_rows"].as_u64()==Some(u64::from(spec.capacity_rows)),mismatch());
        let (operand,required)=scratch_bytes(spec).ok_or_else(mismatch)?;
        ensure!(spec.scratch.get(operand).copied().unwrap_or(0)>=required,mismatch());
    }
    if spec.name.ends_with("_decode") {
        ensure!(spec.capacity_rows<=64,mismatch());
        if spec.name=="minimax_attention_decode" {
            ensure!(spec.geometry["max_batch"].as_u64().is_some_and(|n|n>=u64::from(spec.capacity_rows)),mismatch());
        }
    }
    Ok(())
}

/// Every MiniMax coordinator export, resolved once (on the current device,
/// [`Self::load`]) and checked against the family ABI and geometry. A launch
/// takes the caller's stream after checking each operand's device, extent for
/// the live rows and alignment; the caller's owners keep the operands live.
pub struct MiniMaxPrograms<'a> {
    library: &'a NativeLibrary,
    device: i32,
    programs: BTreeMap<&'static str, Program<'a>>,
}

impl<'a> MiniMaxPrograms<'a> {
    /// Resolves and loads the inventory on `device`, which must be current.
    pub fn load(programs: &'a Programs<'a>, device: i32) -> Result<Self> {
        let loaded = definitions().into_iter().map(|(name, pointers)| {
            let spec=programs.spec(name).map_err(|_|MiniMaxNativeError::Program{program:name.into(),reason:"missing coordinator export"})?;
            ensure!(spec.scalar_kinds==scalar_kinds(name), MiniMaxNativeError::Program{program:name.into(),reason:"scalar ABI mismatch"});
            ensure!(spec.pointers.iter().map(String::as_str).eq(pointers.iter().copied()),
                MiniMaxNativeError::Program{program:name.into(),reason:"pointer ABI mismatch"});
            validate_geometry(spec)?;
            Ok((name, native(name,programs.program(name, pointers))?))
        }).collect::<Result<_>>()?;
        Ok(Self { library: programs.library(), device, programs: loaded })
    }
    pub fn library(&self) -> &'a NativeLibrary { self.library }
    pub fn device(&self) -> i32 { self.device }
    /// Every operand lives on this table's device.
    pub fn check_device(&self, buffers: &[CuteafdDeviceBuffer]) -> Result<()> {
        ensure!(buffers.iter().all(|b| b.device_id == self.device), "MiniMax operands belong to another device");
        Ok(())
    }
    pub fn capacity(&self, name: &str) -> Result<usize> {
        Ok(self.programs.get(name).context("missing MiniMax program")?.spec().capacity_rows as usize)
    }
    /// `minimax_mxfp8_{projection}`, or its split-K decode program when it
    /// has one and `rows` fit it.
    pub fn projection(&self, projection: &str, rows: usize) -> String {
        let decode = format!("minimax_mxfp8_{projection}_decode");
        match self.programs.get(decode.as_str()) {
            Some(program) if rows <= program.spec().capacity_rows as usize => decode,
            _ => format!("minimax_mxfp8_{projection}"),
        }
    }
    /// # Safety
    /// `stream` is a live stream of this table's device, and every operand is
    /// live device memory of its extent until `stream` reaches the launch.
    pub unsafe fn launch(&self, name: &str, operands: &[CuteafdDeviceBuffer], scalars: &[Scalar],
        stream: *mut c_void) -> Result<()> {
        let program = self.programs.get(name).ok_or_else(||MiniMaxNativeError::Program{program:name.into(),reason:"missing coordinator export"})?;
        let spec = program.spec();
        ensure!(operands.len() == spec.pointers.len(), MiniMaxNativeError::Program{program:name.into(),reason:"pointer ABI mismatch"});
        let scalar = |i: usize| -> Result<usize> { match scalars.get(i) {
            Some(Scalar::I32(v)) if *v > 0 => Ok(*v as usize),
            Some(Scalar::I64(v)) if *v > 0 => usize::try_from(*v).context("MiniMax scalar overflow"),
            _ => return Err(MiniMaxNativeError::Program{program:name.into(),reason:"invalid positive integer extent"}.into()),
        } };
        let rows = scalar(0)?;
        ensure!(matches!(name,"minimax_rope"|"minimax_scale_prep") || spec.capacity_rows == 0 || rows <= spec.capacity_rows as usize,
            MiniMaxNativeError::Program{program:name.into(),reason:"live rows exceed exported capacity"});
        let mut required = vec![0; operands.len()];
        if name == "minimax_scale_prep" { let bytes = rows.checked_mul(scalar(1)?).context("scale extent overflow")?;
            ensure!(rows % 128 == 0 && scalar(1)? % 4 == 0, "MiniMax scale tile geometry"); required.fill(bytes);
        } else if name == "minimax_rope" { required.fill(rows.checked_mul(64).context("rope extent overflow")?);
        } else if name == "minimax_norm" { required[0] = rows * 6144 * 4; required[1] = 6144 * 2; required[6] = rows * 6144 * 2;
        } else if name.ends_with("norm_rope") {
            let heads = if name == "minimax_q_norm_rope" { 64 } else if name == "minimax_index_k_norm_rope" { 1 } else { 4 };
            required[0] = rows * heads * 128 * 2; required[1] = 256; required[3]=64; required[4]=64;
            required[5] = rows * 8; required[6] = required[0];
        } else if name.starts_with("minimax_mxfp8_") {
            let (n,k) = projection_shape(name).context("unknown MiniMax projection")?;
            required = vec![rows*k*2, n*k, n*k/32, 0, rows*n*2];
        } else { match name {
            "minimax_kv_split" => required = vec![rows*2048, rows*1024, rows*1024],
            "minimax_swiglu_i3072" => required = vec![rows*12288, rows*6144],
            "minimax_swiglu_i12288" => required = vec![rows*49152, rows*24576],
            "minimax_router_scores" => required = vec![rows*12288, 128*6144*4, rows*512],
            "minimax_route" => required = vec![rows*512,512,rows*16,rows*16],
            "minimax_residual" => required = vec![rows*24576, rows*12288, rows*24576],
            "minimax_index_q_quant" => required = vec![rows*1024, 0, rows*512, rows*16],
            "minimax_index_k_pack" => required = vec![rows*256, rows*8, 8448, 0],
            "minimax_kv_pack" => required = vec![rows*1024,rows*1024,rows*8,rows*4,16,16,32768,32768],
            "minimax_index_decode" => {
                let (width,stride,pages) = (scalar(1)?,scalar(2)?,scalar(3)?);
                ensure!(width<=stride && width<=spec.geometry["max_pages"].as_u64().unwrap() as usize
                    && pages<=i32::MAX as usize, "MiniMax index page extents");
                required = vec![rows*512, rows*16, pages*8448, rows*stride*4, rows*4, 4*spec.capacity_rows as usize*16*4,0];
            }
            "minimax_attention_decode" => {
                let (batch,stride,pages) = (scalar(1)?,scalar(2)?,scalar(3)?);
                ensure!(batch == rows && batch <= 64 && pages<=i32::MAX as usize, "MiniMax decode batch extent");
                required = vec![rows*16384,pages*32768,pages*32768,batch*16,batch*16,batch*stride*4,
                    batch*4,(batch+1)*4,4*spec.capacity_rows as usize*16*4,rows*16384,rows*256,0];
            }
            "minimax_index_prefill" => {
                let keys = scalar(1)?;
                ensure!(keys <= spec.geometry["max_pages"].as_u64().unwrap() as usize*64, "MiniMax prefill key extent");
                let padded = keys.next_multiple_of(256);
                required = vec![rows*512, rows*16, padded*128, padded*4, rows*4, rows*4, 4*spec.capacity_rows as usize*16*4, 0];
            }
            "minimax_attention_extend" => {
                let (batch,stride,pages) = (scalar(1)?,scalar(2)?,scalar(3)?);
                ensure!(batch <= rows && batch as u64 <= spec.geometry["max_batch"].as_u64().unwrap()
                    && pages<=i32::MAX as usize, "MiniMax extend batch extent");
                required = vec![rows*16384,pages*32768,pages*32768,batch*16,batch*16,batch*stride*4,
                    batch*4,(batch+1)*4,4*spec.capacity_rows as usize*16*4,rows*16384,rows*256,0];
            }
            _ => anyhow::bail!("unsupported MiniMax operation: {name}"),
        } }
        for (i, operand) in operands.iter().enumerate() {
            let scratch = spec.scratch.get(&spec.pointers[i]).copied().unwrap_or(0) as usize;
            ensure!(operand.bytes >= required[i].max(scratch), "MiniMax operand {} too small: {name}", spec.pointers[i]);
            ensure!(required[i].max(scratch)==0 || operand.ptr as usize%16==0,
                MiniMaxNativeError::Operand("program operand is not 16-byte aligned"));
        }
        self.check_device(operands)?;
        let pointers: Vec<_> = operands.iter().map(|b| b.ptr).collect();
        // SAFETY: family ABI and extents were checked; the caller keeps every
        // operand's allocation live until the stream reaches this launch.
        native(name,unsafe { program.launch(&pointers, scalars, stream) })
    }
    /// Gathers `count` pages of an MSA index cache (their physical ids in
    /// `pages`) into the contiguous key rows and FP32 scales the prefill
    /// scorer reads: page j becomes rows [64j, 64j + 64).
    ///
    /// # Safety
    /// As [`Self::launch`]; `pages` holds `count` pool pages of `cache`.
    pub unsafe fn index_stage(&self, cache: CuteafdDeviceBuffer, pages: CuteafdDeviceBuffer, count: usize,
        keys: CuteafdDeviceBuffer, scales: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<()> {
        ensure!(count > 0 && count <= i32::MAX as usize && pages.bytes >= count * 4 && keys.bytes >= count * 8192
            && scales.bytes >= count * 256 && cache.bytes % 8448 == 0, MiniMaxNativeError::Operand("index stage extents"));
        ensure!([cache, pages, keys, scales].iter().all(|b| b.ptr as usize % 16 == 0),
            MiniMaxNativeError::Operand("program operand is not 16-byte aligned"));
        self.check_device(&[cache, pages, keys, scales])?;
        type Stage = unsafe extern "C" fn(*const c_void, *const i32, u32, *mut c_void, *mut c_void, *mut c_void) -> i32;
        // SAFETY: the symbol's C ABI is fixed by minimax_index_stage.cu; load_library required it.
        let stage = unsafe { self.library.lib.get::<Stage>(INDEX_STAGE.as_bytes()) }
            .map_err(|_| MiniMaxNativeError::Program { program: INDEX_STAGE.into(), reason: "missing shared execution export" })?;
        // SAFETY: checked extents and alignment; the page ids are the batch's
        // validated pool pages, and the caller keeps every operand live.
        let status = unsafe { stage(cache.ptr, pages.ptr.cast(), count as u32, keys.ptr, scales.ptr, stream) };
        ensure!(status == 0, MiniMaxNativeError::Runtime { operation: "index stage".into(),
            source: anyhow::anyhow!("CUDA status {status}") });
        Ok(())
    }
}

pub fn linear_shape(name: &str) -> Option<(usize,usize)> { Some(match name {
    "q" => (8192,6144), "kv" => (1024,6144), "o" => (6144,8192),
    "index_q" => (512,6144), "index_k" => (128,6144), "shared_gate_up" => (6144,6144),
    "shared_down" => (6144,3072), "dense_gate_up" => (24576,6144), "dense_down" => (6144,12288), _ => return None,
}) }

#[cfg(test)]
mod tests {
    use super::*;
    fn spec(name:&str,pointers:&[&str])->ProgramSpec {
        let capacity=if name.ends_with("_decode") {64} else if matches!(name,"minimax_index_prefill"|"minimax_attention_extend") {1024} else {4096};
        let mut spec=ProgramSpec {name:name.into(),pointers:pointers.iter().map(|s|s.to_string()).collect(),
            scalar_kinds:scalar_kinds(name).to_vec(),scratch:Default::default(),capacity_rows:capacity,geometry:geometry(name)};
        spec.geometry["max_rows"]=capacity.into();
        if matches!(name,"minimax_index_decode"|"minimax_index_prefill") {spec.geometry["max_pages"]=16384.into();}
        if matches!(name,"minimax_attention_decode"|"minimax_attention_extend") {spec.geometry["max_batch"]=64.into();}
        if name.starts_with("minimax_mxfp8_") {spec.geometry["splits"]=if name.ends_with("_decode") {4} else {1}.into();}
        if let Some((operand,bytes))=scratch_bytes(&spec) {spec.scratch.insert(operand.into(),bytes);}
        spec
    }
    #[test] fn minimax_projection_inventory_is_native_and_fixed() {
        assert_eq!(linear_shape("dense_down"),Some((6144,12288)));
        assert_eq!(linear_shape("kv"),Some((1024,6144)));
        assert!(linear_shape("bf16").is_none());
        let names: std::collections::BTreeSet<_> = definitions().iter().map(|(n,_)|*n).collect();
        assert_eq!(names.len(),33);
        assert_eq!(projection_shape("minimax_mxfp8_index_k_decode"),Some((128,6144)));
        assert!(projection_shape("minimax_mxfp8_bf16_decode").is_none());
        assert!(names.iter().all(|name|name.starts_with("minimax_")));
    }

    #[test] fn minimax_rejects_geometry_and_scratch_that_change_native_numerics() -> Result<()> {
        for (name,pointers) in definitions() {validate_geometry(&spec(name,pointers))?;}
        for (name,bytes) in [("minimax_mxfp8_q",26_739_712),("minimax_mxfp8_dense_down",53_478_400),
            ("minimax_index_decode",25_202_688),("minimax_attention_decode",1_280),
            ("minimax_index_prefill",134_799_872),("minimax_attention_extend",792_576),
            // 64 rows: stage scratch plus four FP32 planes of 64 x 8192.
            ("minimax_mxfp8_q_decode",393_216+12_288+24_576+1_024+64*8192*4*4)] {
            let mut program=spec(name,&[]);
            let (operand,actual)=scratch_bytes(&program).unwrap();assert_eq!(actual,bytes);
            program.scratch.insert(operand.into(),actual-1);assert!(validate_geometry(&program).is_err());
        }
        let mut linear=spec("minimax_mxfp8_q",&[]);
        linear.geometry["activation_encoding"]="bf16".into();assert!(validate_geometry(&linear).is_err());
        let mut norm=spec("minimax_q_norm_rope",&[]);
        norm.geometry["zero_centered"]=false.into();assert!(validate_geometry(&norm).is_err());
        let mut index=spec("minimax_index_decode",&[]);
        index.geometry["topk"]=8.into();assert!(validate_geometry(&index).is_err());
        let mut attention=spec("minimax_attention_decode",&[]);
        attention.geometry["max_batch"]=32.into();assert!(validate_geometry(&attention).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires an explicit CPU build slot and native fixture directory"]
    fn minimax_native_fixture_validates_launches() -> Result<()> {
        use crate::native_library_lifetime_fixture::Fixture;
        use std::sync::atomic::{AtomicI32,AtomicUsize,Ordering};
        static LAUNCHES:AtomicUsize=AtomicUsize::new(0);
        static STATUS:AtomicI32=AtomicI32::new(0);
        unsafe extern "C" fn launch(_:u32,_:*const *mut c_void,_:*const u64,_:*mut c_void)->i32 {
            LAUNCHES.fetch_add(1,Ordering::Relaxed);STATUS.load(Ordering::Relaxed)
        }
        let fixture=Fixture::build()?;
        let library=fixture.load()?;
        let specs=definitions().into_iter().map(|(name,pointers)|spec(name,pointers)).collect();
        let fixture_programs=Programs::fixture(&library,specs,launch);
        library.cuda_set_device(0)?;
        let programs=MiniMaxPrograms::load(&fixture_programs,0)?;
        let stream=std::ptr::null_mut();
        let output=library.alloc_device_buffer(64)?;
        let input=library.alloc_device_buffer(6144*4+16)?;
        let weight=library.alloc_device_buffer(6144*2)?;
        let norm=library.alloc_device_buffer(6144*2)?;
        library.cuda_set_device(1)?;
        let other=library.alloc_device_buffer(6144*4)?;
        library.cuda_set_device(0)?;
        let operands=|x,out|[x,weight,weight,weight,weight,weight,out];
        // SAFETY: the fixture dispatcher only counts launches; no operand is read.
        unsafe {
            assert!(programs.launch("minimax_norm",&operands(input,output),&[Scalar::I32(1)],stream).is_err());
            assert!(programs.launch("minimax_norm",&operands(input,norm),&[Scalar::I32(4097)],stream).is_err());
            let unaligned=CuteafdDeviceBuffer{ptr:input.ptr.cast::<u8>().wrapping_add(1).cast(),bytes:6144*4,..input};
            assert!(programs.launch("minimax_norm",&operands(unaligned,norm),&[Scalar::I32(1)],stream).is_err());
            assert!(programs.launch("minimax_norm",&operands(other,norm),&[Scalar::I32(1)],stream).is_err());
            assert_eq!(LAUNCHES.load(Ordering::Relaxed),0,"invalid launch reached the native dispatcher");
            programs.launch("minimax_norm",&operands(input,norm),&[Scalar::I32(1)],stream)?;
            assert_eq!(LAUNCHES.load(Ordering::Relaxed),1);
            STATUS.store(17,Ordering::Relaxed);
            let error=programs.launch("minimax_norm",&operands(input,norm),&[Scalar::I32(1)],stream).unwrap_err();
            assert!(matches!(error.downcast_ref::<MiniMaxNativeError>(),Some(MiniMaxNativeError::Runtime{operation,..}) if operation=="minimax_norm"));
            assert!(format!("{error:#}").contains("status 17"));
        }
        STATUS.store(0,Ordering::Relaxed);
        Ok(())
    }
}
