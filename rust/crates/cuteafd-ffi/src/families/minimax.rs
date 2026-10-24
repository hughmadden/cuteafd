//! Retained device operands for the MiniMax coordinator ABI.
mod dspark;

use crate::{CuteafdDeviceBuffer, CuteafdTieredPlanes, NativeLibrary, programs::{Program, ProgramSpec, Programs, Scalar, VocabularyHead}};
use anyhow::{ensure, Context, Result};
use std::{cell::RefCell, collections::BTreeMap, ffi::c_void, path::Path, rc::Rc};

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

fn on_device<T>(library: &NativeLibrary, device: i32, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let previous = native("get device",library.cuda_get_device())?;
    native("select device",library.cuda_set_device(device))?;
    let result = native("device operation",f());
    let restored=native("restore device",library.cuda_set_device(previous));
    result.and_then(|value|{restored?;Ok(value)})
}

/// A device allocation, or one range of a tiered plane set (freed with the set).
struct Storage<'a> { library: &'a NativeLibrary, buffer: CuteafdDeviceBuffer, planes: Option<Rc<PlaneSet<'a>>> }
impl Drop for Storage<'_> {
    fn drop(&mut self) {
        if self.planes.is_some() || self.library.is_quarantined_after_failed_drain() { return; }
        if let Err(error) = on_device(self.library, self.buffer.device_id,
            || self.library.free_device_buffer(&mut self.buffer)) {
            tracing::error!(%error, "releasing MiniMax device storage");
        }
    }
}

/// Every range of one tiered allocation; released after the last view of any range.
struct PlaneSet<'a> { library: &'a NativeLibrary, planes: CuteafdTieredPlanes }
impl Drop for PlaneSet<'_> {
    fn drop(&mut self) {
        if self.library.is_quarantined_after_failed_drain() { return; }
        let (library, device) = (self.library, self.planes.device_id);
        if let Err(error) = on_device(library, device, || library.free_tiered_planes(&mut self.planes)) {
            tracing::error!(%error, "releasing MiniMax tiered planes");
        }
    }
}

#[derive(Clone)]
pub struct Buffer<'a> { storage: Rc<Storage<'a>>, view: CuteafdDeviceBuffer }
impl<'a> Buffer<'a> {
    pub fn new(library: &'a NativeLibrary, device: i32, bytes: usize) -> Result<Self> {
        ensure!(bytes > 0, MiniMaxNativeError::Operand("empty allocation"));
        let buffer = on_device(library, device, || library.alloc_device_buffer(bytes))?;
        Ok(Self { storage: Rc::new(Storage { library, buffer, planes: None }), view: buffer })
    }
    /// One buffer per range of a new tiered plane set ([`CuteafdTieredPlanes`]):
    /// `device_bytes` of device memory, `host_bytes` of pinned host memory, then
    /// `shared_bytes` that every range aliases. Kernels address a range like any
    /// device buffer.
    pub fn tiered(library: &'a NativeLibrary, device: i32, count: usize, device_bytes: usize,
        host_bytes: usize, shared_bytes: usize) -> Result<Vec<Self>> {
        ensure!(count > 0 && device_bytes + host_bytes + shared_bytes > 0, MiniMaxNativeError::Operand("empty tiered planes"));
        let planes = on_device(library, device, || library.tiered_planes(count, device_bytes, host_bytes, shared_bytes))?;
        let views = (0..count).map(|i| planes.plane(i)).collect::<Option<Vec<_>>>()
            .ok_or(MiniMaxNativeError::Operand("tiered plane count"))?;
        let set = Rc::new(PlaneSet { library, planes });
        Ok(views.into_iter().map(|buffer| Self { storage: Rc::new(Storage { library, buffer,
            planes: Some(set.clone()) }), view: buffer }).collect())
    }
    pub fn raw(&self) -> CuteafdDeviceBuffer { self.view }
    pub fn view(&self, offset: usize, bytes: usize) -> Result<Self> {
        ensure!(offset.checked_add(bytes).is_some_and(|end| end <= self.view.bytes), MiniMaxNativeError::Operand("view exceeds storage"));
        Ok(Self { storage: self.storage.clone(), view: CuteafdDeviceBuffer {
            ptr: self.view.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..self.view } })
    }
    pub fn upload(&self, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= self.view.bytes, "MiniMax upload exceeds storage");
        on_device(self.storage.library, self.view.device_id, || self.storage.library.copy_h2d(
            CuteafdDeviceBuffer { bytes: bytes.len(), ..self.view }, bytes))
    }
}

/// Trusted generated native library, checked for the MiniMax coordinator inventory.
pub fn load_library(path: &Path) -> Result<NativeLibrary> {
    // SAFETY: the caller chooses the installed engine module; resolving the
    // inventory below rejects libraries that cannot execute this family.
    let library = native("library load",unsafe { NativeLibrary::load(path) })?;
    { let programs = native("program inventory",library.programs())?;
        for (name, _) in definitions() { programs.spec(name).map_err(|_|MiniMaxNativeError::Program {
            program:name.into(),reason:"missing coordinator export" })?; } }
    for name in ["cuteafd_cuda_accumulate_bf16_to_f32_async", "cuteafd_add_tp2_shared_async", INDEX_STAGE, KV_STAGE, KV_TRANSLATE,
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
const KV_STAGE: &str = "cuteafd_minimax_kv_stage_async";
const KV_TRANSLATE: &str = "cuteafd_minimax_kv_translate_async";

/// Staging scratch of a host-placed cache (`minimax_kv_stage.cu`): one slot-map
/// entry per pool page, one head mask per staging page and a 16-byte counter.
pub fn kv_stage_scratch_bytes(pool_pages: usize, staging_pages: usize) -> usize {
    (pool_pages * 4).next_multiple_of(16) + (staging_pages * 4).next_multiple_of(16) + 16
}

/// One layer's staging of host K/V pages ([`Commands::kv_stage`]).
#[derive(Debug, Clone, Copy)]
pub struct KvStage {
    /// Selection rows the q2k buffer is laid out for (the export's capacity).
    pub q2k_rows: usize,
    pub rows: usize,
    /// Decode: one table and visible length per row; extend: one shared table
    /// and the cache length after the rows.
    pub per_row: bool,
    pub stride: usize,
    /// Page ids below this are device memory.
    pub resident: usize,
    pub pool_pages: usize,
    /// Staging page ids `base..base + capacity` of every plane.
    pub base: usize,
    pub capacity: usize,
}

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

pub struct Commands<'a> {
    library: &'a NativeLibrary,
    device: i32,
    stream: *mut c_void,
    programs: BTreeMap<&'static str, Program<'a>>,
    identity:Rc<()>,
    retained: RefCell<Vec<Buffer<'a>>>,
    owners: RefCell<Vec<Rc<dyn Retained + 'a>>>,
}
trait Retained {}
impl<T> Retained for T {}
impl<'a> Commands<'a> {
    pub fn new(library: &'a NativeLibrary, device: i32, programs: &'a Programs<'a>) -> Result<Self> {
        let loaded = on_device(library, device, || definitions().into_iter().map(|(name, pointers)| {
            let spec=programs.spec(name).map_err(|_|MiniMaxNativeError::Program{program:name.into(),reason:"missing coordinator export"})?;
            ensure!(spec.scalar_kinds==scalar_kinds(name), MiniMaxNativeError::Program{program:name.into(),reason:"scalar ABI mismatch"});
            ensure!(spec.pointers.iter().map(String::as_str).eq(pointers.iter().copied()),
                MiniMaxNativeError::Program{program:name.into(),reason:"pointer ABI mismatch"});
            validate_geometry(spec)?;
            Ok((name, native(name,programs.program(name, pointers))?))
        }).collect::<Result<_>>())?;
        let stream = on_device(library, device, || library.cuda_stream_create())?;
        Ok(Self { library, device, stream, programs: loaded, identity:Rc::new(()),retained: RefCell::new(Vec::new()),owners:RefCell::new(Vec::new()) })
    }
    pub fn raw(&self) -> *mut c_void { self.stream }
    pub fn library(&self) -> &'a NativeLibrary { self.library }
    pub fn device(&self) -> i32 { self.device }
    fn keep(&self, buffers: &[&Buffer<'a>]) -> Result<()> {
        ensure!(buffers.iter().all(|b| b.view.device_id == self.device
            && std::ptr::eq(b.storage.library, self.library)), "MiniMax operands belong to another device/library");
        self.retained.borrow_mut().extend(buffers.iter().map(|b| (*b).clone()));
        Ok(())
    }
    pub fn retain(&self, buffers: &[&Buffer<'a>]) -> Result<()> { self.keep(buffers) }
    pub fn retain_owner<T:'a>(&self, owner: Rc<T>) { self.owners.borrow_mut().push(owner); }
    pub fn drain(&self) -> Result<()> {
        // SAFETY: this owner created the stream and retains every operand.
        let result=on_device(self.library, self.device, || unsafe { self.library.cuda_stream_synchronize(self.stream) });
        if result.is_err() { self.library.quarantine_module_after_failed_drain(); return result; }
        on_device(self.library,self.device,||{
            self.retained.borrow_mut().clear();
            self.owners.borrow_mut().clear();
            Ok(())
        })
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
    pub fn launch(&self, name: &str, operands: &[&Buffer<'a>], scalars: &[Scalar]) -> Result<()> {
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
            ensure!(operand.view.bytes >= required[i].max(scratch), "MiniMax operand {} too small: {name}", spec.pointers[i]);
            ensure!(required[i].max(scratch)==0 || operand.view.ptr as usize%16==0,
                MiniMaxNativeError::Operand("program operand is not 16-byte aligned"));
        }
        self.keep(operands)?;
        let pointers: Vec<_> = operands.iter().map(|b| b.view.ptr).collect();
        // SAFETY: family ABI and extents were checked; the stream retains the
        // actual allocation owners, including every sliced operand, until drain.
        native(name,on_device(self.library, self.device, || unsafe { program.launch(&pointers, scalars, self.stream) }))
    }
    /// Gathers `count` pages of an MSA index cache (their physical ids in
    /// `pages`) into the contiguous key rows and FP32 scales the prefill
    /// scorer reads: page j becomes rows [64j, 64j + 64).
    pub fn index_stage(&self, cache: &Buffer<'a>, pages: &Buffer<'a>, count: usize,
        keys: &Buffer<'a>, scales: &Buffer<'a>) -> Result<()> {
        ensure!(count > 0 && count <= i32::MAX as usize && pages.view.bytes >= count * 4 && keys.view.bytes >= count * 8192
            && scales.view.bytes >= count * 256 && cache.view.bytes % 8448 == 0, MiniMaxNativeError::Operand("index stage extents"));
        ensure!([cache, pages, keys, scales].iter().all(|b| b.view.ptr as usize % 16 == 0),
            MiniMaxNativeError::Operand("program operand is not 16-byte aligned"));
        self.keep(&[cache, pages, keys, scales])?;
        type Stage = unsafe extern "C" fn(*const c_void, *const i32, u32, *mut c_void, *mut c_void, *mut c_void) -> i32;
        // SAFETY: the symbol's C ABI is fixed by minimax_index_stage.cu; load_library required it.
        let stage = unsafe { self.library.lib.get::<Stage>(INDEX_STAGE.as_bytes()) }
            .map_err(|_| MiniMaxNativeError::Program { program: INDEX_STAGE.into(), reason: "missing shared execution export" })?;
        // SAFETY: checked extents and alignment; the page ids are the batch's
        // validated pool pages, and every operand is retained until drain.
        let status = on_device(self.library, self.device, || Ok(unsafe {
            stage(cache.view.ptr, pages.view.ptr.cast(), count as u32, keys.view.ptr, scales.view.ptr, self.stream) }))?;
        ensure!(status == 0, MiniMaxNativeError::Runtime { operation: "index stage".into(),
            source: anyhow::anyhow!("CUDA status {status}") });
        Ok(())
    }
    /// Copies the head slices of every host page (id >= `stage.resident`) the
    /// layer's selections read into the staging pages of `key`/`value`
    /// (`minimax_kv_stage.cu`). `scratch` ([`kv_stage_scratch_bytes`]) is zero
    /// on entry; `stats` (two u64) accumulates copied slices and host reads.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_stage(&self, stage: KvStage, q2k: &Buffer<'a>, table: &Buffer<'a>, lengths: &Buffer<'a>,
        key: &Buffer<'a>, value: &Buffer<'a>, scratch: &Buffer<'a>, stats: &Buffer<'a>) -> Result<()> {
        let tables = if stage.per_row { stage.rows } else { 1 };
        let plane = stage.base.checked_add(stage.capacity).and_then(|n| n.checked_mul(32768));
        ensure!(stage.rows > 0 && stage.rows <= stage.q2k_rows && stage.stride > 0 && stage.capacity > 0
            && stage.resident <= stage.pool_pages && stage.pool_pages <= stage.base && stage.base + stage.capacity <= i32::MAX as usize
            && q2k.view.bytes >= 4 * stage.q2k_rows * 16 * 4 && table.view.bytes >= tables * stage.stride * 4
            && lengths.view.bytes >= tables * 4 && plane.is_some_and(|n| key.view.bytes >= n && value.view.bytes >= n)
            && scratch.view.bytes >= kv_stage_scratch_bytes(stage.pool_pages, stage.capacity) && stats.view.bytes >= 16,
            MiniMaxNativeError::Operand("kv stage extents"));
        ensure!([q2k, table, lengths, key, value, scratch, stats].iter().all(|b| b.view.ptr as usize % 16 == 0),
            MiniMaxNativeError::Operand("program operand is not 16-byte aligned"));
        self.keep(&[q2k, table, lengths, key, value, scratch, stats])?;
        type Stage = unsafe extern "C" fn(*const i32, u32, u32, *const i32, u32, i32, *const i32, *mut c_void,
            *mut c_void, i32, i32, i32, *mut i32, *mut i32, *mut i32, *mut u64, *mut c_void) -> i32;
        // SAFETY: the symbol's C ABI is fixed by minimax_kv_stage.cu; load_library required it.
        let call = unsafe { self.library.lib.get::<Stage>(KV_STAGE.as_bytes()) }
            .map_err(|_| MiniMaxNativeError::Program { program: KV_STAGE.into(), reason: "missing shared execution export" })?;
        let map = scratch.view.ptr.cast::<i32>();
        // SAFETY: the scratch layout matches kv_stage_scratch_bytes (16-byte aligned parts).
        let heads = unsafe { map.byte_add((stage.pool_pages * 4).next_multiple_of(16)) };
        // SAFETY: as above; the counter follows the head masks.
        let count = unsafe { heads.byte_add((stage.capacity * 4).next_multiple_of(16)) };
        // SAFETY: checked extents and alignment; every operand is retained until drain.
        let status = on_device(self.library, self.device, || Ok(unsafe { call(q2k.view.ptr.cast(), stage.q2k_rows as u32,
            stage.rows as u32, table.view.ptr.cast(), stage.stride as u32, i32::from(stage.per_row), lengths.view.ptr.cast(),
            key.view.ptr, value.view.ptr, stage.resident as i32, stage.base as i32, stage.capacity as i32, map, heads, count,
            stats.view.ptr.cast(), self.stream) }))?;
        ensure!(status == 0, MiniMaxNativeError::Runtime { operation: "kv stage".into(),
            source: anyhow::anyhow!("CUDA status {status}") });
        Ok(())
    }
    /// `out[i]` = the staging page of `table[i]` when [`Commands::kv_stage`]
    /// copied it this layer, else `table[i]`.
    pub fn kv_translate(&self, table: &Buffer<'a>, out: &Buffer<'a>, entries: usize, scratch: &Buffer<'a>,
        resident: usize, base: usize, pool_pages: usize) -> Result<()> {
        ensure!(entries > 0 && table.view.bytes >= entries * 4 && out.view.bytes >= entries * 4
            && scratch.view.bytes >= pool_pages * 4 && resident <= pool_pages && pool_pages <= base && base <= i32::MAX as usize,
            MiniMaxNativeError::Operand("kv translate extents"));
        ensure!([table, out, scratch].iter().all(|b| b.view.ptr as usize % 16 == 0),
            MiniMaxNativeError::Operand("program operand is not 16-byte aligned"));
        self.keep(&[table, out, scratch])?;
        type Translate = unsafe extern "C" fn(*const i32, *mut i32, u64, *const i32, i32, i32, *mut c_void) -> i32;
        // SAFETY: the symbol's C ABI is fixed by minimax_kv_stage.cu; load_library required it.
        let call = unsafe { self.library.lib.get::<Translate>(KV_TRANSLATE.as_bytes()) }
            .map_err(|_| MiniMaxNativeError::Program { program: KV_TRANSLATE.into(), reason: "missing shared execution export" })?;
        // SAFETY: checked extents and alignment; every operand is retained until drain.
        let status = on_device(self.library, self.device, || Ok(unsafe { call(table.view.ptr.cast(), out.view.ptr.cast(),
            entries as u64, scratch.view.ptr.cast(), resident as i32, base as i32, self.stream) }))?;
        ensure!(status == 0, MiniMaxNativeError::Runtime { operation: "kv translate".into(),
            source: anyhow::anyhow!("CUDA status {status}") });
        Ok(())
    }
    pub fn copy(&self, destination: &Buffer<'a>, source: &Buffer<'a>, bytes: usize) -> Result<()> {
        ensure!(bytes <= destination.view.bytes && bytes <= source.view.bytes, "MiniMax copy extent");
        self.keep(&[destination, source])?;
        // SAFETY: checked views are retained by this stream through completion.
        on_device(self.library, self.device, || unsafe { self.library.copy_d2d_async(destination.raw(), source.raw(), bytes, self.stream) })
    }
    pub fn zero(&self, buffer: &Buffer<'a>) -> Result<()> {
        self.keep(&[buffer])?;
        // SAFETY: this stream retains the complete initialized destination.
        on_device(self.library, self.device, || unsafe { self.library.cuda_zero_bytes_async(buffer.raw(), buffer.view.bytes, self.stream) })
    }
    pub fn upload(&self, buffer: &Buffer<'a>, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= buffer.view.bytes, "MiniMax ordered upload extent");
        self.keep(&[buffer])?;
        // SAFETY: ordered upload owns its host staging; destination is retained.
        on_device(self.library, self.device, || unsafe { self.library.copy_h2d_ordered(
            CuteafdDeviceBuffer { bytes: bytes.len(), ..buffer.raw() }, bytes, self.stream) })
    }
    pub fn widen(&self, source: &Buffer<'a>, destination: &Buffer<'a>, count: usize) -> Result<()> {
        // Only the live rows are accumulated into, so only they are cleared.
        self.zero(&destination.view(0, count.checked_mul(4).context("MiniMax widen extent")?)?)?;
        self.keep(&[source, destination])?;
        // SAFETY: the native routine checks both extents; retained distinct
        // allocations are initialized by the preceding stream operations.
        on_device(self.library, self.device, || unsafe { self.library.cuda_accumulate_bf16_to_f32_async(
            source.raw(), destination.raw(), count, self.stream) })
    }
    pub fn add_bf16(&self, a: &Buffer<'a>, b: &Buffer<'a>, out: &Buffer<'a>, count: usize) -> Result<()> {
        self.keep(&[a,b,out])?;
        // SAFETY: the binding checks shapes, and all three owners remain live.
        on_device(self.library, self.device, || unsafe { self.library.bf16_add()?.launch(a.raw(),b.raw(),out.raw(),count,self.stream) })
    }
}
impl Drop for Commands<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            self.library.quarantine_module_after_failed_drain();
            tracing::error!(%error, "MiniMax stream failed to drain before owner destruction");
            std::process::abort();
        }
        // SAFETY: the owned stream has drained and no operand can be in use.
        let _ = on_device(self.library, self.device, || unsafe { self.library.cuda_stream_destroy(self.stream) });
    }
}

pub fn linear_shape(name: &str) -> Option<(usize,usize)> { Some(match name {
    "q" => (8192,6144), "kv" => (1024,6144), "o" => (6144,8192),
    "index_q" => (512,6144), "index_k" => (128,6144), "shared_gate_up" => (6144,6144),
    "shared_down" => (6144,3072), "dense_gate_up" => (24576,6144), "dense_down" => (6144,12288), _ => return None,
}) }

struct HeadOwner<'a> { handle: VocabularyHead<'a>, workspace: Buffer<'a>, stream:Rc<()> }
pub struct Head<'a> { owner:Rc<HeadOwner<'a>>, rows: usize, vocab: usize }
impl<'a> Head<'a> {
    pub fn new(commands: &Commands<'a>, workspace: Buffer<'a>, rows: usize, vocab: usize) -> Result<Self> {
        ensure!(workspace.view.bytes >= 4<<20 && rows > 0 && rows <= 4096, "MiniMax head admission");
        commands.keep(&[&workspace])?;
        // SAFETY: the owner retains its workspace until the handle is dropped.
        let handle = on_device(commands.library, commands.device, || unsafe { commands.library.vocabulary_head_rows(
            workspace.view.ptr,6144,rows as u32,vocab as u32) })?;
        Ok(Self { owner:Rc::new(HeadOwner{handle,workspace,stream:commands.identity.clone()}), rows, vocab })
    }
    pub fn launch(&self, commands: &Commands<'a>, input: &Buffer<'a>, weight: &Buffer<'a>,
        output: &Buffer<'a>, rows: usize, tensor: bool) -> Result<()> {
        ensure!(Rc::ptr_eq(&self.owner.stream,&commands.identity),MiniMaxNativeError::Operand("head used on another stream"));
        ensure!(rows>0 && rows<=self.rows && input.view.bytes>=rows*6144*2
            && weight.view.bytes>=self.vocab*6144*2 && output.view.bytes>=rows*self.vocab*4, "MiniMax head operands");
        commands.keep(&[input,weight,output,&self.owner.workspace])?;
        commands.retain_owner(self.owner.clone());
        // SAFETY: the head's fixed geometry and live extents agree; all operand
        // owners remain retained through the execution stream's drain.
        on_device(commands.library, commands.device, || unsafe {
            if tensor { self.owner.handle.launch_tensor_invariant(input.view.ptr.cast(),weight.view.ptr.cast(),output.view.ptr.cast(),rows as u32,commands.stream) }
            else { self.owner.handle.launch(input.view.ptr.cast(),weight.view.ptr.cast(),output.view.ptr.cast(),rows as u32,commands.stream) }
        })
    }
}

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
    fn minimax_native_fixture_retains_views_and_validates_launches() -> Result<()> {
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
        let programs=Programs::fixture(&library,specs,launch);
        let commands=Commands::new(&library,0,&programs)?;
        struct Owner(Rc<std::cell::Cell<bool>>);
        impl Drop for Owner { fn drop(&mut self) { self.0.set(true); } }
        let retired=Rc::new(std::cell::Cell::new(false));
        let owner=Rc::new(Owner(retired.clone()));
        commands.retain_owner(owner.clone());drop(owner);
        assert!(!retired.get(),"queued launch lost its module owner");
        let source=Buffer::new(&library,0,512)?;
        source.upload(&(0..512).map(|n|(n%251) as u8).collect::<Vec<_>>())?;
        let weak=Rc::downgrade(&source.storage);
        let view=source.view(128,64)?;
        let output=Buffer::new(&library,0,64)?;
        commands.copy(&output,&view,64)?;
        drop(source);drop(view);
        assert!(weak.upgrade().is_some(),"queued copy lost its source owner");
        commands.drain()?;
        assert!(weak.upgrade().is_none(),"completed copy retained its retired source");
        assert!(retired.get(),"completed stream retained its retired module owner");
        let mut bytes=vec![0;64];library.copy_d2h(&mut bytes,output.raw())?;
        assert_eq!(bytes,(128..192).map(|n|(n%251) as u8).collect::<Vec<_>>());
        assert!(output.view(63,2).is_err());
        assert!(output.view(usize::MAX,2).is_err());
        let other=Buffer::new(&library,1,64)?;
        assert!(commands.copy(&output,&other,64).is_err());
        let input=Buffer::new(&library,0,6144*4+16)?;
        let weight=Buffer::new(&library,0,6144*2)?;
        let norm=Buffer::new(&library,0,6144*2)?;
        assert!(commands.launch("minimax_norm",&[&input,&weight,&weight,&weight,&weight,&weight,&output],&[Scalar::I32(1)]).is_err());
        assert!(commands.launch("minimax_norm",&[&input,&weight,&weight,&weight,&weight,&weight,&norm],&[Scalar::I32(4097)]).is_err());
        let unaligned=input.view(1,6144*4)?;
        assert!(commands.launch("minimax_norm",&[&unaligned,&weight,&weight,&weight,&weight,&weight,&norm],&[Scalar::I32(1)]).is_err());
        assert_eq!(LAUNCHES.load(Ordering::Relaxed),0,"invalid launch reached the native dispatcher");
        commands.launch("minimax_norm",&[&input,&weight,&weight,&weight,&weight,&weight,&norm],&[Scalar::I32(1)])?;
        assert_eq!(LAUNCHES.load(Ordering::Relaxed),1);
        STATUS.store(17,Ordering::Relaxed);
        let error=commands.launch("minimax_norm",&[&input,&weight,&weight,&weight,&weight,&weight,&norm],&[Scalar::I32(1)]).unwrap_err();
        assert!(matches!(error.downcast_ref::<MiniMaxNativeError>(),Some(MiniMaxNativeError::Runtime{operation,..}) if operation=="minimax_norm"));
        assert!(format!("{error:#}").contains("status 17"));
        STATUS.store(0,Ordering::Relaxed);
        commands.drain()?;
        Ok(())
    }
}
