//! MiMo V2 coordinator weights, packed for the exported mimo_* programs.
//!
//! The checkpoint's FP8 projections, `w_qkv` and the dense `w_gate_up` /
//! `w_down`, stay FP8 and have no other copy: `{w}_fp8` (the E4M3 bytes,
//! `[N, K]`) with the block grid expanded to one FP32 scale per output row and
//! 128-wide K block (exact, the programs widen to `bf16(w * s)`), row major
//! for decode programs (`{w}_scale`, `[N, K/128]`) and K-block major for
//! prefill programs (`{w}_kscale`, `[K/128, N]`). Decode programs run the FP8 GEMVs up to
//! 32 rows and W8A16 GEMMs above (bitwise the former BF16 programs over the
//! dequantized weights); prefill programs run W8A8 or W8A16. Scale grids are
//! 128x128 blocks except the full-attention `k_proj` [768, 4096], whose
//! [8, 32] grid is per KV head (each 192-row head a 128-row then a 64-row
//! block). `w_qkv` is `[q_proj; k_proj; v_proj]` (V2.6 Pro: the fused,
//! TP-interleaved `qkv_proj` de-interleaved with every key padded to 256
//! rows, see `FusedQkvLayout`).
//!
//! BF16 operands: `w_o` (BF16 in the release; `fp8_decode && fp8_o_proj` adds
//! an E4M3 copy with per-row x 128-K scales `[N, K/128]` for decode rows), the
//! norms, sinks and LM head (`fp8_head`: the same per-row quantization for
//! decode rows); the FP32 router weight (Flash) becomes `w_hilo = [bf16(w);
//! bf16(w - bf16(w))]` for the router program's two FP32-accumulated BF16
//! products, a BF16 one (V2.6 Pro) is `w_router` as stored.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::mimo_v2::{FusedQkvLayout, MimoAttention, MimoV2Config};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;

pub(crate) struct MimoLayer<'a> {
    pub attention: MimoAttention,
    pub dense: bool,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

impl MimoLayer<'_> {
    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    pub fn has(&self, operand: &str) -> bool {
        self.operands.contains_key(operand)
    }

    /// The router scores program's weight operand: `w_hilo` (FP32 weight split
    /// into BF16 hi + lo, Flash) or `w_router` (BF16 as stored, V2.6 Pro).
    pub fn router_operand(&self) -> Result<(&'static str, *mut c_void)> {
        for name in ["w_hilo", "w_router"] {
            if let Some(weight) = self.operands.get(name) {
                return Ok((name, weight.buffer.ptr));
            }
        }
        anyhow::bail!("layer has no router weight")
    }

    /// `operand`, or `fallback` when the layer has no such weight (an FP8
    /// operand the program then does not read: it runs with `fp8_rows` 0).
    pub fn ptr_or(&self, operand: &str, fallback: &str) -> Result<*mut c_void> {
        self.operands.get(operand).map_or_else(|| self.ptr(fallback), |a| Ok(a.buffer.ptr))
    }

    /// Device bytes of the layer's operands.
    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct MimoWeights<'a> {
    pub layers: Vec<MimoLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    pub head: DeviceAllocation<'a>,
    /// E4M3 LM head and its per-row x 128-K scales (`fp8_head`).
    pub head_fp8: Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>,
}

pub(crate) struct MimoLoader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    pub stream: *mut c_void,
    /// The checkpoint's tensor-parallel degree (fused `qkv_proj` row shards).
    pub checkpoint_tp: usize,
    pub fp8_decode: bool,
    pub fp8_head: bool,
    pub fp8_o_proj: bool,
    /// Scale rule of copies quantized from BF16 (o_proj, the LM head).
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
}

/// Scale-grid row of every weight row: uniform 128-row blocks, or per
/// 192-row head (a 128-row then a 64-row block).
fn scale_rows(name: &str, rows: usize, grid_rows: usize) -> Result<Vec<usize>> {
    if rows.div_ceil(128) == grid_rows {
        return Ok((0..rows).map(|r| r / 128).collect());
    }
    let head = 192;
    ensure!(rows % head == 0 && grid_rows == rows / head * head.div_ceil(128),
        "{name}: no block layout maps {rows} rows onto {grid_rows} scale rows");
    Ok((0..rows).map(|r| r / head * head.div_ceil(128) + r % head / 128).collect())
}

/// An FP8-only weight: E4M3 values and its per-row x 128-K scales row major
/// (`[N, K/128]`, decode programs) and K-block major (`[K/128, N]`, prefill).
struct Fp8Copy<'a> {
    values: DeviceAllocation<'a>,
    scale: DeviceAllocation<'a>,
    kscale: DeviceAllocation<'a>,
}

impl<'a> Fp8Copy<'a> {
    fn insert(self, ops: &mut HashMap<&'static str, DeviceAllocation<'a>>, names: [&'static str; 3]) {
        ops.insert(names[0], self.values);
        ops.insert(names[1], self.scale);
        ops.insert(names[2], self.kscale);
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 4];
    for (dst, v) in out.chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// Row-major `[N, KB]` FP32 scales as K-block-major `[KB, N]` bytes.
fn kmajor(row_scales: &[f32], k_blocks: usize) -> Vec<u8> {
    let n = row_scales.len() / k_blocks;
    let mut out = vec![0u8; row_scales.len() * 4];
    for (b, block) in out.chunks_exact_mut(n * 4).enumerate() {
        for (r, value) in block.chunks_exact_mut(4).enumerate() {
            value.copy_from_slice(&row_scales[r * k_blocks + b].to_le_bytes());
        }
    }
    out
}

impl<'a> MimoLoader<'a> {
    fn tensor(&self, name: &str) -> Result<&CheckpointTensor> {
        let at = self.checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
        Ok(&self.checkpoint.tensors[at])
    }

    fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        let tensor = self.tensor(name)?;
        let mut bytes = vec![0u8; tensor.meta.byte_length as usize];
        std::fs::File::open(self.checkpoint.snapshot.join(&tensor.shard))?
            .read_exact_at(&mut bytes, tensor.meta.byte_offset)
            .with_context(|| format!("reading {name}"))?;
        Ok((bytes, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
    }

    /// Reads tensor `name`'s bytes into `out[at..]` (one positioned read into the
    /// operand's buffer: no second copy) and returns its byte length, dtype and shape.
    fn read_into(&self, name: &str, out: &mut [u8], at: usize) -> Result<(usize, DType, Vec<usize>)> {
        let tensor = self.tensor(name)?;
        let length = tensor.meta.byte_length as usize;
        ensure!(at + length <= out.len(), "{name}: {length} bytes past the operand buffer");
        std::fs::File::open(self.checkpoint.snapshot.join(&tensor.shard))?
            .read_exact_at(&mut out[at..at + length], tensor.meta.byte_offset)
            .with_context(|| format!("reading {name}"))?;
        Ok((length, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = DeviceAllocation::new(self.library, bytes.len().max(256))?;
        self.library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    }

    /// The row-concatenation of 2-D `names` as one BF16 operand.
    fn rows(&self, names: &[String]) -> Result<DeviceAllocation<'a>> {
        Ok(self.rows_fp8(names, false)?.0)
    }

    /// `rows`, and with `fp8` also its E4M3 copy with per-row x 128-K FP32
    /// scales: FP8 tensors keep their bytes (block grid expanded per row),
    /// BF16 ones are quantized per row and 128-K block on the GPU. Each
    /// tensor is read once.
    #[allow(clippy::type_complexity)]
    fn rows_fp8(&self, names: &[String], fp8: bool)
        -> Result<(DeviceAllocation<'a>, Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>)> {
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum();
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        ensure!(!fp8 || cols % 128 == 0, "{names:?}: FP8 copies need K % 128 == 0");
        let k_blocks = cols.div_ceil(128);
        let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
        let copy = if fp8 {
            Some((DeviceAllocation::new(self.library, rows * cols)?,
                DeviceAllocation::new(self.library, rows * k_blocks * 4)?))
        } else {
            None
        };
        let at = |buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize| CuteafdDeviceBuffer {
            // SAFETY: callers keep offset + bytes inside the allocation.
            ptr: unsafe { buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..buffer
        };
        let mut row = 0;
        // Uploads the stream still reads; released after the final synchronize.
        let mut staged = Vec::new();
        for (name, (bytes, dtype, shape)) in &tensors {
            let dest = |first: usize, count: usize| at(out.buffer, (row + first) * cols * 2, count * cols * 2);
            match dtype {
                DType::Bf16 => {
                    self.library.copy_h2d(dest(0, shape[0]), bytes)?;
                    if let Some((q, s)) = &copy {
                        // SAFETY: the BF16 rows and their E4M3 / scale destinations are live;
                        // the stream drains below.
                        unsafe {
                            self.library.fp8_quant_rule(dest(0, shape[0]).ptr,
                                at(q.buffer, row * cols, shape[0] * cols).ptr,
                                at(s.buffer, row * k_blocks * 4, shape[0] * k_blocks * 4).ptr, shape[0], cols,
                                true, self.fp8_scales.code(), self.stream)?;
                        }
                    }
                }
                DType::F8E4M3 => {
                    let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                    ensure!(scale_dtype == DType::F32, "{name}: block scales must be FP32");
                    ensure!(scale_shape == [scale_shape[0], k_blocks], "{name}: unexpected scale grid {scale_shape:?}");
                    // Row blocks: uniform 128, or per 192-row head (128 + 64).
                    let (block, per_block_rows) = if shape[0].div_ceil(128) == scale_shape[0] {
                        (shape[0], shape[0].div_ceil(128))
                    } else {
                        let head = 192;
                        ensure!(shape[0] % head == 0 && scale_shape[0] == shape[0] / head * head.div_ceil(128),
                            "{name}: no block layout maps {} rows onto {} scale rows", shape[0], scale_shape[0]);
                        (head, head.div_ceil(128))
                    };
                    let (w, s) = (self.upload(bytes)?, self.upload(&scale)?);
                    for chunk in 0..shape[0] / block {
                        // SAFETY: chunk rows of the FP8 weight, their scale rows and the
                        // destination rows are live; the stream drains before `w`/`s` drop.
                        unsafe {
                            self.library.fp8_block_dequant(
                                w.buffer.ptr.cast::<u8>().add(chunk * block * cols).cast(),
                                s.buffer.ptr.cast::<u8>().add(chunk * per_block_rows * k_blocks * 4).cast(),
                                dest(chunk * block, block).ptr, block, cols, self.stream)?;
                        }
                    }
                    if let Some((q, s8)) = &copy {
                        let grid: Vec<f32> =
                            scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
                        let mut expanded = Vec::with_capacity(shape[0] * k_blocks * 4);
                        for r in scale_rows(name, shape[0], grid.len() / k_blocks)? {
                            for v in &grid[r * k_blocks..(r + 1) * k_blocks] {
                                expanded.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                        self.library.copy_h2d(at(s8.buffer, row * k_blocks * 4, expanded.len()), &expanded)?;
                        // SAFETY: both regions hold these E4M3 rows; ordered on the loader stream.
                        unsafe {
                            self.library.copy_d2d_async(at(q.buffer, row * cols, shape[0] * cols), w.buffer,
                                shape[0] * cols, self.stream)?;
                        }
                    }
                    staged.push((w, s));
                }
                other => anyhow::bail!("{name}: unsupported coordinator dtype {other:?}"),
            }
            row += shape[0];
        }
        // SAFETY: the loader owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        drop(staged);
        Ok((out, copy))
    }

    fn has(&self, name: &str) -> bool {
        self.checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name)).is_ok()
    }

    /// The row-concatenation of the FP8 checkpoint weights `names` as their
    /// E4M3 bytes `[N, K]` and FP32 per-row x 128-K scales (each row's
    /// block-grid value; exact), row major and K-block major.
    fn fp8_kmajor(&self, names: &[String]) -> Result<Fp8Copy<'a>> {
        let total: u64 = names.iter().map(|n| self.tensor(n).map(|t| t.meta.byte_length)).sum::<Result<u64>>()?;
        crate::shared::memory::staging::with_staging(total as usize, |values| {
            // Per row of the concatenation: its tensor's grid and that grid's block row.
            let (mut grids, mut rows) = (Vec::new(), Vec::<(usize, usize)>::new());
            let (mut cols, mut at) = (None, 0usize);
            for name in names {
                let (length, dtype, shape) = self.read_into(name, values, at)?;
                at += length;
                ensure!(dtype == DType::F8E4M3 && shape.len() == 2 && shape[1] % 128 == 0
                    && cols.is_none_or(|c| c == shape[1]),
                    "{name}: the mimo programs take this weight as an FP8 checkpoint tensor (the official FP8 \
                     release), found {dtype:?} {shape:?}");
                cols = Some(shape[1]);
                let k_blocks = shape[1] / 128;
                let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                ensure!(scale_dtype == DType::F32 && scale_shape.len() == 2 && scale_shape[1] == k_blocks,
                    "{name}: expected FP32 block scales, found {scale_dtype:?} {scale_shape:?}");
                grids.push(scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<f32>>());
                rows.extend(scale_rows(name, shape[0], scale_shape[0])?.into_iter().map(|r| (grids.len() - 1, r)));
            }
            let k_blocks = cols.context("no FP8 rows")? / 128;
            // Row major (decode) and K-block major (prefill) copies of each row's grid values.
            let n = rows.len();
            let mut scale = vec![0u8; k_blocks * n * 4];
            for (row, &(grid, r)) in scale.chunks_exact_mut(k_blocks * 4).zip(&rows) {
                for (value, v) in row.chunks_exact_mut(4).zip(&grids[grid][r * k_blocks..(r + 1) * k_blocks]) {
                    value.copy_from_slice(&v.to_le_bytes());
                }
            }
            let mut kscale = vec![0u8; k_blocks * n * 4];
            for (b, block) in kscale.chunks_exact_mut(n * 4).enumerate() {
                for (value, &(grid, r)) in block.chunks_exact_mut(4).zip(&rows) {
                    value.copy_from_slice(&grids[grid][r * k_blocks + b].to_le_bytes());
                }
            }
            Ok(Fp8Copy { values: self.upload(values)?, scale: self.upload(&scale)?, kscale: self.upload(&kscale)? })
        })
    }

    /// V2.6 Pro's fused `qkv_proj` (FP8, TP-interleaved row shards with their
    /// own 128x128 grids) in the coordinator's `[q; k; v]` layout with keys
    /// `cfg.qkv_key_stride()` rows apart (256: each 192-row key zero-padded):
    /// the E4M3 rows and FP32 per-row x 128-K scales, row major and K-block
    /// major (exactly the checkpoint's values; padding rows keep zero values and scales).
    fn fused_qkv(&self, cfg: &MimoV2Config, attention: MimoAttention, name: &str) -> Result<Fp8Copy<'a>> {
        let layout = FusedQkvLayout::new(cfg, attention, self.checkpoint_tp)?;
        let source_bytes = self.tensor(name)?.meta.byte_length as usize;
        let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
        let stride = cfg.qkv_key_stride();
        let width = layout.padded_rows(stride);
        let cols = source_bytes / layout.rows().max(1);
        crate::shared::memory::staging::with_staging_pair(width * cols, source_bytes, |values, bytes| {
            let (_, dtype, shape) = self.read_into(name, bytes, 0)?;
            let k_blocks = cols.div_ceil(128);
            ensure!(dtype == DType::F8E4M3 && shape == [layout.rows(), cols] && scale_dtype == DType::F32
                && scale_shape == [layout.scale_rows(), k_blocks] && cols % 128 == 0,
                "{name}: expected E4M3 [{}, {cols}] with FP32 [{}, {}] scales for checkpoint TP {}, found {dtype:?} \
                 {shape:?} / {scale_dtype:?} {scale_shape:?}", layout.rows(), layout.scale_rows(), k_blocks,
                self.checkpoint_tp);
            let segments = if stride == layout.k {
                layout.segments()
            } else {
                ensure!(layout.k == cfg.head_dim && stride % 128 == 0 && layout.q % 128 == 0 && layout.v % 128 == 0,
                    "{name}: padded keys need one KV head per checkpoint shard and 128-row query/value shards");
                layout.segments_with_key_stride(stride)
            };
            let grid: Vec<f32> = scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            // The padded E4M3 rows and each row's shard-grid scales; padding rows are zero.
            let mut covered = vec![false; width];
            let mut row_scales = vec![0f32; width * k_blocks];
            for segment in &segments {
                values[segment.dest_row * cols..][..segment.rows * cols]
                    .copy_from_slice(&bytes[segment.source_row * cols..][..segment.rows * cols]);
                covered[segment.dest_row..segment.dest_row + segment.rows].fill(true);
                for r in 0..segment.rows {
                    row_scales[(segment.dest_row + r) * k_blocks..][..k_blocks]
                        .copy_from_slice(&grid[(segment.scale_row + r / 128) * k_blocks..][..k_blocks]);
                }
            }
            for (row, _) in covered.iter().enumerate().filter(|(_, &c)| !c) {
                values[row * cols..(row + 1) * cols].fill(0);
            }
            Ok(Fp8Copy { values: self.upload(values)?, scale: self.upload(&f32_bytes(&row_scales))?,
                kscale: self.upload(&kmajor(&row_scales, k_blocks))? })
        })
    }

    fn one(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let (bytes, dtype, shape) = self.raw(name)?;
        if shape.len() == 2 && dtype == DType::F8E4M3 {
            return self.rows(&[name.to_string()]);
        }
        self.upload(&bytes)
    }

    /// `[bf16(w); bf16(w - bf16(w))]` of the FP32 router weight.
    fn router_hilo(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let (bytes, dtype, shape) = self.raw(name)?;
        ensure!(dtype == DType::F32 && shape.len() == 2, "{name}: the MiMo router weight is FP32 [E, H]");
        let bf16 = |x: f32| -> u16 {
            // Round to nearest even, as torch's float -> bfloat16.
            let bits = u64::from(x.to_bits());
            ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let values: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut out = Vec::with_capacity(values.len() * 4);
        let hi: Vec<u16> = values.iter().map(|&x| bf16(x)).collect();
        for h in &hi {
            out.extend_from_slice(&h.to_le_bytes());
        }
        for (&x, &h) in values.iter().zip(&hi) {
            out.extend_from_slice(&bf16(x - f32::from_bits(u32::from(h) << 16)).to_le_bytes());
        }
        self.upload(&out)
    }

    pub fn layer(&self, cfg: &MimoV2Config, layer: usize) -> Result<MimoLayer<'a>> {
        self.block(cfg, &format!("model.layers.{layer}"), cfg.attention[layer], cfg.dense[layer],
            "post_attention_layernorm")
    }

    /// MTP layer `k` (`model.mtp.layers.{k}`): an SWA decoder layer with a
    /// dense MLP (`pre_mlp_layernorm` is its post-attention norm).
    pub fn mtp_layer(&self, cfg: &MimoV2Config, k: usize) -> Result<MimoLayer<'a>> {
        let layer = self.block(cfg, &format!("model.mtp.layers.{k}"), MimoAttention::Sliding, true, "pre_mlp_layernorm");
        crate::shared::memory::staging::release_staging();
        layer
    }

    /// One MTP block's extra weights: `eh_proj` (BF16 [H, 2H]), `enorm`,
    /// `hnorm` and `final_layernorm`.
    pub fn mtp_extras(&self, k: usize) -> Result<[DeviceAllocation<'a>; 4]> {
        let p = format!("model.mtp.layers.{k}");
        Ok([self.one(&format!("{p}.eh_proj.weight"))?, self.one(&format!("{p}.enorm.weight"))?,
            self.one(&format!("{p}.hnorm.weight"))?, self.one(&format!("{p}.final_layernorm.weight"))?])
    }

    fn block(&self, cfg: &MimoV2Config, p: &str, attention: MimoAttention, dense: bool, post: &str)
        -> Result<MimoLayer<'a>> {
        let layer = p;
        let mut ops: HashMap<&'static str, DeviceAllocation<'a>> = HashMap::new();
        ops.insert("input_norm", self.one(&format!("{p}.input_layernorm.weight"))?);
        ops.insert("post_norm", self.one(&format!("{p}.{post}.weight"))?);
        let fused = format!("{p}.self_attn.qkv_proj.weight");
        let qkv = [format!("{p}.self_attn.q_proj.weight"), format!("{p}.self_attn.k_proj.weight"),
            format!("{p}.self_attn.v_proj.weight")];
        // `w_o` (BF16 in the release) plus, for decode rows, its per-row FP8 copy.
        let both = |ops: &mut HashMap<&'static str, DeviceAllocation<'a>>, key: &'static str, names: &[String],
            fp8: bool| -> Result<()> {
            let (bf16, copy) = self.rows_fp8(names, fp8)?;
            ops.insert(key, bf16);
            if let Some((q, s)) = copy {
                let (q_key, s_key) = match key {
                    "w_o" => ("w_o_fp8", "w_o_scale"),
                    other => anyhow::bail!("{other} has no BF16 + FP8 layout"),
                };
                ops.insert(q_key, q);
                ops.insert(s_key, s);
            }
            Ok(())
        };
        let copy = if self.has(&fused) { self.fused_qkv(cfg, attention, &fused)? } else { self.fp8_kmajor(&qkv)? };
        copy.insert(&mut ops, ["w_qkv_fp8", "w_qkv_scale", "w_qkv_kscale"]);
        both(&mut ops, "w_o", &[format!("{p}.self_attn.o_proj.weight")], self.fp8_decode && self.fp8_o_proj)?;
        let sinks = match attention {
            MimoAttention::Full => cfg.full_sinks,
            MimoAttention::Sliding => cfg.swa_sinks,
        };
        ensure!(sinks == (attention == MimoAttention::Sliding),
            "layer {layer}: the mimo programs take sinks on SWA layers only");
        if sinks {
            ops.insert("sinks", self.one(&format!("{p}.self_attn.attention_sink_bias"))?);
        }
        if dense {
            self.fp8_kmajor(&[format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")])?
                .insert(&mut ops, ["w_gate_up_fp8", "w_gate_up_scale", "w_gate_up_kscale"]);
            self.fp8_kmajor(&[format!("{p}.mlp.down_proj.weight")])?
                .insert(&mut ops, ["w_down_fp8", "w_down_scale", "w_down_kscale"]);
        } else {
            let router = format!("{p}.mlp.gate.weight");
            if self.tensor(&router)?.meta.dtype == DType::Bf16 {
                ops.insert("w_router", self.one(&router)?);
            } else {
                ops.insert("w_hilo", self.router_hilo(&router)?);
            }
            ops.insert("gate.bias", self.one(&format!("{p}.mlp.gate.e_score_correction_bias"))?);
        }
        Ok(MimoLayer { attention, dense, operands: ops })
    }

    /// Layers `0..layers` (all of them unless the caller stops early).
    pub fn model(&self, cfg: &MimoV2Config, layers: usize) -> Result<MimoWeights<'a>> {
        let (head, head_fp8) = self.rows_fp8(&["lm_head.weight".to_string()], self.fp8_head)?;
        let weights = MimoWeights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            norm: self.one("model.norm.weight")?,
            head,
            head_fp8,
        };
        crate::shared::memory::staging::release_staging();
        Ok(weights)
    }
}
