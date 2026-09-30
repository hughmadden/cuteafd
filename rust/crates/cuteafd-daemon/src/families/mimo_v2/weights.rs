//! MiMo V2 coordinator weights, packed for the exported mimo_* programs.
//!
//! Every operand is BF16: FP8 checkpoint tensors are dequantized on the GPU
//! with their FP32 block scales (re-quantizing to power-of-two scales costs
//! +0.043 nats mean NLL on the golden prompt), BF16 tensors are copied.
//! Scale grids are 128x128 blocks except the full-attention `k_proj`
//! [768, 4096], whose [8, 32] grid is per KV head: each 192-row head is a
//! 128-row block then a 64-row block, so it is dequantized head by head.
//! `w_qkv` is `[q_proj; k_proj; v_proj]` written in place (V2.6 Pro: the
//! fused, TP-interleaved `qkv_proj` de-interleaved with every key padded to
//! 256 rows, see `FusedQkvLayout`); the FP32 router weight (Flash) becomes
//! `w_hilo = [bf16(w); bf16(w - bf16(w))]` for the router program's two
//! FP32-accumulated BF16 products, a BF16 one (V2.6 Pro) is `w_router` as
//! stored.
//!
//! FP8 decode copies (`fp8_decode`): `{w}_fp8` (E4M3 `[N, K]`) and
//! `{w}_scale` (FP32 `[N, K/128]`, one scale per output row and 128-wide K
//! block) of `w_qkv`, `w_o` and the dense `w_gate_up`/`w_down`, read by the
//! decode programs for steps of at most 16 rows. FP8 checkpoint tensors keep
//! their own E4M3 bytes, their block grids expanded per row (exact: the
//! MMA widens to the same `bf16(w * s)` the BF16 operand holds); BF16
//! tensors (`o_proj`) are quantized per row and 128-K block. `fp8_head`: the
//! same per-row quantization of the BF16 LM head.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::mimo_v2::{FusedQkvLayout, MimoAttention, MimoV2Config};
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

    /// V2.6 Pro's fused `qkv_proj` (FP8, TP-interleaved row shards with their
    /// own 128x128 grids) in the coordinator's `[q; k; v]` layout with keys
    /// `cfg.qkv_key_stride()` rows apart (256: each 192-row key zero-padded so
    /// every checkpoint block is a whole 128-row block). Returns the BF16
    /// weight and, with `fp8`, the E4M3 rows and FP32 per-row x 128-K scales
    /// in that layout (the decode producers' operands; exactly the
    /// checkpoint's values).
    fn fused_qkv(&self, cfg: &MimoV2Config, attention: MimoAttention, name: &str, fp8: bool)
        -> Result<(DeviceAllocation<'a>, Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>)> {
        let layout = FusedQkvLayout::new(cfg, attention, self.checkpoint_tp)?;
        let (bytes, dtype, shape) = self.raw(name)?;
        let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
        let cols = shape[1];
        let k_blocks = cols.div_ceil(128);
        ensure!(dtype == DType::F8E4M3 && shape == [layout.rows(), cols] && scale_dtype == DType::F32
            && scale_shape == [layout.scale_rows(), k_blocks] && cols % 128 == 0,
            "{name}: expected E4M3 [{}, {cols}] with FP32 [{}, {}] scales for checkpoint TP {}, found {dtype:?} {shape:?} / \
             {scale_dtype:?} {scale_shape:?}", layout.rows(), layout.scale_rows(), k_blocks, self.checkpoint_tp);
        let stride = cfg.qkv_key_stride();
        let segments = if stride == layout.k {
            layout.segments()
        } else {
            ensure!(layout.k == cfg.head_dim && stride % 128 == 0 && layout.q % 128 == 0 && layout.v % 128 == 0,
                "{name}: padded keys need one KV head per checkpoint shard and 128-row query/value shards");
            layout.segments_with_key_stride(stride)
        };
        let width = layout.padded_rows(stride);
        // The padded E4M3 rows and grid (unused rows and blocks stay zero).
        let mut values = vec![0u8; width * cols];
        let mut grid = vec![0u8; width.div_ceil(128) * k_blocks * 4];
        let uniform = stride % 128 == 0;
        for segment in &segments {
            values[segment.dest_row * cols..][..segment.rows * cols]
                .copy_from_slice(&bytes[segment.source_row * cols..][..segment.rows * cols]);
            if uniform {
                let blocks = segment.rows.div_ceil(128) * k_blocks * 4;
                grid[segment.dest_row / 128 * k_blocks * 4..][..blocks]
                    .copy_from_slice(&scale[segment.scale_row * k_blocks * 4..][..blocks]);
            }
        }
        let out = DeviceAllocation::new(self.library, width * cols * 2)?;
        self.library.cuda_zero_bytes(out.buffer, out.buffer.bytes)?;
        let w = self.upload(&values)?;
        // The decode programs' per-row x 128-K scales: each row takes its
        // shard grid's block row (padding rows keep 0).
        let row_scales = if fp8 {
            let mut rows = vec![0u8; width * k_blocks * 4];
            for segment in &segments {
                for r in 0..segment.rows {
                    rows[(segment.dest_row + r) * k_blocks * 4..][..k_blocks * 4]
                        .copy_from_slice(&scale[(segment.scale_row + r / 128) * k_blocks * 4..][..k_blocks * 4]);
                }
            }
            Some(self.upload(&rows)?)
        } else {
            None
        };
        if uniform {
            let s = self.upload(&grid)?;
            // SAFETY: `w`, `s` and `out` hold the padded layout; the stream drains before return.
            unsafe {
                self.library.fp8_block_dequant(w.buffer.ptr, s.buffer.ptr, out.buffer.ptr, width, cols, self.stream)?;
                self.library.cuda_stream_synchronize(self.stream)?;
            }
            return Ok((out, row_scales.map(|s| (w, s))));
        }
        let s = self.upload(&scale)?;
        for segment in &segments {
            // SAFETY: the segment's source rows (already at their destination in `w`),
            // their scale rows and the destination rows lie inside `w`, `s` and `out`.
            unsafe {
                self.library.fp8_block_dequant(
                    w.buffer.ptr.cast::<u8>().add(segment.dest_row * cols).cast(),
                    s.buffer.ptr.cast::<u8>().add(segment.scale_row * k_blocks * 4).cast(),
                    out.buffer.ptr.cast::<u8>().add(segment.dest_row * cols * 2).cast(),
                    segment.rows, cols, self.stream)?;
            }
        }
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        Ok((out, row_scales.map(|s| (w, s))))
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
        self.block(cfg, &format!("model.mtp.layers.{k}"), MimoAttention::Sliding, true, "pre_mlp_layernorm")
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
        let both = |ops: &mut HashMap<&'static str, DeviceAllocation<'a>>, key: &'static str, names: &[String],
            fp8: bool| -> Result<()> {
            let (bf16, copy) = self.rows_fp8(names, fp8)?;
            ops.insert(key, bf16);
            if let Some((q, s)) = copy {
                let (q_key, s_key) = match key {
                    "w_qkv" => ("w_qkv_fp8", "w_qkv_scale"),
                    "w_o" => ("w_o_fp8", "w_o_scale"),
                    "w_gate_up" => ("w_gate_up_fp8", "w_gate_up_scale"),
                    _ => ("w_down_fp8", "w_down_scale"),
                };
                ops.insert(q_key, q);
                ops.insert(s_key, s);
            }
            Ok(())
        };
        if self.has(&fused) {
            let (bf16, fp8) = self.fused_qkv(cfg, attention, &fused, self.fp8_decode)?;
            ops.insert("w_qkv", bf16);
            if let Some((values, scales)) = fp8 {
                ops.insert("w_qkv_fp8", values);
                ops.insert("w_qkv_scale", scales);
            }
        } else {
            both(&mut ops, "w_qkv", &qkv, self.fp8_decode)?;
        }
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
            let gate_up = [format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")];
            both(&mut ops, "w_gate_up", &gate_up, self.fp8_decode)?;
            both(&mut ops, "w_down", &[format!("{p}.mlp.down_proj.weight")], self.fp8_decode)?;
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
        Ok(MimoWeights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            norm: self.one("model.norm.weight")?,
            head,
            head_fp8,
        })
    }
}
