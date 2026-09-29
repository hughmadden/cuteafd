//! MiMo V2 coordinator weights, packed for the exported mimo_* programs.
//!
//! Every operand is BF16: FP8 checkpoint tensors are dequantized on the GPU
//! with their FP32 block scales (re-quantizing to power-of-two scales costs
//! +0.043 nats mean NLL on the golden prompt), BF16 tensors are copied.
//! Scale grids are 128x128 blocks except the full-attention `k_proj`
//! [768, 4096], whose [8, 32] grid is per KV head: each 192-row head is a
//! 128-row block then a 64-row block, so it is dequantized head by head.
//! `w_qkv` is `[q_proj; k_proj; v_proj]` written in place (V2.6 Pro: the
//! fused, TP-interleaved `qkv_proj` de-interleaved, see `FusedQkvLayout`); the
//! FP32 router weight (Flash) becomes `w_hilo = [bf16(w); bf16(w - bf16(w))]`
//! for the router program's two FP32-accumulated BF16 products, a BF16 one
//! (V2.6 Pro) is `w_router` as stored.
use crate::v41_memory::DeviceAllocation;
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
    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
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
}

pub(crate) struct MimoWeights<'a> {
    pub layers: Vec<MimoLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    pub head: DeviceAllocation<'a>,
}

pub(crate) struct MimoLoader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    pub stream: *mut c_void,
    /// The checkpoint's tensor-parallel degree (fused `qkv_proj` row shards).
    pub checkpoint_tp: usize,
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
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum();
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
        let mut row = 0;
        for (name, (bytes, dtype, shape)) in &tensors {
            let dest = |first: usize, count: usize| CuteafdDeviceBuffer {
                // SAFETY: rows row+first..row+first+count lie inside `out`.
                ptr: unsafe { out.buffer.ptr.cast::<u8>().add((row + first) * cols * 2) }.cast(),
                bytes: count * cols * 2,
                ..out.buffer
            };
            match dtype {
                DType::Bf16 => self.library.copy_h2d(dest(0, shape[0]), bytes)?,
                DType::F8E4M3 => {
                    let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                    ensure!(scale_dtype == DType::F32, "{name}: block scales must be FP32");
                    let k_blocks = cols.div_ceil(128);
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
                    // SAFETY: the engine owns this stream.
                    unsafe { self.library.cuda_stream_synchronize(self.stream)? };
                }
                other => anyhow::bail!("{name}: unsupported coordinator dtype {other:?}"),
            }
            row += shape[0];
        }
        Ok(out)
    }

    fn has(&self, name: &str) -> bool {
        self.checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name)).is_ok()
    }

    /// V2.6 Pro's fused `qkv_proj` (FP8, TP-interleaved row shards with their
    /// own 128x128 grids) dequantized into the de-interleaved `[q; k; v]`.
    fn fused_qkv(&self, cfg: &MimoV2Config, attention: MimoAttention, name: &str) -> Result<DeviceAllocation<'a>> {
        let layout = FusedQkvLayout::new(cfg, attention, self.checkpoint_tp)?;
        let (bytes, dtype, shape) = self.raw(name)?;
        let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
        let cols = shape[1];
        ensure!(dtype == DType::F8E4M3 && shape == [layout.rows(), cols] && scale_dtype == DType::F32
            && scale_shape == [layout.scale_rows(), cols.div_ceil(128)],
            "{name}: expected E4M3 [{}, {cols}] with FP32 [{}, {}] scales for checkpoint TP {}, found {dtype:?} {shape:?} / \
             {scale_dtype:?} {scale_shape:?}", layout.rows(), layout.scale_rows(), cols.div_ceil(128), self.checkpoint_tp);
        let out = DeviceAllocation::new(self.library, layout.rows() * cols * 2)?;
        let (w, s) = (self.upload(&bytes)?, self.upload(&scale)?);
        let k_blocks = cols.div_ceil(128);
        for segment in layout.segments() {
            // SAFETY: the segment's source rows, their scale rows and the destination
            // rows lie inside `w`, `s` and `out`; the stream drains before `w`/`s` drop.
            unsafe {
                self.library.fp8_block_dequant(
                    w.buffer.ptr.cast::<u8>().add(segment.source_row * cols).cast(),
                    s.buffer.ptr.cast::<u8>().add(segment.scale_row * k_blocks * 4).cast(),
                    out.buffer.ptr.cast::<u8>().add(segment.dest_row * cols * 2).cast(),
                    segment.rows, cols, self.stream)?;
            }
        }
        // SAFETY: the engine owns this stream.
        unsafe { self.library.cuda_stream_synchronize(self.stream)? };
        Ok(out)
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
        let p = format!("model.layers.{layer}");
        let attention = cfg.attention[layer];
        let dense = cfg.dense[layer];
        let mut ops: HashMap<&'static str, DeviceAllocation<'a>> = HashMap::new();
        ops.insert("input_norm", self.one(&format!("{p}.input_layernorm.weight"))?);
        ops.insert("post_norm", self.one(&format!("{p}.post_attention_layernorm.weight"))?);
        let fused = format!("{p}.self_attn.qkv_proj.weight");
        ops.insert("w_qkv", if self.has(&fused) {
            self.fused_qkv(cfg, attention, &fused)?
        } else {
            self.rows(&[format!("{p}.self_attn.q_proj.weight"), format!("{p}.self_attn.k_proj.weight"),
                format!("{p}.self_attn.v_proj.weight")])?
        });
        ops.insert("w_o", self.one(&format!("{p}.self_attn.o_proj.weight"))?);
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
            ops.insert("w_gate_up", self.rows(&[format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")])?);
            ops.insert("w_down", self.one(&format!("{p}.mlp.down_proj.weight"))?);
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
        Ok(MimoWeights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            norm: self.one("model.norm.weight")?,
            head: self.one("lm_head.weight")?,
        })
    }
}
