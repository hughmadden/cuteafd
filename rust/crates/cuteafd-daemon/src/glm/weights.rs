//! GLM 5.x coordinator weights, packed for the exported glm_* programs.
//!
//! Every operand has a BF16 copy: FP8 checkpoint tensors are dequantized on
//! the GPU with their FP32 128x128 block scales (re-quantizing to the
//! power-of-two scales b12x's FP8 linears take costs 0.025 nats), BF16
//! tensors are copied. The weights the decode (`_m64`) programs read in FP8
//! (`w_qkv_a`, `w_q_b`, `w_iq`, `w_o`, `w_gate_up`, `w_down`) also stay
//! resident as the checkpoint's E4M3 bytes (`{w}_fp8`) with their FP32 scale
//! grids (`{w}_scale`, `[ceil(N/128), K/128]`); each checkpoint tensor is
//! read once for both. Row-concatenated operands (`w_qkv_a`, `w_ik`,
//! `w_gate_up`) are written in place (every part but the last is a whole
//! number of 128-row blocks, so the scale grids concatenate too); `kv_b` is
//! split per head into `w_uk` (transposed) and `w_uv` on the host.
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::glm_dsa::{GlmDsaConfig, GlmIndexer};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;

pub(crate) struct GlmLayer<'a> {
    pub dense: bool,
    pub full_indexer: bool,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

/// The `{w}_fp8` / `{w}_scale` operand names of a decode FP8 weight.
pub(crate) fn fp8_operand_names(name: &str) -> (&'static str, &'static str) {
    match name {
        "w_qkv_a" => ("w_qkv_a_fp8", "w_qkv_a_scale"),
        "w_q_b" => ("w_q_b_fp8", "w_q_b_scale"),
        "w_iq" => ("w_iq_fp8", "w_iq_scale"),
        "w_o" => ("w_o_fp8", "w_o_scale"),
        "w_gate_up" => ("w_gate_up_fp8", "w_gate_up_scale"),
        "w_down" => ("w_down_fp8", "w_down_scale"),
        other => unreachable!("{other} has no FP8 decode operand"),
    }
}

impl GlmLayer<'_> {
    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }
}

pub(crate) struct GlmWeights<'a> {
    pub layers: Vec<GlmLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    pub head: DeviceAllocation<'a>,
}

pub(crate) struct GlmLoader<'a> {
    pub library: &'a NativeLibrary,
    pub catalog: &'a OfficialV41Catalog,
    pub stream: *mut c_void,
}

impl<'a> GlmLoader<'a> {
    fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        let tensor = self.catalog.tensor(name)?;
        let mut bytes = vec![0u8; tensor.metadata.byte_length as usize];
        std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
            .read_exact_at(&mut bytes, tensor.metadata.byte_offset)
            .with_context(|| format!("reading {name}"))?;
        Ok((bytes, tensor.metadata.dtype.clone(), tensor.metadata.shape.clone()))
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

    /// The row-concatenation of 2-D `names` as BF16 and, with `keep_fp8`, also
    /// as E4M3 `[N, K]` plus the FP32 block-scale grid `[ceil(N/128), K/128]`
    /// (every part must then be an FP8 checkpoint weight).
    #[allow(clippy::type_complexity)]
    fn rows_fp8(&self, names: &[String], keep_fp8: bool)
        -> Result<(DeviceAllocation<'a>, Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>)> {
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum();
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
        let k_blocks = cols.div_ceil(128);
        let fp8 = if keep_fp8 {
            ensure!(tensors.iter().all(|(_, (_, dtype, _))| *dtype == DType::F8E4M3),
                "{names:?}: the decode programs take these weights as FP8 checkpoint tensors");
            ensure!(tensors[..tensors.len() - 1].iter().all(|(_, (_, _, s))| s[0] % 128 == 0),
                "{names:?}: only the last concatenated weight may end inside a 128-row block");
            Some((DeviceAllocation::new(self.library, rows * cols)?,
                DeviceAllocation::new(self.library, rows.div_ceil(128) * k_blocks * 4)?))
        } else {
            None
        };
        let mut row = 0;
        for (name, (bytes, dtype, shape)) in &tensors {
            let dest = CuteafdDeviceBuffer {
                // SAFETY: rows row..row+shape[0] lie inside `out`.
                ptr: unsafe { out.buffer.ptr.cast::<u8>().add(row * cols * 2) }.cast(),
                bytes: shape[0] * cols * 2,
                ..out.buffer
            };
            match dtype {
                DType::Bf16 => self.library.copy_h2d(dest, bytes)?,
                DType::F8E4M3 => {
                    let (scale, scale_dtype, _) = self.raw(&format!("{}_scale_inv", name))?;
                    ensure!(scale_dtype == DType::F32, "{name}: block scales must be FP32");
                    ensure!(scale.len() == shape[0].div_ceil(128) * k_blocks * 4, "{name}: unexpected scale grid");
                    let temporaries;
                    let (w, s) = match &fp8 {
                        // The resident FP8 copy and scale grid at this part's rows.
                        Some((w8, s8)) => {
                            let sub = |alloc: &DeviceAllocation<'a>, offset: usize, bytes: usize| CuteafdDeviceBuffer {
                                // SAFETY: rows row..row+shape[0] (and their scale rows) lie inside `alloc`.
                                ptr: unsafe { alloc.buffer.ptr.cast::<u8>().add(offset) }.cast(),
                                bytes,
                                ..alloc.buffer
                            };
                            let (w, s) = (sub(w8, row * cols, bytes.len()), sub(s8, row / 128 * k_blocks * 4, scale.len()));
                            self.library.copy_h2d(w, bytes)?;
                            self.library.copy_h2d(s, &scale)?;
                            (w.ptr, s.ptr)
                        }
                        None => {
                            temporaries = (self.upload(bytes)?, self.upload(&scale)?);
                            (temporaries.0.buffer.ptr, temporaries.1.buffer.ptr)
                        }
                    };
                    // SAFETY: fp8 [shape], its scale grid and the destination rows
                    // are live; the stream is drained before the temporaries drop.
                    unsafe {
                        self.library.fp8_block_dequant(w, s, dest.ptr, shape[0], cols, self.stream)?;
                        self.library.cuda_stream_synchronize(self.stream)?;
                    }
                }
                other => anyhow::bail!("{name}: unsupported coordinator dtype {other:?}"),
            }
            row += shape[0];
        }
        Ok((out, fp8))
    }

    fn one(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let (bytes, dtype, shape) = self.raw(name)?;
        if shape.len() == 2 && dtype == DType::F8E4M3 {
            return self.rows(&[name.to_string()]);
        }
        self.upload(&bytes)
    }

    /// `w_uk[h,c,d] = kv_b[h*(D+V)+d, c]` and `w_uv[h,v,c] = kv_b[h*(D+V)+D+v, c]`.
    fn kv_b(&self, cfg: &GlmDsaConfig, prefix: &str) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let (heads, d, v, c) = (cfg.heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        let kv_b = self.rows(&[format!("{prefix}.self_attn.kv_b_proj.weight")])?;
        let mut host = vec![0u8; heads * (d + v) * c * 2];
        self.library.copy_d2h(&mut host, kv_b.buffer)?;
        let at = |row: usize, col: usize| &host[(row * c + col) * 2..][..2];
        let mut uk = Vec::with_capacity(heads * c * d * 2);
        let mut uv = Vec::with_capacity(heads * v * c * 2);
        for h in 0..heads {
            for col in 0..c {
                for dd in 0..d {
                    uk.extend_from_slice(at(h * (d + v) + dd, col));
                }
            }
            uv.extend_from_slice(&host[(h * (d + v) + d) * c * 2..(h * (d + v) + d + v) * c * 2]);
        }
        Ok((self.upload(&uk)?, self.upload(&uv)?))
    }

    pub fn layer(&self, cfg: &GlmDsaConfig, layer: usize) -> Result<GlmLayer<'a>> {
        let p = format!("model.layers.{layer}");
        let dense = layer < cfg.first_moe_layer;
        let full_indexer = cfg.indexers[layer] == GlmIndexer::Full;
        let mut ops: HashMap<&'static str, DeviceAllocation<'a>> = HashMap::new();
        // BF16 plus the FP8 copy and scale grid the decode programs read.
        let with_fp8 = |ops: &mut HashMap<&'static str, DeviceAllocation<'a>>, name: &'static str,
            parts: &[String]| -> Result<()> {
            let (bf16, fp8) = self.rows_fp8(parts, true)?;
            let (w8, scale) = fp8.context("FP8 operands")?;
            let (w8_name, scale_name) = fp8_operand_names(name);
            ops.insert(name, bf16);
            ops.insert(w8_name, w8);
            ops.insert(scale_name, scale);
            Ok(())
        };
        ops.insert("input_norm", self.one(&format!("{p}.input_layernorm.weight"))?);
        ops.insert("post_norm", self.one(&format!("{p}.post_attention_layernorm.weight"))?);
        with_fp8(&mut ops, "w_qkv_a", &[format!("{p}.self_attn.q_a_proj.weight"),
            format!("{p}.self_attn.kv_a_proj_with_mqa.weight")])?;
        ops.insert("q_a_norm", self.one(&format!("{p}.self_attn.q_a_layernorm.weight"))?);
        ops.insert("kv_a_norm", self.one(&format!("{p}.self_attn.kv_a_layernorm.weight"))?);
        with_fp8(&mut ops, "w_q_b", &[format!("{p}.self_attn.q_b_proj.weight")])?;
        let (uk, uv) = self.kv_b(cfg, &p)?;
        ops.insert("w_uk", uk);
        ops.insert("w_uv", uv);
        with_fp8(&mut ops, "w_o", &[format!("{p}.self_attn.o_proj.weight")])?;
        if full_indexer {
            with_fp8(&mut ops, "w_iq", &[format!("{p}.self_attn.indexer.wq_b.weight")])?;
            ops.insert("w_ik", self.rows(&[format!("{p}.self_attn.indexer.wk.weight"),
                format!("{p}.self_attn.indexer.weights_proj.weight")])?);
            ops.insert("k_norm_w", self.one(&format!("{p}.self_attn.indexer.k_norm.weight"))?);
            ops.insert("k_norm_b", self.one(&format!("{p}.self_attn.indexer.k_norm.bias"))?);
        }
        let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        with_fp8(&mut ops, "w_gate_up", &[format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")])?;
        with_fp8(&mut ops, "w_down", &[format!("{mlp}.down_proj.weight")])?;
        if !dense {
            ops.insert("gate", self.one(&format!("{p}.mlp.gate.weight"))?);
            ops.insert("gate.bias", self.one(&format!("{p}.mlp.gate.e_score_correction_bias"))?);
        }
        Ok(GlmLayer { dense, full_indexer, operands: ops })
    }

    /// Layers `0..layers` (all of them unless the caller stops early).
    pub fn model(&self, cfg: &GlmDsaConfig, layers: usize) -> Result<GlmWeights<'a>> {
        Ok(GlmWeights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            norm: self.one("model.norm.weight")?,
            head: self.one("lm_head.weight")?,
        })
    }
}
