//! GLM 5.x coordinator weights, packed for the exported glm_* programs.
//!
//! Every operand is BF16: FP8 checkpoint tensors are dequantized on the GPU
//! with their FP32 128x128 block scales (re-quantizing to the power-of-two
//! scales b12x's FP8 linears take costs 0.025 nats), BF16 tensors are copied.
//! Row-concatenated operands (`w_qkv_a`, `w_ik`, `w_gate_up`) are written in
//! place; `kv_b` is split per head into `w_uk` (transposed) and `w_uv` on
//! the host.
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

impl GlmLayer<'_> {
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
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum();
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
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
                    let (w, s) = (self.upload(bytes)?, self.upload(&scale)?);
                    // SAFETY: fp8 [shape], its scale grid and the destination rows
                    // are live; the stream is drained before the temporaries drop.
                    unsafe {
                        self.library.fp8_block_dequant(w.buffer.ptr, s.buffer.ptr, dest.ptr, shape[0], cols, self.stream)?;
                        self.library.cuda_stream_synchronize(self.stream)?;
                    }
                }
                other => anyhow::bail!("{name}: unsupported coordinator dtype {other:?}"),
            }
            row += shape[0];
        }
        Ok(out)
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
        ops.insert("input_norm", self.one(&format!("{p}.input_layernorm.weight"))?);
        ops.insert("post_norm", self.one(&format!("{p}.post_attention_layernorm.weight"))?);
        ops.insert("w_qkv_a", self.rows(&[format!("{p}.self_attn.q_a_proj.weight"),
            format!("{p}.self_attn.kv_a_proj_with_mqa.weight")])?);
        ops.insert("q_a_norm", self.one(&format!("{p}.self_attn.q_a_layernorm.weight"))?);
        ops.insert("kv_a_norm", self.one(&format!("{p}.self_attn.kv_a_layernorm.weight"))?);
        ops.insert("w_q_b", self.one(&format!("{p}.self_attn.q_b_proj.weight"))?);
        let (uk, uv) = self.kv_b(cfg, &p)?;
        ops.insert("w_uk", uk);
        ops.insert("w_uv", uv);
        ops.insert("w_o", self.one(&format!("{p}.self_attn.o_proj.weight"))?);
        if full_indexer {
            ops.insert("w_iq", self.one(&format!("{p}.self_attn.indexer.wq_b.weight"))?);
            ops.insert("w_ik", self.rows(&[format!("{p}.self_attn.indexer.wk.weight"),
                format!("{p}.self_attn.indexer.weights_proj.weight")])?);
            ops.insert("k_norm_w", self.one(&format!("{p}.self_attn.indexer.k_norm.weight"))?);
            ops.insert("k_norm_b", self.one(&format!("{p}.self_attn.indexer.k_norm.bias"))?);
        }
        let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        ops.insert("w_gate_up", self.rows(&[format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")])?);
        ops.insert("w_down", self.one(&format!("{mlp}.down_proj.weight"))?);
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
