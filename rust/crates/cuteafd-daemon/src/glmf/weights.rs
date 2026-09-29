//! GLM 5.3 Flash coordinator weights, packed for the exported glmf_* programs.
//!
//! Every matrix operand is BF16: the EXL3 publications store the dense
//! tensors in BF16 (equal to the official BF16 release), FP8 checkpoint
//! tensors are dequantized on the GPU with their FP32 128x128 block scales.
//! Packing (see the glmf program docstrings): KDA `w_in = [q; k; v; f_a; g_a;
//! b]`, `w_fg = [f_b; g_b]`, `conv_w` FP32 `[3D, 4]`; MLA `w_qkv_a = [q_a;
//! kv_a]`, `kv_b` split per head into `w_uk [N, 512, 256]` (transposed key
//! rows) and `w_uv [N, 256, 512]`; mHC `fn` widened to FP32; dense and shared
//! `w_gate_up = [gate; up]`.
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::glm_next::{GlmNextAttention, GlmNextConfig};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;

pub(crate) const PREFIX: &str = "model.language_model.";

pub(crate) struct GlmfLayer<'a> {
    pub attention: GlmNextAttention,
    pub dense: bool,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

impl GlmfLayer<'_> {
    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct GlmfWeights<'a> {
    pub layers: Vec<GlmfLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    pub head: DeviceAllocation<'a>,
}

pub(crate) struct GlmfLoader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    pub stream: *mut c_void,
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl<'a> GlmfLoader<'a> {
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

    /// The row-concatenation of 2-D `names` as one BF16 operand (FP8 blocks dequantized).
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
                    ensure!(scale_shape == [shape[0].div_ceil(128), cols.div_ceil(128)],
                        "{name}: unexpected scale grid {scale_shape:?}");
                    let (w, s) = (self.upload(bytes)?, self.upload(&scale)?);
                    // SAFETY: the FP8 weight, its scales and the destination rows are live;
                    // the stream drains before `w`/`s` drop.
                    unsafe {
                        self.library.fp8_block_dequant(w.buffer.ptr, s.buffer.ptr, dest(0, shape[0]).ptr, shape[0],
                            cols, self.stream)?;
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

    /// A BF16 (or FP32) tensor widened to FP32.
    fn f32(&self, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let mut values = Vec::new();
        for name in names {
            let (bytes, dtype, _) = self.raw(name)?;
            match dtype {
                DType::Bf16 => values.extend(bf16_to_f32(&bytes)),
                DType::F32 => values.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))),
                other => anyhow::bail!("{name}: expected BF16 or FP32, found {other:?}"),
            }
        }
        self.upload(&f32_bytes(&values))
    }

    /// `kv_b_proj [N*(nope+v), 512]` -> `w_uk [N, 512, nope]` (key rows transposed per head)
    /// and `w_uv [N, v, 512]`.
    fn absorbed(&self, cfg: &GlmNextConfig, name: &str) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let (bytes, dtype, shape) = self.raw(name)?;
        let (n, nope, v, lat) = (cfg.heads, cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        ensure!(dtype == DType::Bf16 && shape == [n * (nope + v), lat], "{name}: expected BF16 [{}, {lat}]",
            n * (nope + v));
        let at = |row: usize, col: usize| -> [u8; 2] {
            let i = (row * lat + col) * 2;
            [bytes[i], bytes[i + 1]]
        };
        let mut uk = Vec::with_capacity(n * lat * nope * 2);
        let mut uv = Vec::with_capacity(n * v * lat * 2);
        for h in 0..n {
            let base = h * (nope + v);
            for c in 0..lat {
                for d in 0..nope {
                    uk.extend_from_slice(&at(base + d, c));
                }
            }
            for r in 0..v {
                uv.extend_from_slice(&bytes[(base + nope + r) * lat * 2..(base + nope + r + 1) * lat * 2]);
            }
        }
        Ok((self.upload(&uk)?, self.upload(&uv)?))
    }

    pub fn layer(&self, cfg: &GlmNextConfig, layer: usize) -> Result<GlmfLayer<'a>> {
        let p = format!("{PREFIX}layers.{layer}");
        let attention = cfg.attention[layer];
        let dense = cfg.dense[layer];
        let mut ops: HashMap<&'static str, DeviceAllocation<'a>> = HashMap::new();
        for site in ["attn", "ffn"] {
            let (fn_, scale, base): (&'static str, &'static str, &'static str) = if site == "attn" {
                ("attn.fn", "attn.scale", "attn.base")
            } else {
                ("ffn.fn", "ffn.scale", "ffn.base")
            };
            ops.insert(fn_, self.f32(&[format!("{p}.hc_{site}_fn")])?);
            ops.insert(scale, self.f32(&[format!("{p}.hc_{site}_scale")])?);
            ops.insert(base, self.f32(&[format!("{p}.hc_{site}_base")])?);
        }
        ops.insert("input_norm", self.one(&format!("{p}.input_layernorm.weight"))?);
        ops.insert("post_norm", self.one(&format!("{p}.post_attention_layernorm.weight"))?);
        let a = |name: &str| format!("{p}.self_attn.{name}");
        match attention {
            GlmNextAttention::Kda => {
                ops.insert("w_in", self.rows(&["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj"]
                    .map(|n| a(&format!("{n}.weight"))))?);
                ops.insert("w_fg", self.rows(&[a("f_b_proj.weight"), a("g_b_proj.weight")])?);
                // [3D, 1, 4] each -> FP32 [3D, 4].
                ops.insert("conv_w", self.f32(&["q", "k", "v"].map(|n| a(&format!("{n}_conv1d.weight"))))?);
                ops.insert("a_log", self.f32(&[a("A_log")])?);
                ops.insert("dt_bias", self.f32(&[a("dt_bias")])?);
                ops.insert("o_norm", self.one(&a("o_norm.weight"))?);
                ops.insert("w_o", self.one(&a("o_proj.weight"))?);
            }
            GlmNextAttention::Mla => {
                ops.insert("w_qkv_a", self.rows(&[a("q_a_proj.weight"), a("kv_a_proj_with_mqa.weight")])?);
                ops.insert("q_a_norm", self.one(&a("q_a_layernorm.weight"))?);
                ops.insert("kv_a_norm", self.one(&a("kv_a_layernorm.weight"))?);
                ops.insert("w_q_b", self.one(&a("q_b_proj.weight"))?);
                let (uk, uv) = self.absorbed(cfg, &a("kv_b_proj.weight"))?;
                ops.insert("w_uk", uk);
                ops.insert("w_uv", uv);
                ops.insert("w_o", self.one(&a("o_proj.weight"))?);
                let i = |name: &str| a(&format!("indexer.{name}"));
                ops.insert("w_iq", self.one(&i("wq_b.weight"))?);
                ops.insert("w_ik", self.rows(&[i("wk.weight"), i("weights_proj.weight"),
                    i("index_kpool_compress_gate")])?);
                ops.insert("k_norm_w", self.one(&i("k_norm.weight"))?);
                ops.insert("k_norm_b", self.one(&i("k_norm.bias"))?);
                ops.insert("ape", self.one(&i("index_kpool_compress_ape"))?);
            }
        }
        if dense {
            ops.insert("w_gate_up", self.rows(&[format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")])?);
            ops.insert("w_down", self.one(&format!("{p}.mlp.down_proj.weight"))?);
        } else {
            ops.insert("gate", self.one(&format!("{p}.mlp.gate.weight"))?);
            ops.insert("gate.bias", self.f32(&[format!("{p}.mlp.gate.e_score_correction_bias")])?);
            ops.insert("w_gate_up", self.rows(&[format!("{p}.mlp.shared_experts.gate_proj.weight"),
                format!("{p}.mlp.shared_experts.up_proj.weight")])?);
            ops.insert("w_down", self.one(&format!("{p}.mlp.shared_experts.down_proj.weight"))?);
        }
        Ok(GlmfLayer { attention, dense, operands: ops })
    }

    /// Layers `0..layers` (all of them unless the caller stops early).
    pub fn model(&self, cfg: &GlmNextConfig, layers: usize) -> Result<GlmfWeights<'a>> {
        Ok(GlmfWeights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            norm: self.one(&format!("{PREFIX}norm.weight"))?,
            head: self.one("lm_head.weight")?,
        })
    }
}
