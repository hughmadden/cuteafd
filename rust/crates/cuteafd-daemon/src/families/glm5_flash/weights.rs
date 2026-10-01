//! GLM 5.3 Flash coordinator weights, packed for the exported glmf_* programs.
//!
//! The MLA (`q_a|kv_a`, `q_b`, `o_proj`), dense and shared-expert projections
//! are FP8 only: the official FP8 release's E4M3 bytes and FP32 128x128 block
//! scales (`--fp8-snapshot`), else 128x128 blocks quantized from the BF16
//! checkpoint at load. Every other matrix operand is BF16 (the EXL3
//! publications store the dense tensors in BF16, equal to the official BF16
//! release; FP8 checkpoint tensors are dequantized on the GPU with their FP32
//! 128x128 block scales), plus the optional per-row FP8 copies of the KDA
//! projections and the LM head.
//! Packing (see the glmf program docstrings): KDA `w_in = [q; k; v; f_a; g_a;
//! b]`, `w_fg = [f_b; g_b]`, `conv_w` FP32 `[3D, 4]`; MLA `w_qkv_a = [q_a;
//! kv_a]`, `kv_b` split per head into `w_uk [N, 512, 256]` (transposed key
//! rows) and `w_uv [N, 256, 512]`; mHC `fn` widened to FP32; dense and shared
//! `w_gate_up = [gate; up]`. A ModelOpt NVFP4 dense MLP (nvidia/GLM-5.3-Flash-
//! NVFP4, layers 0-2) stays NVFP4: `nvfp4_w{1,3,2}` packed E2M1 and
//! `nvfp4_s{1,3,2}` its E4M3 scales then FP32 weight_scale_2 and input_scale,
//! the one-expert layout of the `fp8-glmfdense-nvfp4` package.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
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
    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    /// An FP8 operand, or `fallback`'s pointer when the layer has none (the
    /// program then runs with `fp8_rows` 0 and never reads it).
    pub fn ptr_or(&self, operand: &str, fallback: &str) -> Result<*mut c_void> {
        match self.operands.get(operand) {
            Some(allocation) => Ok(allocation.buffer.ptr),
            None => self.ptr(fallback),
        }
    }

    pub fn has(&self, operand: &str) -> bool {
        self.operands.contains_key(operand)
    }

    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct GlmfWeights<'a> {
    pub layers: Vec<GlmfLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    pub head: DeviceAllocation<'a>,
    /// E4M3 LM head with per-row x 128-K scales, for decode rows (--fp8-head).
    pub head_fp8: Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>,
}

pub(crate) struct GlmfLoader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    pub stream: *mut c_void,
    /// The official FP8 release: MLA/dense/shared weights (their only copies,
    /// 128x128 blocks) come from it when given, else are quantized from BF16.
    pub fp8_source: Option<&'a Checkpoint>,
    pub kda_fp8: super::fp8::KdaFp8,
    pub fp8_head: bool,
    /// Numerics gate only: KDA projections rounded through NVFP4 (Some(search)) and kept in BF16.
    pub kda_nvfp4: Option<bool>,
    /// Scale rule of copies quantized from BF16.
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
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

    /// A ModelOpt NVFP4 dense MLP at `mlp` as the one-expert fp8_moe operands
    /// (see the module docstring); false when the checkpoint stores it otherwise.
    fn nvfp4_dense(&self, cfg: &GlmNextConfig, mlp: &str, ops: &mut HashMap<&'static str, DeviceAllocation<'a>>)
        -> Result<bool> {
        if self.tensor(&format!("{mlp}.gate_proj.weight"))?.meta.dtype != DType::U8 {
            return Ok(false);
        }
        let (h, i) = (cfg.hidden, cfg.dense_intermediate);
        for (proj, w_key, s_key, rows, cols) in [("gate_proj", "nvfp4_w1", "nvfp4_s1", i, h),
            ("up_proj", "nvfp4_w3", "nvfp4_s3", i, h), ("down_proj", "nvfp4_w2", "nvfp4_s2", h, i)] {
            let name = format!("{mlp}.{proj}");
            let (weight, dtype, shape) = self.raw(&format!("{name}.weight"))?;
            ensure!(dtype == DType::U8 && shape == [rows, cols / 2], "{name}.weight: expected packed E2M1 U8 [{rows}, {}], \
                found {dtype:?} {shape:?}", cols / 2);
            let (mut scales, dtype, shape) = self.raw(&format!("{name}.weight_scale"))?;
            ensure!(dtype == DType::F8E4M3 && shape == [rows, cols / 16], "{name}.weight_scale: expected E4M3 \
                [{rows}, {}], found {dtype:?} {shape:?}", cols / 16);
            for scalar in ["weight_scale_2", "input_scale"] {
                let (bytes, dtype, _) = self.raw(&format!("{name}.{scalar}"))?;
                ensure!(dtype == DType::F32 && bytes.len() == 4, "{name}.{scalar}: expected one FP32 value");
                scales.extend_from_slice(&bytes);
            }
            ops.insert(w_key, self.upload(&weight)?);
            ops.insert(s_key, self.upload(&scales)?);
        }
        Ok(true)
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

    /// The row-concatenation of 2-D `names` as E4M3 bytes and FP32 scales in
    /// `layout`: the FP8 source checkpoint's own blocks when it stores the
    /// tensors as FP8 (block layout), else quantized from BF16.
    fn fp8(&self, names: &[String], layout: super::fp8::Layout) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        if layout == super::fp8::Layout::Block {
            if let Some(copy) = self.fp8_blocks(names)? {
                return Ok(copy);
            }
        }
        let (values, scales, _) = self.fp8_host(names, layout)?;
        Ok((self.upload(&values)?, self.upload(&scales)?))
    }

    /// The FP8 source's own E4M3 blocks and FP32 128x128 grids of `names`
    /// (row-concatenated) on the device, read through this thread's staging
    /// buffer; None when a part is not an FP8 tensor there (quantized instead).
    fn fp8_blocks(&self, names: &[String]) -> Result<Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let checkpoint = self.fp8_source.unwrap_or(self.checkpoint);
        let find = |name: &str| -> Result<&'a CheckpointTensor> {
            let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("FP8 checkpoint has no tensor {name}"))?;
            Ok(&checkpoint.tensors[at])
        };
        let tensors = names.iter().map(|n| find(n)).collect::<Result<Vec<_>>>()?;
        if tensors.iter().any(|t| t.meta.dtype != DType::F8E4M3 || t.meta.shape.len() != 2) {
            return Ok(None);
        }
        let total: usize = tensors.iter().map(|t| t.meta.byte_length as usize).sum();
        crate::shared::memory::staging::with_staging(total, |values| {
            let (mut grid, mut at) = (Vec::new(), 0usize);
            for (i, (name, tensor)) in names.iter().zip(&tensors).enumerate() {
                let shape = &tensor.meta.shape;
                ensure!(shape[1] == tensors[0].meta.shape[1] && (i + 1 == names.len() || shape[0] % 128 == 0),
                    "{names:?}: FP8 rows must share columns and fill whole 128-row blocks but the last");
                let length = tensor.meta.byte_length as usize;
                std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?
                    .read_exact_at(&mut values[at..at + length], tensor.meta.byte_offset)
                    .with_context(|| format!("reading {name}"))?;
                at += length;
                let scale = find(&format!("{name}_scale_inv"))?;
                ensure!(scale.meta.dtype == DType::F32 && scale.meta.shape == [shape[0] / 128, shape[1].div_ceil(128)],
                    "{name}: expected FP32 128x128 block scales, found {:?} {:?}", scale.meta.dtype, scale.meta.shape);
                let mut bytes = vec![0u8; scale.meta.byte_length as usize];
                std::fs::File::open(checkpoint.snapshot.join(&scale.shard))?
                    .read_exact_at(&mut bytes, scale.meta.byte_offset)?;
                grid.extend_from_slice(&bytes);
            }
            Ok(Some((self.upload(values)?, self.upload(&grid)?)))
        })
    }

    /// Per-row FP8 copy of `names` (Row128) plus its scales K-block major
    /// (`[K/128, N]`, the prefill GEMMs' layout): (values, row scales, K-major scales).
    fn fp8_rows_kmajor(&self, names: &[String])
        -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let (values, scales, cols) = self.fp8_host(names, super::fp8::Layout::Row128)?;
        let (kb, n) = (cols / 128, values.len() / cols);
        let mut kmajor = vec![0u8; scales.len()];
        for row in 0..n {
            for b in 0..kb {
                kmajor[(b * n + row) * 4..][..4].copy_from_slice(&scales[(row * kb + b) * 4..][..4]);
            }
        }
        Ok((self.upload(&values)?, self.upload(&scales)?, self.upload(&kmajor)?))
    }

    /// Host bytes of [`Self::fp8`]: E4M3 values, FP32 scales, and the column count.
    fn fp8_host(&self, names: &[String], layout: super::fp8::Layout) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        use super::fp8::{quantize, Layout};
        let source = |name: &str| -> Result<(Vec<u8>, DType, Vec<usize>)> {
            match self.fp8_source {
                Some(checkpoint) => {
                    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                        .map_err(|_| anyhow::anyhow!("FP8 checkpoint has no tensor {name}"))?;
                    let tensor = &checkpoint.tensors[at];
                    let mut bytes = vec![0u8; tensor.meta.byte_length as usize];
                    std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?
                        .read_exact_at(&mut bytes, tensor.meta.byte_offset)?;
                    Ok((bytes, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
                }
                None => self.raw(name),
            }
        };
        let (mut values, mut scales) = (Vec::new(), Vec::<u8>::new());
        let mut cols = None;
        for name in names {
            let (bytes, dtype, shape) = source(name)?;
            ensure!(shape.len() == 2 && cols.is_none_or(|c| c == shape[1]), "{name}: FP8 rows must share columns");
            cols = Some(shape[1]);
            match dtype {
                DType::F8E4M3 if layout == Layout::Block => {
                    let (scale, scale_dtype, scale_shape) = source(&format!("{name}_scale_inv"))?;
                    ensure!(scale_dtype == DType::F32 && shape[0] % 128 == 0
                        && scale_shape == [shape[0] / 128, shape[1].div_ceil(128)],
                        "{name}: expected FP32 128x128 block scales, found {scale_dtype:?} {scale_shape:?}");
                    values.extend_from_slice(&bytes);
                    scales.extend_from_slice(&scale);
                }
                DType::Bf16 => {
                    let (q, s) = quantize(&bytes, shape[0], shape[1], layout, self.fp8_scales);
                    values.extend_from_slice(&q);
                    scales.extend(s.iter().flat_map(|v| v.to_le_bytes()));
                }
                other => anyhow::bail!("{name}: cannot make an FP8 {layout:?} copy of {other:?}"),
            }
        }
        Ok((values, scales, cols.context("no FP8 rows")?))
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
                let w_in = ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj"].map(|n| a(&format!("{n}.weight")));
                let kda_layout = match self.kda_fp8 {
                    super::fp8::KdaFp8::Off => None,
                    super::fp8::KdaFp8::Channel => Some(super::fp8::Layout::Channel),
                    super::fp8::KdaFp8::Row128 => Some(super::fp8::Layout::Row128),
                };
                if kda_layout == Some(super::fp8::Layout::Row128) {
                    // Per-row copies also serve the block-FP8 prefill GEMMs (K-major scales).
                    let (q, s, k) = self.fp8_rows_kmajor(&w_in)?;
                    ops.insert("w_in_fp8", q);
                    ops.insert("w_in_scale", s);
                    ops.insert("w_in_kscale", k);
                    let (q, s, k) = self.fp8_rows_kmajor(&[a("o_proj.weight")])?;
                    ops.insert("w_o_fp8", q);
                    ops.insert("w_o_scale", s);
                    ops.insert("w_o_kscale", k);
                } else if let Some(layout) = kda_layout {
                    let (q, s) = self.fp8(&w_in, layout)?;
                    ops.insert("w_in_fp8", q);
                    ops.insert("w_in_scale", s);
                    let (q, s) = self.fp8(&[a("o_proj.weight")], layout)?;
                    ops.insert("w_o_fp8", q);
                    ops.insert("w_o_scale", s);
                }
                if let Some(search) = self.kda_nvfp4 {
                    let mut bytes = Vec::new();
                    for name in w_in.iter().chain([a("o_proj.weight")].iter()) {
                        let (raw, dtype, shape) = self.raw(name)?;
                        ensure!(dtype == DType::Bf16, "{name}: NVFP4 gate needs BF16");
                        let rounded = super::fp8::nvfp4_roundtrip(&raw, shape[0], shape[1], search);
                        if name.ends_with("o_proj.weight") {
                            ops.insert("w_o_nvfp4", self.upload(&rounded)?);
                        } else {
                            bytes.extend(rounded);
                        }
                    }
                    ops.insert("w_in", self.upload(&bytes)?);
                } else {
                    ops.insert("w_in", self.rows(&w_in)?);
                }
                ops.insert("w_fg", self.rows(&[a("f_b_proj.weight"), a("g_b_proj.weight")])?);
                // [3D, 1, 4] each -> FP32 [3D, 4].
                ops.insert("conv_w", self.f32(&["q", "k", "v"].map(|n| a(&format!("{n}_conv1d.weight"))))?);
                ops.insert("a_log", self.f32(&[a("A_log")])?);
                ops.insert("dt_bias", self.f32(&[a("dt_bias")])?);
                ops.insert("o_norm", self.one(&a("o_norm.weight"))?);
                let w_o = match ops.remove("w_o_nvfp4") {
                    Some(rounded) => rounded,
                    None => self.one(&a("o_proj.weight"))?,
                };
                ops.insert("w_o", w_o);
            }
            GlmNextAttention::Mla => {
                ops.insert("q_a_norm", self.one(&a("q_a_layernorm.weight"))?);
                ops.insert("kv_a_norm", self.one(&a("kv_a_layernorm.weight"))?);
                let (uk, uv) = self.absorbed(cfg, &a("kv_b_proj.weight"))?;
                ops.insert("w_uk", uk);
                ops.insert("w_uv", uv);
                let block = super::fp8::Layout::Block;
                let (q, s) = self.fp8(&[a("q_a_proj.weight"), a("kv_a_proj_with_mqa.weight")], block)?;
                ops.insert("w_qkv_a_fp8", q);
                ops.insert("w_qkv_a_scale", s);
                let (q, s) = self.fp8(&[a("q_b_proj.weight")], block)?;
                ops.insert("w_q_b_fp8", q);
                ops.insert("w_q_b_scale", s);
                let (q, s) = self.fp8(&[a("o_proj.weight")], block)?;
                ops.insert("w_o_fp8", q);
                ops.insert("w_o_scale", s);
                let i = |name: &str| a(&format!("indexer.{name}"));
                ops.insert("w_iq", self.one(&i("wq_b.weight"))?);
                ops.insert("w_ik", self.rows(&[i("wk.weight"), i("weights_proj.weight"),
                    i("index_kpool_compress_gate")])?);
                ops.insert("k_norm_w", self.one(&i("k_norm.weight"))?);
                ops.insert("k_norm_b", self.one(&i("k_norm.bias"))?);
                ops.insert("ape", self.one(&i("index_kpool_compress_ape"))?);
            }
        }
        let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        if !(dense && self.nvfp4_dense(cfg, &mlp, &mut ops)?) {
            let block = super::fp8::Layout::Block;
            let (q, s) = self.fp8(&[format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")], block)?;
            ops.insert("w_gate_up_fp8", q);
            ops.insert("w_gate_up_scale", s);
            let (q, s) = self.fp8(&[format!("{mlp}.down_proj.weight")], block)?;
            ops.insert("w_down_fp8", q);
            ops.insert("w_down_scale", s);
        }
        if !dense {
            ops.insert("gate", self.one(&format!("{p}.mlp.gate.weight"))?);
            ops.insert("gate.bias", self.f32(&[format!("{p}.mlp.gate.e_score_correction_bias")])?);
        }
        Ok(GlmfLayer { attention, dense, operands: ops })
    }

    /// Layers `0..layers` (all of them unless the caller stops early).
    pub fn model(&self, cfg: &GlmNextConfig, layers: usize) -> Result<GlmfWeights<'a>> {
        let weights = GlmfWeights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            norm: self.one(&format!("{PREFIX}norm.weight"))?,
            head: self.one("lm_head.weight")?,
            head_fp8: if self.fp8_head {
                Some(self.fp8(&["lm_head.weight".to_string()], super::fp8::Layout::Row128)?)
            } else {
                None
            },
        };
        crate::shared::memory::staging::release_staging();
        Ok(weights)
    }
}
