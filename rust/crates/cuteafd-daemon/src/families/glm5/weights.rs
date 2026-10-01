//! GLM 5.x coordinator weights, packed for the exported glm_* programs.
//!
//! The checkpoint's FP8 projections stay FP8: E4M3 bytes (`{w}_fp8`) with
//! their FP32 128x128 scale grids (`{w}_scale`, `[ceil(N/128), K/128]`) are
//! the only copies of `w_qkv_a`, `w_q_b`, `w_iq`, `w_o`, `w_gate_up` and
//! `w_down` (decode programs: FP8 GEMV up to 16 rows, W8A16 GEMMs above, bitwise
//! the BF16 programs over the dequantized weights; prefill programs: W8A8 or
//! W8A16). Prefill reads `q_a|kv_a` (2624 rows, the last block partial) with
//! per-row scales K-block major (`w_qkv_a_kscale`, `[K/128, N]`). `kv_b` is
//! split per head on the host into E4M3 `w_uk_fp8 [N, 512, 192]` (key rows
//! transposed) and `w_uv_fp8 [N, 256, 512]` with one FP32 scale per weight
//! row and 64-wide K tile (`w_uk_scale [N, 512, 3]`, `w_uv_scale [N, 256, 8]`:
//! heads span 3.5 blocks of the grid). Row-concatenated operands (`w_qkv_a`,
//! `w_gate_up`) concatenate their grids (every part but the last is a whole
//! number of 128-row blocks). BF16 operands: norms, the router, the LM head and
//! `w_ik` (`wk` FP8 dequantized on the GPU with its scales, `weights_proj` BF16).
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::glm5::{GlmDsaConfig, GlmIndexer};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;

pub(crate) struct GlmLayer<'a> {
    pub dense: bool,
    pub full_indexer: bool,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

/// The `{w}_fp8` / `{w}_scale` operand names of an FP8 weight, and its
/// per-row K-block-major scales `{w}_kscale` (prefill `w_qkv_a` only).
pub(crate) fn fp8_operand_names(name: &str) -> (&'static str, &'static str, &'static str) {
    match name {
        "w_qkv_a" => ("w_qkv_a_fp8", "w_qkv_a_scale", "w_qkv_a_kscale"),
        "w_q_b" => ("w_q_b_fp8", "w_q_b_scale", ""),
        "w_iq" => ("w_iq_fp8", "w_iq_scale", ""),
        "w_o" => ("w_o_fp8", "w_o_scale", ""),
        "w_gate_up" => ("w_gate_up_fp8", "w_gate_up_scale", ""),
        "w_down" => ("w_down_fp8", "w_down_scale", ""),
        "w_uk" => ("w_uk_fp8", "w_uk_scale", ""),
        "w_uv" => ("w_uv_fp8", "w_uv_scale", ""),
        other => unreachable!("{other} has no FP8 operand"),
    }
}

impl GlmLayer<'_> {
    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    /// Device bytes of the layer's operands.
    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
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

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 4];
    for (dst, v) in out.chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    out
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

    /// The row-concatenation of 2-D `names` as one BF16 operand (FP8 parts
    /// dequantized on the GPU with their FP32 block scales).
    fn rows(&self, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum();
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
        let k_blocks = cols.div_ceil(128);
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

    /// Reads tensor `name`'s bytes into `out[at..]` (one positioned read into the
    /// operand's buffer: no second copy) and returns its byte length, dtype and shape.
    fn read_into(&self, name: &str, out: &mut [u8], at: usize) -> Result<(usize, DType, Vec<usize>)> {
        let tensor = self.catalog.tensor(name)?;
        let length = tensor.metadata.byte_length as usize;
        ensure!(at + length <= out.len(), "{name}: {length} bytes past the operand buffer");
        std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
            .read_exact_at(&mut out[at..at + length], tensor.metadata.byte_offset)
            .with_context(|| format!("reading {name}"))?;
        Ok((length, tensor.metadata.dtype.clone(), tensor.metadata.shape.clone()))
    }

    /// Reads the row-concatenation of the FP8 checkpoint weights `names` into
    /// this thread's staging buffer and hands `body` the E4M3 bytes `[N, K]`,
    /// the FP32 block grids `[ceil(N/128), K/128]` (every part but the last a
    /// whole number of 128-row blocks), N and K.
    fn with_fp8<T>(&self, names: &[String], body: impl FnOnce(&[u8], Vec<f32>, usize, usize) -> Result<T>)
        -> Result<T> {
        let total: u64 = names.iter().map(|n| self.catalog.tensor(n).map(|t| t.metadata.byte_length))
            .sum::<Result<u64>>()?;
        crate::shared::memory::staging::with_staging(total as usize, |values| {
            let mut grid = Vec::new();
            let (mut rows, mut cols, mut at) = (0usize, None, 0usize);
            for (i, name) in names.iter().enumerate() {
                let (length, dtype, shape) = self.read_into(name, values, at)?;
                at += length;
                ensure!(dtype == DType::F8E4M3 && shape.len() == 2,
                    "{name}: the glm programs take this weight as an FP8 checkpoint tensor, found {dtype:?} {shape:?}");
                ensure!(cols.is_none_or(|c| c == shape[1]), "{names:?} do not share columns");
                ensure!(i + 1 == names.len() || shape[0] % 128 == 0,
                    "{names:?}: only the last concatenated weight may end inside a 128-row block");
                cols = Some(shape[1]);
                let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                ensure!(scale_dtype == DType::F32 && scale_shape == [shape[0].div_ceil(128), shape[1].div_ceil(128)],
                    "{name}: expected FP32 128x128 block scales, found {scale_dtype:?} {scale_shape:?}");
                grid.extend(scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
                rows += shape[0];
            }
            body(values, grid, rows, cols.context("no FP8 rows")?)
        })
    }

    /// E4M3 `[N, K]` and the FP32 block grid of `names` on the device.
    fn fp8(&self, names: &[String]) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        self.with_fp8(names, |values, grid, _, _| Ok((self.upload(values)?, self.upload(&f32_bytes(&grid))?)))
    }

    /// `w_uk [N, 512, D]` / `w_uv [N, V, 512]` E4M3 slices of the FP8 `kv_b`
    /// with one scale per weight row and 64-wide K tile (`[N, 512, D/64]`,
    /// `[N, V, 512/64]`): `w_uk[h,c,d] = kv_b[h*(D+V)+d, c]`,
    /// `w_uv[h,v,c] = kv_b[h*(D+V)+D+v, c]`.
    #[allow(clippy::type_complexity)]
    fn kv_b(&self, cfg: &GlmDsaConfig, prefix: &str)
        -> Result<((DeviceAllocation<'a>, DeviceAllocation<'a>), (DeviceAllocation<'a>, DeviceAllocation<'a>))> {
        let (heads, d, v, c) = (cfg.heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        self.with_fp8(&[format!("{prefix}.self_attn.kv_b_proj.weight")], |kv_b, grid, rows, cols| {
        ensure!(rows == heads * (d + v) && cols == c && d % 64 == 0 && c % 128 == 0,
            "{prefix}: kv_b_proj [{rows}, {cols}] does not match {heads} heads x ({d} + {v}) x {c}");
        let kb = c / 128;
        let scale = |row: usize, col: usize| grid[(row / 128) * kb + col / 128];
        // Each 64-row K tile of a key slice lies in one 128-row block (448 = 3.5 x 128).
        ensure!((d + v) % 64 == 0, "{prefix}: head rows must be whole 64-row tiles");
        let (mut uk, mut uk_s) = (vec![0u8; heads * c * d], vec![0f32; heads * c * d / 64]);
        let (mut uv, mut uv_s) = (vec![0u8; heads * v * c], vec![0f32; heads * v * c / 64]);
        // One head per task: key rows transposed in 64 x 64 tiles, value rows copied.
        std::thread::scope(|scope| {
            let parts = uk.chunks_mut(c * d).zip(uk_s.chunks_mut(c * d / 64)).zip(uv.chunks_mut(v * c))
                .zip(uv_s.chunks_mut(v * c / 64)).enumerate();
            let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).clamp(1, 16);
            let mut groups: Vec<Vec<_>> = (0..threads).map(|_| Vec::new()).collect();
            for (h, part) in parts {
                groups[h % threads].push((h, part));
            }
            for group in groups {
                let (kv_b, scale) = (&kv_b, &scale);
                scope.spawn(move || {
                    for (h, (((uk, uk_s), uv), uv_s)) in group {
                        let base = h * (d + v);
                        for c0 in (0..c).step_by(64) {
                            for d0 in (0..d).step_by(64) {
                                for dd in d0..d0 + 64 {
                                    let src = &kv_b[(base + dd) * c..][..c];
                                    for col in c0..c0 + 64 {
                                        uk[col * d + dd] = src[col];
                                    }
                                }
                            }
                        }
                        for col in 0..c {
                            for t in 0..d / 64 {
                                uk_s[col * (d / 64) + t] = scale(base + t * 64, col);
                            }
                        }
                        uv.copy_from_slice(&kv_b[(base + d) * c..(base + d + v) * c]);
                        for r in 0..v {
                            for t in 0..c / 64 {
                                uv_s[r * (c / 64) + t] = scale(base + d + r, t * 64);
                            }
                        }
                    }
                });
            }
        });
        Ok(((self.upload(&uk)?, self.upload(&f32_bytes(&uk_s))?), (self.upload(&uv)?, self.upload(&f32_bytes(&uv_s))?)))
        })
    }

    fn one(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let (bytes, dtype, shape) = self.raw(name)?;
        if shape.len() == 2 && dtype == DType::F8E4M3 {
            return self.rows(&[name.to_string()]);
        }
        self.upload(&bytes)
    }

    pub fn layer(&self, cfg: &GlmDsaConfig, layer: usize) -> Result<GlmLayer<'a>> {
        let p = format!("model.layers.{layer}");
        let dense = layer < cfg.first_moe_layer;
        let full_indexer = cfg.indexers[layer] == GlmIndexer::Full;
        let mut ops: HashMap<&'static str, DeviceAllocation<'a>> = HashMap::new();
        // The checkpoint's E4M3 bytes and FP32 scale grid: the only copy.
        let with_fp8 = |ops: &mut HashMap<&'static str, DeviceAllocation<'a>>, name: &'static str,
            parts: &[String]| -> Result<()> {
            let (w8, scale) = self.fp8(parts)?;
            let (w8_name, scale_name, _) = fp8_operand_names(name);
            ops.insert(w8_name, w8);
            ops.insert(scale_name, scale);
            Ok(())
        };
        ops.insert("input_norm", self.one(&format!("{p}.input_layernorm.weight"))?);
        ops.insert("post_norm", self.one(&format!("{p}.post_attention_layernorm.weight"))?);
        let qkv_a = [format!("{p}.self_attn.q_a_proj.weight"), format!("{p}.self_attn.kv_a_proj_with_mqa.weight")];
        let (values, scale, kscale) = self.with_fp8(&qkv_a, |values, grid, rows, cols| {
            // Prefill: per-row scales K-block major (the block-FP8 GEMM wants whole 128-row blocks).
            let kb = cols.div_ceil(128);
            let grid_ref = &grid;
            let kscale: Vec<f32> = (0..kb).flat_map(|b| (0..rows).map(move |r| grid_ref[(r / 128) * kb + b])).collect();
            Ok((self.upload(values)?, self.upload(&f32_bytes(&grid))?, self.upload(&f32_bytes(&kscale))?))
        })?;
        ops.insert("w_qkv_a_fp8", values);
        ops.insert("w_qkv_a_scale", scale);
        ops.insert("w_qkv_a_kscale", kscale);
        ops.insert("q_a_norm", self.one(&format!("{p}.self_attn.q_a_layernorm.weight"))?);
        ops.insert("kv_a_norm", self.one(&format!("{p}.self_attn.kv_a_layernorm.weight"))?);
        with_fp8(&mut ops, "w_q_b", &[format!("{p}.self_attn.q_b_proj.weight")])?;
        let ((uk, uk_scale), (uv, uv_scale)) = self.kv_b(cfg, &p)?;
        ops.insert("w_uk_fp8", uk);
        ops.insert("w_uk_scale", uk_scale);
        ops.insert("w_uv_fp8", uv);
        ops.insert("w_uv_scale", uv_scale);
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
        let weights = GlmWeights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            norm: self.one("model.norm.weight")?,
            head: self.one("lm_head.weight")?,
        };
        crate::shared::memory::staging::release_staging();
        Ok(weights)
    }
}
