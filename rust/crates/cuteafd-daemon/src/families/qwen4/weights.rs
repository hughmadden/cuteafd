//! Qwen 3.8 Flash Next coordinator weights, packed for the exported qwen4_* programs.
//!
//! Every Qwen checkpoint (BF16, FP8, NVFP4, EXL3) stores the coordinator
//! tensors in BF16. Packing (see the qwen4 program docstrings): each
//! hyper-connection site `w_di = [input_mix_weight_down; block_inject_weight]`
//! (the model's final mixer has no injection), GDN `w_in = [in_proj_qkv;
//! in_proj_z; in_proj_b; in_proj_a]` with FP32 `conv_w [10240, 4]`, `a_log`
//! and `dt_bias`, full attention `w_in = [q_proj; k_proj; v_proj;
//! indexer.index_qk_proj]`, the shared expert `w_gate_up = [gate_proj;
//! up_proj; shared_expert_gate; 15 zero rows]`, PLE `w_kv = [key_proj;
//! value_proj]` and FP32 `conv_w [10240, 4]`.
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use cuteafd_loader::families::qwen4::{Qwen4Attention, Qwen4Config};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;

pub(crate) const PREFIX: &str = "model.language_model.";
/// Rows of the shared expert's packed gate/up/gate-logit operand.
pub(crate) const SHARED_ROWS: usize = 1296;

pub(crate) struct Qwen4Layer<'a> {
    pub attention: Qwen4Attention,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

impl Qwen4Layer<'_> {
    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    pub fn has(&self, operand: &str) -> bool {
        self.operands.contains_key(operand)
    }

    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct Qwen4Weights<'a> {
    pub layers: Vec<Qwen4Layer<'a>>,
    /// The final hyper-connection mixer: norm, w_down, w_up.
    pub mixer: [DeviceAllocation<'a>; 3],
    pub head: DeviceAllocation<'a>,
    /// The native MTP layer (`mtp.*`), when loaded.
    pub mtp: Option<MtpWeights<'a>>,
}

/// Qwen's MTP drafter (vLLM `Qwen4ExpMultiTokenPredictor`): one full-attention
/// decoder layer (`mtp.layers.0`, no PLE) fed by the `residual_linear_shared`
/// feedback of the target's pre-mixer streams and the next token's embedding
/// (shared `embed_tokens`), its own stream mixer, and the shared `lm_head`.
pub(crate) struct MtpWeights<'a> {
    pub layer: Qwen4Layer<'a>,
    /// `pre_fc_norm_hidden` [4H], `pre_fc_norm_embedding` [H], `fc_hidden`, `fc_embedding` [H, H].
    pub norm_hidden: DeviceAllocation<'a>,
    pub norm_embed: DeviceAllocation<'a>,
    pub fc_hidden: DeviceAllocation<'a>,
    pub fc_embed: DeviceAllocation<'a>,
    /// `mtp.hyper_connection_mixer`: norm, w_down, w_up.
    pub mixer: [DeviceAllocation<'a>; 3],
    /// The drafts' E4M3 copy of the shared `lm_head` (per-row x 128-K FP32
    /// scales) and its scales; the target keeps the BF16 head.
    pub head_fp8: Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>,
}

impl MtpWeights<'_> {
    pub fn bytes(&self) -> usize {
        let [a, b, c] = &self.mixer;
        self.layer.bytes() + [&self.norm_hidden, &self.norm_embed, &self.fc_hidden, &self.fc_embed, a, b, c]
            .iter().map(|t| t.buffer.bytes).sum::<usize>()
    }
}

pub(crate) struct Qwen4Loader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    /// Also keep E4M3 copies (FP32 128x128 block scales) of the large GDN and
    /// attention projections for the decode programs (`*_fp8_m64`).
    pub fp8_decode: bool,
    /// Scale rule of the E4M3 copies (decode projections, MTP draft head).
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    pub stream: *mut c_void,
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl<'a> Qwen4Loader<'a> {
    pub fn tensor(&self, name: &str) -> Result<&CheckpointTensor> {
        cuteafd_ffi::memory_ledger::tensor(name);
        let at = self.checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
        Ok(&self.checkpoint.tensors[at])
    }

    pub fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
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

    /// The row-concatenation of BF16 2-D `names` plus `pad_rows` zero rows as one operand.
    fn rows(&self, names: &[String], pad_rows: usize) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum::<usize>() + pad_rows;
        let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
        self.library.cuda_zero_bytes(out.buffer, out.buffer.bytes)?;
        let mut row = 0;
        for (name, (bytes, dtype, shape)) in &tensors {
            ensure!(shape.len() == 2 && shape[1] == cols, "{names:?} do not share columns");
            ensure!(*dtype == DType::Bf16, "{name}: coordinator tensors must be BF16, found {dtype:?}");
            let dest = CuteafdDeviceBuffer {
                // SAFETY: rows row..row+shape[0] lie inside `out`.
                ptr: unsafe { out.buffer.ptr.cast::<u8>().add(row * cols * 2) }.cast(),
                bytes: shape[0] * cols * 2,
                ..out.buffer
            };
            self.library.copy_h2d(dest, bytes)?;
            row += shape[0];
        }
        Ok(out)
    }

    fn one(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let (bytes, dtype, _) = self.raw(name)?;
        ensure!(dtype == DType::Bf16, "{name}: coordinator tensors must be BF16, found {dtype:?}");
        self.upload(&bytes)
    }

    /// A BF16 tensor widened to FP32.
    fn f32(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("f32");
        let (bytes, dtype, _) = self.raw(name)?;
        let values = match dtype {
            DType::Bf16 => bf16_to_f32(&bytes),
            DType::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            other => anyhow::bail!("{name}: expected BF16 or FP32, found {other:?}"),
        };
        self.upload(&f32_bytes(&values))
    }

    fn hc(&self, prefix: &str, inject: bool) -> Result<[DeviceAllocation<'a>; 3]> {
        let mut di = vec![format!("{prefix}.input_mix_weight_down.weight")];
        if inject {
            di.push(format!("{prefix}.block_inject_weight.weight"));
        }
        Ok([self.one(&format!("{prefix}.hc_norm.weight"))?, self.rows(&di, 0)?,
            self.one(&format!("{prefix}.input_mix_weight_up.weight"))?])
    }

    pub fn layer(&self, cfg: &Qwen4Config, layer: usize) -> Result<Qwen4Layer<'a>> {
        self.layer_at(cfg, &format!("{PREFIX}layers.{layer}"), cfg.attention[layer], cfg.ple_layers.contains(&layer))
    }

    /// The decoder layer under `p` (a target layer or `mtp.layers.0`).
    fn layer_at(&self, cfg: &Qwen4Config, p: &str, attention: Qwen4Attention, ple: bool) -> Result<Qwen4Layer<'a>> {
        let mut ops: HashMap<&'static str, DeviceAllocation<'a>> = HashMap::new();
        for (site, names) in [("attn_hyper_connection", ["attn.norm", "attn.w_di", "attn.w_up"]),
            ("mlp_hyper_connection", ["mlp.norm", "mlp.w_di", "mlp.w_up"])] {
            for (name, value) in names.into_iter().zip(self.hc(&format!("{p}.{site}"), true)?) {
                ops.insert(name, value);
            }
        }
        match attention {
            Qwen4Attention::Gdn => {
                let a = |name: &str| format!("{p}.linear_attn.{name}");
                ops.insert("w_in", self.rows(&["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
                    .map(|n| a(&format!("{n}.weight"))), 0)?);
                // [10240, 1, 4] -> FP32 [10240, 4].
                ops.insert("conv_w", self.f32(&a("conv1d.weight"))?);
                ops.insert("a_log", self.f32(&a("A_log"))?);
                ops.insert("dt_bias", self.f32(&a("dt_bias"))?);
                ops.insert("norm_w", self.one(&a("norm.weight"))?);
                ops.insert("w_out", self.one(&a("out_proj.weight"))?);
            }
            Qwen4Attention::Full => {
                let a = |name: &str| format!("{p}.self_attn.{name}");
                // (E4M3 copies of w_in / w_o are added below.)
                ops.insert("w_in", self.rows(&[a("q_proj.weight"), a("k_proj.weight"), a("v_proj.weight"),
                    a("indexer.index_qk_proj.weight")], 0)?);
                ops.insert("q_norm", self.one(&a("q_norm.weight"))?);
                ops.insert("k_norm", self.one(&a("k_norm.weight"))?);
                ops.insert("iq_norm", self.one(&a("indexer.q_layernorm.weight"))?);
                ops.insert("ik_norm", self.one(&a("indexer.k_layernorm.weight"))?);
                ops.insert("w_o", self.one(&a("o_proj.weight"))?);
            }
        }
        let m = |name: &str| format!("{p}.mlp.{name}");
        ops.insert("gate", self.one(&m("gate.weight"))?);
        let shared = [m("shared_expert.gate_proj.weight"), m("shared_expert.up_proj.weight"), m("shared_expert_gate.weight")];
        let packed = 2 * cfg.shared_intermediate + 1;
        ensure!(packed <= SHARED_ROWS, "shared expert rows exceed the packed operand");
        ops.insert("shared.w_gate_up", self.rows(&shared, SHARED_ROWS - packed)?);
        ops.insert("shared.w_down", self.one(&m("shared_expert.down_proj.weight"))?);
        if ple {
            let e = |name: &str| format!("{p}.ple.{name}");
            ops.insert("ple.w_kv", self.rows(&[e("key_proj.weight"), e("value_proj.weight")], 0)?);
            ops.insert("ple.norm_key", self.one(&e("norm_key.weight"))?);
            ops.insert("ple.norm_query", self.one(&e("norm_query.weight"))?);
            ops.insert("ple.norm_conv", self.one(&e("norm_conv.weight"))?);
            ops.insert("ple.conv_w", self.f32(&e("conv1d.weight"))?);
        }
        if self.fp8_decode {
            let (first, second): (&'static str, &'static str) = match attention {
                Qwen4Attention::Gdn => ("w_in", "w_out"),
                Qwen4Attention::Full => ("w_in", "w_o"),
            };
            for name in [first, second] {
                let (rows, cols) = match (attention, name) {
                    (Qwen4Attention::Gdn, "w_in") => (cfg.gdn_conv_width() + cfg.gdn_value_width() + 2 * cfg.gdn_value_heads, cfg.hidden),
                    (Qwen4Attention::Gdn, _) => (cfg.hidden, cfg.gdn_value_width()),
                    (Qwen4Attention::Full, "w_in") => (cfg.attn_in_width(), cfg.hidden),
                    (Qwen4Attention::Full, _) => (cfg.hidden, cfg.heads * cfg.head_dim),
                };
                let weight = ops.get(name).context("projection to quantize")?.buffer.ptr;
                let q = DeviceAllocation::new(self.library, rows * cols)?;
                let scale = DeviceAllocation::new(self.library, rows.div_ceil(128) * cols.div_ceil(128) * 4)?;
                // SAFETY: the BF16 weight, the E4M3 copy and the scales are live device
                // buffers of these shapes; the stream drains before they are used.
                unsafe {
                    self.library.fp8_quant_rule(weight, q.buffer.ptr, scale.buffer.ptr, rows, cols, false,
                        self.fp8_scales.code(), self.stream)?;
                    self.library.cuda_stream_synchronize(self.stream)?;
                }
                let (fp8, scales): (&'static str, &'static str) = match name {
                    "w_in" => ("w_in_fp8", "w_in_scale"),
                    "w_out" => ("w_out_fp8", "w_out_scale"),
                    _ => ("w_o_fp8", "w_o_scale"),
                };
                ops.insert(fp8, q);
                ops.insert(scales, scale);
            }
        }
        Ok(Qwen4Layer { attention, operands: ops })
    }

    /// Layers `0..layers` (all of them unless the caller stops early), and the
    /// MTP layer with `mtp` (with an E4M3 draft head when `fp8_head`).
    pub fn model(&self, cfg: &Qwen4Config, layers: usize, mtp: bool, fp8_head: bool) -> Result<Qwen4Weights<'a>> {
        let (head, dtype, shape) = self.raw("lm_head.weight")?;
        ensure!(dtype == DType::Bf16 && shape == [cfg.vocab_size, cfg.hidden], "lm_head must be BF16 [vocab, hidden]");
        let mtp = if mtp {
            let mut weights = self.mtp(cfg)?;
            if fp8_head {
                let started = std::time::Instant::now();
                let (q, scales) = crate::families::glm5_flash::fp8::quantize(&head, cfg.vocab_size, cfg.hidden,
                    crate::families::glm5_flash::fp8::Layout::Row128, self.fp8_scales);
                weights.head_fp8 = Some((self.upload(&q)?, self.upload(&f32_bytes(&scales))?));
                tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "MTP draft head quantized to E4M3");
            }
            Some(weights)
        } else {
            None
        };
        Ok(Qwen4Weights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            mixer: self.hc(&format!("{PREFIX}hyper_connection_mixer"), false)?,
            head: self.upload(&head)?,
            mtp,
        })
    }

    pub fn mtp(&self, cfg: &Qwen4Config) -> Result<MtpWeights<'a>> {
        ensure!(cfg.mtp_layers == 1, "the MTP drafter runs one MTP layer (config has {})", cfg.mtp_layers);
        ensure!(self.tensor("mtp.fc_hidden.weight").is_ok(), "the checkpoint has no MTP weights (mtp.*)");
        Ok(MtpWeights {
            layer: self.layer_at(cfg, "mtp.layers.0", Qwen4Attention::Full, false)?,
            norm_hidden: self.one("mtp.pre_fc_norm_hidden.weight")?,
            norm_embed: self.one("mtp.pre_fc_norm_embedding.weight")?,
            fc_hidden: self.one("mtp.fc_hidden.weight")?,
            fc_embed: self.one("mtp.fc_embedding.weight")?,
            mixer: self.hc("mtp.hyper_connection_mixer", false)?,
            head_fp8: None,
        })
    }
}
