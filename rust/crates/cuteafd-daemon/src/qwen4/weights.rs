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
use crate::v41_memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use cuteafd_loader::qwen4_exp::{Qwen4Attention, Qwen4Config};
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

    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct Qwen4Weights<'a> {
    pub layers: Vec<Qwen4Layer<'a>>,
    /// The final hyper-connection mixer: norm, w_down, w_up.
    pub mixer: [DeviceAllocation<'a>; 3],
    pub head: DeviceAllocation<'a>,
}

pub(crate) struct Qwen4Loader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl<'a> Qwen4Loader<'a> {
    pub fn tensor(&self, name: &str) -> Result<&CheckpointTensor> {
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
        let (bytes, dtype, _) = self.raw(name)?;
        ensure!(dtype == DType::Bf16, "{name}: coordinator tensors must be BF16, found {dtype:?}");
        self.upload(&bytes)
    }

    /// A BF16 tensor widened to FP32.
    fn f32(&self, name: &str) -> Result<DeviceAllocation<'a>> {
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
        let p = format!("{PREFIX}layers.{layer}");
        let attention = cfg.attention[layer];
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
        if cfg.ple_layers.contains(&layer) {
            let e = |name: &str| format!("{p}.ple.{name}");
            ops.insert("ple.w_kv", self.rows(&[e("key_proj.weight"), e("value_proj.weight")], 0)?);
            ops.insert("ple.norm_key", self.one(&e("norm_key.weight"))?);
            ops.insert("ple.norm_query", self.one(&e("norm_query.weight"))?);
            ops.insert("ple.norm_conv", self.one(&e("norm_conv.weight"))?);
            ops.insert("ple.conv_w", self.f32(&e("conv1d.weight"))?);
        }
        Ok(Qwen4Layer { attention, operands: ops })
    }

    /// Layers `0..layers` (all of them unless the caller stops early).
    pub fn model(&self, cfg: &Qwen4Config, layers: usize) -> Result<Qwen4Weights<'a>> {
        Ok(Qwen4Weights {
            layers: (0..layers.min(cfg.layers)).map(|l| self.layer(cfg, l)).collect::<Result<_>>()?,
            mixer: self.hc(&format!("{PREFIX}hyper_connection_mixer"), false)?,
            head: self.one("lm_head.weight")?,
        })
    }
}
