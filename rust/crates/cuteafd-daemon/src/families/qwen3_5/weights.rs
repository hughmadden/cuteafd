//! Native dense BF16/E4M3 operands with checkpoint-format scales and BF16 gates.
use crate::shared::memory::device::Allocation;
use crate::shared::qwen::weights::WeightReader;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_loader::families::qwen::HybridAttention;
use cuteafd_loader::families::qwen3_5::Qwen35Config;
#[cfg(test)]
use cuteafd_loader::{plan::checkpoint::Checkpoint, serving_capacity::WeightArenaLayout};
use std::{collections::HashMap, ffi::c_void};

const PREFIX: &str = "model.language_model.";

use cuteafd_loader::families::qwen3_5::weight_layout::{layer_requests, native_operands,
    validate_native_matrices, Storage};
pub(crate) use cuteafd_loader::families::qwen3_5::weight_layout::DenseFormat;
#[cfg(test)]
use cuteafd_loader::families::qwen3_5::weight_layout::{layer_operands, mtp_arena, target_arena};

pub(crate) struct DenseLayer<'a> {
    pub(crate) attention: HybridAttention,
    operands: HashMap<&'static str, Allocation<'a>>,
}

impl DenseLayer<'_> {
    pub(crate) fn ptr(&self, name: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(name).with_context(|| format!("dense Qwen layer has no operand {name}"))?.buffer.ptr)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct DenseWeights<'a> {
    pub(crate) layers: Vec<DenseLayer<'a>>,
    pub(crate) norm: Allocation<'a>,
    pub(crate) head: Allocation<'a>,
}

#[cfg(test)]
fn packed_native_layout(tensors: &[(Vec<u8>, DType, Vec<usize>)], storage: Storage) -> Result<Vec<u8>> {
    validate_native_matrices(tensors.iter().map(|(bytes, dtype, shape)|
        (dtype, shape.as_slice(), bytes.len() as u64)), storage)?;
    let mut out = Vec::new();
    for (bytes, dtype, _) in tensors {
        if storage == Storage::Scales && *dtype == DType::Bf16 {
            out.extend(bytes.chunks_exact(2).flat_map(|v|
                (u32::from(u16::from_le_bytes([v[0], v[1]])) << 16).to_le_bytes()));
        } else { out.extend_from_slice(bytes); }
    }
    ensure!(!out.is_empty(), "empty native matrix packing");
    Ok(out)
}

impl<'a> WeightReader<'a> {
    pub(crate) fn dense_bytes(&self, cfg: &Qwen35Config, norm_dtype: DType, _format: DenseFormat)
        -> Result<cuteafd_loader::placement::families::qwen3_5::NativeWeights> {
        cuteafd_loader::placement::families::qwen3_5::NativeWeights::native(self.checkpoint, cfg, norm_dtype)
    }

    /// Validate every header before uploading any weights.
    pub(crate) fn dense_native(&self, cfg: &Qwen35Config, norm_dtype: DType, format: DenseFormat) -> Result<DenseWeights<'a>> {
        self.dense_bytes(cfg, norm_dtype, format)?;
        let norm_name = format!("{PREFIX}norm.weight");
        let head_name = if cfg.tied_embeddings { format!("{PREFIX}embed_tokens.weight") } else { "lm_head.weight".into() };
        let layers = cfg.attention.iter().enumerate().map(|(index, &kind)|
            self.dense_layer(&format!("{PREFIX}layers.{index}"), kind, format)).collect::<Result<_>>()?;
        Ok(DenseWeights { layers, norm: self.bf16(&norm_name)?, head: self.bf16(&head_name)? })
    }

    pub(crate) fn dense_layer_bytes(&self, prefix: &str, attention: HybridAttention, format: DenseFormat) -> Result<u64> {
        layer_requests(self.checkpoint, prefix, attention, format)?.into_iter().try_fold(0u64, |total, (_, bytes)| {
            total.checked_add(u64::try_from(bytes)?).context("dense Qwen layer size overflow")
        })
    }

    /// MTP and target blocks share the checkpoint-native projection packing.
    pub(crate) fn dense_layer(&self, prefix: &str, attention: HybridAttention, format: DenseFormat) -> Result<DenseLayer<'a>> {
        let operands = native_operands(prefix, attention, format).into_iter().map(|spec| {
                let weight = match spec.storage {
                    Storage::Bf16 if spec.tensors.len() > 1 => self.rows(&spec.tensors, 0)?,
                    Storage::Bf16 => self.bf16(&spec.tensors[0])?,
                    Storage::F32 => self.f32(&spec.tensors[0])?,
                    Storage::Fp8 | Storage::Scales => {
                        let headers = spec.tensors.iter().map(|name| self.tensor(name)).collect::<Result<Vec<_>>>()?;
                        validate_native_matrices(headers.iter().map(|tensor|
                            (&tensor.meta.dtype, tensor.meta.shape.as_slice(), tensor.meta.byte_length)), spec.storage)?;
                        self.native(&spec.tensors, spec.storage == Storage::Scales)?
                    },
                    Storage::NativeNorm => self.native(&spec.tensors, false)?,
                };
            Ok((spec.name, weight))
        }).collect::<Result<_>>()?;
        Ok(DenseLayer { attention, operands })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(format: DenseFormat, tied_embeddings: bool) -> (Checkpoint, Qwen35Config) {
        use cuteafd_loader::{families::qwen::{GdnGeometry, GdnOutputGate}, plan::checkpoint::CheckpointTensor,
            SafetensorsTensorMetadata};
        let cfg = Qwen35Config { hidden: 128, intermediate: 128, vocab: 128,
            attention: vec![HybridAttention::Gdn, HybridAttention::Full], heads: 1, kv_heads: 1, head_dim: 256,
            gdn: GdnGeometry { key_heads: 1, value_heads: 1, key_dim: 128, value_dim: 128,
                conv_kernel: 4, output_gate: GdnOutputGate::Silu }, rope_dim: 64, rope_theta: 1e7,
            rms_norm_eps: 1e-6, max_context: 4096, mtp_layers: 1, tied_embeddings };
        let mut tensors = std::collections::BTreeMap::new();
        let mut add = |name: String, dtype: DType, shape: Vec<usize>| {
            let width = match dtype { DType::F8E4M3 => 1, DType::F32 => 4, _ => 2 };
            let byte_length = (shape.iter().product::<usize>() * width) as u64;
            tensors.insert(name.clone(), CheckpointTensor { shard: "model.safetensors".into(),
                meta: SafetensorsTensorMetadata { name, dtype, shape, byte_offset: 0, byte_length } });
        };
        for (prefix, kind) in [(format!("{PREFIX}layers.0"), HybridAttention::Gdn),
            (format!("{PREFIX}layers.1"), HybridAttention::Full), ("mtp.layers.0".into(), HybridAttention::Full)] {
            for spec in native_operands(&prefix, kind, format) {
                for name in spec.tensors {
                    let (dtype, shape) = match spec.storage {
                        Storage::Fp8 => (DType::F8E4M3, vec![128, 128]),
                        Storage::Scales => (DType::Bf16, vec![1, 1]),
                        Storage::F32 => (DType::Bf16, vec![129]),
                        Storage::NativeNorm => (DType::Bf16, vec![128]),
                        Storage::Bf16 => (DType::Bf16, vec![128, 128]),
                    };
                    add(name, dtype, shape);
                }
            }
        }
        for name in [format!("{PREFIX}embed_tokens.weight"), "lm_head.weight".into()] {
            add(name, DType::Bf16, vec![128, 128]);
        }
        add(format!("{PREFIX}norm.weight"), DType::Bf16, vec![128]);
        for name in ["norm", "pre_fc_norm_hidden", "pre_fc_norm_embedding"] {
            add(format!("mtp.{name}.weight"), DType::Bf16, vec![128]);
        }
        add("mtp.fc.weight".into(), DType::Bf16, vec![128, 256]);
        (Checkpoint { view: None, snapshot: Default::default(), config: serde_json::json!({}), quantize_config: None,
            weight_map: Default::default(), tensors: tensors.into_values().collect(), missing_shards: vec![], shard_bytes: 0 }, cfg)
    }

    fn check_layout(layout: &WeightArenaLayout) {
        let mut views = layout.views().values().copied().collect::<Vec<_>>();
        views.sort_by_key(|view| view.offset());
        let mut end = 0;
        for view in views {
            assert_eq!(view.offset() % 256, 0);
            assert!(view.offset() >= end);
            assert!(view.bytes() >= 256);
            end = view.offset() + view.bytes();
        }
        assert_eq!(end, layout.bytes());
        assert_eq!(layout.bytes(), layout.payload_bytes() + layout.padding_bytes());
    }

    #[test]
    fn arena_layouts_preserve_native_formats_floor_and_separate_mtp_ownership() -> Result<()> {
        for format in [DenseFormat::Bf16, DenseFormat::Fp8] {
            for tied in [false, true] {
                let (mut checkpoint, cfg) = fixture(format, tied);
                let target = target_arena(&checkpoint, &cfg, DType::Bf16, format)?;
                let mtp = mtp_arena(&checkpoint, &cfg, format)?;
                check_layout(&target);
                check_layout(&mtp);
                assert_ne!(target.id(), mtp.id());
                assert_eq!(target.views().len(), if format.fp8() { 30 } else { 21 });
                assert_eq!(mtp.views().len(), if format.fp8() { 16 } else { 12 });
                assert!(target.views().keys().all(|name| !name.starts_with("mtp.")));
                assert!(mtp.views().keys().all(|name| name.starts_with("mtp.")));
                assert_eq!(target.view("embedding.table").unwrap().bytes(), 32768);
                assert_eq!(target.view("head.weight").unwrap().bytes(), 32768);
                let p = format!("{PREFIX}layers.0");
                assert_eq!(target.view(&format!("{p}/a_log")).unwrap().bytes(), 516);
                assert!(target.padding_bytes() > 0);
                for tensor in &mut checkpoint.tensors {
                    if ["conv1d.weight", "A_log", "dt_bias"].iter().any(|suffix| tensor.meta.name.ends_with(suffix)) {
                        tensor.meta.dtype = DType::F32;
                        tensor.meta.byte_length *= 2;
                    }
                }
                assert_eq!(target_arena(&checkpoint, &cfg, DType::Bf16, format)?, target);
                if format.fp8() {
                    assert_eq!(target.view(&format!("{p}/w_qkvz_scale")).unwrap().bytes(), 256);
                    assert!(target.view(&format!("{p}/w_in")).is_none());
                    for tensor in &mut checkpoint.tensors {
                        if tensor.meta.name.ends_with("weight_scale_inv") {
                            tensor.meta.dtype = DType::F32;
                            tensor.meta.byte_length *= 2;
                        }
                    }
                    assert_eq!(target_arena(&checkpoint, &cfg, DType::Bf16, format)?, target);
                    assert_eq!(mtp_arena(&checkpoint, &cfg, format)?, mtp);
                }
                for tensor in &mut checkpoint.tensors {
                    if tensor.meta.name.ends_with("linear_attn.norm.weight") {
                        tensor.meta.dtype = DType::F32;
                        tensor.meta.byte_length *= 2;
                    }
                }
                let fp32_norm = target_arena(&checkpoint, &cfg, DType::F32, format)?;
                assert_eq!(fp32_norm.view(&format!("{p}/norm_w")).unwrap().bytes(), 512);
                assert!(target_arena(&checkpoint, &cfg, DType::Bf16, format).is_err());
                checkpoint.tensors.retain(|t| !t.meta.name.ends_with("linear_attn.A_log"));
                assert!(target_arena(&checkpoint, &cfg, DType::F32, format).is_err());
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "reads installed checkpoint headers; set CUTEAFD_QWEN_DENSE_ARENA_SNAPSHOT"]
    fn installed_arena_inventory() -> Result<()> {
        let snapshot = std::env::var("CUTEAFD_QWEN_DENSE_ARENA_SNAPSHOT")?;
        let checkpoint = Checkpoint::open(std::path::Path::new(&snapshot))?;
        let cfg = Qwen35Config::from_hf(&checkpoint.config)?;
        let gate = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{PREFIX}layers.0.mlp.gate_proj.weight"))
            .context("missing first gate")?;
        let format = if gate.meta.dtype == DType::F8E4M3 { DenseFormat::Fp8 } else { DenseFormat::Bf16 };
        let norm = checkpoint.tensors.iter().find(|t| t.meta.name == format!("{PREFIX}layers.0.linear_attn.norm.weight"))
            .context("missing GDN norm")?.meta.dtype.clone();
        let mut layouts = vec![target_arena(&checkpoint, &cfg, norm, format)?];
        if cfg.mtp_layers == 1 { layouts.push(mtp_arena(&checkpoint, &cfg, format)?); }
        for layout in layouts {
            check_layout(&layout);
            let large = layout.views().values().filter(|view| view.bytes() > 1 << 20).collect::<Vec<_>>();
            let slack = |bytes: usize| bytes.div_ceil(2 << 20) * (2 << 20) - bytes;
            println!("{}: allocations={} payload={} large_count={} large_bytes={} before_slack={} arena={} padding={} tail={}",
                layout.id(), layout.views().len(), layout.payload_bytes(), large.len(),
                large.iter().map(|v| v.bytes()).sum::<usize>(), large.iter().map(|v| slack(v.bytes())).sum::<usize>(),
                layout.bytes(), layout.padding_bytes(), slack(layout.bytes()));
        }
        Ok(())
    }

    #[test]
    fn fp8_operands_keep_raw_weight_bytes_exact_scales_and_bf16_gates() -> Result<()> {
        let specs = native_operands("layer", HybridAttention::Gdn, DenseFormat::Fp8);
        let find = |name| specs.iter().find(|spec| spec.name == name).unwrap();
        assert_eq!(find("w_ba").tensors, ["layer.linear_attn.in_proj_b.weight", "layer.linear_attn.in_proj_a.weight"]);
        assert_eq!(find("w_ba").storage, Storage::Bf16);
        assert_eq!(find("w_qkvz_scale").tensors, ["layer.linear_attn.in_proj_qkv.weight_scale_inv",
            "layer.linear_attn.in_proj_z.weight_scale_inv"]);
        assert!(specs.iter().all(|spec| !matches!(spec.name, "w_in" | "w_out")));
        assert_eq!(packed_native_layout(&[(vec![1, 2], DType::F8E4M3, vec![1, 2]),
            (vec![3, 4], DType::F8E4M3, vec![1, 2])], Storage::Fp8)?, vec![1, 2, 3, 4]);
        let bits = [0x3f80u16, 0x8000, 0x0001, 0x7f7f];
        let bytes = bits.iter().flat_map(|value| value.to_le_bytes()).collect();
        let widened = packed_native_layout(&[(bytes, DType::Bf16, vec![2, 2])], Storage::Scales)?;
        assert_eq!(widened, bits.iter().flat_map(|&value| (u32::from(value) << 16).to_le_bytes()).collect::<Vec<_>>());
        assert!(packed_native_layout(&[(vec![0; 2], DType::Bf16, vec![1, 1])], Storage::Fp8).is_err());
        assert!(packed_native_layout(&[(vec![0], DType::F8E4M3, vec![1, 2])], Storage::Fp8).is_err());
        Ok(())
    }

    #[test]
    fn dense_operands_keep_native_projection_order_and_norm_precision() {
        for kind in [HybridAttention::Gdn, HybridAttention::Full] {
            let specs = layer_operands("layer", kind);
            let names = |operand: &str| specs.iter().find(|s| s.name == operand).unwrap().tensors.clone();
            assert_eq!(names("ffn.w_gate_up"), ["layer.mlp.gate_proj.weight", "layer.mlp.up_proj.weight"]);
            if kind == HybridAttention::Gdn {
                assert_eq!(names("w_in"), ["layer.linear_attn.in_proj_qkv.weight", "layer.linear_attn.in_proj_z.weight",
                    "layer.linear_attn.in_proj_b.weight", "layer.linear_attn.in_proj_a.weight"]);
                assert_eq!(specs.iter().find(|s| s.name == "norm_w").unwrap().storage, Storage::NativeNorm);
                assert_eq!(specs.iter().filter(|s| s.storage == Storage::F32).count(), 3);
            } else {
                assert_eq!(names("w_in"), ["layer.self_attn.q_proj.weight", "layer.self_attn.k_proj.weight", "layer.self_attn.v_proj.weight"]);
            }
            assert!(specs.iter().all(|s| !s.name.contains("expert") && !s.name.contains("hc") && !s.name.contains("index")));
        }
    }
}
