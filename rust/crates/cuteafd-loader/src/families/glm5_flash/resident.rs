//! Resident GLM Flash operands: the loader's concatenations, conversions and rank ownership.
use super::{GlmNextAttention, GlmNextConfig};
use crate::plan::Checkpoint;
use cuteafd_core::DType;

#[derive(Debug, Clone, Copy)]
pub struct GlmfRepresentation {
    pub kda_fp8: bool,
    pub head: GlmfHeadFormat,
    pub output_shard: bool,
}
impl Default for GlmfRepresentation {
    fn default() -> Self { Self { kda_fp8: false, head: GlmfHeadFormat::Auto, output_shard: false } }
}

/// The one resident form of the vocabulary head `lm_head.weight` [vocab, hidden] that target
/// logits and the drafter share (`--fp8-head`, launcher `GLM5_FLASH_FP8_HEAD`): the checkpoint's
/// BF16 (`false`, also `bf16`/`off`), an E4M3 copy with FP32 scales per row and 128-wide K block
/// (`true`, also `fp8`/`on`), or `packed`: the exact 12-bit copy of the BF16 values
/// (`cuteafd_core::packed_head`), packed and verified at load from the landed BF16, which it
/// then releases. Target logits read it exactly (both planes and the exception table, FP32
/// sums); the drafter reads its coarse plane. A packed head serves one coordinator GPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GlmfHeadFormat {
    /// Resolve after reading checkpoint headers and selecting the coordinator topology.
    #[default]
    Auto,
    Bf16,
    Fp8,
    Packed,
}

impl std::str::FromStr for GlmfHeadFormat {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "false" | "bf16" | "off" => Ok(Self::Bf16),
            "true" | "fp8" | "on" => Ok(Self::Fp8),
            "packed" => Ok(Self::Packed),
            _ => Err(format!("{value:?}: expected auto, false (bf16), true (fp8) or packed")),
        }
    }
}

impl GlmfHeadFormat {
    /// Auto preserves the checkpoint head; conversions require an explicit choice.
    pub fn resolve(self, native_head: bool, _ranks: usize) -> Self {
        if native_head { Self::Bf16 }
        else if self == Self::Auto {
            Self::Bf16
        } else { self }
    }

    /// The resolved `fp8-head` setting: the boolean spellings for BF16 and FP8, `packed`.
    pub fn setting(self) -> &'static str {
        match self { Self::Auto => "auto", Self::Bf16 => "false", Self::Fp8 => "true", Self::Packed => "packed" }
    }

    /// The head's name in plans and logs.
    pub fn name(self) -> &'static str {
        match self { Self::Auto => "auto", Self::Bf16 => "BF16", Self::Fp8 => "FP8", Self::Packed => "exact 12-bit" }
    }

    /// Resident bytes of a `[vocab, hidden]` BF16 head in this form, as the loader allocates them.
    pub fn resident_bytes(self, vocab: u64, hidden: u64) -> u64 {
        match self {
            Self::Auto => panic!("resolve GLM Flash head before sizing"),
            Self::Bf16 => (vocab * hidden * 2).max(256),
            Self::Fp8 => (vocab * hidden).max(256) + (vocab * hidden.div_ceil(128) * 4).max(256),
            Self::Packed => cuteafd_core::packed_head::PackedHeadLayout::bytes(vocab, hidden),
        }
    }

    /// Bytes beside the resident form while loading: the landed BF16 source a packed head
    /// packs from (released once the pack verifies). The FP8 copy quantizes from pinned host
    /// rows and keeps no device source.
    pub fn load_source_bytes(self, vocab: u64, hidden: u64) -> u64 {
        if self == Self::Packed { Self::Bf16.resident_bytes(vocab, hidden) } else { 0 }
    }

    /// GEMV scratch of one projection site of up to `rows` rows on an `sms`-SM device: only
    /// the packed head's (`cuteafd_packed_head_workspace`); the FP8 program has none.
    pub fn scratch_bytes(self, rows: u64, vocab: u64, hidden: u64, sms: u64) -> u64 {
        if self == Self::Packed { cuteafd_core::packed_head::workspace_bytes(rows, hidden, vocab, sms) } else { 0 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GlmfResidentRank {
    pub embedding: u64,
    pub weights: u64,
}

/// Native EXLR8 allocations from the shared planner's already-owned target plans.
/// Shared experts have separate format-level arenas; MTP never enters these plans.
pub fn native_resident(selection: &crate::plan::exlr8::dense::DenseSelection, layers: usize,
    plans: &[crate::plan::exlr8::dense::DensePlan], shared_bytes: &[u64])
    -> anyhow::Result<Vec<GlmfResidentRank>> {
    use crate::plan::exlr8::dense::resident_native;
    anyhow::ensure!(matches!(plans.len(), 1 | 2) && plans.len() == shared_bytes.len(),
        "Flash native resident owners must match one/two shared expert ranks");
    plans.iter().zip(shared_bytes).enumerate().map(|(rank, (plan, &shared))| {
        anyhow::ensure!(plan.world == plans.len() && plan.rank == rank,
            "Flash native resident plan rank/world mismatch");
        let mut resident = GlmfResidentRank { embedding: 0, weights: shared };
        for (group, _, bytes) in resident_native(selection, layers, plan)? {
            let total = if group == "embedding" { &mut resident.embedding } else { &mut resident.weights };
            *total = total.checked_add(bytes).ok_or_else(|| anyhow::anyhow!("Flash native resident overflow"))?;
        }
        Ok(resident)
    }).collect()
}

/// No tensor data is read. FP8 block scales and FP32 auxiliaries are counted as uploaded, not
/// as the source checkpoint stores them. Routed experts, media and external drafts are separate.
pub fn resident_weights(checkpoint: &Checkpoint, cfg: &GlmNextConfig, layers: usize, ranks: usize,
    repr: GlmfRepresentation) -> Result<Vec<GlmfResidentRank>, String> {
    resident_weights_placed(checkpoint, cfg, layers, ranks, repr,
        crate::placement::AttentionPlacement::Heads, 0)
}

/// Candidate MLA ownership with KDA and FFN operands still head-split. This
/// inventory includes each rank's router replica and does not advertise an unqualified executor.
pub fn resident_weights_placed(checkpoint: &Checkpoint, cfg: &GlmNextConfig, layers: usize, ranks: usize,
    repr: GlmfRepresentation, attention: crate::placement::AttentionPlacement, first_owner: usize)
    -> Result<Vec<GlmfResidentRank>, String> {
    use crate::placement::AttentionPlacement;
    if !(1..=2).contains(&ranks) || first_owner >= ranks
        || (attention != AttentionPlacement::Heads && ranks != 2) {
        return Err("GLM Flash attention weights require valid one/two-rank ownership".into());
    }
    let owned = attention == AttentionPlacement::Layers;
    let core = super::core::CoreInventory::from_checkpoint(cfg, checkpoint).map_err(|e| e.to_string())?;
    let tensor = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .ok_or_else(|| format!("GLM Flash resident operand missing: {name}"));
    let elements = |name: &str| -> Result<u64, String> {
        tensor(name)?.meta.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d as u64)
            .ok_or_else(|| format!("GLM Flash operand overflow: {name}")))
    };
    let mut out = vec![GlmfResidentRank::default(); ranks];
    out[0].embedding = tensor("model.language_model.embed_tokens.weight")?.meta.byte_length.max(256);
    let fp8_ranks = |names: &[String], divisor: usize, row128: bool| -> Result<u64, String> {
        let mut values = 0;
        let mut scales = 0;
        for name in names {
            let t = tensor(name)?;
            let shape = &t.meta.shape;
            if shape.len() != 2 { return Err(format!("GLM Flash FP8 operand is not a matrix: {name}")); }
            let n = shape[0] as u64;
            let k = shape[1] as u64;
            values += n * k;
            scales += if row128 { n * k.div_ceil(128) * 4 } else { n.div_ceil(128) * k.div_ceil(128) * 4 };
        }
        Ok((values / divisor as u64).max(256) + (scales / divisor as u64).max(256))
    };
    let fp8 = |names: &[String], split: bool, row128: bool| fp8_ranks(names, if split { ranks } else { 1 }, row128);
    out[0].weights = tensor("model.language_model.norm.weight")?.meta.byte_length.max(256)
        + if let Some(head) = core.get_weight("lm_head") {
            head.resident_bytes(1).map_err(|e| e.to_string())?
        } else if core.native.get("lm_head").is_some_and(|head| head.fp8) {
            fp8(&["lm_head.weight".into()], false, false)?
        } else {
            match repr.head.resolve(core.get_weight("lm_head").is_some(), ranks) {
                GlmfHeadFormat::Auto => unreachable!("head resolved"),
                GlmfHeadFormat::Fp8 => fp8(&["lm_head.weight".into()], false, true)?,
                GlmfHeadFormat::Bf16 => tensor("lm_head.weight")?.meta.byte_length.max(256),
                GlmfHeadFormat::Packed => {
                    let head = tensor("lm_head.weight")?;
                    if ranks != 1 { return Err("the exact 12-bit GLM Flash head serves one coordinator GPU".into()); }
                    match (&head.meta.dtype, head.meta.shape.as_slice()) {
                        (DType::Bf16, &[vocab, hidden]) => {
                            cuteafd_core::packed_head::PackedHeadLayout::new(vocab, hidden).map_err(|e| e.to_string())?;
                            GlmfHeadFormat::Packed.resident_bytes(vocab as u64, hidden as u64)
                        }
                        _ => return Err("the exact 12-bit GLM Flash head packs a BF16 [vocab, hidden] lm_head".into()),
                    }
                }
            }
        };
    let mut mla_owner = first_owner;
    for layer in 0..layers.min(cfg.layers) {
        let owner = mla_owner;
        if cfg.attention[layer] == GlmNextAttention::Mla { mla_owner ^= 1; }
        let native = core.layer_weights(cfg, layer, ranks, owned.then_some(owner as u8)
            .filter(|_| cfg.attention[layer] == GlmNextAttention::Mla)).map_err(|e| e.to_string())?;
        let native_attention = core.layer_native(cfg, layer, ranks, owned.then_some(owner as u8)
            .filter(|_| cfg.attention[layer] == GlmNextAttention::Mla)).map_err(|e| e.to_string())?;
        let p = format!("model.language_model.layers.{layer}");
        let a = |s: &str| format!("{p}.self_attn.{s}");
        let one = |s: &str| -> Result<u64, String> { Ok(tensor(s)?.meta.byte_length.max(256)) };
        let f32 = |names: &[String], split: bool| -> Result<u64, String> {
            let n = names.iter().map(|name| elements(name)).collect::<Result<Vec<_>, _>>()?.into_iter().sum::<u64>();
            Ok((n * 4 / if split { ranks as u64 } else { 1 }).max(256))
        };
        for rank in 0..ranks {
            let mut bytes = native_attention[rank].values().try_fold(0u64, |bytes, layout|
                layout.bytes().map(|n| bytes + n)).map_err(|e| e.to_string())?;
            for site in ["attn", "ffn"] {
                for suffix in ["fn", "scale", "base"] { bytes += f32(&[format!("{p}.hc_{site}_{suffix}")], false)?; }
            }
            bytes += one(&format!("{p}.input_layernorm.weight"))? + one(&format!("{p}.post_attention_layernorm.weight"))?;
            match cfg.attention[layer] {
                GlmNextAttention::Kda => {
                    if core.get_weight(&a("q_proj")).is_some() || core.native.contains_key(&a("q_proj")) {
                        bytes += native[rank].iter().filter(|(name, _)| name.starts_with("self_attn."))
                            .map(|(_, layout)| layout.bytes as u64).sum::<u64>();
                        let aux = elements(&a("f_a_proj.weight"))? + elements(&a("g_a_proj.weight"))?
                            + elements(&a("b_proj.weight"))? / ranks as u64;
                        bytes += (aux * 2).max(256);
                    } else {
                        let inputs: Vec<_> = ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj"]
                            .iter().map(|s| a(&format!("{s}.weight"))).collect();
                        let mut input_values = 0;
                        for (i, name) in inputs.iter().enumerate() {
                            input_values += elements(name)? / if ranks > 1 && !matches!(i, 3 | 4) { ranks as u64 } else { 1 };
                        }
                        bytes += if repr.kda_fp8 { input_values.max(256) + (input_values / 128 * 4).max(256) }
                            else { (input_values * 2).max(256) };
                        let output_split = ranks > 1 && !(repr.kda_fp8 && repr.output_shard);
                        bytes += if repr.kda_fp8 { fp8(&[a("o_proj.weight")], output_split, true)? }
                            else { (elements(&a("o_proj.weight"))? * 2 / ranks as u64).max(256) };
                    }
                    bytes += ((elements(&a("f_b_proj.weight"))? + elements(&a("g_b_proj.weight"))?) * 2 / ranks as u64).max(256);
                    bytes += f32(&["q", "k", "v"].map(|s| a(&format!("{s}_conv1d.weight"))), ranks > 1)?;
                    bytes += f32(&[a("A_log")], ranks > 1)? + f32(&[a("dt_bias")], ranks > 1)?;
                    bytes += one(&a("o_norm.weight"))?;
                }
                GlmNextAttention::Mla => {
                    if !owned || rank == owner {
                        let parts = if owned { 1 } else { ranks };
                        bytes += one(&a("q_a_layernorm.weight"))? + one(&a("kv_a_layernorm.weight"))?;
                        bytes += (cfg.heads * cfg.qk_nope_dim * cfg.kv_lora_rank * 2 / parts) as u64;
                        bytes += (cfg.heads * cfg.v_head_dim * cfg.kv_lora_rank * 2 / parts) as u64;
                        if core.get_weight(&a("q_a_proj")).is_some() || core.native.contains_key(&a("q_a_proj")) {
                            bytes += native[rank].iter().filter(|(name, _)| name.starts_with("self_attn."))
                                .map(|(_, layout)| layout.bytes as u64).sum::<u64>();
                        } else {
                            bytes += fp8(&[a("q_a_proj.weight"), a("kv_a_proj_with_mqa.weight")], false, false)?;
                            bytes += fp8_ranks(&[a("q_b_proj.weight")], parts, false)?
                                + fp8_ranks(&[a("o_proj.weight")], parts, false)?;
                        }
                        bytes += one(&a("indexer.wq_b.weight"))?;
                        bytes += ["wk.weight", "weights_proj.weight", "index_kpool_compress_gate"].iter()
                            .map(|s| elements(&a(&format!("indexer.{s}")))).collect::<Result<Vec<_>, _>>()?.into_iter().sum::<u64>() * 2;
                        for s in ["k_norm.weight", "k_norm.bias", "index_kpool_compress_ape"] { bytes += one(&a(&format!("indexer.{s}")))?; }
                    }
                }
            }
            let mlp = if cfg.dense[layer] { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
            let gate = format!("{mlp}.gate_proj.weight");
            if core.staged(gate.strip_suffix(".weight").unwrap()) {
                bytes += native[rank].iter().filter(|(name, _)| name.starts_with("mlp."))
                    .map(|(_, layout)| layout.bytes as u64).sum::<u64>();
            } else if cfg.dense[layer] && tensor(&gate)?.meta.dtype == DType::U8 {
                if rank == 0 {
                    for s in ["gate_proj", "up_proj", "down_proj"] {
                        bytes += one(&format!("{mlp}.{s}.weight"))?;
                        bytes += (tensor(&format!("{mlp}.{s}.weight_scale"))?.meta.byte_length + 8).max(256);
                    }
                }
            } else {
                bytes += fp8(&[gate, format!("{mlp}.up_proj.weight")], ranks > 1, false)?;
                bytes += fp8(&[format!("{mlp}.down_proj.weight")], ranks > 1, false)?;
            }
            if !cfg.dense[layer] {
                bytes += one(&format!("{p}.mlp.gate.weight"))?;
                bytes += f32(&[format!("{p}.mlp.gate.e_score_correction_bias")], false)?;
            }
            out[rank].weights += bytes;
        }
    }
    Ok(out)
}

/// Router operands newly replicated on GPU1 by the TP2 executor, counted before loading.
pub fn router_replica_bytes(checkpoint: &Checkpoint, cfg: &GlmNextConfig, layers: usize) -> Result<u64, String> {
    let mut bytes = 0u64;
    for layer in 0..layers.min(cfg.layers) {
        if cfg.dense[layer] { continue; }
        let prefix = format!("model.language_model.layers.{layer}.mlp.gate");
        for (suffix, f32) in [("weight", false), ("e_score_correction_bias", true)] {
            let name = format!("{prefix}.{suffix}");
            let tensor = checkpoint.tensors.iter().find(|t| t.meta.name == name)
                .ok_or_else(|| format!("GLM Flash router operand missing: {name}"))?;
            let resident = if f32 { tensor.meta.shape.iter().product::<usize>() as u64 * 4 }
                else { tensor.meta.byte_length };
            bytes = bytes.checked_add(resident.max(256)).ok_or_else(|| "GLM Flash router bytes overflow".to_string())?;
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_fixture_charges_borrowable_head_only_on_lead() {
        use crate::plan::checkpoint::CheckpointTensor;
        let config = crate::plan::testing::glm5_flash_config(2);
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let tensors = [("model.language_model.embed_tokens.weight", vec![64, 4096]),
            ("model.language_model.norm.weight", vec![4096]), ("lm_head.weight", vec![64, 4096])]
            .into_iter().map(|(name, shape)| CheckpointTensor { shard: "fixture".into(),
                meta: crate::SafetensorsTensorMetadata { name: name.into(), dtype: DType::Bf16,
                    byte_length: shape.iter().product::<usize>() as u64 * 2, shape, byte_offset: 0 } }).collect();
        let checkpoint = Checkpoint { view: None, snapshot: "fixture".into(), config, quantize_config: None,
            weight_map: Default::default(), tensors, missing_shards: vec![], shard_bytes: 0 };
        // Zero target layers isolates the head representation and ownership contract.
        let single = resident_weights(&checkpoint, &cfg, 0, 1, Default::default()).unwrap();
        let split = resident_weights(&checkpoint, &cfg, 0, 2, Default::default()).unwrap();
        assert_eq!(single[0].embedding, 524_288);
        assert_eq!(single[0].weights, 532_480); // BF16 norm and source head.
        assert_eq!(split[0].weights, 532_480); // Same source precision on TP2.
        assert_eq!((split[1].embedding, split[1].weights), (0, 0));
        assert!(resident_weights(&checkpoint, &cfg, 2, 1, Default::default()).unwrap_err().contains("resident operand missing"));
        // Each head form charges its own allocation and nothing else; the exact 12-bit one is
        // PackedHeadLayout's single allocation and serves one GPU.
        for (head, bytes) in [(GlmfHeadFormat::Bf16, 524_288), (GlmfHeadFormat::Fp8, 270_336),
            (GlmfHeadFormat::Packed, 395_776)] {
            let repr = GlmfRepresentation { head, ..Default::default() };
            let single = resident_weights(&checkpoint, &cfg, 0, 1, repr).unwrap();
            assert_eq!(single[0].weights, 8192 + bytes, "{head:?}");
            assert_eq!(head.resident_bytes(64, 4096), bytes);
        }
        assert_eq!(GlmfHeadFormat::Packed.resident_bytes(64, 4096),
            cuteafd_core::packed_head::PackedHeadLayout::new(64, 4096).unwrap().total_bytes() as u64);
        let packed = GlmfRepresentation { head: GlmfHeadFormat::Packed, ..Default::default() };
        assert!(resident_weights(&checkpoint, &cfg, 0, 2, packed).unwrap_err().contains("one coordinator GPU"));
    }

    /// `--fp8-head` keeps its boolean spellings and adds `packed`; GLM 5.3 Flash's
    /// [154880, 4096] head is 1.27 GB BF16, 654 MB FP8 and 956 MB exact 12-bit, plus the
    /// landed BF16 while packing and the GEMV's 512 KiB of scratch below 606 SMs.
    #[test]
    fn head_formats_parse_report_and_charge_the_flash_head() {
        use GlmfHeadFormat::{Bf16, Fp8, Packed};
        for (spellings, head) in [(["false", "bf16", "off"], Bf16), (["true", "fp8", "on"], Fp8)] {
            for spelling in spellings { assert_eq!(spelling.parse::<GlmfHeadFormat>(), Ok(head)); }
        }
        assert_eq!("packed".parse::<GlmfHeadFormat>(), Ok(Packed));
        assert!("maybe".parse::<GlmfHeadFormat>().unwrap_err().contains("packed"));
        assert_eq!([Bf16, Fp8, Packed].map(GlmfHeadFormat::setting), ["false", "true", "packed"]);
        assert_eq!([Bf16, Fp8, Packed].map(GlmfHeadFormat::name), ["BF16", "FP8", "exact 12-bit"]);
        assert_eq!("auto".parse::<GlmfHeadFormat>(), Ok(GlmfHeadFormat::Auto));
        assert_eq!(GlmfHeadFormat::default(), GlmfHeadFormat::Auto);
        assert_eq!(GlmfRepresentation::default().head, GlmfHeadFormat::Auto);
        for (native, ranks, resolved) in [(false, 1, Bf16), (false, 2, Bf16), (true, 1, Bf16), (true, 2, Bf16)] {
            assert_eq!(GlmfHeadFormat::Auto.resolve(native, ranks), resolved);
        }
        let (vocab, hidden) = (154_880, 4096);
        assert_eq!([Bf16, Fp8, Packed].map(|head| head.resident_bytes(vocab, hidden)),
            [1_268_776_960, 654_213_120, 956_074_752]);
        assert_eq!([Bf16, Fp8, Packed].map(|head| head.load_source_bytes(vocab, hidden)), [0, 0, 1_268_776_960]);
        for sms in [48, 170, 188] {
            assert_eq!([Bf16, Fp8, Packed].map(|head| head.scratch_bytes(256, vocab, hidden, sms)), [0, 0, 524_288]);
        }
    }

    #[test]
    fn owned_mla_inventory_preserves_kda_ffn_and_global_operands() {
        use crate::placement::AttentionPlacement::{Context, Heads, Layers};
        use crate::plan::testing::{glm5_flash_config, glm5_flash_tensors, write_snapshot};
        let dir = tempfile::tempdir().unwrap();
        let config = glm5_flash_config(6);
        write_snapshot(dir.path(), &config, &glm5_flash_tensors(&config), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let heads = resident_weights(&checkpoint, &cfg, 6, 2, Default::default()).unwrap();
        let context = resident_weights_placed(&checkpoint, &cfg, 6, 2, Default::default(), Context, 0).unwrap();
        assert_eq!(heads.iter().map(|r| (r.weights, r.embedding)).collect::<Vec<_>>(),
            context.iter().map(|r| (r.weights, r.embedding)).collect::<Vec<_>>());
        let owned0 = resident_weights_placed(&checkpoint, &cfg, 6, 2, Default::default(), Layers, 0).unwrap();
        let owned1 = resident_weights_placed(&checkpoint, &cfg, 6, 2, Default::default(), Layers, 1).unwrap();
        let mut previous = [0; 2];
        for layers in 0..=6 {
            // Hold head precision fixed while checking only attention ownership.
            let single = resident_weights(&checkpoint, &cfg, layers, 1,
                GlmfRepresentation { head: GlmfHeadFormat::Fp8, ..Default::default() }).unwrap();
            let split = resident_weights(&checkpoint, &cfg, layers, 2, Default::default()).unwrap();
            for first in [0, 1] {
                let owned = resident_weights_placed(&checkpoint, &cfg, layers, 2, Default::default(), Layers, first).unwrap();
                assert_eq!(owned.iter().map(|r| r.embedding).collect::<Vec<_>>(), split.iter().map(|r| r.embedding).collect::<Vec<_>>());
                let total = |ranks: &[GlmfResidentRank]| ranks.iter().map(|r| r.weights).sum::<u64>();
                let saving = total(&split) - total(&owned);
                if layers > 0 && cfg.attention[layers - 1] == GlmNextAttention::Kda {
                    assert_eq!(saving, previous[first], "KDA must remain unchanged");
                }
                if layers > 0 && cfg.attention[layers - 1] == GlmNextAttention::Mla {
                    assert!(saving > previous[first], "replicated MLA operands are removed");
                }
                previous[first] = saving;
                assert!(total(&owned) >= single[0].weights, "FFN metadata remains replicated");
            }
        }
        assert_eq!(owned0[0].weights - owned1[0].weights, owned1[1].weights - owned0[1].weights);
        for (ranks, mode, first) in [(1, Layers, 0), (2, Context, 2), (0, Heads, 0)] {
            assert!(resident_weights_placed(&checkpoint, &cfg, 6, ranks, Default::default(), mode, first).is_err());
        }
    }

    #[test]
    #[ignore = "requires published native Flash checkpoint headers; CPU only"]
    fn published_native_residency_consumes_owned_plans_without_mtp_or_duplicate_vocabulary() {
        use crate::plan::exlr8::{dense::{DensePlan, DenseRecord, DenseRole, DenseSelection}, residency::{ExpertSet, MetadataInput}};
        use std::collections::BTreeMap;
        let snapshot = std::path::PathBuf::from(std::env::var_os("CUTEAFD_EXLR8_FLASH_SNAPSHOT").expect("Flash snapshot"));
        let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json")).unwrap();
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let selection = DenseSelection::from_hf(&config).unwrap();
        let records = DenseRecord::inventory(&snapshot).unwrap().into_iter().filter(|record|
            selection.role(&record.tensor, cfg.layers).unwrap() != DenseRole::Excluded).collect::<Vec<_>>();
        let metadata = MetadataInput::open(&snapshot, "k4").unwrap();
        let routed = cfg.dense.iter().enumerate().filter_map(|(layer, dense)| (!dense).then_some(layer)).collect::<Vec<_>>();
        let costs = crate::plan::exlr8::routed_expert_costs(&metadata, &routed).unwrap();
        assert_eq!(costs.len(), routed.len());
        for (&layer, cost) in routed.iter().zip(&costs) {
            assert_eq!(cost.whole.resident, metadata.plan(layer, 1, 0, ExpertSet::Routed).unwrap().resident_bytes as u64);
            for rank in 0..2 {
                assert_eq!(cost.half[rank].resident, metadata.plan(layer, 2, rank, ExpertSet::Routed).unwrap().resident_bytes as u64);
                assert_eq!(cost.half[rank].staging, 0);
            }
            assert!(cost.tp2 && cost.spark_ok);
        }
        for invalid in [vec![0], vec![cfg.layers], vec![3, 3], vec![4, 3]] {
            assert!(crate::plan::exlr8::routed_expert_costs(&metadata, &invalid).is_err());
            assert!(crate::plan::exlr8::shared_expert_resident(&metadata, &invalid, 1).is_err());
        }
        assert!(crate::plan::exlr8::shared_expert_resident(&metadata, &routed, 0).is_err());
        for world in [1, 2] {
            // Owner facts come from outside admission. No Flash policy runs here.
            for owners in [BTreeMap::new(), if world == 2 { cfg.attention.iter().enumerate()
                .filter(|(_, kind)| **kind == GlmNextAttention::Mla).map(|(layer, _)| (layer, 1)).collect()
                } else { BTreeMap::new() }] {
                let plans = (0..world).map(|rank| DensePlan::for_rank_owned(&records, world, rank, &owners).unwrap())
                    .collect::<Vec<_>>();
                let layers = cfg.dense.iter().enumerate().filter_map(|(layer, dense)| (!dense).then_some(layer)).collect::<Vec<_>>();
                let shared = crate::plan::exlr8::shared_expert_resident(&metadata, &layers, world).unwrap();
                for (rank, &bytes) in shared.iter().enumerate() {
                    let expected = layers.iter().map(|&layer|
                        metadata.plan(layer, world, rank, ExpertSet::Shared).unwrap().resident_bytes as u64).sum::<u64>();
                    assert_eq!(bytes, expected);
                }
                let resident = native_resident(&selection, cfg.layers, &plans, &shared).unwrap();
                for (rank, result) in resident.iter().enumerate() {
                    assert_eq!(result.weights + result.embedding, plans[rank].resident_bytes as u64 + shared[rank]);
                    assert!(!plans[rank].tensors.iter().any(|slice| slice.tensor.contains(&format!("layers.{}.", cfg.layers))));
                    if rank == 0 { assert!(result.embedding > 0); } else { assert_eq!(result.embedding, 0); }
                }
                assert!(native_resident(&selection, cfg.layers, &plans, &[]).is_err());
                let mut invalid = plans.clone(); invalid[0].rank = world;
                assert!(native_resident(&selection, cfg.layers, &invalid, &shared).is_err());
                invalid = plans.clone(); invalid[0].resident_bytes += 1;
                assert!(native_resident(&selection, cfg.layers, &invalid, &shared).is_err());
            }
        }
    }

    #[test]
    fn real_exl3_and_fp8_headers_match_rc3_uploaded_operands() {
        let home = std::env::var_os("HF_HOME").map(std::path::PathBuf::from)
            .unwrap_or_else(|| "/mnt/sparknest/hf-home".into());
        for model in ["wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3.25-Core-FP8", "zai-org/GLM-5.3-Flash"] {
            let Ok(resolved) = crate::resolve_snapshot_at_revision(model, Some(&home), None) else { continue };
            let Some(path) = resolved.snapshot_path else { continue };
            let checkpoint = Checkpoint::open(&path).unwrap();
            let cfg = GlmNextConfig::from_hf(&checkpoint.config).unwrap();
            // Original retains all BF16 KDA projections and the BF16 shared head.
            let single = resident_weights(&checkpoint, &cfg, cfg.layers, 1, Default::default()).unwrap();
            assert_eq!((single[0].embedding, single[0].weights),
                (1_268_776_960, 14_034_716_416), "{model}");
            let packed = GlmfRepresentation { head: GlmfHeadFormat::Packed, ..Default::default() };
            let packed = resident_weights(&checkpoint, &cfg, cfg.layers, 1, packed).unwrap();
            assert_eq!(packed[0].weights, 14_034_716_416 - 1_268_776_960 + 956_074_752, "{model}");
            let split = resident_weights(&checkpoint, &cfg, cfg.layers, 2, Default::default()).unwrap();
            assert_eq!((split[0].weights, split[1].weights),
                (7_936_512_768, 6_568_588_800 + router_replica_bytes(&checkpoint, &cfg, cfg.layers).unwrap()), "{model}");
            assert_eq!(split[1].embedding, 0);
        }
    }
}
