//! Native source operands and CPU head arithmetic for the bundled MiniMax drafter.
use super::config::MiniMaxM3DsparkConfig;
use crate::formats::native_dense::NativeDenseLayout;
use crate::plan::checkpoint::Checkpoint;
use crate::serving_capacity::CacheGeometryError;
use cuteafd_core::dense_attention::{GqaGeometry, NormWeight};
use cuteafd_core::{DType, FfnActivation};
use serde_json::Value;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const PREFIX: &str = "language_model.model.dspark";
pub const TARGET_LAYERS: [usize; 5] = [3, 17, 31, 45, 59];
pub const BORROWED_EMBEDDING: &str = "language_model.model.embed_tokens.weight";
pub const BORROWED_HEAD: &str = "language_model.lm_head.weight";

#[derive(Debug, Error)]
pub enum DsparkError {
    #[error("MiniMax DSpark configuration: {0}")]
    Config(String),
    #[error("MiniMax DSpark checkpoint: {0:#}")]
    Checkpoint(#[source] anyhow::Error),
    #[error(transparent)]
    Headers(#[from] CacheGeometryError),
    #[error("MiniMax DSpark unexpected tensor {0}")]
    UnexpectedTensor(String),
    #[error("MiniMax DSpark head arithmetic: {0}")]
    Arithmetic(&'static str),
    #[error("MiniMax DSpark memory: {0}")]
    Memory(&'static str),
}

#[derive(Debug, Clone)]
pub struct DsparkContract {
    pub snapshot: PathBuf,
    pub config: MiniMaxM3DsparkConfig,
    pub weights: NativeDenseLayout,
    pub hidden: usize,
    pub vocab_size: usize,
    pub geometry: GqaGeometry,
    pub norm_weight: NormWeight,
    pub norm_eps: f32,
    pub rope_theta: f64,
    pub activation: FfnActivation,
    /// The MiniMax DSpark architecture defaults to noncausal draft blocks.
    pub causal: bool,
    pub sample_from_anchor: bool,
}

/// Drafter context rings when none are requested: a ring is assigned to a
/// request for its life, and the draft policy stops paying past a few
/// concurrent drafting sequences (expert fan-in grows with drafted rows).
pub fn default_ring_slots(sequences: usize) -> usize { sequences.min(8) }

/// The drafter's own allocations; target step scratch and the borrowed head
/// and embedding remain in the target ledger. `slots` counts context rings,
/// which the server maps onto scheduler slots (at most one per request).
#[derive(Debug, Clone)]
pub struct DraftMemory {
    pub buffers: std::collections::BTreeMap<&'static str, usize>,
    pub weights_bytes: u64,
    pub scratch_bytes: u64,
    pub ring_bytes: u64,
    pub ring_plane_bytes: usize,
    pub bytes: u64,
    pub sequences: usize,
    pub slots: usize,
    pub rows: usize,
}

impl DraftMemory {
    pub fn new(contract: &DsparkContract, slots: usize, rows: usize, sms: usize) -> Result<Self, DsparkError> {
        if !(1..=64).contains(&slots) || !(7..=4096).contains(&rows) || !(1..=(i32::MAX / 2) as usize).contains(&sms) {
            return Err(DsparkError::Memory("requires 1..64 slots, 7..4096 rows and a positive SM count"));
        }
        let cfg = &contract.config;
        let g = contract.geometry;
        let sequences = slots.min(32).min(rows / cfg.block);
        let r = sequences * cfg.block;
        // Native draft_attention.cu allocates the larger WMMA/scalar layout.
        // Each 32-query slice owns one partial output and two softmax scalars.
        let slices = (g.query_heads / g.kv_heads * cfg.block).div_ceil(32);
        let max_keys = cfg.sliding_window + cfg.block;
        let partitions = max_keys.div_ceil(32).min((2 * sms).div_ceil(g.kv_heads * slices));
        let scalar_rows = if g.qk_dim == 64 { 32 } else { 16 };
        let partial_rows = (slices * partitions * 32).max(max_keys.min(1024 + 16).div_ceil(128) * scalar_rows);
        let attention_workspace = sequences * g.kv_heads * partial_rows * (g.qk_dim + 2) * 4;
        // glmf_dspark.cu: 296 FP32 maxima/uint32 ids per sequence plus
        // one uint32 scratch token for every drafted row.
        let markov_workspace = sequences * 296 * 8 + sequences * cfg.block * 4;
        let h = contract.hidden;
        let buffers = std::collections::BTreeMap::from([
            ("taps", rows * TARGET_LAYERS.len() * h * 2), ("tap", rows * h * 2),
            ("input", r * h * 2), ("ids", r * 4), ("tables", 3 * sequences * 4),
            ("anchors", sequences * 4), ("tokens", r * 4), ("confidence", r * 4),
            ("markov_norms", contract.vocab_size * 4), ("markov_workspace", markov_workspace),
            ("attention_workspace", attention_workspace),
        ]).into_iter().map(|(name, bytes)| (name, bytes.max(256))).collect::<std::collections::BTreeMap<_, _>>();
        let ring_plane_bytes = slots * cfg.sliding_window * g.kv_heads * g.qk_dim * 2;
        let ring_bytes = (cfg.layers * 2 * ring_plane_bytes) as u64;
        let scratch_bytes = buffers.values().map(|&n| n as u64).sum::<u64>();
        // Every fused matrix already exceeds the allocation floor; only the
        // confidence bias pads from two bytes to 256.
        let weights_bytes = contract.weights.resident_bytes;
        let bytes = weights_bytes.checked_add(scratch_bytes).and_then(|n| n.checked_add(ring_bytes))
            .ok_or(DsparkError::Memory("allocation sum overflow"))?;
        Ok(Self { buffers, weights_bytes, scratch_bytes, ring_bytes, ring_plane_bytes, bytes, sequences, slots, rows })
    }
}

fn require(value: &Value, key: &str, expected: Value) -> Result<(), DsparkError> {
    if value[key] == expected { return Ok(()); }
    Err(DsparkError::Config(format!("{key}: expected {expected}, found {}", value[key])))
}

fn explicit_bool(value: &Value, key: &str) -> Result<Option<bool>, DsparkError> {
    let Some(v) = value.get(key) else { return Ok(None); };
    v.as_bool().map(Some).ok_or_else(|| DsparkError::Config(format!("{key} must be boolean")))
}

fn explicit_policy(root: &Value, keys: &[(&str, bool)]) -> Result<Option<bool>, DsparkError> {
    let text = root.get("text_config").filter(|v| v.is_object()).unwrap_or(root);
    let mut policy = None;
    for object in [root, text, &text["dflash_config"], &text["dspark_config"]] {
        for (key, inverted) in keys {
            if let Some(value) = explicit_bool(object, key)? {
                let value = value ^ *inverted;
                if policy.is_some_and(|prior| prior != value) {
                    return Err(DsparkError::Config(format!("conflicting explicit {key} policy")));
                }
                policy = Some(value);
            }
        }
    }
    Ok(policy)
}

impl DsparkContract {
    /// The target owns the embedding and LM head; this opens only `dspark/`.
    pub fn open(target_snapshot: &Path) -> Result<Self, DsparkError> {
        let checkpoint = Checkpoint::open(&target_snapshot.join("dspark")).map_err(DsparkError::Checkpoint)?;
        Self::from_checkpoint(&checkpoint)
    }

    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self, DsparkError> {
        let config = MiniMaxM3DsparkConfig::from_hf(&checkpoint.config)
            .map_err(|error| DsparkError::Config(format!("{error:#}")))?;
        Self::release_geometry(&checkpoint.config, &config)?;
        let declarations = declarations();
        let expected: std::collections::BTreeSet<_> = declarations.iter().map(|(name, ..)| name.as_str()).collect();
        for name in checkpoint.weight_map.keys() {
            if !expected.contains(name.as_str()) { return Err(DsparkError::UnexpectedTensor(name.clone())); }
        }
        let weights = NativeDenseLayout::from_headers(checkpoint, declarations)?;
        Ok(Self {
            snapshot: checkpoint.snapshot.clone(), config, weights, hidden: 6144, vocab_size: 200064,
            geometry: GqaGeometry { query_heads: 64, kv_heads: 4, qk_dim: 128, value_dim: 128, rotary_dim: 64 },
            norm_weight: NormWeight::ZeroCentered, norm_eps: 1e-6, rope_theta: 1e7,
            activation: FfnActivation::SwigluOai { alpha: 1.702, limit: 7.0 },
            // DSparkMiniMaxDraftModel supplies causal=false; its DSpark worker
            // defaults sample_from_anchor=true when the release omits both.
            causal: explicit_policy(&checkpoint.config, &[("is_causal", false), ("causal", false), ("sliding_window_non_causal", true)])?.unwrap_or(false),
            sample_from_anchor: explicit_policy(&checkpoint.config, &[("sample_from_anchor", false)])?.unwrap_or(true),
        })
    }

    pub fn release_geometry(root: &Value, config: &MiniMaxM3DsparkConfig) -> Result<(), DsparkError> {
        if (config.layers, config.block, config.markov_rank, config.mask_token, config.sliding_window, config.ffn_intermediate)
            != (5, 7, 256, 200058, 4096, 12288)
            || config.target_layers != TARGET_LAYERS || !config.confidence_head {
            return Err(DsparkError::Config("drafter geometry differs from the bundled release".into()));
        }
        let v = root.get("text_config").filter(|v| v.is_object()).unwrap_or(root);
        for (key, expected) in [("hidden_size", 6144), ("vocab_size", 200064), ("num_attention_heads", 64),
            ("num_key_value_heads", 4), ("head_dim", 128), ("rotary_dim", 64), ("dense_intermediate_size", 12288)] {
            require(v, key, expected.into())?;
        }
        for (key, expected) in [("rms_norm_eps", 1e-6), ("rope_theta", 1e7), ("partial_rotary_factor", 0.5),
            ("swiglu_alpha", 1.702), ("swiglu_limit", 7.0)] {
            if v[key].as_f64() != Some(expected) { return Err(DsparkError::Config(format!("{key}: expected {expected}, found {}", v[key]))); }
        }
        for key in ["use_gemma_norm", "use_qk_norm", "confidence_head_with_markov"] { require(v, key, true.into())?; }
        for (key, expected) in [("qk_norm_type", "per_head"), ("hidden_act", "swigluoai"), ("dspark_aux_gemm_precision", "bf16")] {
            require(v, key, expected.into())?;
        }
        require(v, "attention_output_gate", false.into())?;
        let quant = &root["quantization_config"];
        require(quant, "quant_method", "mxfp8".into())?;
        require(quant, "activation_scheme", "dynamic".into())?;
        require(quant, "weight_block_size", serde_json::json!([1, 32]))?;
        Ok(())
    }
}

/// Checkpoint-native operands; the borrowed target embedding and LM head are absent.
pub fn declarations() -> Vec<(String, Vec<usize>, DType)> {
    let (h, rank, vocab, d) = (6144, 256, 200064, PREFIX);
    let mut out = vec![
        (format!("{d}.fc.weight"), vec![h, TARGET_LAYERS.len() * h], DType::Bf16),
        (format!("{d}.final_norm.weight"), vec![h], DType::Bf16),
        (format!("{d}.hidden_norm.weight"), vec![h], DType::Bf16),
        (format!("{d}.confidence_head.proj.weight"), vec![1, h + rank], DType::Bf16),
        (format!("{d}.confidence_head.proj.bias"), vec![1], DType::Bf16),
    ];
    for w in ["markov_w1", "markov_w2"] { out.push((format!("{d}.markov_head.{w}.weight"), vec![vocab, rank], DType::Bf16)); }
    for layer in 0..5 {
        let base = format!("{d}.layers.{layer}.decoder_layer");
        for norm in ["input_layernorm", "post_attention_layernorm"] { out.push((format!("{base}.{norm}.weight"), vec![h], DType::Bf16)); }
        for norm in ["q_norm", "k_norm"] { out.push((format!("{base}.self_attn.{norm}.weight"), vec![128], DType::Bf16)); }
        for (stem, n, k) in [("self_attn.q_proj", 8192, h), ("self_attn.k_proj", 512, h),
            ("self_attn.v_proj", 512, h), ("self_attn.o_proj", h, 8192),
            ("mlp.gate_proj", 12288, h), ("mlp.up_proj", 12288, h), ("mlp.down_proj", h, 12288)] {
            out.push((format!("{base}.{stem}.weight"), vec![n, k], DType::F8E4M3));
            out.push((format!("{base}.{stem}.weight_scale_inv"), vec![n, k / 32], DType::U8));
        }
    }
    out
}

/// One slot's logits plus the vanilla Markov bias, without a scale or softmax.
pub fn markov_logits(logits: &[f32], previous: &[f32], w2: &[f32]) -> Result<Vec<f32>, DsparkError> {
    if logits.is_empty() || previous.is_empty() || logits.len().checked_mul(previous.len()) != Some(w2.len()) {
        return Err(DsparkError::Arithmetic("Markov matrix shape"));
    }
    Ok(logits.iter().zip(w2.chunks_exact(previous.len())).map(|(logit, row)|
        logit + previous.iter().zip(row).map(|(a, b)| a * b).sum::<f32>()).collect())
}

/// Slot zero conditions on the anchor; later slots condition on the prior draft.
pub fn markov_greedy(logits: &[f32], vocab: usize, rank: usize, w1: &[f32], w2: &[f32], anchor: u32)
    -> Result<Vec<u32>, DsparkError> {
    if vocab == 0 || rank == 0 || vocab > u32::MAX as usize || logits.len() % vocab != 0
        || vocab.checked_mul(rank) != Some(w1.len()) || w1.len() != w2.len() || anchor as usize >= vocab {
        return Err(DsparkError::Arithmetic("Markov block shape or anchor"));
    }
    let mut previous = anchor as usize;
    let mut tokens = Vec::with_capacity(logits.len() / vocab);
    for row in logits.chunks_exact(vocab) {
        let scores = markov_logits(row, &w1[previous * rank..(previous + 1) * rank], w2)?;
        // Strict comparison preserves the first vocabulary id on ties.
        previous = scores.iter().enumerate().fold(0, |best, (id, score)| if *score > scores[best] { id } else { best });
        tokens.push(previous as u32);
    }
    Ok(tokens)
}

/// The hidden input is final-normalized; Markov features use the previous token.
pub fn confidence(hidden: &[f32], previous: &[f32], weight: &[f32], bias: f32) -> Result<f32, DsparkError> {
    if hidden.is_empty() || previous.is_empty() || hidden.len().checked_add(previous.len()) != Some(weight.len()) {
        return Err(DsparkError::Arithmetic("confidence vector shape"));
    }
    let logit = hidden.iter().chain(previous).zip(weight).map(|(x, w)| x * w).sum::<f32>() + bias;
    Ok(if logit >= 0.0 { 1.0 / (1.0 + (-logit).exp()) } else { let exp = logit.exp(); exp / (1.0 + exp) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::minimax_m3::config::tests::release_dspark;
    use crate::plan::testing;

    fn checkpoint() -> (tempfile::TempDir, Checkpoint) {
        let dir = tempfile::tempdir().unwrap();
        testing::write_snapshot(dir.path(), &release_dspark(), &testing::minimax_m3_dspark_tensors(), None);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        (dir, checkpoint)
    }

    #[test]
    fn native_release_headers() {
        let (_dir, mut checkpoint) = checkpoint();
        if let Some(snapshot) = std::env::var_os("CUTEAFD_MINIMAX_HEADER_SNAPSHOT") {
            checkpoint = Checkpoint::open(&PathBuf::from(snapshot).join("dspark")).unwrap();
        }
        let contract = DsparkContract::from_checkpoint(&checkpoint).unwrap();
        assert_eq!(contract.weights.tensors.len(), 97);
        assert_eq!(contract.weights.tensors.iter().map(|t| t.meta.byte_length).sum::<u64>(), 2_301_852_674);
        assert_eq!(contract.weights.tensors.iter().filter(|t| t.meta.dtype == DType::F8E4M3).count(), 35);
        assert_eq!(contract.weights.tensors.iter().filter(|t| t.meta.dtype == DType::U8).count(), 35);
        assert!(contract.weights.tensors.iter().all(|t| t.meta.name != BORROWED_EMBEDDING && t.meta.name != BORROWED_HEAD));
        assert_eq!(contract.config.target_layers, TARGET_LAYERS);
        assert_eq!((contract.geometry.query_heads, contract.geometry.kv_heads, contract.geometry.qk_dim,
            contract.geometry.rotary_dim, contract.norm_weight), (64, 4, 128, 64, NormWeight::ZeroCentered));
        assert_eq!(contract.activation, FfnActivation::SwigluOai { alpha: 1.702, limit: 7.0 });
        assert!(!contract.causal);
        assert!(contract.sample_from_anchor);
    }

    #[test]
    fn draft_memory_matches_native_slices_padding_and_ring_planes() {
        let (_dir, checkpoint) = checkpoint();
        let contract = DsparkContract::from_checkpoint(&checkpoint).unwrap();
        let layout = DraftMemory::new(&contract, 16, 256, 188).unwrap();
        assert_eq!((layout.sequences, layout.slots, layout.rows), (16, 16, 256));
        assert_eq!(layout.weights_bytes, 2_301_852_928);
        assert_eq!(layout.buffers["attention_workspace"], 16 * 4 * (4 * 24 * 32) * 130 * 4);
        assert_eq!(layout.buffers["markov_workspace"], 16 * (296 * 8 + 7 * 4));
        assert_eq!(layout.buffers["taps"], 256 * 5 * 6144 * 2);
        assert_eq!(layout.buffers["tables"], 256);
        assert_eq!(layout.buffers["anchors"], 256);
        assert_eq!(layout.ring_plane_bytes, 16 * 4096 * 512 * 2);
        assert_eq!(layout.ring_bytes, 10 * layout.ring_plane_bytes as u64);
        assert_eq!(layout.bytes, layout.weights_bytes + layout.scratch_bytes + layout.ring_bytes);
        assert_eq!(layout.scratch_bytes, layout.buffers.values().map(|&b| b as u64).sum::<u64>());
        let smaller = DraftMemory::new(&contract, 16, 256, 170).unwrap();
        assert_eq!(smaller.buffers["attention_workspace"], 16 * 4 * (4 * 22 * 32) * 130 * 4);
        assert_eq!(layout.bytes - smaller.bytes, 8_519_680);
        let scalar = DraftMemory::new(&contract, 1, 7, 1).unwrap();
        assert_eq!(scalar.buffers["attention_workspace"], 4 * (9 * 16) * 130 * 4);
        assert_eq!(DraftMemory::new(&contract, 64, 4096, 188).unwrap().sequences, 32);
        assert_eq!(DraftMemory::new(&contract, 64, 13, 188).unwrap().sequences, 1);
        assert!(DraftMemory::new(&contract, 16, 6, 188).is_err());
        assert!(DraftMemory::new(&contract, 0, 256, 188).is_err());
        assert!(DraftMemory::new(&contract, 16, 256, 0).is_err());
    }

    #[test]
    fn rejects_wrong_weight_dtype_shape_and_extent() {
        let (_dir, mut checkpoint) = checkpoint();
        let at = checkpoint.tensors.iter().position(|t| t.meta.name.ends_with("layers.0.decoder_layer.self_attn.q_proj.weight")).unwrap();
        let original = checkpoint.tensors[at].clone();
        checkpoint.tensors[at].meta.dtype = DType::Bf16;
        assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::Headers(_))));
        checkpoint.tensors[at] = original.clone();
        checkpoint.tensors[at].meta.shape[0] /= 2;
        assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::Headers(_))));
        checkpoint.tensors[at] = original;
        checkpoint.tensors[at].meta.byte_length -= 1;
        assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::Headers(_))));
    }

    #[test]
    fn rejects_converted_scales_missing_weights_and_target_copies() {
        let (_dir, mut checkpoint) = checkpoint();
        let at = checkpoint.tensors.iter().position(|t| t.meta.name.ends_with("layers.0.decoder_layer.self_attn.q_proj.weight_scale_inv")).unwrap();
        checkpoint.tensors[at].meta.dtype = DType::F32;
        assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::Headers(_))));
        checkpoint.tensors.remove(at);
        assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::Headers(_))));
        checkpoint.weight_map.insert(BORROWED_HEAD.into(), "model.safetensors".into());
        assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::UnexpectedTensor(_))));
    }

    #[test]
    fn rejects_geometry_and_numerics_drift() {
        let (_dir, mut checkpoint) = checkpoint();
        let original = checkpoint.config.clone();
        for (key, value) in [("num_attention_heads", 32.into()), ("rotary_dim", 128.into()),
            ("use_gemma_norm", false.into()), ("qk_norm_type", "all_heads".into()),
            ("confidence_head_with_markov", false.into()), ("dspark_block_size", 8.into()),
            ("dspark_target_layer_ids", serde_json::json!([2, 16, 30, 44, 58]))] {
            checkpoint.config = original.clone();
            checkpoint.config["text_config"][key] = value;
            assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::Config(_))), "{key}");
        }
        checkpoint.config = original;
        checkpoint.config["quantization_config"]["weight_block_size"] = serde_json::json!([128, 128]);
        assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::Config(_))));
    }

    #[test]
    fn retains_explicit_mask_and_anchor_policy() {
        let (_dir, mut checkpoint) = checkpoint();
        checkpoint.config["text_config"]["dflash_config"]["sliding_window_non_causal"] = true.into();
        checkpoint.config["text_config"]["dspark_config"]["sample_from_anchor"] = true.into();
        let contract = DsparkContract::from_checkpoint(&checkpoint).unwrap();
        assert_eq!((contract.causal, contract.sample_from_anchor), (false, true));
        checkpoint.config["text_config"]["is_causal"] = true.into();
        assert!(matches!(DsparkContract::from_checkpoint(&checkpoint), Err(DsparkError::Config(_))));
    }

    #[test]
    fn markov_bias_matches_scalar_reference() {
        let (logits, previous, w2) = ([0.5, -1.0, 0.25], [2.0, -0.5], [0.25, 1.0, -2.0, 4.0, 1.0, -0.5]);
        let got = markov_logits(&logits, &previous, &w2).unwrap();
        assert_eq!(got, [0.5, -7.0, 2.5]);
        for (v, actual) in got.into_iter().enumerate() {
            let expected = logits[v] as f64 + (0..2).map(|r| previous[r] as f64 * w2[v * 2 + r] as f64).sum::<f64>();
            assert!((actual as f64 - expected).abs() < 1e-7);
        }
        assert!(markov_logits(&logits, &[], &w2).is_err());
    }

    #[test]
    fn markov_block_conditions_each_successor_on_its_predecessor() {
        let w1 = [1.0, 2.0, -1.0];
        let w2 = [0.0, 1.0, -1.0];
        let logits = [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0];
        assert_eq!(markov_greedy(&logits, 3, 1, &w1, &w2, 0).unwrap(), [1, 2, 2]);
        assert_eq!(markov_greedy(&[0.0; 3], 3, 1, &[0.0; 3], &w2, 2).unwrap(), [0]);
        assert!(markov_greedy(&logits, 3, 1, &w1, &w2, 3).is_err());
    }

    #[test]
    fn confidence_matches_scalar_reference_and_uses_markov_features() {
        let (hidden, previous, weight, bias) = ([2.0, -1.0], [0.5, -2.0], [0.25, 1.0, 2.0, -0.5], -0.75);
        let expected = 1.0 / (1.0 + (-0.75f64).exp());
        assert!((confidence(&hidden, &previous, &weight, bias).unwrap() as f64 - expected).abs() < 1e-7);
        assert_eq!(confidence(&hidden, &previous, &[0.0; 4], 0.0).unwrap(), 0.5);
        assert_eq!(confidence(&hidden, &previous, &[0.0; 4], -1000.0).unwrap(), 0.0);
        assert_eq!(confidence(&hidden, &previous, &[0.0; 4], 1000.0).unwrap(), 1.0);
        assert!(confidence(&hidden, &previous, &[0.0; 3], bias).is_err());
    }
}
