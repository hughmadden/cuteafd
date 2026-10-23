//! MiniMax M3 model arguments from `config.json` (`MiniMaxM3SparseForConditionalGeneration`,
//! `model_type=minimax_m3_vl`), the reader shared by the planner and runtime.
//!
//! Semantics follow transformers' `modular_minimax_m3_vl.py` (the checkpoint ships
//! only its configuration code):
//! - Attention is GQA with per-head Gemma QK norms (`x * (1 + w)`) and NeoX RoPE on
//!   the first `rotary_dim` channels of each head.
//! - `sparse_attention_config.sparse_attention_freq[layer]` selects MiniMax Sparse
//!   Attention (MSA): a lightning indexer (`sparse_num_index_heads` heads, one per KV
//!   head, against one shared `sparse_index_dim` key) scores every earlier token,
//!   max-pools the scores into `sparse_block_size` blocks, forces the
//!   `sparse_local_block` blocks ending at the query's own block, and keeps the top
//!   `sparse_topk_blocks` per query row and KV head. The index branch has no values.
//! - `moe_layer_freq[layer]` selects the MoE (sigmoid router plus a selection-only
//!   bias, renormalized, times `routed_scaling_factor`, plus one shared expert);
//!   other layers run a dense MLP of `dense_intermediate_size`.
//! - Every FFN is SwiGLU-OAI: `gate.clamp(max=L) * sigmoid(alpha * gate.clamp(max=L))
//!   * (up.clamp(-L, L) + 1)`.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// The text model.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxM3Config {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Leading channels of each query/key head that RoPE rotates (NeoX halves).
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// `moe[layer]`: a routed MoE plus shared expert instead of the dense MLP.
    pub moe: Vec<bool>,
    pub dense_intermediate: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub experts: usize,
    pub topk: usize,
    pub routed_scale: f64,
    pub swiglu_alpha: f64,
    pub swiglu_limit: f64,
    /// `sparse[layer]`: MSA (every release layer); otherwise full causal GQA.
    pub sparse: Vec<bool>,
    pub msa: MsaConfig,
    pub max_position: usize,
    pub tie_embeddings: bool,
    pub bos: u32,
    pub eos: Vec<u32>,
}

/// MiniMax Sparse Attention selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsaConfig {
    /// Index query heads, one per KV head (each KV group selects its own blocks).
    pub index_heads: usize,
    pub index_dim: usize,
    /// Keys pooled into one scored block (the max over the block's token scores).
    pub block: usize,
    /// Blocks each query row and KV head attends, the local blocks included.
    pub topk_blocks: usize,
    /// Blocks ending at the query's own block that are always selected.
    pub local_blocks: usize,
}

impl MsaConfig {
    /// Keys one query row of one KV head attends once the context holds
    /// `topk_blocks` whole blocks.
    pub fn attended_keys(&self) -> usize { self.topk_blocks * self.block }
}

/// The bundled vision tower (CLIP encoder with 3D RoPE, 2x2 patch merge, projector).
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxM3VisionConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub patch: usize,
    pub image_size: usize,
    pub rope_theta: f64,
    pub layer_norm_eps: f64,
    pub spatial_merge: usize,
    pub temporal_patch: usize,
    pub max_segment_frames: usize,
    /// Text width the projector emits (the text hidden size).
    pub projection: usize,
    pub projector_hidden: usize,
    pub image_token: u32,
    pub video_token: u32,
    /// `(height, width)` tilings `dynamic_res` chooses from.
    pub grid_pinpoints: Vec<(usize, usize)>,
}

impl MiniMaxM3VisionConfig {
    /// Patches of one `image_size` tile before the merge.
    pub fn tile_patches(&self) -> usize { (self.image_size / self.patch).pow(2) }
    /// Text tokens one tile becomes after the spatial merge.
    pub fn tile_tokens(&self) -> usize { self.tile_patches() / (self.spatial_merge * self.spatial_merge) }
}

/// The bundled DSpark drafter (`dspark/config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxM3DsparkConfig {
    pub layers: usize,
    /// Tokens drafted per block (the anchor's successors).
    pub block: usize,
    pub markov_rank: usize,
    pub mask_token: u32,
    /// Target layers whose outputs the drafter reads, ascending.
    pub target_layers: Vec<usize>,
    pub sliding_window: usize,
    pub ffn_intermediate: usize,
    pub confidence_head: bool,
}

fn int(value: &Value, key: &str) -> Result<usize> {
    let n = value[key].as_u64().with_context(|| format!("MiniMax M3 config lacks {key}"))?;
    ensure!(n > 0 && n <= 1 << 24, "invalid MiniMax M3 {key}: {n}");
    Ok(n as usize)
}

fn float(value: &Value, key: &str) -> Result<f64> {
    value[key].as_f64().filter(|x| x.is_finite()).with_context(|| format!("MiniMax M3 config lacks {key}"))
}

fn flags(value: &Value, key: &str, layers: usize) -> Result<Vec<bool>> {
    let list = value[key].as_array().with_context(|| format!("MiniMax M3 config lacks {key}"))?;
    ensure!(list.len() == layers, "{key} has {} entries for {layers} layers", list.len());
    list.iter().map(|x| match x.as_u64() {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => bail!("{key} entry {x} is not 0 or 1"),
    }).collect()
}

fn token_ids(value: &Value) -> Vec<u32> {
    match value {
        Value::Array(ids) => ids.iter().filter_map(|id| id.as_u64().map(|id| id as u32)).collect(),
        other => other.as_u64().map(|id| vec![id as u32]).unwrap_or_default(),
    }
}

impl MiniMaxM3Config {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        ensure!(root["model_type"] == "minimax_m3_vl", "MiniMax M3 requires model_type=minimax_m3_vl, found {}", root["model_type"]);
        let v = root.get("text_config").filter(|v| v.is_object()).context("MiniMax M3 requires text_config")?;
        let layers = int(v, "num_hidden_layers")?;
        let (heads, kv_heads, head_dim) = (int(v, "num_attention_heads")?, int(v, "num_key_value_heads")?, int(v, "head_dim")?);
        ensure!(heads % kv_heads == 0, "MiniMax M3 heads {heads} are not a multiple of KV heads {kv_heads}");
        let rotary_dim = int(v, "rotary_dim")?;
        ensure!(rotary_dim <= head_dim && rotary_dim % 2 == 0, "MiniMax M3 rotary_dim {rotary_dim} must be even and at most head_dim {head_dim}");
        if let Some(factor) = v["partial_rotary_factor"].as_f64() {
            ensure!((factor * head_dim as f64 - rotary_dim as f64).abs() < 1e-9,
                "partial_rotary_factor {factor} disagrees with rotary_dim {rotary_dim} of head_dim {head_dim}");
        }
        ensure!(v["use_gemma_norm"].as_bool() == Some(true), "MiniMax M3 norms must be Gemma (1 + w) RMSNorms");
        ensure!(v["use_qk_norm"].as_bool() == Some(true) && v["qk_norm_type"] == "per_head",
            "MiniMax M3 requires per-head QK norms, found use_qk_norm={} qk_norm_type={}", v["use_qk_norm"], v["qk_norm_type"]);
        ensure!(v["attention_output_gate"].as_bool() != Some(true), "MiniMax M3 attention output gates are not part of this release");
        ensure!(v["hidden_act"] == "swigluoai", "MiniMax M3 FFNs must be swigluoai, found {}", v["hidden_act"]);
        ensure!(v["scoring_func"] == "sigmoid" && v["use_routing_bias"].as_bool() == Some(true),
            "MiniMax M3 routing must be sigmoid with a selection bias");
        ensure!(v["n_shared_experts"].as_u64() == Some(1), "MiniMax M3 requires one shared expert, found {}", v["n_shared_experts"]);
        let moe = flags(v, "moe_layer_freq", layers)?;
        let sparse_cfg = &v["sparse_attention_config"];
        let sparse = if sparse_cfg["use_sparse_attention"].as_bool() == Some(true) {
            flags(sparse_cfg, "sparse_attention_freq", layers)?
        } else {
            vec![false; layers]
        };
        let msa = MsaConfig {
            index_heads: int(sparse_cfg, "sparse_num_index_heads")?,
            index_dim: int(sparse_cfg, "sparse_index_dim")?,
            block: int(sparse_cfg, "sparse_block_size")?,
            topk_blocks: int(sparse_cfg, "sparse_topk_blocks")?,
            local_blocks: int(sparse_cfg, "sparse_local_block")?,
        };
        ensure!(msa.index_heads == kv_heads, "MSA index heads {} must equal KV heads {kv_heads} (one selection per KV group)", msa.index_heads);
        ensure!(msa.index_dim >= rotary_dim, "MSA index dim {} is narrower than rotary_dim {rotary_dim}", msa.index_dim);
        ensure!(msa.local_blocks <= msa.topk_blocks, "MSA local blocks {} exceed top-k {}", msa.local_blocks, msa.topk_blocks);
        ensure!(sparse_cfg["sparse_score_type"] == "max", "MSA block score must be max, found {}", sparse_cfg["sparse_score_type"]);
        ensure!(sparse_cfg["sparse_init_block"].as_u64().unwrap_or(0) == 0, "MSA sink (init) blocks are not part of this release");
        if let Some(values) = sparse_cfg["sparse_disable_index_value"].as_array() {
            ensure!(values.iter().all(|x| x.as_u64() == Some(1)), "MSA index values must be disabled on every layer");
        }
        let eos = token_ids(&v["eos_token_id"]);
        Ok(Self {
            vocab_size: int(v, "vocab_size")?,
            hidden: int(v, "hidden_size")?,
            layers,
            heads,
            kv_heads,
            head_dim,
            rotary_dim,
            rope_theta: float(v, "rope_theta")?,
            rms_norm_eps: float(v, "rms_norm_eps")?,
            moe,
            dense_intermediate: int(v, "dense_intermediate_size")?,
            moe_intermediate: int(v, "intermediate_size")?,
            shared_intermediate: int(v, "shared_intermediate_size")?,
            experts: int(v, "num_local_experts")?,
            topk: int(v, "num_experts_per_tok")?,
            routed_scale: float(v, "routed_scaling_factor")?,
            swiglu_alpha: float(v, "swiglu_alpha")?,
            swiglu_limit: float(v, "swiglu_limit")?,
            sparse,
            msa,
            max_position: int(v, "max_position_embeddings")?,
            tie_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
            bos: v["bos_token_id"].as_u64().unwrap_or(200_034) as u32,
            eos: if eos.is_empty() { vec![200_020] } else { eos },
        })
    }

    /// The geometry this build's kernels are written for (MiniMax-M3.1-Flash).
    pub fn release_geometry(&self) -> Result<()> {
        let got = (self.hidden, self.layers, self.heads, self.kv_heads, self.head_dim, self.rotary_dim,
            self.experts, self.topk, self.moe_intermediate, self.shared_intermediate, self.dense_intermediate);
        ensure!(got == (6144, 60, 64, 4, 128, 64, 128, 4, 3072, 3072, 12288),
            "MiniMax M3 geometry (hidden, layers, heads, kv heads, head dim, rotary, experts, top-k, moe/shared/dense \
             intermediate) {got:?} is not the MiniMax-M3.1-Flash release this build serves");
        ensure!(self.msa == MsaConfig { index_heads: 4, index_dim: 128, block: 128, topk_blocks: 16, local_blocks: 1 },
            "MiniMax M3 MSA {:?} is not the release selection", self.msa);
        ensure!(self.sparse.iter().all(|s| *s), "MiniMax-M3.1-Flash runs MSA on every layer");
        ensure!(!self.tie_embeddings, "MiniMax M3 ships an untied LM head");
        Ok(())
    }

    pub fn moe_layers(&self) -> usize { self.moe.iter().filter(|m| **m).count() }

    /// KV heads' query group width.
    pub fn group(&self) -> usize { self.heads / self.kv_heads }

    /// `1 / sqrt(head_dim)`.
    pub fn softmax_scale(&self) -> f64 { (self.head_dim as f64).powf(-0.5) }
}

impl MiniMaxM3VisionConfig {
    pub fn read(snapshot: &Path) -> Result<Option<Self>> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Option<Self>> {
        let Some(v) = root.get("vision_config").filter(|v| v.is_object()) else { return Ok(None) };
        ensure!(v["hidden_act"] == "gelu" && root["projector_hidden_act"] == "gelu", "MiniMax M3 vision uses exact GELU");
        ensure!(v["position_embedding_type"] == "rope" && v["rope_mode"] == "3d", "MiniMax M3 vision requires 3D RoPE");
        ensure!(root["vision_feature_layer"].as_i64() == Some(-1) && root["vision_feature_select_strategy"] == "full",
            "MiniMax M3 vision features come from the last layer, every patch");
        let compression = &v["img_token_compression_config"];
        ensure!(compression["image_token_compression_method"] == "patch_merge", "MiniMax M3 vision merges patches");
        let pinpoints = root["image_grid_pinpoints"].as_str().context("MiniMax M3 lacks image_grid_pinpoints")?;
        let grid_pinpoints = pinpoints.trim_matches(|c| c == '[' || c == ']').split("),").map(|pair| {
            let pair = pair.trim().trim_start_matches('(').trim_end_matches(')');
            let (h, w) = pair.split_once(',').with_context(|| format!("image_grid_pinpoints entry {pair:?}"))?;
            Ok((h.trim().parse()?, w.trim().parse()?))
        }).collect::<Result<Vec<(usize, usize)>>>()?;
        let config = Self {
            hidden: int(v, "hidden_size")?,
            heads: int(v, "num_attention_heads")?,
            layers: int(v, "num_hidden_layers")?,
            intermediate: int(v, "intermediate_size")?,
            patch: int(v, "patch_size")?,
            image_size: int(v, "image_size")?,
            rope_theta: float(v, "rope_theta")?,
            layer_norm_eps: float(v, "layer_norm_eps")?,
            spatial_merge: int(compression, "spatial_merge_size")?,
            temporal_patch: int(compression, "temporal_patch_size")?,
            max_segment_frames: int(v, "vision_segment_max_frames")?,
            projection: int(v, "projection_dim")?,
            projector_hidden: int(root, "projector_hidden_size")?,
            image_token: int(root, "image_token_index")? as u32,
            video_token: int(root, "video_token_index")? as u32,
            grid_pinpoints,
        };
        ensure!(config.hidden % config.heads == 0, "vision hidden {} is not a multiple of heads {}", config.hidden, config.heads);
        ensure!(config.image_size % (config.patch * config.spatial_merge) == 0,
            "vision tile {} is not whole merged patches of {}x{}", config.image_size, config.patch, config.spatial_merge);
        ensure!(config.grid_pinpoints.iter().all(|(h, w)| h % (config.image_size / 2) == 0 && w % (config.image_size / 2) == 0),
            "image_grid_pinpoints must be multiples of half a tile");
        Ok(Some(config))
    }
}

impl MiniMaxM3DsparkConfig {
    /// `dspark/config.json` beside the target, when the release bundles one.
    pub fn read(snapshot: &Path) -> Result<Option<Self>> {
        let path = snapshot.join("dspark").join("config.json");
        if !path.is_file() { return Ok(None) }
        Self::from_hf(&read_json(&path)?).map(Some)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        let v = root.get("text_config").filter(|v| v.is_object()).unwrap_or(root);
        ensure!(v["architectures"].as_array().is_some_and(|a| a.iter().any(|x| x == "DSparkMiniMaxDraftModel")),
            "MiniMax M3 drafter must be DSparkMiniMaxDraftModel");
        ensure!(v["dspark_markov_head_type"] == "vanilla", "MiniMax DSpark Markov head must be vanilla");
        ensure!(v["dspark_use_dense_ffn"].as_bool() == Some(true), "MiniMax DSpark layers must be dense");
        ensure!(v["dspark_hybrid_context_fusion"].as_bool() != Some(true), "MiniMax DSpark hybrid context fusion is not part of this release");
        let layers = int(v, "num_hidden_layers")?;
        let kinds = v["layer_types"].as_array().context("MiniMax DSpark lacks layer_types")?;
        ensure!(kinds.len() == layers && kinds.iter().all(|k| k == "sliding_attention"), "MiniMax DSpark layers must all be sliding");
        let target_layers = v["dspark_target_layer_ids"].as_array().context("MiniMax DSpark lacks dspark_target_layer_ids")?
            .iter().map(|x| x.as_u64().map(|x| x as usize).with_context(|| format!("target layer {x}"))).collect::<Result<Vec<_>>>()?;
        ensure!(!target_layers.is_empty() && target_layers.windows(2).all(|w| w[0] < w[1]), "DSpark target layers must ascend");
        Ok(Self {
            layers,
            block: int(v, "dspark_block_size")?,
            markov_rank: int(v, "dspark_markov_rank")?,
            mask_token: int(v, "dspark_noise_token_id")? as u32,
            target_layers,
            sliding_window: int(v, "sliding_window")?,
            ffn_intermediate: int(v, "dspark_ffn_hidden_size")?,
            confidence_head: v["enable_confidence_head"].as_bool() == Some(true),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn release_config() -> Value { serde_json::from_str(include_str!("fixtures/config.json")).unwrap() }
    pub(crate) fn release_dspark() -> Value { serde_json::from_str(include_str!("fixtures/dspark-config.json")).unwrap() }

    #[test]
    fn release_text_config() {
        let cfg = MiniMaxM3Config::from_hf(&release_config()).unwrap();
        cfg.release_geometry().unwrap();
        assert_eq!(cfg.moe_layers(), 57);
        assert_eq!(&cfg.moe[..4], &[false, false, false, true]);
        assert_eq!(cfg.group(), 16);
        assert_eq!(cfg.msa.attended_keys(), 2048);
        assert_eq!(cfg.max_position, 1 << 20);
        assert_eq!((cfg.swiglu_alpha, cfg.swiglu_limit, cfg.routed_scale), (1.702, 7.0, 2.0));
        assert_eq!((cfg.bos, cfg.eos.as_slice()), (200_034, &[200_020][..]));
    }

    #[test]
    fn release_vision_and_drafter() {
        let vision = MiniMaxM3VisionConfig::from_hf(&release_config()).unwrap().unwrap();
        assert_eq!((vision.hidden, vision.layers, vision.patch, vision.spatial_merge, vision.temporal_patch), (1280, 32, 14, 2, 2));
        assert_eq!((vision.tile_patches(), vision.tile_tokens()), (2304, 576));
        assert_eq!(vision.grid_pinpoints.len(), 36);
        assert_eq!(vision.grid_pinpoints[1], (336, 672));
        assert_eq!((vision.image_token, vision.video_token), (200_025, 200_026));
        let draft = MiniMaxM3DsparkConfig::from_hf(&release_dspark()).unwrap();
        assert_eq!(draft.target_layers, [3, 17, 31, 45, 59]);
        assert_eq!((draft.layers, draft.block, draft.markov_rank, draft.sliding_window), (5, 7, 256, 4096));
        assert!(draft.confidence_head);
    }

    #[test]
    fn rejects_drifted_configs() {
        let mutate = |f: &dyn Fn(&mut Value)| { let mut c = release_config(); f(&mut c); MiniMaxM3Config::from_hf(&c) };
        assert!(mutate(&|c| c["text_config"]["sparse_attention_config"]["sparse_score_type"] = "mean".into()).is_err());
        assert!(mutate(&|c| c["text_config"]["sparse_attention_config"]["sparse_num_index_heads"] = 8.into()).is_err());
        assert!(mutate(&|c| c["text_config"]["hidden_act"] = "silu".into()).is_err());
        assert!(mutate(&|c| c["text_config"]["moe_layer_freq"].as_array_mut().unwrap().pop().map(|_| ()).unwrap()).is_err());
        let small = mutate(&|c| c["text_config"]["num_local_experts"] = 64.into()).unwrap();
        assert!(small.release_geometry().is_err());
    }
}
