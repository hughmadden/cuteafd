//! Execution constraints are family facts; checkpoint shard sides describe storage only.
use super::{DenseRecord, DenseEncoding};
use crate::families::{glm5::GlmDsaConfig, glm5_flash::{GlmNextConfig, GlmNextAttention}};
use crate::formats::trellis_core::CoreSplit;
use crate::kda::KdaRole;
use anyhow::{ensure, Context};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DenseAxis { Output, Input }

impl From<CoreSplit> for DenseExecution {
    fn from(split: CoreSplit) -> Self {
        match split {
            CoreSplit::Whole => Self::Replicated,
            CoreSplit::Output => Self::Split(DenseAxis::Output),
            CoreSplit::Input => Self::Split(DenseAxis::Input),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct KvBGeometry {
    pub heads: usize,
    pub nope: usize,
    pub values: usize,
    pub latent: usize,
}

/// The four resident operands produced by the existing native FP8 absorption job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBLayout {
    pub first_head: usize,
    pub heads: usize,
    pub geometry: KvBGeometry,
    pub allocations: [usize; 4],
}

impl KvBGeometry {
    pub fn layout(self, world: usize, rank: usize) -> anyhow::Result<KvBLayout> {
        ensure!(world > 0 && rank < world && self.heads > 0 && self.heads % world == 0
            && self.nope > 0 && self.nope % 64 == 0 && self.values > 0
            && self.latent > 0 && self.latent % 64 == 0, "invalid kv_b head/scale geometry");
        let heads = self.heads / world;
        let bytes = |dims: &[usize]| super::product(dims);
        Ok(KvBLayout { first_head: rank * heads, heads, geometry: self, allocations: [
            bytes(&[heads, self.latent, self.nope])?,
            bytes(&[heads, self.latent, self.nope / 64, 4])?,
            bytes(&[heads, self.values, self.latent])?,
            bytes(&[heads, self.values, self.latent / 64, 4])?,
        ] })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DenseExecution {
    RankZero,
    Replicated,
    Split(DenseAxis),
    AbsorbHeads(KvBGeometry),
}

/// A kernel-required lossless load conversion, shared by admission and loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DenseTransform { Identity, WidenBf16 }

impl DenseTransform {
    /// Exact native allocator request for the resident representation.
    pub fn allocation_bytes(self, component: &super::DenseComponent) -> anyhow::Result<usize> {
        Ok(super::allocation(self.resident_bytes(component)?))
    }

    pub fn resident_bytes(self, component: &super::DenseComponent) -> anyhow::Result<usize> {
        let bytes = component.resident_bytes()?;
        match self {
            Self::Identity => Ok(bytes),
            Self::WidenBf16 => {
                self.read_window(component)?;
                bytes.checked_mul(2).context("native widening extent overflow")
            }
        }
    }

    /// The reader and admission use the same source rows. Widening a flat
    /// operand uses BF16-element rows so even large operands stage boundedly.
    pub fn read_window(self, component: &super::DenseComponent) -> anyhow::Result<super::DenseWindow> {
        if self == Self::WidenBf16 {
            ensure!(component.dtype == "BF16" && component.source.bytes > 0 && component.source.bytes % 2 == 0,
                "native widening requires complete BF16 elements");
            if let Some(window) = &component.window {
                ensure!(window.row_bytes % 2 == 0 && window.cols.start % 2 == 0 && window.cols.end % 2 == 0,
                    "native widening window cuts BF16 elements");
            }
        }
        Ok(component.window.clone().unwrap_or_else(|| match self {
            Self::Identity => super::DenseWindow { row_bytes: component.source.bytes,
                rows: 0..1, cols: 0..component.source.bytes },
            Self::WidenBf16 => super::DenseWindow { row_bytes: 2,
                rows: 0..component.source.bytes / 2, cols: 0..2 },
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DenseOwnership {
    pub execution: DenseExecution,
    /// Only this MLA layer's attention records may follow a planner-selected whole-layer owner.
    pub whole_mla: Option<usize>,
    pub transform: DenseTransform,
}

impl DenseOwnership {
    pub const fn new(execution: DenseExecution) -> Self {
        Self { execution, whole_mla: None, transform: DenseTransform::Identity }
    }
}

pub(super) enum DenseFamily { Glm(GlmDsaConfig), Flash(GlmNextConfig) }

/// Serving selects only the target backbone and its two vocabulary tables.
/// Native MTP records remain checkpoint inventory, never external-drafter residency.
#[derive(Debug, Clone)]
pub struct DenseSelection {
    prefix: &'static str,
    layers: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabularyRole { Embedding, Head }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseRole { Backbone, Vocabulary(VocabularyRole), Excluded }

impl DenseSelection {
    /// Native reader and physical-owner names, independent of the family root.
    pub const PROJECTIONS: [(&'static str, &'static str); 19] = [
        ("q_a", "self_attn.q_a_proj"),
        ("kv_a", "self_attn.kv_a_proj_with_mqa"),
        ("q_b", "self_attn.q_b_proj"),
        ("o", "self_attn.o_proj"),
        ("index_q", "self_attn.indexer.wq_b"),
        ("index_k", "self_attn.indexer.wk"),
        ("index_w", "self_attn.indexer.weights_proj"),
        ("gate", "mlp.gate_proj"),
        ("up", "mlp.up_proj"),
        ("down", "mlp.down_proj"),
        ("router", "mlp.gate"),
        ("q", "self_attn.q_proj"),
        ("k", "self_attn.k_proj"),
        ("v", "self_attn.v_proj"),
        ("beta", "self_attn.b_proj"),
        ("f_a", "self_attn.f_a_proj"),
        ("g_a", "self_attn.g_a_proj"),
        ("f_b", "self_attn.f_b_proj"),
        ("g_b", "self_attn.g_b_proj"),
    ];

    pub fn from_hf(config: &serde_json::Value) -> anyhow::Result<Self> {
        Ok(match DenseFamily::from_hf(config)? {
            DenseFamily::Glm(cfg) => Self { prefix: "model.", layers: cfg.layers },
            DenseFamily::Flash(cfg) => Self { prefix: "model.language_model.", layers: cfg.layers },
        })
    }

    pub fn vocabulary_names(&self) -> [String; 2] {
        [format!("{}embed_tokens.weight", self.prefix), "lm_head.weight".into()]
    }

    pub fn norm_name(&self) -> String { format!("{}norm.weight", self.prefix) }

    pub fn layer_prefix(&self, layer: usize) -> anyhow::Result<String> {
        ensure!(layer < self.layers, "native dense layer outside config");
        Ok(format!("{}layers.{layer}", self.prefix))
    }

    /// The same bounded selection feeds allocation admission and the dense reader.
    /// Whole-MLA ownership is deliberately not a family selection policy.
    pub fn backbone(&self, records: &[DenseRecord], layers: usize) -> anyhow::Result<Vec<DenseRecord>> {
        ensure!(layers > 0 && layers <= self.layers, "native dense backbone extent outside config");
        let selected = records.iter().filter_map(|record| match self.role(&record.tensor, layers) {
            Ok(DenseRole::Backbone) => Some(Ok(record.clone())),
            Ok(_) => None,
            Err(error) => Some(Err(error)),
        }).collect::<anyhow::Result<Vec<_>>>()?;
        ensure!(!selected.is_empty(), "empty native dense backbone selection");
        Ok(selected)
    }

    /// Family-root-independent resident group for an admitted target operand.
    pub fn resident_group(&self, tensor: &str, layers: usize) -> anyhow::Result<&'static str> {
        Ok(match self.role(tensor, layers)? {
            DenseRole::Vocabulary(VocabularyRole::Embedding) => "embedding",
            DenseRole::Vocabulary(VocabularyRole::Head) => "lm_head",
            DenseRole::Excluded => anyhow::bail!("{tensor}: excluded from target residency"),
            DenseRole::Backbone => {
                let root = format!("{}layers.", self.prefix);
                let leaf = tensor.strip_prefix(&root).and_then(|rest| rest.split_once('.'))
                    .map(|(_, leaf)| leaf);
                match leaf {
                    Some(leaf) if leaf.starts_with("self_attn.indexer.") => "indexer",
                    Some(leaf) if leaf.starts_with("self_attn.") => "attention",
                    Some("mlp.gate.weight" | "mlp.gate.e_score_correction_bias") => "router",
                    Some(leaf) if leaf.starts_with("mlp.") => "dense",
                    _ => "norm",
                }
            }
        })
    }

    /// Role describes target use, independently of the record's native encoding.
    pub fn role(&self, tensor: &str, layers: usize) -> anyhow::Result<DenseRole> {
        ensure!(layers > 0 && layers <= self.layers, "native dense backbone extent outside config");
        let [embedding, head] = self.vocabulary_names();
        if tensor == embedding { return Ok(DenseRole::Vocabulary(VocabularyRole::Embedding)); }
        if tensor == head { return Ok(DenseRole::Vocabulary(VocabularyRole::Head)); }
        if tensor == self.norm_name() { return Ok(DenseRole::Backbone); }
        if let Some(rest) = tensor.strip_prefix(&format!("{}layers.", self.prefix)) {
            let (layer, _) = rest.split_once('.').context("native dense tensor lacks layer leaf")?;
            if layer.parse::<usize>()? < layers { return Ok(DenseRole::Backbone); }
        }
        Ok(DenseRole::Excluded)
    }
}

impl DenseFamily {
    pub(super) fn from_hf(config: &serde_json::Value) -> anyhow::Result<Self> {
        let text = config.get("text_config").unwrap_or(config);
        match text["model_type"].as_str() {
            Some("glm_moe_dsa") => Ok(Self::Glm(GlmDsaConfig::from_hf(config)?)),
            Some("glm5_next_text") => Ok(Self::Flash(GlmNextConfig::from_hf(config)?)),
            _ => anyhow::bail!("unsupported native dense family config"),
        }
    }

    pub(super) fn ownership(&self, tensor: &str) -> anyhow::Result<DenseOwnership> {
        let (prefix, flash) = match self {
            Self::Glm(_) => ("model.", false),
            Self::Flash(_) => ("model.language_model.", true),
        };
        if tensor == "lm_head.weight" || ["embed_tokens.weight", "norm.weight"].iter()
            .any(|leaf| tensor.strip_prefix(prefix) == Some(*leaf)) {
            return Ok(DenseOwnership::new(DenseExecution::RankZero));
        }
        let root = format!("{prefix}layers.");
        let (layer, leaf) = tensor.strip_prefix(&root).and_then(|s| s.split_once('.'))
            .with_context(|| format!("{tensor}: no native dense execution constraint"))?;
        let layer: usize = layer.parse()?;
        let (layers, mla, kv) = match self {
            Self::Glm(cfg) => (cfg.layers, true, KvBGeometry { heads: cfg.heads,
                nope: cfg.qk_nope_head_dim, values: cfg.v_head_dim, latent: cfg.kv_lora_rank }),
            Self::Flash(cfg) => (cfg.layers, cfg.attention.get(layer) == Some(&GlmNextAttention::Mla),
                KvBGeometry { heads: cfg.heads, nope: cfg.qk_nope_dim,
                    values: cfg.v_head_dim, latent: cfg.kv_lora_rank }),
        };
        ensure!(layer < layers, "{tensor}: dense layer outside backbone");
        let stem = tensor.strip_suffix(".weight").unwrap_or(tensor);
        let split = match self {
            Self::Glm(cfg) => crate::families::glm5::core::split(cfg, stem),
            Self::Flash(cfg) => crate::families::glm5_flash::core::split(cfg, stem),
        };
        let execution = if leaf == "self_attn.kv_b_proj.weight" {
            ensure!(mla, "{tensor}: kv_b requires MLA");
            DenseExecution::AbsorbHeads(kv)
        } else if leaf.starts_with("self_attn.indexer.") {
            // All index heads execute on each participating attention rank.
            DenseExecution::Replicated
        } else if leaf == "mlp.gate.weight" || leaf == "mlp.gate.e_score_correction_bias" {
            if flash { DenseExecution::Replicated } else { DenseExecution::RankZero }
        } else if let Some(split) = split {
            split.into()
        } else if flash && !mla && leaf.starts_with("self_attn.") {
            let role = match leaf.trim_start_matches("self_attn.") {
                "f_a_proj.weight" => KdaRole::Fa, "g_a_proj.weight" => KdaRole::Ga,
                "f_b_proj.weight" => KdaRole::Fb, "g_b_proj.weight" => KdaRole::Gb,
                "b_proj.weight" => KdaRole::Beta,
                "q_conv1d.weight" => KdaRole::ConvQ, "k_conv1d.weight" => KdaRole::ConvK,
                "v_conv1d.weight" => KdaRole::ConvV, "A_log" => KdaRole::ALog,
                "dt_bias" => KdaRole::DtBias, "o_norm.weight" => KdaRole::ONorm,
                _ => anyhow::bail!("{tensor}: no KDA execution constraint"),
            };
            role.split().into()
        } else {
            ensure!(!leaf.contains("_proj.") && !leaf.contains(".experts."),
                "{tensor}: no native projection execution constraint");
            DenseExecution::Replicated
        };
        let transform = if flash && matches!(leaf, "hc_attn_fn" | "hc_ffn_fn"
            | "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight") {
            DenseTransform::WidenBf16
        } else { DenseTransform::Identity };
        Ok(DenseOwnership { execution, transform,
            whole_mla: (flash && mla && leaf.starts_with("self_attn.")).then_some(layer) })
    }
}

impl DenseRecord {
    pub(super) fn validate_ownership(&self) -> anyhow::Result<()> {
        if self.ownership.transform == DenseTransform::WidenBf16 {
            ensure!(self.encoding == DenseEncoding::Bf16 && self.native_blocks == 0 && self.components.len() == 1,
                "{}: widening requires one plain BF16 source", self.tensor);
            self.ownership.transform.resident_bytes(&self.components[0])?;
        }
        if let DenseExecution::AbsorbHeads(g) = self.ownership.execution {
            ensure!(self.encoding == DenseEncoding::BlockFp8
                && self.shape == [super::product(&[g.heads, g.nope.checked_add(g.values).context("kv_b rows overflow")?])?, g.latent],
                "{}: kv_b absorption shape/encoding mismatch", self.tensor);
            g.layout(1, 0)?;
        }
        Ok(())
    }
}
