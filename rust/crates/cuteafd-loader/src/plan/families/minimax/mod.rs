//! MiniMax M3 (`MiniMaxM3SparseForConditionalGeneration`, `minimax_m3_vl`): MSA
//! block-sparse GQA on every layer, a sigmoid-routed SwiGLU-OAI MoE over
//! compressed-tensors NVFP4 experts (W4A4, calibrated input scales), MXFP8
//! (E4M3, UE8M0 per 1x32) attention, indexer, dense and shared projections, a
//! bundled CLIP-style vision tower and a bundled DSpark drafter (`dspark/`).
//! The planner validates the checkpoint's own config and tensor headers.
use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use cuteafd_core::FfnActivation;

use crate::families::minimax_m3::config::{MiniMaxM3Config, MiniMaxM3DsparkConfig, MiniMaxM3VisionConfig};
use crate::plan::{checkpoint::Checkpoint, family::*, format::{self, QuantOperand, ScaleEncoding}, names::indexed, spec::*};

pub mod coordinator;
pub mod admission;
mod placement;
#[cfg(test)]
mod tests;

pub struct MiniMaxM3;
pub static MINIMAX_M3: MiniMaxM3 = MiniMaxM3;

const TEXT: &str = "language_model.model.";
const HEAD: &str = "language_model.lm_head.";
const DRAFT: &str = "language_model.model.dspark.";

/// The NVFP4 expert package MiniMax M3 runs: H6144/E128/K4/I3072 W4A4 with
/// compressed-tensors divisor global scales and the SwiGLU-OAI epilogue.
pub const PACKAGE: &str = "minimax:nvfp4";

/// How the release stores one logical weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Storage {
    Bf16,
    F32,
    /// E4M3 with UE8M0 scales per row and 32 columns.
    Mxfp8,
    /// Compressed-tensors NVFP4 with a calibrated input scale.
    Nvfp4,
}

#[derive(Debug, Clone)]
struct Contract {
    role: TensorRole,
    /// The logical `[N, K]` (or the plain tensor's shape).
    shape: Vec<usize>,
    storage: Storage,
    /// A `bias` member beside the weight (vision and projector linears).
    bias: bool,
}

pub(super) struct Model {
    cfg: MiniMaxM3Config,
    draft: Option<MiniMaxM3DsparkConfig>,
    spec: ModelSpec,
    /// Logical weights by stem.
    contracts: BTreeMap<String, Contract>,
    /// Header byte counts by component, the bundled drafter's included.
    bytes: BTreeMap<Component, u64>,
    /// Every expected tensor has a readable header.
    complete: bool,
}

fn text(name: &str) -> Option<&str> { name.strip_prefix(TEXT) }

impl Model {
    fn open(checkpoint: &Checkpoint) -> Result<Self> {
        let cfg = MiniMaxM3Config::from_hf(&checkpoint.config)?;
        cfg.release_geometry()?;
        let vision = MiniMaxM3VisionConfig::from_hf(&checkpoint.config)?;
        let draft = MiniMaxM3DsparkConfig::read(&checkpoint.snapshot)?;
        let (h, v, d) = (cfg.hidden, cfg.vocab_size, cfg.head_dim);
        let mut contracts = BTreeMap::new();
        let mut add = |stem: String, role: TensorRole, shape: Vec<usize>, storage, bias| {
            contracts.insert(stem, Contract { role, shape, storage, bias });
        };
        add(format!("{TEXT}embed_tokens"), TensorRole::new(Component::Embedding), vec![v, h], Storage::Bf16, false);
        add(format!("{HEAD}").trim_end_matches('.').to_owned(), TensorRole::new(Component::LmHead), vec![v, h], Storage::Bf16, false);
        add(format!("{TEXT}norm"), TensorRole::new(Component::Norm), vec![h], Storage::Bf16, false);
        for layer in 0..cfg.layers {
            let base = format!("{TEXT}layers.{layer}");
            let at = |c| TensorRole::layer(c, layer);
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{base}.{norm}"), at(Component::Norm), vec![h], Storage::Bf16, false);
            }
            let a = format!("{base}.self_attn");
            for (proj, n, k) in [("q", cfg.heads * d, h), ("k", cfg.kv_heads * d, h), ("v", cfg.kv_heads * d, h), ("o", h, cfg.heads * d)] {
                add(format!("{a}.{proj}_proj"), at(Component::Attention), vec![n, k], Storage::Mxfp8, false);
            }
            for norm in ["q_norm", "k_norm"] { add(format!("{a}.{norm}"), at(Component::Attention), vec![d], Storage::Bf16, false); }
            if cfg.sparse[layer] {
                let i = cfg.msa.index_dim;
                add(format!("{a}.index_q_proj"), at(Component::Indexer), vec![cfg.msa.index_heads * i, h], Storage::Mxfp8, false);
                add(format!("{a}.index_k_proj"), at(Component::Indexer), vec![i, h], Storage::Mxfp8, false);
                for norm in ["index_q_norm", "index_k_norm"] { add(format!("{a}.{norm}"), at(Component::Indexer), vec![i], Storage::Bf16, false); }
            }
            let ffn = |prefix: String, inter: usize, component, add: &mut dyn FnMut(String, TensorRole, Vec<usize>, Storage, bool)| {
                for (proj, n, k) in [("gate", inter, h), ("up", inter, h), ("down", h, inter)] {
                    add(format!("{prefix}.{proj}_proj"), TensorRole::layer(component, layer), vec![n, k], Storage::Mxfp8, false);
                }
            };
            if cfg.moe[layer] {
                let m = format!("{base}.block_sparse_moe");
                add(format!("{m}.gate"), at(Component::Router), vec![cfg.experts, h], Storage::F32, false);
                add(format!("{m}.e_score_correction_bias"), at(Component::Router), vec![cfg.experts], Storage::F32, false);
                ffn(format!("{m}.shared_experts"), cfg.shared_intermediate, Component::SharedExpert, &mut add);
                for expert in 0..cfg.experts {
                    let role = TensorRole::expert(Component::RoutedExpert, layer, expert);
                    let e = format!("{m}.experts.{expert}");
                    for (proj, n, k) in [("w1", cfg.moe_intermediate, h), ("w3", cfg.moe_intermediate, h), ("w2", h, cfg.moe_intermediate)] {
                        add(format!("{e}.{proj}"), role.clone(), vec![n, k], Storage::Nvfp4, false);
                    }
                }
            } else {
                ffn(format!("{base}.mlp"), cfg.dense_intermediate, Component::DenseFfn, &mut add);
            }
        }
        if let Some(vc) = &vision {
            let vision_role = TensorRole::new(Component::Vision);
            let (vh, vi, p, t) = (vc.hidden, vc.intermediate, vc.patch, vc.temporal_patch);
            let tower = "vision_tower.vision_model";
            add(format!("{tower}.embeddings.patch_embedding"), vision_role.clone(), vec![vh, 3, t, p, p], Storage::F32, false);
            add(format!("{tower}.pre_layrnorm"), vision_role.clone(), vec![vh], Storage::Bf16, true);
            for layer in 0..vc.layers {
                let base = format!("{tower}.encoder.layers.{layer}");
                for norm in ["layer_norm1", "layer_norm2"] { add(format!("{base}.{norm}"), vision_role.clone(), vec![vh], Storage::Bf16, true); }
                for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    add(format!("{base}.self_attn.{proj}"), vision_role.clone(), vec![vh, vh], Storage::Bf16, true);
                }
                add(format!("{base}.mlp.fc1"), vision_role.clone(), vec![vi, vh], Storage::Bf16, true);
                add(format!("{base}.mlp.fc2"), vision_role.clone(), vec![vh, vi], Storage::Bf16, true);
            }
            let merged = vc.projector_hidden * vc.spatial_merge * vc.spatial_merge;
            for (stem, n, k) in [("multi_modal_projector.linear_1", vc.projector_hidden, vh),
                ("multi_modal_projector.linear_2", vc.projection, vc.projector_hidden),
                ("patch_merge_mlp.linear_1", vc.projector_hidden, merged),
                ("patch_merge_mlp.linear_2", vc.projection, vc.projector_hidden)] {
                add(stem.into(), vision_role.clone(), vec![n, k], Storage::Bf16, true);
            }
        }
        let mut model = Self { spec: spec(&cfg, vision.is_some(), draft.as_ref(), checkpoint), cfg, draft, contracts,
            bytes: BTreeMap::new(), complete: checkpoint.missing_shards.is_empty() };
        let mut stems: BTreeMap<&str, Vec<&crate::SafetensorsTensorMetadata>> = BTreeMap::new();
        for tensor in &checkpoint.tensors { stems.entry(format::split_stem(&tensor.meta.name).0).or_default().push(&tensor.meta); }
        for (stem, contract) in &model.contracts {
            let found = stems.contains_key(stem.as_str()) || stems.get(format::split_stem(stem).0)
                .is_some_and(|members| members.iter().any(|m| &m.name == stem));
            if !found {
                let named = [format!("{stem}.weight"), format!("{stem}.weight_packed"), stem.clone()];
                ensure!(named.iter().any(|n| checkpoint.contains_tensor(n)), "MiniMax M3 checkpoint lacks {stem} ({:?})", contract.role.component);
                model.complete = false;
            }
        }
        for tensor in &checkpoint.tensors {
            let Some(role) = MINIMAX_M3.classify(&model.spec, &tensor.meta.name) else { continue };
            *model.bytes.entry(role.component).or_default() += tensor.meta.byte_length;
        }
        if model.draft.is_some() {
            let drafter = Checkpoint::open(&checkpoint.snapshot.join("dspark")).context("MiniMax DSpark headers")?;
            for tensor in &drafter.tensors {
                ensure!(tensor.meta.name.starts_with(DRAFT), "unexpected MiniMax DSpark tensor {}", tensor.meta.name);
                *model.bytes.entry(Component::Speculator).or_default() += tensor.meta.byte_length;
            }
            model.complete &= drafter.missing_shards.is_empty();
        }
        Ok(model)
    }

    pub(super) fn cfg(&self) -> &MiniMaxM3Config { &self.cfg }

    fn contract(&self, stem: &str) -> Option<&Contract> {
        self.contracts.get(stem).or_else(|| stem.strip_suffix(".weight").and_then(|s| self.contracts.get(s)))
    }
}

fn spec(cfg: &MiniMaxM3Config, vision: bool, draft: Option<&MiniMaxM3DsparkConfig>, checkpoint: &Checkpoint) -> ModelSpec {
    let layers = (0..cfg.layers).map(|layer| LayerSpec {
        attention: if cfg.sparse[layer] {
            AttentionKind::BlockSparseGqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim,
                block: cfg.msa.block, topk_blocks: cfg.msa.topk_blocks, index_dim: cfg.msa.index_dim }
        } else {
            AttentionKind::Gqa { heads: cfg.heads, kv_heads: cfg.kv_heads, head_dim: cfg.head_dim }
        },
        ffn: if cfg.moe[layer] { FfnKind::Moe } else { FfnKind::Dense { intermediate: cfg.dense_intermediate } },
        rope: Some(RopeSpec { dims: cfg.rotary_dim, theta: cfg.rope_theta }),
    }).collect();
    let mut notes = vec![
        format!("MSA: {} index heads (one per KV head) over one {}-wide key, {}-token blocks max-pooled, top-{} with {} local \
            block forced: {} keys per row and KV head", cfg.msa.index_heads, cfg.msa.index_dim, cfg.msa.block, cfg.msa.topk_blocks,
            cfg.msa.local_blocks, cfg.msa.attended_keys()),
        format!("per-head Gemma QK norms, NeoX RoPE on {} of {} head channels (theta {})", cfg.rotary_dim, cfg.head_dim, cfg.rope_theta),
        format!("SwiGLU-OAI FFNs (alpha {}, limit {}); routed experts compressed-tensors NVFP4 W4A4 (divisor global scales); \
            attention, indexer, dense and shared projections MXFP8 E4M3 with UE8M0 1x32 scales", cfg.swiglu_alpha, cfg.swiglu_limit),
    ];
    if let Some(draft) = draft {
        notes.push(format!("bundled DSpark drafter (dspark/): {} sliding-{} dense layers, block {}, Markov rank {}, taps {:?}",
            draft.layers, draft.sliding_window, draft.block, draft.markov_rank, draft.target_layers));
    }
    ModelSpec {
        family: "minimax_m3",
        architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
        hidden: cfg.hidden,
        vocab: cfg.vocab_size,
        layers,
        moe: Some(MoeSpec {
            experts: cfg.experts,
            top_k: cfg.topk,
            intermediate: cfg.moe_intermediate,
            shared_experts: 1,
            shared_intermediate: cfg.shared_intermediate,
            scoring: "sigmoid; FP32 router, bias for selection only; renormalize unbiased top-k".into(),
            routed_scaling: Some(cfg.routed_scale),
            groups: None,
        }),
        speculator: draft.map(|d| SpeculatorSpec::Dspark { stages: d.layers, experts: 0, top_k: 0, target_layers: d.target_layers.clone() }),
        tables: Vec::new(),
        vision,
        notes,
    }
}

impl Family for MiniMaxM3 {
    fn id(&self) -> &'static str { "minimax_m3" }
    fn runtime(&self) -> RuntimeStatus { RuntimeStatus::Planned }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|a| a == "MiniMaxM3SparseForConditionalGeneration")
    }
    fn open(&self, checkpoint: &Checkpoint) -> Result<Box<dyn FamilyModel>, ConfigError> {
        Ok(Box::new(Model::open(checkpoint).map_err(ConfigError::from_anyhow)?))
    }
    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        if ["vision_tower.", "multi_modal_projector.", "patch_merge_mlp."].iter().any(|p| name.starts_with(p)) {
            return Some(TensorRole::new(Component::Vision));
        }
        if name.starts_with(HEAD) { return Some(TensorRole::new(Component::LmHead)); }
        let rest = text(name)?;
        if rest.starts_with("embed_tokens.") { return Some(TensorRole::new(Component::Embedding)); }
        if rest.starts_with("norm.") { return Some(TensorRole::new(Component::Norm)); }
        if rest.starts_with("dspark.") { return Some(TensorRole::new(Component::Speculator)); }
        let (layer, rest) = indexed(rest, "layers.")?;
        if layer >= spec.layers.len() { return None; }
        if let Some(expert) = rest.strip_prefix("block_sparse_moe.experts.")
            .and_then(|r| r.split_once('.')).and_then(|(e, _)| e.parse().ok()) {
            return Some(TensorRole::expert(Component::RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("block_sparse_moe.gate.") || rest == "block_sparse_moe.e_score_correction_bias" { Component::Router }
            else if rest.starts_with("block_sparse_moe.shared_experts.") { Component::SharedExpert }
            else if rest.starts_with("mlp.") { Component::DenseFfn }
            else if rest.starts_with("self_attn.index_") { Component::Indexer }
            else if rest.starts_with("self_attn.") { Component::Attention }
            else if rest.starts_with("input_layernorm.") || rest.starts_with("post_attention_layernorm.") { Component::Norm }
            else { return None; };
        Some(TensorRole::layer(component, layer))
    }
    fn component_hint(&self, component: Component) -> Option<Hint> {
        match component {
            Component::Attention | Component::Indexer => Some(Hint {
                what: "MiniMax M3 native MSA execution awaits device qualification".into(),
                how: "Use the matching MiniMax PROGRAMS.json for single-device WIP admission; serving uses FP8 K/V planes, \
                    8448-byte index pages and causal paged tiles. Qualify real-weight numerics and cache replay before release.".into(),
            }),
            Component::RoutedExpert => Some(Hint {
                what: "MiniMax M3 native NVFP4 W4A4 expert execution awaits device qualification".into(),
                how: "Install fp8-minimax-nvfp4a4 with SwiGLU-OAI activation code 3; use mount loading for divisor scales, \
                    Spark TP2/TP3/TP4 and TP1 for onboarded bottom routed layers. Push loading is unsupported.".into(),
            }),
            Component::Vision => Some(Hint {
                what: "MiniMax M3 images and video run through the bundled tower on the RTX or a Spark rank".into(),
                how: "Build with the D80 vision attention export; qualify the tower with python/tools/qualify/minimax_m3/qualify-vision.py and the served path with scripts/qualify/minimax_m3/qualify-native-vision.py.".into(),
            }),
            Component::Speculator => Some(Hint {
                what: "MiniMax M3 DSpark drafting awaits device qualification".into(),
                how: "The bundled dspark/ drafter uses native MXFP8 layers, target taps 3/17/31/45/59 and target verification. \
                    Qualify real-weight attention, Markov/confidence heads and acceptance before changing the launcher default from off.".into(),
            }),
            _ => None,
        }
    }
    fn optional(&self, component: Component) -> bool {
        matches!(component, Component::Vision | Component::Speculator)
    }
}

impl FamilyModel for Model {
    fn spec(&self) -> &ModelSpec { &self.spec }
    /// Placement reads the image's PROGRAMS.json, so every placed layout is the native ledger.
    fn native_serving_admission(&self) -> bool { true }
    fn activation(&self, layer: usize, _shared: bool) -> Option<FfnActivation> {
        (layer < self.cfg.layers).then_some(FfnActivation::SwigluOai {
            alpha: self.cfg.swiglu_alpha as f32, limit: self.cfg.swiglu_limit as f32 })
    }
    fn spark_budget_bytes(&self) -> u64 { placement::SPARK_BUDGET_BYTES }
    fn cache_geometry(&self, options: crate::serving_capacity::CacheOptions)
        -> Result<Option<crate::serving_capacity::FamilyCacheGeometry>, crate::serving_capacity::CacheGeometryError> {
        if options.coordinator_ranks != 1 || options.attention_placement
            .is_some_and(|p| p != crate::placement::AttentionPlacement::Heads) {
            return Err(crate::serving_capacity::CacheGeometryError::Unsupported { family: "minimax_m3", what: "one coordinator GPU" });
        }
        let cache = crate::families::minimax_m3::cache::MiniMaxCacheLayout::new(&self.cfg, self.cfg.max_position as u64, options.minimax_kv)?;
        Ok(Some(cache.geometry()))
    }
    fn accepts(&self, role: &TensorRole, stem: &str, operand: &mut QuantOperand) -> Result<(), String> {
        let contract = self.contract(stem)
            .ok_or_else(|| format!("{stem}: outside the MiniMax M3 tensor contract ({})", super::describe(operand)))?;
        super::require(contract.role.component == role.component && contract.role.layer == role.layer, ||
            format!("{stem}: classified {:?}, the contract says {:?}", role.component, contract.role.component))?;
        let shape = operand.logical == contract.shape;
        let bias = operand.companions == if contract.bias { vec!["bias".to_owned()] } else { Vec::new() };
        let ok = shape && bias && match contract.storage {
            Storage::Bf16 => operand.is_plain(&[format::Encoding::Bf16]) && operand.input_scale.is_none(),
            Storage::F32 => operand.is_plain(&[format::Encoding::F32]) && operand.input_scale.is_none(),
            Storage::Mxfp8 => operand.is_fp8_row_groups(32, &[ScaleEncoding::Ue8m0]) && operand.global_scale.is_none()
                && operand.input_scale.is_none(),
            Storage::Nvfp4 => operand.is_compressed_nvfp4() && operand.input_scale.is_some(),
        };
        super::require(ok, || format!("{stem}: expected {:?} {:?}{}, found {}", contract.storage, contract.shape,
            if contract.bias { " with bias" } else { "" }, super::describe(operand)))
    }
    fn experts(&self, operand: &QuantOperand) -> Option<ExpertContract> {
        (operand.is_compressed_nvfp4() && operand.input_scale.is_some()).then(|| ExpertContract {
            package: PACKAGE.into(),
            block: 16,
            spark_worlds: crate::plan::experts::spark_worlds(PACKAGE, self.cfg.moe_intermediate),
            local: Err("MiniMax M3 routed experts (216 GiB) cannot all be resident on the coordinators; \
                bottom layers onboard to the RTX beside 2, 3, 4 or 6 Spark ranks".into()),
        })
    }
    fn placement(&self, input: &crate::plan::PlanInput, report: &mut crate::plan::PlanReport,
        options: &crate::plan::layout::LayoutOptions, requested: crate::plan::ExpertPlacement)
        -> Option<anyhow::Result<cuteafd_core::memory_layout::MemoryLayout>> {
        Some(placement::layout(self, input, report, options, requested))
    }
}
