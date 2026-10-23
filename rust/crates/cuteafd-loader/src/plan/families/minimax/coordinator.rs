//! Native source-weight contract for the MiniMax M3 coordinator: every
//! RTX-resident text operand in its checkpoint format (MXFP8 payload and UE8M0
//! scale codes land unchanged), how each rank of a KV-head split holds it, and
//! the layer arithmetic the native programs must reproduce. Routed experts,
//! vision and the drafter are separate owners.
use super::Model;
use crate::formats::native_dense::NativeDenseLayout;
use crate::plan::{checkpoint::{Checkpoint, CheckpointTensor}, spec::Component};
use anyhow::{ensure, Context, Result};
use cuteafd_core::dense_attention::{GqaGeometry, NormWeight, RotaryArithmetic, RouterBiasOrder, SigmoidRouter};
use cuteafd_core::{DType, FfnActivation};

#[derive(Debug, Clone)]
pub struct CoordinatorContract {
    pub weights: NativeDenseLayout,
    /// Gemma norms: `x * (1 + w)`.
    pub norm_weight: NormWeight,
    pub norm_eps: f32,
    pub router: SigmoidRouter,
    pub layers: Vec<LayerContract>,
}

#[derive(Debug, Clone)]
pub struct LayerContract {
    pub layer: usize,
    pub geometry: GqaGeometry,
    pub theta: f64,
    /// Dense layers' MLP (0-2); routed layers' shared expert otherwise.
    pub dense_activation: Option<FfnActivation>,
    pub shared_activation: Option<FfnActivation>,
    pub msa: Option<crate::families::minimax_m3::config::MsaConfig>,
}

impl CoordinatorContract {
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let model = Model::open(checkpoint)?;
        let cfg = model.cfg();
        let mut declarations = Vec::new();
        for (stem, contract) in &model.contracts {
            if matches!(contract.role.component, Component::RoutedExpert | Component::Vision | Component::Speculator) { continue; }
            match contract.storage {
                super::Storage::Mxfp8 => {
                    let [n, k] = contract.shape[..] else { anyhow::bail!("{stem}: MXFP8 operand must be a matrix") };
                    declarations.push((format!("{stem}.weight"), vec![n, k], DType::F8E4M3));
                    declarations.push((format!("{stem}.weight_scale_inv"), vec![n, k / 32], DType::U8));
                }
                super::Storage::Bf16 | super::Storage::F32 => {
                    let dtype = if contract.storage == super::Storage::Bf16 { DType::Bf16 } else { DType::F32 };
                    let name = if stem.ends_with("e_score_correction_bias") { stem.clone() } else { format!("{stem}.weight") };
                    declarations.push((name, contract.shape.clone(), dtype));
                }
                super::Storage::Nvfp4 => anyhow::bail!("{stem}: NVFP4 outside the routed experts"),
            }
        }
        let weights = NativeDenseLayout::from_headers(checkpoint, declarations)?;
        let activation = FfnActivation::SwigluOai { alpha: cfg.swiglu_alpha as f32, limit: cfg.swiglu_limit as f32 };
        let layers = (0..cfg.layers).map(|layer| Ok(LayerContract {
            layer,
            geometry: GqaGeometry { query_heads: cfg.heads, kv_heads: cfg.kv_heads, qk_dim: cfg.head_dim,
                value_dim: cfg.head_dim, rotary_dim: cfg.rotary_dim }.validate()?,
            theta: cfg.rope_theta,
            dense_activation: (!cfg.moe[layer]).then_some(activation),
            shared_activation: cfg.moe[layer].then_some(activation),
            msa: cfg.sparse[layer].then_some(cfg.msa),
        })).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights,
            norm_weight: NormWeight::ZeroCentered,
            norm_eps: cfg.rms_norm_eps as f32,
            // The reference gathers the unbiased sigmoid scores of the biased
            // selection, renormalizes them and scales by routed_scaling_factor.
            router: SigmoidRouter { topk: cfg.topk, factor: cfg.routed_scale as f32, epsilon: 0.0,
                bias_order: RouterBiasOrder::GatherOriginal },
            layers,
        })
    }
}

/// How a rank of a KV-head split holds an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSlice {
    Replicated,
    /// Output rows (query, KV and index-query heads).
    Rows,
    /// Input columns (the output projection over the rank's query heads).
    Cols,
}

#[derive(Debug, Clone)]
pub struct WeightOperand {
    /// The resident operand (`...gate_up_proj.weight` fuses gate then up rows,
    /// the SwiGLU-OAI order; scale codes fuse the same way).
    pub name: String,
    pub sources: Vec<CheckpointTensor>,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub slice: WeightSlice,
    pub allocation_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct RankWeightLayout {
    pub rank: usize,
    pub ranks: usize,
    pub operands: Vec<WeightOperand>,
    pub resident_bytes: u64,
}

const ROW_SPLIT: [&str; 4] = [".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.index_q_proj."];

impl RankWeightLayout {
    /// One GPU holds everything; with two, attention is split by KV head
    /// (query, KV and index-query rows; output columns) and the rest is
    /// replicated, except the embedding on rank 0 and the head on the last rank.
    pub fn from_checkpoint(checkpoint: &Checkpoint, rank: usize, ranks: usize) -> Result<Self> {
        ensure!(matches!(ranks, 1 | 2) && rank < ranks, "MiniMax M3 coordinator rank geometry");
        let contract = CoordinatorContract::from_checkpoint(checkpoint)?;
        let sources: std::collections::BTreeMap<_, _> = contract.weights.tensors.iter().map(|t| (t.meta.name.as_str(), t)).collect();
        let mut operands = Vec::new();
        for source in &contract.weights.tensors {
            let name = &source.meta.name;
            if name.contains(".up_proj.") { continue; }
            if name == "language_model.model.embed_tokens.weight" && rank != 0 { continue; }
            if name == "language_model.lm_head.weight" && rank + 1 != ranks { continue; }
            let mut selected = vec![source.clone()];
            let mut shape = source.meta.shape.clone();
            let mut output = name.clone();
            if let Some((base, suffix)) = name.split_once(".gate_proj.") {
                let up = sources.get(format!("{base}.up_proj.{suffix}").as_str()).context("MiniMax gate without up")?;
                ensure!(up.meta.shape == shape && up.meta.dtype == source.meta.dtype, "{name}: gate/up operands disagree");
                shape[0] *= 2;
                selected.push((*up).clone());
                output = format!("{base}.gate_up_proj.{suffix}");
            }
            let slice = if ranks == 1 { WeightSlice::Replicated }
                else if ROW_SPLIT.iter().any(|s| name.contains(s)) { WeightSlice::Rows }
                else if name.contains(".self_attn.o_proj.") { WeightSlice::Cols }
                else { WeightSlice::Replicated };
            match slice {
                WeightSlice::Rows => { ensure!(shape[0] % ranks == 0, "{name}: uneven head split"); shape[0] /= ranks; }
                WeightSlice::Cols => {
                    ensure!(shape.len() == 2 && shape[1] % ranks == 0, "{name}: uneven column split");
                    shape[1] /= ranks;
                }
                WeightSlice::Replicated => {}
            }
            let element = match source.meta.dtype { DType::Bf16 => 2u64, DType::F32 => 4, DType::F8E4M3 | DType::U8 => 1,
                ref other => anyhow::bail!("{name}: unsupported coordinator operand {other:?}") };
            let bytes = shape.iter().try_fold(element, |n, d| n.checked_mul(*d as u64)).context("operand size overflow")?.max(256);
            operands.push(WeightOperand { name: output, sources: selected, dtype: source.meta.dtype.clone(), shape, slice, allocation_bytes: bytes });
        }
        let resident_bytes = operands.iter().try_fold(0u64, |n, o| n.checked_add(o.allocation_bytes)).context("rank weight overflow")?;
        Ok(Self { rank, ranks, operands, resident_bytes })
    }

    /// Mapped token lookup owns the embedding; the untied output head stays resident.
    pub fn with_host_embedding(mut self, checkpoint: &Checkpoint) -> Result<Self> {
        const EMBEDDING: &str = "language_model.model.embed_tokens.weight";
        checkpoint.require_untied_embedding(EMBEDDING)?;
        self.operands.retain(|operand| operand.name != EMBEDDING);
        self.resident_bytes = self.operands.iter().try_fold(0u64, |n, o| n.checked_add(o.allocation_bytes))
            .context("rank weight overflow")?;
        Ok(self)
    }

    /// Caller supplies every non-weight cost; nothing missing is inferred as zero.
    pub fn check_budget(&self, available: u64, reservations: &[u64]) -> Result<u64> {
        let total = reservations.iter().try_fold(self.resident_bytes, |n, v| n.checked_add(*v))
            .context("MiniMax coordinator reservation overflow")?;
        ensure!(total <= available, "MiniMax coordinator weights and reservations need {total} bytes, available {available}");
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing;

    #[test]
    fn coordinator_operands_and_split() {
        let dir = tempfile::tempdir().unwrap();
        testing::write_minimax_m3_snapshot(dir.path(), true, true);
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let contract = CoordinatorContract::from_checkpoint(&checkpoint).unwrap();
        assert_eq!((contract.norm_weight, contract.router.topk, contract.router.factor), (NormWeight::ZeroCentered, 4, 2.0));
        assert_eq!(contract.layers.iter().filter(|l| l.dense_activation.is_some()).count(), 3);
        assert!(contract.weights.tensors.iter().all(|t| !t.meta.name.contains(".experts.") && !t.meta.name.starts_with("vision")));
        let one = RankWeightLayout::from_checkpoint(&checkpoint, 0, 1).unwrap();
        let find = |l: &RankWeightLayout, name: &str| l.operands.iter().find(|o| o.name == name).cloned().unwrap();
        let shared = find(&one, "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_up_proj.weight");
        assert_eq!((shared.shape.as_slice(), shared.sources.len()), (&[6144, 6144][..], 2));
        let scales = find(&one, "language_model.model.layers.0.mlp.gate_up_proj.weight_scale_inv");
        assert_eq!((scales.dtype, scales.shape.as_slice()), (DType::U8, &[24576, 192][..]));
        let [r0, r1] = [0, 1].map(|rank| RankWeightLayout::from_checkpoint(&checkpoint, rank, 2).unwrap());
        assert_eq!(find(&r0, "language_model.model.layers.5.self_attn.q_proj.weight").shape, [4096, 6144]);
        assert_eq!(find(&r1, "language_model.model.layers.5.self_attn.o_proj.weight_scale_inv").shape, [6144, 128]);
        assert_eq!(find(&r1, "language_model.model.layers.5.self_attn.index_k_proj.weight").slice, WeightSlice::Replicated);
        assert!(r0.operands.iter().any(|o| o.name.ends_with("embed_tokens.weight")) && !r0.operands.iter().any(|o| o.name.contains("lm_head")));
        assert!(r1.operands.iter().any(|o| o.name.contains("lm_head")));
        // A split saves the attention heads, not the replicated MLPs.
        assert!(r0.resident_bytes < one.resident_bytes && r0.resident_bytes > one.resident_bytes / 2);
    }

    #[test]
    fn minimax_host_embedding_removes_only_the_input_table() -> Result<()> {
        let dir = tempfile::tempdir()?;
        testing::write_minimax_m3_snapshot(dir.path(), false, false);
        let mut checkpoint = Checkpoint::open(dir.path())?;
        let gpu = RankWeightLayout::from_checkpoint(&checkpoint, 0, 1)?;
        let host = gpu.clone().with_host_embedding(&checkpoint)?;
        let embedding = gpu.operands.iter().find(|o| o.name.ends_with("embed_tokens.weight")).unwrap();
        assert_eq!(gpu.resident_bytes - host.resident_bytes, embedding.allocation_bytes);
        assert_eq!(host.operands.len() + 1, gpu.operands.len());
        assert!(!host.operands.iter().any(|o| o.name.ends_with("embed_tokens.weight")));
        let head = |layout: &RankWeightLayout| layout.operands.iter()
            .find(|o| o.name == "language_model.lm_head.weight").unwrap().allocation_bytes;
        assert_eq!(head(&gpu), head(&host));
        checkpoint.config["tie_word_embeddings"] = serde_json::Value::Bool(true);
        assert!(gpu.with_host_embedding(&checkpoint).is_err());
        Ok(())
    }
}
