//! GLM 5.x: MLA + DeepSeek Sparse Attention indexers (glm_moe_dsa) and the
//! hybrid Kimi-Delta-Attention + DSA variant with mHC (glm5_next). HF names,
//! optionally under `model.language_model.`.
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{Family, Hint, RuntimeStatus};
use crate::plan::format::WeightFormat;
use crate::plan::names::indexed;
use crate::plan::spec::*;

pub struct Glm {
    id: &'static str,
    architecture: &'static str,
    runtime: RuntimeStatus,
}

/// GLM 5.x: serve-glm / glm-golden on the glm coordinator programs.
pub static GLM_DSA: Glm =
    Glm { id: "glm_dsa", architecture: "GlmMoeDsaForCausalLM", runtime: RuntimeStatus::Serving };
/// GLM 5.3 Flash: serve-glmf / glmf-golden on the glmf coordinator programs.
pub static GLM_NEXT: Glm =
    Glm { id: "glm_next", architecture: "Glm5NextForConditionalGeneration", runtime: RuntimeStatus::Serving };

fn str_list(config: &Value, key: &str) -> Vec<String> {
    config
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

fn strip_model_prefix(name: &str) -> Option<&str> {
    name.strip_prefix("model.language_model.").or_else(|| name.strip_prefix("model."))
}

impl Family for Glm {
    fn id(&self) -> &'static str {
        self.id
    }
    fn runtime(&self) -> RuntimeStatus {
        self.runtime
    }

    fn executes(&self, component: Component, format: &WeightFormat) -> bool {
        if self.runtime != RuntimeStatus::Serving {
            return false;
        }
        use WeightFormat::*;
        // BF16/F32 coordinator tensors (FP8 128x128 blocks with FP32 scales read
        // as they are); routed experts from EXL3 packages (glm: exl3-glm-k45,
        // glmf: exl3-glmf-k34) or the checkpoint's FP8 (fp8-glm, fp8-glmf).
        let exl3_bits = if self.id == "glm_dsa" { 4..=5 } else { 3..=4 };
        match component {
            Component::RoutedExpert => {
                matches!(format, Fp8Block { block: (128, 128) })
                    || matches!(format, Exl3 { bits } if exl3_bits.contains(bits))
            }
            Component::Speculator | Component::SpeculatorExpert | Component::Vision => false,
            _ => matches!(format, Fp8Block { block: (128, 128) } | Bf16 | F32),
        }
    }

    fn optional(&self, component: Component) -> bool {
        // Text serving runs without the native MTP layer and the vision tower
        // (speculation uses a DFlash2 drafter checkpoint).
        matches!(component, Component::Speculator | Component::SpeculatorExpert | Component::Vision)
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|arch| arch == self.architecture)
    }

    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec> {
        let text = checkpoint.text_config();
        let layers = usize_field(text, "num_hidden_layers")?;
        let mlp_types = str_list(text, "mlp_layer_types");
        let layer_types = str_list(text, "layer_types");
        let indexer_types = str_list(text, "indexer_types");
        let first_dense = opt_usize_field(text, "first_k_dense_replace").unwrap_or(0);
        let dense_intermediate = opt_usize_field(text, "intermediate_size").unwrap_or(0);
        let layer_specs = (0..layers)
            .map(|layer| {
                let dense = mlp_types.get(layer).map_or(layer < first_dense, |t| t == "dense");
                let kind = layer_types.get(layer).map(String::as_str).unwrap_or("deepseek_sparse_attention");
                let attention = if kind == "linear_attention" {
                    AttentionKind::Kda
                } else {
                    AttentionKind::MlaDsa {
                        indexer: indexer_types.get(layer).map_or(true, |t| t == "full"),
                    }
                };
                LayerSpec {
                    attention,
                    ffn: if dense {
                        FfnKind::Dense { intermediate: dense_intermediate }
                    } else {
                        FfnKind::Moe
                    },
                }
            })
            .collect();
        let mtp = opt_usize_field(text, "num_nextn_predict_layers").unwrap_or(0);
        let mut notes = Vec::new();
        if let Some(hc) = opt_usize_field(text, "hc_mult") {
            notes.push(format!(
                "mHC width {hc}, Sinkhorn {} iterations, final collapse {}",
                opt_usize_field(text, "hc_sinkhorn_iters").unwrap_or(0),
                if self.id == "glm_next" { "unweighted mean" } else { "hc_head" },
            ));
        }
        let rope = opt_usize_field(text, "qk_rope_head_dim").unwrap_or(0);
        let kv_lora = opt_usize_field(text, "kv_lora_rank").unwrap_or(0);
        notes.push(format!(
            "MLA q_lora {} kv_lora {kv_lora} rope {rope}, qk nope {} v {} (absorbed query {}; FP8 latent record \
             {} B: E4M3 + 4 FP32 group scales{})",
            opt_usize_field(text, "q_lora_rank").unwrap_or(0),
            opt_usize_field(text, "qk_nope_head_dim").unwrap_or(0),
            opt_usize_field(text, "v_head_dim").unwrap_or(0),
            kv_lora + rope,
            kv_lora + 4 * (kv_lora / 128) + 2 * rope,
            if rope > 0 { " + BF16 RoPE" } else { ", no RoPE" },
        ));
        let kpool = opt_usize_field(text, "index_kpool").unwrap_or(1);
        notes.push(format!(
            "DSA index top-k {}{}",
            opt_usize_field(text, "index_topk").unwrap_or(0),
            if kpool > 1 {
                format!(
                    " tokens = {} pools of {kpool} (gated softmax pool keys + learned position bias){}; \
                     dense causal up to {} tokens",
                    opt_usize_field(text, "index_topk").unwrap_or(0) / kpool,
                    if text.get("index_kpool_always_select_tail").and_then(Value::as_bool) == Some(true) {
                        ", plus the open tail pool"
                    } else {
                        ""
                    },
                    opt_usize_field(text, "index_topk").unwrap_or(0) + kpool - 1,
                )
            } else {
                String::new()
            },
        ));
        if let Some(linear) = text.get("linear_attn_config") {
            let heads = opt_usize_field(linear, "num_heads").unwrap_or(0);
            let dim = opt_usize_field(linear, "head_dim").unwrap_or(0);
            let kda = layer_types.iter().filter(|t| *t == "linear_attention").count();
            notes.push(format!(
                "KDA {heads} heads x {dim}, short conv {}, gate lower bound {}; recurrent state {:.1} MiB per \
                 sequence ({kda} layers x FP32 {heads}x{dim}x{dim}) plus conv state",
                opt_usize_field(linear, "short_conv_kernel_size").unwrap_or(0),
                linear.get("gate_lower_bound").and_then(Value::as_f64).unwrap_or(0.0),
                (kda * heads * dim * dim * 4) as f64 / (1u64 << 20) as f64,
            ));
        }
        if let Some(limit) = text.get("swiglu_limit").and_then(Value::as_f64) {
            notes.push(format!(
                "SwiGLU clamp {limit} (gate <= {limit}, |up| <= {limit}) in dense, shared and routed experts"
            ));
        }
        Ok(ModelSpec {
            family: self.id,
            architecture: self.architecture.into(),
            hidden: usize_field(text, "hidden_size")?,
            vocab: usize_field(text, "vocab_size")?,
            layers: layer_specs,
            moe: Some(MoeSpec {
                experts: usize_field(text, "n_routed_experts")?,
                top_k: usize_field(text, "num_experts_per_tok")?,
                intermediate: usize_field(text, "moe_intermediate_size")?,
                shared_experts: opt_usize_field(text, "n_shared_experts").unwrap_or(0),
                shared_intermediate: usize_field(text, "moe_intermediate_size")?,
                scoring: text.get("scoring_func").and_then(Value::as_str).unwrap_or("sigmoid").into(),
                routed_scaling: text.get("routed_scaling_factor").and_then(Value::as_f64),
                groups: match (opt_usize_field(text, "n_group"), opt_usize_field(text, "topk_group")) {
                    (Some(n), Some(k)) if n > 1 => Some((n, k)),
                    _ => None,
                },
            }),
            speculator: (mtp > 0).then_some(SpeculatorSpec::NativeMtp { layers: mtp }),
            tables: Vec::new(),
            vision: checkpoint.config.get("vision_config").is_some(),
            notes,
        })
    }

    fn classify(&self, spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        if name == "lm_head.weight" {
            return Some(TensorRole::new(LmHead));
        }
        if name.starts_with("model.visual.") || name.starts_with("visual.") || name.starts_with("model.vision") {
            return Some(TensorRole::new(Vision));
        }
        let name = strip_model_prefix(name)?;
        match name {
            "embed_tokens.weight" => return Some(TensorRole::new(Embedding)),
            "norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        let (layer, rest) = indexed(name, "layers.")?;
        // The layer after the backbone is the native MTP layer.
        if layer >= spec.layers.len() {
            if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
                return Some(TensorRole::expert(SpeculatorExpert, layer, expert));
            }
            return Some(TensorRole::layer(Speculator, layer));
        }
        if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
            return Some(TensorRole::expert(RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("self_attn.indexer.") {
            Indexer
        } else if rest.starts_with("self_attn.") {
            Attention
        } else if rest.starts_with("mlp.gate.") {
            Router
        } else if rest.starts_with("mlp.shared_experts.") {
            SharedExpert
        } else if rest.starts_with("mlp.") {
            DenseFfn
        } else if rest.starts_with("hc_") {
            HyperConnection
        } else if rest.ends_with("layernorm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        let glmrt = "../glmrt (the GLM-5.3 engine this family ports from)";
        let (what, how) = match (self.id, component) {
            ("glm_next", Component::Attention) => (
                "hybrid attention: Kimi Delta Attention (linear) layers plus MLA+DSA layers".to_string(),
                "Runs as the glmf programs (b12x integration/cuteafd/glmf.py: token-sequential KDA \
                 recurrence, no-RoPE MLA over 528-byte FP8 records, pooled indexer) in \
                 cuteafd-daemon src/glmf. Faster prefill: b12x sequence/kda_prefill (chunked) for \
                 the recurrence.".to_string(),
            ),
            ("glm_next", Component::Speculator) | ("glm_next", Component::SpeculatorExpert) => (
                "native MTP layer 45 (not run); DFlash2 drafter incoai/GLM-5.3-Flash-DFlash2".to_string(),
                "serve-glmf verifies copy-window drafts only. KDA state needs a rollback for any \
                 speculator: serve-glmf backs it up per verify and replays kept rows (glmf/serve.rs).".to_string(),
            ),
            ("glm_next", Component::Indexer) => (
                "DSA indexer over 4-token key pools".to_string(),
                "Pool keys are a per-channel softmax over each complete pool of LayerNorm(wk x) weighted by \
                 index_kpool_compress_gate x + ape; score = sum_h w_h relu(q_h . k_pool) / sqrt(128); top \
                 index_topk/kpool pools expand to tokens, plus the open tail pool. Up to index_topk + kpool - 1 \
                 tokens every token is selected, so short contexts are dense causal MLA.".to_string(),
            ),
            ("glm_next", Component::RoutedExpert) | ("glm_next", Component::SpeculatorExpert) => (
                "top-8 of 288 sigmoid routed experts with SwiGLU clamp 10".to_string(),
                "EXL3 K3/K4 checkpoints: the glm:exl3 recipe at hidden 4096 / inter 2048 / 288 experts with \
                 the clamp enabled (python/tools/aot/package_v41_exl3_aot.py, exl3_cross_sm121.py for Sparks). \
                 FP8 checkpoints: the fp8 routed family (native/cmake/shared/fp8_moe.cmake) at this geometry.".to_string(),
            ),
            (_, Component::Attention) | (_, Component::Indexer) => (
                "MLA with DeepSeek Sparse Attention indexer".to_string(),
                format!("Port from {glmrt}: native/cuda/kernels/dsa_indexer.cu and the real_full \
                 attention/mla.rs + attention/residual/dsa_indexer.rs path. b12x: attention/sparse_mla, \
                 dsa_indexer. Shared indexer layers reuse the previous full layer's selection \
                 (indexer_types)."),
            ),
            (_, Component::RoutedExpert) => (
                "top-8 sigmoid routed experts".to_string(),
                format!("Spark expertd-native is specialized to V4.1 geometry. Export b12x fused_moe for \
                 this geometry (see the model spec) and format; {glmrt} has mixed EXL3 K3/K4 top-8 \
                 dispatch (native/cuda/kernels/b12x_mixed_aot.h, b12x_direct.cu) and loader layouts \
                 (loader exl3_format.rs Glm53Exl3MixedTp4LayerLayout)."),
            ),
            (_, Component::Router) => (
                "sigmoid noaux_tc router with e_score_correction_bias, routed scale 2.5".to_string(),
                format!("{glmrt} native/cuda/kernels/router.cu implements the top-8 path."),
            ),
            (_, Component::Speculator) | (_, Component::SpeculatorExpert) => (
                "native MTP layer; DFlash2 external drafter preferred".to_string(),
                format!("{glmrt} commands/real_full/{{mtp,dflash*}}.rs. DFlash2 (incoai/GLM-5.3-DFlash2) \
                 is the measured best speculator for GLM-5.3."),
            ),
            (_, Component::DenseFfn) | (_, Component::SharedExpert) => (
                "FP8 dense and shared FFN on the coordinator".to_string(),
                "b12x gemm block_fp8_linear covers 128x128 FP8 at M small; reuse V4.1's shared-expert \
                 path with this geometry.".to_string(),
            ),
            (_, Component::HyperConnection) => (
                "mHC hyper-connections".to_string(),
                "V4.1 mHC kernels (v41_hc) apply at this width.".to_string(),
            ),
            _ => return None,
        };
        Some(Hint { what, how })
    }
}
