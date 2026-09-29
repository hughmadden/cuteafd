//! Xiaomi MiMo V2: hybrid full / sliding-window GQA with attention sinks,
//! sigmoid top-8 experts without shared experts, dense-FFN MTP layers.
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{Family, Hint, RuntimeStatus};
use crate::plan::names::indexed;
use crate::plan::spec::*;

const GIB: f64 = (1u64 << 30) as f64;

pub struct MiMo;
pub static MIMO_V2: MiMo = MiMo;

const ARCHITECTURES: &[&str] = &["MiMoV2FlashForCausalLM", "MiMoV2ForCausalLM"];

fn usize_list(config: &Value, key: &str) -> Vec<usize> {
    config
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
        .unwrap_or_default()
}

impl Family for MiMo {
    fn id(&self) -> &'static str {
        "mimo_v2"
    }
    fn runtime(&self) -> RuntimeStatus {
        RuntimeStatus::Planned
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|arch| ARCHITECTURES.contains(&arch.as_str()))
    }

    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec> {
        let text = checkpoint.text_config();
        let layers = usize_field(text, "num_hidden_layers")?;
        let hybrid = usize_list(text, "hybrid_layer_pattern");
        let moe_freq = usize_list(text, "moe_layer_freq");
        let window = opt_usize_field(text, "sliding_window_size")
            .or_else(|| opt_usize_field(text, "sliding_window"))
            .unwrap_or(0);
        let sinks = text.get("add_swa_attention_sink_bias").and_then(Value::as_bool).unwrap_or(false);
        let layer_specs = (0..layers)
            .map(|layer| LayerSpec {
                attention: if hybrid.get(layer) == Some(&1) {
                    AttentionKind::SlidingGqa {
                        heads: opt_usize_field(text, "swa_num_attention_heads").unwrap_or(0),
                        kv_heads: opt_usize_field(text, "swa_num_key_value_heads").unwrap_or(0),
                        head_dim: opt_usize_field(text, "swa_head_dim").unwrap_or(0),
                        window,
                        sinks,
                    }
                } else {
                    AttentionKind::Gqa {
                        heads: opt_usize_field(text, "num_attention_heads").unwrap_or(0),
                        kv_heads: opt_usize_field(text, "num_key_value_heads").unwrap_or(0),
                        head_dim: opt_usize_field(text, "head_dim").unwrap_or(0),
                    }
                },
                ffn: if moe_freq.get(layer) == Some(&1) {
                    FfnKind::Moe
                } else {
                    FfnKind::Dense { intermediate: opt_usize_field(text, "intermediate_size").unwrap_or(0) }
                },
            })
            .collect();
        let mtp = checkpoint
            .tensors
            .iter()
            .filter_map(|t| indexed(&t.meta.name, "model.mtp.layers.").map(|(i, _)| i))
            .max()
            .map_or(0, |max| max + 1);
        Ok(ModelSpec {
            family: "mimo_v2",
            architecture: checkpoint.architectures().first().cloned().unwrap_or_default(),
            hidden: usize_field(text, "hidden_size")?,
            vocab: usize_field(text, "vocab_size")?,
            layers: layer_specs,
            moe: Some(MoeSpec {
                experts: usize_field(text, "n_routed_experts")?,
                top_k: usize_field(text, "num_experts_per_tok")?,
                intermediate: usize_field(text, "moe_intermediate_size")?,
                shared_experts: 0,
                shared_intermediate: 0,
                scoring: text.get("scoring_func").and_then(Value::as_str).unwrap_or("sigmoid").into(),
                routed_scaling: text.get("routed_scaling_factor").and_then(Value::as_f64),
                groups: None,
            }),
            speculator: (mtp > 0).then_some(SpeculatorSpec::NativeMtp { layers: mtp }),
            tables: Vec::new(),
            vision: checkpoint.config.get("vision_config").is_some(),
            notes: {
                let mut notes = vec![format!(
                    "partial rotary {}, attention value scale {}",
                    text.get("partial_rotary_factor").cloned().unwrap_or(Value::Null),
                    text.get("attention_value_scale").cloned().unwrap_or(Value::Null)
                )];
                if checkpoint.tensors.iter().any(|t| t.meta.name.ends_with("self_attn.qkv_proj.weight")) {
                    let tp = crate::mimo_v2::checkpoint_tp(&checkpoint.snapshot).unwrap_or(1);
                    notes.push(format!("fused qkv_proj stored TP{tp}-interleaved ([q|k|v] per row shard, \
                        own 128x128 grid per shard); the engine de-interleaves it (FusedQkvLayout)"));
                }
                if let Some(router) = checkpoint.tensors.iter().find(|t| t.meta.name.ends_with("mlp.gate.weight")) {
                    notes.push(format!("router weight {:?}", router.meta.dtype));
                }
                if checkpoint.snapshot.join("dflash").join("config.json").exists() {
                    notes.push("dflash/: DFlash block drafter (not planned: its own qwen3-style config)".into());
                }
                if let Ok(family) = crate::mimo_v2::MimoV2Config::from_hf(text).and_then(|c| c.program_family().map(str::to_owned)) {
                    notes.push(format!("coordinator programs: family {family} (CUTEAFD_ENABLE_MIMO_AOT, \
                        CUTEAFD_MIMO_GEOMETRIES={family})"));
                }
                notes
            },
        })
    }

    fn classify(&self, _spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        match name {
            "lm_head.weight" => return Some(TensorRole::new(LmHead)),
            "model.embed_tokens.weight" => return Some(TensorRole::new(Embedding)),
            "model.norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        if let Some((layer, _)) = indexed(name, "model.mtp.layers.") {
            return Some(TensorRole::layer(Speculator, layer));
        }
        if name.starts_with("visual.") || name.starts_with("model.visual.") || name.starts_with("audio")
            || name.starts_with("speech_") || name.starts_with("model.audio")
        {
            return Some(TensorRole::new(Vision));
        }
        let (layer, rest) = indexed(name, "model.layers.")?;
        if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
            return Some(TensorRole::expert(RoutedExpert, layer, expert));
        }
        let component = if rest.starts_with("self_attn.") {
            Attention
        } else if rest.starts_with("mlp.gate.") {
            Router
        } else if rest.starts_with("mlp.") {
            DenseFfn
        } else if rest.ends_with("layernorm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn component_hint_for(&self, spec: &ModelSpec, component: Component, formats: &[String]) -> Option<Hint> {
        let moe = spec.moe.as_ref()?;
        let (h, i, e, k) = (spec.hidden, moe.intermediate, moe.experts, moe.top_k);
        let mxfp4 = formats.iter().any(|f| f.starts_with("mxfp4"));
        match component {
            Component::RoutedExpert if mxfp4 => {
                // TP6: whole 32-blocks per rank (352/320 of 2048), stored zero-padded to 128.
                let widest = (i / 32).div_ceil(6) * 32;
                let padded = widest.div_ceil(128) * 128;
                let per_rank = |slice: usize| (spec.moe_layers() * e * 3 * slice * h) as f64 * (0.5 + 1.0 / 32.0) / GIB;
                Some(Hint {
                    what: format!("sigmoid top-{k} routed experts, no shared expert, MXFP4 (packed E2M1 U8 [N, K/2], \
                        even element low nibble, UE8M0 U8 [N, K/32]): H {h}, I {i}, {e} experts"),
                    how: format!("Exact family `mimop:fp8` (b12x fp8_moe weights=mxfp4: E2M1 x 2^(s-127) widened \
                        to BF16, BF16 MMA; packages fp8-mimop tp1 coordinator, tp6/tp2 Spark, \
                        python/tools/package_fp8_moe_aot.py --geometry mimop [--cross-sm121]). Spark layout TP6 \
                        over six ranks: whole 32-blocks per rank ({widest}/{} rows) zero-padded to {padded}, \
                        {:.1} GiB per rank (TP2xEP3: {:.1} GiB, but a decode step reads all of a row's experts \
                        that land on one EP group). Missing: an MXFP4 streaming/TMA GEMM route for large \
                        prefill steps (the grouped GEMV serves every row count).",
                        widest - 32, per_rank(padded), per_rank(i / 2) / 3.0),
                })
            }
            Component::RoutedExpert => Some(Hint {
                what: format!("sigmoid top-{k} routed experts, no shared expert (H {h}, I {i}, {e} experts)"),
                how: format!("Exact family `mimo:fp8` (b12x fp8_moe over E4M3 + FP32 128x128 scales; packages \
                    fp8-mimo tp1/tp2/tp4) for H {h} / I {i} / {e} experts / top-{k}; UE8M0 requantization costs \
                    +0.011 nats mean NLL (V2 Flash), EXL3 is the compact alternative."),
            }),
            // Pro's router weight is BF16 (the bias FP32); Flash's is FP32.
            Component::Router if formats.iter().any(|f| f == "bf16") => Some(Hint {
                what: "sigmoid noaux_tc router with FP32 e_score_correction_bias, BF16 router weight".into(),
                how: "mimop_router_scores (BF16 weight, FP32 accumulation) then cuteafd_router_select \
                    (sigmoid, normalized, no routed scaling).".into(),
            }),
            _ => self.component_hint(component),
        }
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        let (what, how) = match component {
            Component::Attention => (
                "GQA full attention plus 128-token sliding-window GQA with learned sink bias",
                "Coordinator programs exist (b12x integration mimo_{full,swa}_{producer,attention}, mimo_o; \
                 family mimo for V2 Flash, mimop for V2.6 Pro): BF16 KV records, paged full layers, 256-slot SWA \
                 rings, sinks, QK 192 / V 128, NeoX RoPE on 64 dims. FP8 scale grids restart per segment \
                 (fp8-block128x128-segmented): Flash's full-layer k_proj per KV head (128 + 64 rows), Pro's fused \
                 qkv_proj per checkpoint TP shard. Next: FP8 decode weights, multi-row decode tiles for verify \
                 steps, FP8 KV.",
            ),
            Component::RoutedExpert => (
                "sigmoid top-8 routed experts, no shared expert",
                "Needs a Spark expert family at hidden 4096 / intermediate 2048 / 256 experts / top-8: FP8 \
                 E4M3 weights with FP32 128x128 block scales (UE8M0 requantization costs +0.011 nats mean NLL), \
                 or EXL3 quantization. V2.6 Pro stores MXFP4 with E8M0 scales typed U8.",
            ),
            Component::Router => (
                "sigmoid noaux_tc router with e_score_correction_bias (router weight FP32 on Flash)",
                "mimo_router_scores (FP32 weight as BF16 hi + lo, FP32 sums) then cuteafd_router_select \
                 (sigmoid, normalized, no routed scaling).",
            ),
            Component::Speculator => (
                "MTP layers (SWA attention with sinks, dense FFN, eh_proj/enorm/hnorm)",
                "Reuse the mimo/mimop SWA programs and the dense FFN program; add the eh_proj fusion \
                 (BF16 [H, 2H]).",
            ),
            _ => return None,
        };
        Some(Hint { what: what.into(), how: how.into() })
    }
}
