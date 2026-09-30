//! DeepSeek V4 and V4.1: compressed MLA with indexers, mHC hyper-connections,
//! DeepSeek-native tensor names (optionally with HF-named EXL3 experts).
use anyhow::Result;
use serde_json::Value;

use crate::plan::checkpoint::{opt_usize_field, usize_field, Checkpoint};
use crate::plan::family::{Family, Hint, RuntimeStatus};
use crate::plan::format::WeightFormat;
use crate::plan::names::{indexed, indexed_tail};
use crate::plan::spec::*;

pub struct DeepSeek {
    id: &'static str,
    architecture: &'static str,
    runtime: RuntimeStatus,
}

pub static DEEPSEEK_V41: DeepSeek = DeepSeek {
    id: "deepseek_v41",
    architecture: "DeepseekV41ForCausalLM",
    runtime: RuntimeStatus::Serving,
};
pub static DEEPSEEK_V4: DeepSeek = DeepSeek {
    id: "deepseek_v4",
    architecture: "DeepseekV4ForCausalLM",
    runtime: RuntimeStatus::Serving,
};

fn usize_list(config: &Value, key: &str) -> Vec<usize> {
    config
        .get(key)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
        .unwrap_or_default()
}

impl Family for DeepSeek {
    fn id(&self) -> &'static str {
        self.id
    }
    fn runtime(&self) -> RuntimeStatus {
        self.runtime
    }
    fn detect(&self, checkpoint: &Checkpoint) -> bool {
        checkpoint.architectures().iter().any(|arch| arch == self.architecture)
    }

    fn spec(&self, checkpoint: &Checkpoint) -> Result<ModelSpec> {
        let text = checkpoint.text_config();
        let layers = usize_field(text, "num_hidden_layers")?;
        let ratios = usize_list(text, "compress_ratios");
        let index_sources = usize_list(text, "index_source_layer_ids");
        let hash_layers = opt_usize_field(text, "num_hash_layers").unwrap_or(0);
        // A layer carries its own indexer when its tensors say so; V4.1
        // publishes the owners explicitly, V4 puts one on every ratio-4 layer.
        let has_indexer = |layer: usize| {
            let prefix = format!("layers.{layer}.attn.indexer.");
            checkpoint.tensors.iter().any(|t| t.meta.name.starts_with(&prefix))
                || index_sources.contains(&layer)
        };
        let layer_specs = (0..layers)
            .map(|layer| LayerSpec {
                attention: AttentionKind::CompressedMla {
                    ratio: ratios.get(layer).copied().unwrap_or(0),
                    indexer: has_indexer(layer),
                },
                ffn: FfnKind::Moe,
            })
            .collect();
        let dspark_stages = checkpoint
            .tensors
            .iter()
            .filter_map(|t| indexed(&t.meta.name, "mtp.").map(|(index, _)| index))
            .max()
            .map_or(0, |max| max + 1);
        let speculator = (dspark_stages > 0).then(|| SpeculatorSpec::Dspark {
            stages: dspark_stages,
            experts: opt_usize_field(text, "dspark_n_routed_experts")
                .or_else(|| opt_usize_field(text, "n_routed_experts"))
                .unwrap_or(0),
            top_k: opt_usize_field(text, "dspark_num_experts_per_tok")
                .or_else(|| opt_usize_field(text, "num_experts_per_tok"))
                .unwrap_or(0),
            target_layers: usize_list(text, "dspark_target_layer_ids"),
        });
        let engram = usize_list(text, "engram_layer_ids");
        let mut tables = Vec::new();
        if !engram.is_empty() {
            tables.push(MappedTableSpec { name: "engram".into(), layers: engram });
        }
        let mut notes = vec![format!("compress ratios {:?}", dedup(&ratios))];
        if let Some(hc) = opt_usize_field(text, "hc_mult") {
            notes.push(format!("mHC width {hc}"));
        }
        if hash_layers > 0 {
            notes.push(format!("hash-routed layers 0..{hash_layers} (ffn.gate.tid2eid)"));
        }
        if let Some(sources) = text.get("kv_source_layer_ids") {
            notes.push(format!("CED KV sources {sources}"));
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
                scoring: text.get("scoring_func").and_then(Value::as_str).unwrap_or("softmax").into(),
                routed_scaling: text.get("routed_scaling_factor").and_then(Value::as_f64),
                groups: None,
            }),
            speculator,
            tables,
            vision: checkpoint.config.get("vision_config").is_some(),
            notes,
        })
    }

    fn classify(&self, _spec: &ModelSpec, name: &str) -> Option<TensorRole> {
        use Component::*;
        match name {
            "embed.weight" => return Some(TensorRole::new(Embedding)),
            "head.weight" => return Some(TensorRole::new(LmHead)),
            "norm.weight" => return Some(TensorRole::new(Norm)),
            _ => {}
        }
        if name.starts_with("hc_head_") {
            return Some(TensorRole::new(HyperConnection));
        }
        if name.starts_with("vision.") || name.starts_with("aligner.") || name.starts_with("image_") {
            return Some(TensorRole::new(Vision));
        }
        // EXL3 publications store routed experts with HF names.
        if let Some((layer, rest)) = indexed(name, "model.layers.") {
            if let Some((expert, _)) = indexed(rest, "mlp.experts.") {
                return Some(TensorRole::expert(RoutedExpert, layer, expert));
            }
            return Some(TensorRole::layer(Other, layer));
        }
        if let Some((stage, rest)) = indexed_tail(name, "mtp.") {
            if let Some((expert, _)) = indexed(rest, "ffn.experts.")
                .or_else(|| indexed(rest, "mlp.experts."))
            {
                return Some(TensorRole::expert(SpeculatorExpert, stage, expert));
            }
            return Some(TensorRole::layer(Speculator, stage));
        }
        let (layer, rest) = indexed(name, "layers.")?;
        let component = if let Some((expert, _)) = indexed(rest, "ffn.experts.") {
            return Some(TensorRole::expert(RoutedExpert, layer, expert));
        } else if rest.starts_with("attn.compressor.") || rest.starts_with("attn.indexer.compressor.") {
            Compressor
        } else if rest.starts_with("attn.indexer.") {
            Indexer
        } else if rest.starts_with("attn.") {
            Attention
        } else if rest.starts_with("ffn.gate.") {
            Router
        } else if rest.starts_with("ffn.shared_experts.") {
            SharedExpert
        } else if rest.starts_with("hc_") {
            HyperConnection
        } else if rest.starts_with("engram.embed.") {
            MappedTable
        } else if rest.starts_with("engram.") {
            TableProjection
        } else if rest.ends_with("norm.weight") {
            Norm
        } else {
            Other
        };
        Some(TensorRole::layer(component, layer))
    }

    fn executes(&self, component: Component, format: &WeightFormat) -> bool {
        if self.runtime != RuntimeStatus::Serving {
            return false;
        }
        use WeightFormat::*;
        if self.id == "deepseek_v4" {
            // serve-dsv4: native MXFP4 or EXL3 (expertd-native, packages
            // exl3-dsv4*-k*) routed experts on the Sparks, 128x128 block-FP8
            // projections and BF16/F32 tensors on the RTX, integer
            // hash-routing tables.
            return match component {
                Component::RoutedExpert => matches!(format, Mxfp4 { group: 32 } | Exl3 { bits: 2..=4 }),
                Component::Speculator | Component::SpeculatorExpert => false,
                _ => matches!(format, Fp8Block { block: (128, 128) } | Bf16 | F32 | Int),
            };
        }
        match component {
            // Native MXFP4, EXL3 K2-K4 packages and ModelOpt W4A4 NVFP4.
            Component::RoutedExpert | Component::SpeculatorExpert => matches!(
                format,
                Mxfp4 { group: 32 } | Exl3 { bits: 2..=4 } | Nvfp4
            ),
            // Engram rows: FP8 with 1x32 E8M0 scales, or the FP4PLE NVFP4 variant.
            Component::MappedTable => matches!(format, Fp8Block { block: (1, 32) } | Nvfp4),
            _ => matches!(format, Fp8Block { block: (32, 32) } | Bf16 | F32),
        }
    }

    fn optional(&self, component: Component) -> bool {
        self.id == "deepseek_v4" && matches!(component, Component::Speculator | Component::SpeculatorExpert)
    }

    fn component_hint(&self, component: Component) -> Option<Hint> {
        if self.id == "deepseek_v4" && component == Component::RoutedExpert {
            return Some(Hint {
                what: "DeepSeek V4 routed experts in NVFP4".into(),
                how: "Spark expertd-native serves native MXFP4 and EXL3 K2-K4 experts at the checkpoint \
                      geometry (ExpertGeometry, read_expert_catalog; EXL3 packages exl3-dsv4f|dsv4p-k<tiers> \
                      from CUTEAFD_EXPERT_FAMILIES=dsv4p:exl3-k23). ModelOpt NVFP4 needs the V4.1 NVFP4 \
                      contract (loader v41_nvfp4) generalized the same way.".into(),
            });
        }
        if self.runtime == RuntimeStatus::Serving {
            return None;
        }
        let (what, how) = match component {
            Component::Attention | Component::Compressor | Component::Indexer => (
                "DeepSeek V4 compressed MLA (ratios 4/128 alternating), per-layer compressor and indexer",
                "Map each component against the official inference/model.py. V4.1's attention stack (rust daemon v41_attention_*, v41_compressor, v41_index_*, \
                 native/families/deepseek_v41/) is the template; V4 differs in ratio schedule (4 and 128 \
                 instead of 2 and 1), has an indexer compressor per ratio-4 layer, no CED encoder/decoder \
                 KV sharing, and FP8 128x128 blocks instead of 32x32. b12x kernels: \
                 attention/compressed_sparse_mla, dsv4_compressor, dsa_indexer. The legacy ds4rt engine \
                 (../ds4rt) served this family; its real_full attention path is the numerical reference.",
            ),
            Component::Router => (
                "hash routing on layers 0..num_hash_layers (ffn.gate.tid2eid) plus sqrtsoftplus noaux_tc",
                "tid2eid maps token id to its 6 experts for the first layers (no scores); later layers use \
                 V4.1's router with gate.bias. Add a hash-route path to the router kernel launch.",
            ),
            Component::RoutedExpert | Component::SpeculatorExpert => (
                "routed experts at this family's geometry",
                "Spark expertd-native and the V4.1 AOT exports are specialized to hidden 5120 / \
                 intermediate 2304 / 384 experts / top-6. Parameterize python/tools/export_b12x_v41_* \
                 by (hidden, intermediate, experts, top_k, format) and select the variant from the \
                 model spec; V4 Flash is 4096/2048/256/6 MXFP4, V4 Pro EXL3 K2 is 7168/3072/384/6 \
                 (b12x moe fused_moe Trellis).",
            ),
            Component::HyperConnection => (
                "mHC with hc_head_* output mixing",
                "V4.1 mHC kernels (v41_hc) apply; V4 adds the hc_head_{fn,base,scale} output head.",
            ),
            Component::Speculator => (
                "three-stage dSpark drafter",
                "Same structure as V4.1 dSpark (markov_head, confidence_head, main_proj) at this \
                 family's width; reuse v41_dspark with geometry parameters.",
            ),
            _ => return None,
        };
        Some(Hint { what: what.into(), how: how.into() })
    }
}

fn dedup(values: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = values.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}
