//! Index-derived role requirements. File inventories never infer missing headers.
use super::{checkpoint::Checkpoint, family, Component, ExpertPlacement};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadRole {
    Coordinator { local_experts: bool, speculator: bool },
    Spark { rank: usize, world: usize },
    Vision,
    Audio,
    Drafter,
    /// A drafter with separate weights; no target-checkpoint tensors are read.
    ExternalDrafter,
}

impl ReadRole {
    pub fn parse(value: &str, placement: ExpertPlacement, speculator: bool) -> Result<Self> {
        match value {
            "coordinator" | "rtx0" | "rtx1" => Ok(Self::Coordinator {
                local_experts: placement == ExpertPlacement::Local, speculator }),
            "vision" => Ok(Self::Vision),
            "audio" => Ok(Self::Audio),
            "drafter" => Ok(if speculator { Self::Drafter } else { Self::ExternalDrafter }),
            _ => {
                let rank = value.strip_prefix("spark").context("role must be coordinator, rtx0, rtx1, sparkN, vision, audio or drafter")?
                    .parse::<usize>().context("Spark rank must be an integer")?;
                let world = placement.spark_ranks();
                ensure!(rank < world, "spark{rank} is outside TP{world}");
                Ok(Self::Spark { rank, world })
            }
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Coordinator { .. } => "coordinator".into(),
            Self::Spark { rank, world } => format!("spark{rank} (TP{world})"),
            Self::Vision => "vision".into(), Self::Audio => "audio".into(),
            Self::Drafter | Self::ExternalDrafter => "drafter".into(),
        }
    }
}

pub fn required_tensors(checkpoint: &Checkpoint, role: ReadRole) -> Result<BTreeSet<String>> {
    let family = family::detect(checkpoint).context("checkpoint family is not supported")?;
    let model = family.open(checkpoint)?;
    let ownership = match role {
        ReadRole::Coordinator { speculator: true, .. } | ReadRole::Drafter =>
            super::spec::DraftOwnership::selected(family.id(),
                if matches!(family.id(), "deepseek_v4" | "deepseek_v41") { "dspark" } else { "mtp" }),
        _ => super::spec::DraftOwnership::None,
    };
    Ok(checkpoint.tensor_names().filter(|name| {
        let Some(tensor) = family.classify(model.spec(), name) else { return false };
        if !ownership.includes(tensor.component) { return false; }
        match role {
            ReadRole::Coordinator { local_experts, speculator } => match tensor.component {
                Component::RoutedExpert => local_experts,
                Component::Speculator | Component::SpeculatorExpert => speculator,
                Component::Vision | Component::Audio => false,
                _ => true,
            },
            // TP slices the intermediate axis; every rank reads each expert's
            // shard, not a disjoint set of experts.
            ReadRole::Spark { .. } => tensor.component == Component::RoutedExpert,
            ReadRole::Vision => tensor.component == Component::Vision,
            ReadRole::Audio => tensor.component == Component::Audio,
            ReadRole::Drafter => matches!(tensor.component, Component::Speculator | Component::SpeculatorExpert),
            ReadRole::ExternalDrafter => false,
        }
    }).map(str::to_owned).collect())
}

pub fn open_role(snapshot: &Path, role: ReadRole) -> Result<Checkpoint> {
    open_role_view(&super::checkpoint::CheckpointView::select(snapshot, super::checkpoint::AttentionMode::Hqa)?, role)
}

pub fn open_role_view(view: &super::checkpoint::CheckpointView, role: ReadRole) -> Result<Checkpoint> {
    let inventory = Checkpoint::inventory_view(view)?;
    let names = required_tensors(&inventory, role)?;
    Checkpoint::open_view_for_role(view, &role.label(), |name| names.contains(name))
}

#[derive(Debug, Serialize)]
pub struct RequiredFile {
    pub path: String,
    pub bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct FileManifest {
    pub role: String,
    pub snapshot: String,
    pub repo_id: Option<String>,
    pub revision: Option<String>,
    pub files: Vec<RequiredFile>,
    pub total_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub additional_snapshots: Vec<FileManifest>,
}

pub fn manifest(snapshot: &Path, role: ReadRole) -> Result<FileManifest> {
    manifest_roles(snapshot, &[role])
}

/// A host's file list is the union of the processes it runs, never a shard split.
pub fn manifest_roles(snapshot: &Path, roles: &[ReadRole]) -> Result<FileManifest> {
    manifest_roles_view(&super::checkpoint::CheckpointView::select(snapshot, super::checkpoint::AttentionMode::Hqa)?, roles)
}

pub fn manifest_roles_view(view: &super::checkpoint::CheckpointView, roles: &[ReadRole]) -> Result<FileManifest> {
    let snapshot = view.snapshot();
    ensure!(!roles.is_empty(), "a file inventory needs at least one role");
    let inventory = Checkpoint::inventory_view(view)?;
    let mut tensors = BTreeSet::new();
    for role in roles { tensors.extend(required_tensors(&inventory, *role)?); }
    let mut files: BTreeSet<String> = tensors.iter().filter_map(|name| inventory.weight_map.get(name).cloned()).collect();
    if roles.iter().any(|r| matches!(r, ReadRole::Drafter | ReadRole::Coordinator { speculator: true, .. })) {
        // Native speculators read calibrated confidence files by numerical key.
        for entry in std::fs::read_dir(snapshot)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.path().is_file() && name.starts_with("draft-confidence.") && name.ends_with(".json") {
                files.insert(name);
            }
        }
    }
    if roles.contains(&ReadRole::Audio) && snapshot.join("audio_tokenizer").is_dir() {
        collect_directory(snapshot, &snapshot.join("audio_tokenizer"), &mut files)?;
    }
    manifest_from_files_view(snapshot, roles.iter().map(|r| r.label()).collect::<Vec<_>>().join(", "), files, Some(view))
}

/// Standalone drafters/encoders have their own config and naming convention;
/// their complete index is required, without pretending they are text families.
pub fn manifest_standalone(snapshot: &Path, role: &str) -> Result<FileManifest> {
    // A standalone model's own loaders may open trained vectors or other
    // auxiliary payloads, not just entries in a safetensors weight map.
    let mut files = BTreeSet::new();
    collect_directory(snapshot, snapshot, &mut files)?;
    manifest_from_files(snapshot, role.into(), files)
}

pub fn include_directory(manifest: &mut FileManifest, snapshot: &Path, directory: &str) -> Result<()> {
    let mut files = manifest.files.iter().map(|f| f.path.clone()).collect();
    collect_directory(snapshot, &snapshot.join(directory), &mut files)?;
    manifest.files = files.into_iter().map(|path| {
        let bytes = std::fs::metadata(snapshot.join(&path)).ok().filter(|m| m.is_file()).map(|m| m.len());
        RequiredFile { path, bytes }
    }).collect();
    manifest.total_bytes = manifest.files.iter().try_fold(0u64, |total, file| total.checked_add(file.bytes?));
    Ok(())
}

fn collect_directory(root: &Path, directory: &Path, files: &mut BTreeSet<String>) -> Result<()> {
    for entry in std::fs::read_dir(directory).with_context(|| format!("inventory {}", directory.display()))? {
        let path = entry?.path();
        if path.is_dir() { collect_directory(root, &path, files)?; }
        else if path.is_file() {
            files.insert(path.strip_prefix(root)?.to_str().context("non-UTF8 inventory path")?.to_owned());
        }
    }
    Ok(())
}

fn manifest_from_files(snapshot: &Path, role: String, files: BTreeSet<String>) -> Result<FileManifest> {
    manifest_from_files_view(snapshot, role, files, None)
}

fn manifest_from_files_view(snapshot: &Path, role: String, mut files: BTreeSet<String>,
    view: Option<&super::checkpoint::CheckpointView>) -> Result<FileManifest> {
    // Named inputs opened by checkpoint, tokenizer, template and processor
    // loaders. Weight shards remain role-filtered by the checkpoint index.
    for name in ["config.json", "model.safetensors.index.json", "generation_config.json",
        "quantize_config.json", "quantization_config.json", "hf_quant_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "tokenizer.model", "spiece.model", "merges.txt", "vocab.txt", "vocab.json",
        "chat_template.jinja", "preprocessor_config.json", "processor_config.json",
        "chat_template.json", "inference/config.json"] {
        let active = view.is_none_or(|view| !matches!(name, "config.json" | "model.safetensors.index.json"
            | "quantize_config.json" | "quantization_config.json" | "hf_quant_config.json")
            || name == view.config_name() || name == view.index_name() || view.quant_names().iter().any(|q| q == name));
        if active && snapshot.join(name).is_file() { files.insert(name.into()); }
    }
    // A checkpoint-declared index (config.json transformers_weights) is opened instead.
    if let Some(view) = view {
        files.insert(view.config_name().into());
        files.extend(view.quant_names().iter().cloned());
    }
    let index = match view { Some(view) => view.index_name().to_owned(), None => super::checkpoint::index_name(snapshot)? };
    if snapshot.join(&index).is_file() { files.insert(index); }
    let files: Vec<_> = files.into_iter().map(|path| {
        let bytes = std::fs::metadata(snapshot.join(&path)).ok().filter(|m| m.is_file()).map(|m| m.len());
        RequiredFile { path, bytes }
    }).collect();
    let total_bytes = files.iter().try_fold(0u64, |total, file| total.checked_add(file.bytes?));
    let cache_repo = snapshot.parent().filter(|p| p.file_name().is_some_and(|n| n == "snapshots"))
        .and_then(|p| p.parent());
    let repo_id = cache_repo.and_then(|p| p.file_name()).and_then(|n| n.to_str())
        .and_then(|name| name.strip_prefix("models--")).map(|name| name.replace("--", "/"));
    let revision = cache_repo.and_then(|_| snapshot.file_name()).and_then(|n| n.to_str()).map(str::to_owned);
    Ok(FileManifest { role, snapshot: snapshot.display().to_string(), repo_id, revision, files, total_bytes,
        additional_snapshots: Vec::new() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{mimo_flash_config, mimo_flash_tensors, write_safetensors};

    #[test]
    fn directory_additions_preserve_selected_bundle_and_auxiliary_snapshots() -> Result<()> {
        let root = tempfile::tempdir()?;
        std::fs::write(root.path().join("config.json"), b"{}")?;
        std::fs::write(root.path().join("config.compact-attention.json"), b"compact")?;
        std::fs::create_dir(root.path().join("aux"))?;
        std::fs::write(root.path().join("aux/vector.bin"), b"123")?;
        let mut manifest = FileManifest { role: "coordinator".into(), snapshot: root.path().display().to_string(),
            repo_id: None, revision: None, files: vec![RequiredFile {
                path: "config.compact-attention.json".into(), bytes: Some(7) }], total_bytes: Some(7),
            additional_snapshots: vec![manifest_standalone(root.path(), "drafter")?] };
        include_directory(&mut manifest, root.path(), "aux")?;
        assert_eq!(manifest.files.iter().map(|file| file.path.as_str()).collect::<Vec<_>>(),
            ["aux/vector.bin", "config.compact-attention.json"]);
        assert_eq!(manifest.total_bytes, Some(10));
        assert_eq!(manifest.additional_snapshots.len(), 1);
        Ok(())
    }

    #[test]
    fn all_family_loader_inputs_are_in_the_role_inventory() {
        use crate::plan::testing::*;
        let v41: serde_json::Value = serde_json::from_str(include_str!("../families/deepseek_v41/official-v41-config.json")).unwrap();
        let mut v4 = v41.clone();
        v4["architectures"] = serde_json::json!(["DeepseekV4ForCausalLM"]);
        v4["model_type"] = serde_json::json!("deepseek_v4");
        // Use real family geometry fixtures; each role's filtered loader must
        // open only shards the manifest lists, including native speculation.
        for config in [mimo_flash_config(), glm5_config(), glm5_flash_config(2), qwen4_config(4), v41, v4] {
            let dir = tempfile::tempdir().unwrap();
            write_snapshot(dir.path(), &config, &[t("lm_head.weight", "BF16", &[2]), t("norm.weight", "BF16", &[2])], None);
            if config["model_type"] == "deepseek_v4" {
                std::fs::create_dir(dir.path().join("inference")).unwrap();
                std::fs::write(dir.path().join("inference/config.json"), serde_json::json!({
                    "vocab_size": 64, "dim": 128, "moe_inter_dim": 128, "n_layers": 2,
                    "n_heads": 2, "n_routed_experts": 4, "n_shared_experts": 1,
                    "n_activated_experts": 2, "score_func": "sqrtsoftplus", "route_scale": 1.5,
                    "swiglu_limit": 10.0, "q_lora_rank": 128, "head_dim": 128, "rope_head_dim": 64,
                    "o_groups": 1, "o_lora_rank": 128, "window_size": 128, "compress_ratios": [0,4],
                    "compress_rope_theta": 160000, "original_seq_len": 65536, "rope_theta": 10000,
                    "rope_factor": 16, "beta_fast": 32, "beta_slow": 1,
                    "index_n_heads": 2, "index_head_dim": 128, "index_topk": 512,
                    "hc_mult": 4, "hc_sinkhorn_iters": 20
                }).to_string()).unwrap();
            }
            for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"] {
                std::fs::write(dir.path().join(name), b"{}").unwrap();
            }
            let roles = [ReadRole::Coordinator { local_experts: true, speculator: true }, ReadRole::Vision, ReadRole::Audio];
            let manifest = manifest_roles(dir.path(), &roles).unwrap();
            let listed: BTreeSet<_> = manifest.files.iter().map(|f| f.path.as_str()).collect();
            for role in roles {
                for tensor in open_role(dir.path(), role).unwrap().tensors {
                    assert!(listed.contains(tensor.shard.as_str()), "loader input missing: {}", tensor.shard);
                }
            }
            for name in ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "generation_config.json"] {
                assert!(listed.contains(name), "loader input missing: {name}");
            }
        }
    }

    #[test]
    fn standalone_glm_and_glmf_include_all_auxiliary_inputs() {
        for architecture in ["DFlash2DraftModel", "DSparkDraftModel"] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("config.json"), serde_json::json!({"architectures":[architecture]}).to_string()).unwrap();
            std::fs::write(dir.path().join("model.safetensors"), b"fixture").unwrap();
            std::fs::write(dir.path().join("draft-confidence.fp8-w8a16-r1.json"), b"{}").unwrap();
            let manifest = manifest_standalone(dir.path(), "drafter").unwrap();
            for entry in std::fs::read_dir(dir.path()).unwrap() {
                let name = entry.unwrap().file_name().to_string_lossy().into_owned();
                assert!(manifest.files.iter().any(|f| f.path == name), "loader input missing: {name}");
            }
        }
    }

    #[test]
    fn declared_native_headers_survive_index_only_eager_family_role_open() -> Result<()> {
        use crate::plan::{checkpoint::{AttentionMode, CheckpointView}, testing::*};
        let dir = tempfile::tempdir()?;
        write_kimi_snapshot(dir.path(), false);
        let names = ["language_model.model.layers.0.self_attn.o_proj.weight",
            "language_model.model.layers.3.self_attn.q_b_proj.weight"];
        let tensors = kimi_k3_tensors(false).into_iter().filter(|tensor| names.contains(&tensor.0.as_str()))
            .collect::<Vec<_>>();
        let mut index: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.path().join("model.safetensors.index.json"))?)?;
        let declaration = serde_json::json!({"schema":"cuteafd-attention-views-v1", "mode":"hqa",
            "views":{"hqa":{"config":"config.json", "index":"model.safetensors.index.json", "quantize":"quantize_config.json"},
                "compact-attention":{"config":"config.compact-attention.json",
                    "index":"model.compact-attention.safetensors.index.json", "quantize":"quantize_config.compact-attention.json"}},
            "native_modules":"quantize_config.json:native_attention"});
        let mut config = kimi_k3_config();
        config["quantization_config"]["attention_view"] = declaration.clone();
        let native = tensors.iter().map(|(name, dtype, shape)| {
            index["weight_map"][name] = serde_json::json!("native.safetensors");
            (name.strip_suffix(".weight").unwrap().to_owned(), serde_json::json!({"format":"bf16",
                "shard":"native.safetensors", "compact_attention_bits":6, "tensors":{name:{"dtype":dtype,"shape":shape}}}))
        }).collect::<serde_json::Map<_, _>>();
        let quant = serde_json::json!({"attention_view":declaration, "native_attention":native});
        for (name, value) in [("config.json", &config), ("model.safetensors.index.json", &index),
            ("quantize_config.json", &quant)] {
            std::fs::write(dir.path().join(name), serde_json::to_vec(value)?)?;
        }
        let mut shard = tensors.clone();
        shard.push(t("unindexed.weight", "BF16", &[2, 2]));
        write_safetensors(&dir.path().join("native.safetensors"), &shard);
        for mode in [AttentionMode::Hqa, AttentionMode::Fp8Compact] {
            let view = CheckpointView::select(dir.path(), mode)?;
            let started = std::time::Instant::now();
            let inventory = Checkpoint::inventory_view(&view)?;
            let role = ReadRole::Coordinator { local_experts: false, speculator: false };
            // Kimi constructs its core eagerly inside family.open, including in
            // required_tensors before the role has loaded its ordinary headers.
            let selected = required_tensors(&inventory, role)?;
            let opened = open_role_view(&view, role)?;
            eprintln!("native inventory {mode:?}: inventory+required+role {:?}", started.elapsed());
            assert_eq!(inventory.tensors.len(), names.len());
            assert!(!inventory.has_header("unindexed.weight"));
            for name in names {
                assert!(selected.contains(name));
                assert!(inventory.has_header(name));
                let stem = name.strip_suffix(".weight").unwrap();
                assert_eq!(inventory.projection(stem)?.storage, opened.projection(stem)?.storage);
            }
            let role = ReadRole::Spark { rank: 0, world: 6 };
            let selected = required_tensors(&inventory, role)?;
            let opened = open_role_view(&view, role)?;
            let expert_shards = selected.iter().map(|name| &inventory.weight_map[name]).collect::<BTreeSet<_>>();
            let expert_bytes = expert_shards.iter().map(|shard|
                std::fs::metadata(dir.path().join(shard)).map(|meta| meta.len())).sum::<std::io::Result<u64>>()?;
            assert_eq!(opened.shard_bytes, expert_bytes, "classification headers cannot grow role payload bytes");
            assert_eq!(opened.tensors.len(), selected.len());
            let classification = Checkpoint::open_view_for_role(&view, "classification-only", |_| false)?;
            assert_eq!(classification.shard_bytes, 0);
            assert!(classification.tensors.is_empty());
            for name in names {
                assert!(opened.has_header(name), "role checkpoint lost classification header {name}");
                assert!(!selected.contains(name), "native attention is not a Spark read");
                let stem = name.strip_suffix(".weight").unwrap();
                assert_eq!(inventory.projection(stem)?.storage, opened.projection(stem)?.storage);
                assert_eq!(inventory.projection(stem)?.storage, classification.projection(stem)?.storage);
                assert_eq!(inventory.module_numerics(stem)?, opened.module_numerics(stem)?);
            }
            assert!(!opened.has_header("unindexed.weight"));
            assert_eq!(required_tensors(&opened, role)?, selected);
            let recipe = super::super::experts::checkpoint_spark_recipe(&opened, 6, &[1])?;
            assert_eq!(recipe, super::super::experts::checkpoint_spark_recipe(
                &Checkpoint::open_view(&view)?, 6, &[1])?);
            let manifest = manifest_roles_view(&view, &[role])?;
            let payload_files = manifest.files.iter().filter(|file| file.path.ends_with(".safetensors"))
                .map(|file| &file.path).collect::<BTreeSet<_>>();
            assert_eq!(payload_files, expert_shards);
            assert!(!manifest.files.iter().any(|file| file.path == "native.safetensors"));
        }
        write_safetensors(&dir.path().join("native.safetensors"), &tensors[..1]);
        let view = CheckpointView::select(dir.path(), AttentionMode::Hqa)?;
        assert!(Checkpoint::inventory_view(&view).unwrap_err().to_string().contains("native attention header missing"));
        assert!(Checkpoint::open_view_for_role(&view, "spark0", |name| name.contains(".experts."))
            .unwrap_err().to_string().contains("header is missing"));
        let mut malformed = tensors.clone();
        malformed[0].2[0] += 1;
        write_safetensors(&dir.path().join("native.safetensors"), &malformed);
        assert!(Checkpoint::inventory_view(&view).unwrap_err().to_string().contains("disagree with selected safetensors header"));
        assert!(Checkpoint::open_view_for_role(&view, "spark0", |name| name.contains(".experts."))
            .unwrap_err().to_string().contains("disagree with selected safetensors header"));
        Ok(())
    }

    #[test]
    fn hqa_vision_roles_return_only_tower_tensors() -> Result<()> {
        use crate::plan::{checkpoint::{AttentionMode, CheckpointView}, testing::*};
        // All six published HQA families bundle a tower, including ML4's native
        // FP8 attention with a separate block-scale operand.
        for family in ["glm5_flash", "glm5", "kimi_k3", "muse_spark", "step5", "mistral_large4"] {
            let dir = tempfile::tempdir()?;
            let (native, tower) = match family {
                "glm5_flash" => {
                    let config = glm5_flash_config(2);
                    let mut tensors = glm5_flash_tensors(&config);
                    tensors.extend(glm5_flash_vision_tensors());
                    write_snapshot(dir.path(), &config, &tensors, None);
                    (t("model.language_model.layers.0.self_attn.o_proj.weight", "BF16", &[4096, 8192]),
                        "model.visual.patch_embed.proj.weight")
                },
                "glm5" => {
                    let mut tensors = glm5_tensors(|stem, n, k| fp8(stem, n, k, None));
                    tensors.push(t("model.visual.patch_embed.proj.weight", "BF16", &[2, 2]));
                    write_snapshot(dir.path(), &glm5_config(), &tensors, None);
                    (t("model.layers.0.self_attn.o_proj.weight", "BF16", &[6144, 16384]),
                        "model.visual.patch_embed.proj.weight")
                },
                "kimi_k3" => {
                    write_kimi_snapshot(dir.path(), false);
                    (t("language_model.model.layers.0.self_attn.o_proj.weight", "BF16", &[7168, 12288]),
                        "vision_tower.encoder.blocks.0.wqkv.weight")
                },
                "muse_spark" => {
                    write_muse_vision_snapshot(dir.path(), &[]);
                    (t("model.language_model.layers.0.self_attn.o_proj.weight", "BF16", &[8192, 8192]),
                        "model.vision_projection.weight")
                },
                "step5" => {
                    let (config, tensors) = step5_fixture("bf16");
                    write_snapshot(dir.path(), &config, &tensors, None);
                    (t("model.layers.0.self_attn.o_proj.weight", "BF16", &[4096, 12288]),
                        "vision_model.conv1.weight")
                },
                "mistral_large4" => {
                    write_mistral_large4_snapshot(dir.path(), Some(&[]), false);
                    (t("layers.0.attention.wo.weight", "F8_E4M3", &[6144, 8192]),
                        "vision_encoder.patch_conv.weight")
                },
                _ => unreachable!(),
            };
            let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.path().join("config.json"))?)?;
            let mut index: serde_json::Value = serde_json::from_slice(&std::fs::read(
                super::super::checkpoint::index_path(dir.path())?)?)?;
            let index_path = dir.path().join("model.safetensors.index.json");
            config.as_object_mut().unwrap().remove("transformers_weights");
            let declaration = serde_json::json!({"schema":"cuteafd-attention-views-v1", "mode":"hqa",
                "views":{"hqa":{"config":"config.json", "index":"model.safetensors.index.json",
                    "quantize":"quantize_config.json"}, "compact-attention":{"config":"config.compact-attention.json",
                    "index":"model.compact-attention.safetensors.index.json", "quantize":"quantize_config.compact-attention.json"}},
                "native_modules":"quantize_config.json:native_attention"});
            config["quantization_config"]["attention_view"] = declaration.clone();
            let mut quant = std::fs::read(dir.path().join("quantize_config.json")).ok()
                .map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes)).transpose()?.unwrap_or(serde_json::json!({}));
            quant["attention_view"] = declaration;
            let stem = native.0.strip_suffix(".weight").unwrap();
            let weights = if family == "mistral_large4" {
                let cfg = crate::families::mistral_large4::Ml4Config::from_hf(&config)?;
                (0..cfg.layers).flat_map(|layer| [
                    t(format!("layers.{layer}.attention.wq.weight"), "F8_E4M3", &[cfg.heads * cfg.qk_head_dim(), cfg.hidden]),
                    t(format!("layers.{layer}.attention.wo.weight"), "F8_E4M3", &[cfg.hidden, cfg.heads * cfg.v_head_dim]),
                ]).collect::<Vec<_>>()
            } else { vec![native.clone()] };
            let mut natives = Vec::new();
            let mut declarations = serde_json::Map::new();
            for weight in weights {
                let module = weight.0.strip_suffix(".weight").unwrap();
                let mut tensors = vec![weight.clone()];
                let mut entry = serde_json::json!({"format":"bf16", "shard":"native.safetensors", "compact_attention_bits":6});
                if weight.1 == "F8_E4M3" {
                    let scale = format!("{module}.qscale_weight");
                    tensors.push(t(&scale, "F32", &weight.2.iter().map(|n| n.div_ceil(128)).collect::<Vec<_>>()));
                    entry["format"] = serde_json::json!("fp8_e4m3_block_scaled");
                    entry["scale"] = serde_json::json!({"tensor":scale,"dtype":"F32","block_size":[128,128],
                        "semantics":"checkpoint-native: original FP8 weight with its original per-block FP32 scale"});
                }
                entry["tensors"] = serde_json::Value::Object(tensors.iter().map(|(name, dtype, shape)|
                    (name.clone(), serde_json::json!({"dtype":dtype,"shape":shape}))).collect());
                declarations.insert(module.to_owned(), entry);
                // Replace any compact operand and stale native scales in the index.
                index["weight_map"].as_object_mut().unwrap().retain(|name, _| !name.starts_with(&format!("{module}.")));
                for (name, ..) in &tensors { index["weight_map"][name] = serde_json::json!("native.safetensors"); }
                natives.extend(tensors);
            }
            quant["native_attention"] = serde_json::Value::Object(declarations);
            write_safetensors(&dir.path().join("native.safetensors"), &natives);
            for (path, value) in [(dir.path().join("config.json"), config), (index_path, index),
                (dir.path().join("quantize_config.json"), quant)] {
                std::fs::write(path, serde_json::to_vec(&value)?)?;
            }
            let view = CheckpointView::select(dir.path(), AttentionMode::Hqa)?;
            let inventory = Checkpoint::inventory_view(&view)?;
            let selected = required_tensors(&inventory, ReadRole::Vision)?;
            assert!(selected.contains(tower), "{family}: missing fixture tower {tower}");
            let opened = open_role_view(&view, ReadRole::Vision)?;
            assert_eq!(opened.tensors.iter().map(|tensor| tensor.meta.name.clone()).collect::<BTreeSet<_>>(),
                selected, "{family}: classification headers leaked into vision payload");
            assert_eq!(required_tensors(&opened, ReadRole::Vision)?, selected, "{family}: role cannot reopen family");
            let payload_shards = selected.iter().map(|name| &opened.weight_map[name]).collect::<BTreeSet<_>>();
            assert_eq!(opened.shard_bytes, payload_shards.iter().map(|shard|
                std::fs::metadata(dir.path().join(shard)).map(|meta| meta.len())).sum::<std::io::Result<u64>>()?);
            assert!(!payload_shards.iter().any(|shard| shard.as_str() == "native.safetensors"));
            assert_eq!(opened.projection(stem)?.storage, inventory.projection(stem)?.storage);
        }
        Ok(())
    }

    #[test]
    fn mimo_coordinator_ignores_expert_and_disabled_tower_shards() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&mimo_flash_config()).unwrap()).unwrap();
        let tensors = mimo_flash_tensors();
        let mut map = BTreeMap::new();
        let mut coordinator = Vec::new();
        let mut experts = Vec::new();
        for tensor in tensors {
            let shard = if tensor.0.contains(".mlp.experts.") {
                experts.push(tensor.clone()); "experts.safetensors"
            } else { coordinator.push(tensor.clone()); "coordinator.safetensors" };
            map.insert(tensor.0, shard);
        }
        map.insert("model.mtp.layers.0.eh_proj.weight".into(), "mtp.safetensors");
        map.insert("model.visual.weight".into(), "vision.safetensors");
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap()).unwrap();
        write_safetensors(&dir.path().join("coordinator.safetensors"), &coordinator);
        let role = ReadRole::Coordinator { local_experts: false, speculator: false };
        let opened = open_role(dir.path(), role).unwrap();
        assert!(opened.contains_tensor("model.mtp.layers.0.eh_proj.weight"));
        let manifest = manifest(dir.path(), role).unwrap();
        assert!(manifest.files.iter().any(|f| f.path == "coordinator.safetensors"));
        assert!(!manifest.files.iter().any(|f| f.path == "experts.safetensors" || f.path == "mtp.safetensors"));
        let mtp = open_role(dir.path(), ReadRole::Coordinator { local_experts: false, speculator: true })
            .unwrap_err().to_string();
        assert!(mtp.contains("model.mtp.layers.0.eh_proj.weight") && mtp.contains("mtp.safetensors"));
        write_safetensors(&dir.path().join("experts.safetensors"), &experts);
        let spark_role = ReadRole::Spark { rank: 0, world: 4 };
        let spark = open_role(dir.path(), spark_role).unwrap();
        assert_eq!(spark.tensors.len(), experts.len());
        let union = manifest_roles(dir.path(), &[role, spark_role]).unwrap();
        let listed: BTreeSet<_> = union.files.iter().map(|f| f.path.as_str()).collect();
        assert!(listed.contains("coordinator.safetensors") && listed.contains("experts.safetensors"));
        assert!(!listed.contains("mtp.safetensors") && !listed.contains("vision.safetensors"));
        for tensor in opened.tensors.iter().chain(&spark.tensors) {
            assert!(listed.contains(tensor.shard.as_str()), "loader opened an unlisted shard");
        }
        assert!(listed.contains("config.json") && listed.contains("model.safetensors.index.json"));
        std::fs::remove_file(dir.path().join("coordinator.safetensors")).unwrap();
        assert!(open_role(dir.path(), ReadRole::Spark { rank: 3, world: 4 }).is_ok());
        let error = open_role(dir.path(), role).unwrap_err().to_string();
        assert!(error.contains("coordinator.safetensors") && error.contains("coordinator"));
    }
}

pub fn shards(checkpoint: &Checkpoint, names: &BTreeSet<String>) -> BTreeMap<String, Vec<String>> {
    let mut result: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in names {
        if let Some(shard) = checkpoint.weight_map.get(name) {
            result.entry(shard.clone()).or_default().push(name.clone());
        }
    }
    result
}
