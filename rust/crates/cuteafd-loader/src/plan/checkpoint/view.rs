//! One immutable attention bundle, selected before inventory or payload reads.
#[cfg(test)]
use super::HF_INDEX;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, path::{Path, PathBuf}, sync::Arc};

pub const COMPACT_CONFIG: &str = "config.compact-attention.json";
pub const COMPACT_INDEX: &str = "model.compact-attention.safetensors.index.json";
pub const COMPACT_QUANT: &str = "quantize_config.compact-attention.json";
/// Changing the quantizer invalidates only FP8 attention identities.
pub const FP8_RECIPE: &str = "row-major-e4m3-f32-block128-rne-amax448-v1";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionMode {
    /// The checkpoint's default bundle; HQA publications retain native attention.
    #[default]
    Hqa,
    /// The complete alternate bundle, preserving the parent's quantized attention.
    Compact,
    /// The default bundle with deterministic load-time FP8 attention conversion.
    Fp8Compact,
}

#[derive(Debug, Clone)]
pub struct CheckpointView {
    snapshot: PathBuf,
    mode: AttentionMode,
    config_name: String,
    index_name: String,
    quant_names: Vec<String>,
    fingerprint: String,
    metadata_digests: BTreeMap<String, [u8; 32]>,
    config: Arc<Value>,
    quantize_config: Option<Arc<Value>>,
    weight_map: Option<Arc<BTreeMap<String, String>>>,
    attention: Option<Arc<crate::formats::exl3_contract::AttentionContract>>,
}

fn metadata_bytes(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    ensure!(file.metadata()?.len() <= super::MAX_JSON_BYTES, "{} exceeds metadata size limit", path.display());
    let mut bytes = Vec::new();
    file.take(super::MAX_JSON_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= super::MAX_JSON_BYTES, "{} exceeds metadata size limit", path.display());
    Ok(bytes)
}

fn check_references(value: &Value, quant_names: &[String], index: &str, aliases: &[String]) -> Result<()> {
    match value {
        Value::Object(object) => for (key, value) in object {
            if key == "per_module_metadata" {
                let reference = value.as_str().context("per_module_metadata must name a metadata file")?;
                let (file, field) = reference.split_once(':').context("per_module_metadata must name file:field")?;
                ensure!(quant_names.iter().any(|name| name == file) && field == "tensor_storage",
                    "attention bundle references unselected quantization metadata {reference}");
            }
            if key == "transformers_weights" && !value.is_null() {
                ensure!(value.as_str().is_some_and(|name| name == index || aliases.iter().any(|alias| alias == name)), "attention bundle references unselected index {value}");
            }
            check_references(value, quant_names, index, aliases)?;
        },
        Value::Array(values) => for value in values { check_references(value, quant_names, index, aliases)?; },
        _ => (),
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error("attention index alias {alias}: {reason}")]
pub struct InvalidIndexAlias { pub alias: String, pub reason: &'static str }

fn index_aliases(snapshot: &Path, config: &Value, mode: AttentionMode, canonical: &str) -> Result<Vec<String>> {
    let declaration = config.pointer("/quantization_config/attention_view");
    // Compact metadata intentionally omits the HQA declaration; consult only its
    // small default config when an alternate index name needs authorization.
    let default_config;
    let declared_alias = config.get("transformers_weights").and_then(Value::as_str)
        .is_some_and(|name| name != canonical);
    let declaration = if declaration.is_none() && mode == AttentionMode::Compact && declared_alias {
        default_config = serde_json::from_slice::<Value>(&metadata_bytes(&snapshot.join("config.json"))?)?;
        default_config.pointer("/quantization_config/attention_view")
    } else { declaration };
    let key = if mode == AttentionMode::Compact { "compact-attention" } else { "hqa" };
    let aliases: Vec<String> = declaration.and_then(|v| v.get("index_aliases")).and_then(|v| v.get(key))
        .map(|v| serde_json::from_value(v.clone())).transpose()?.unwrap_or_default();
    for alias in &aliases {
        if !alias.ends_with(".index.json") || alias.starts_with('.') || alias.contains(['/', '\\']) {
            return Err(InvalidIndexAlias { alias: alias.clone(), reason: "invalid index filename" }.into());
        }
        let read_map = |name: &str| -> Result<Value> {
            let value: Value = serde_json::from_slice(&metadata_bytes(&snapshot.join(name))?)?;
            let map = value.get("weight_map").filter(|v| v.is_object()).context("index requires weight_map")?;
            Ok(map.clone())
        };
        if read_map(alias)? != read_map(canonical)? {
            return Err(InvalidIndexAlias { alias: alias.clone(), reason: "weight map differs from canonical index" }.into());
        }
    }
    Ok(aliases)
}

impl CheckpointView {
    pub fn select(snapshot: &Path, mode: AttentionMode) -> Result<Self> {
        Self::select_recipe(snapshot, mode, FP8_RECIPE)
    }

    fn select_recipe(snapshot: &Path, mode: AttentionMode, recipe: &str) -> Result<Self> {
        let config_name = if mode == AttentionMode::Compact { COMPACT_CONFIG } else { "config.json" };
        let config_bytes = metadata_bytes(&snapshot.join(config_name))
            .with_context(|| format!("selecting {mode:?} attention bundle: {config_name}"))?;
        let config: Value = serde_json::from_slice(&config_bytes)?;
        let index_name = if mode == AttentionMode::Compact { COMPACT_INDEX.to_owned() }
            else if config.pointer("/quantization_config/attention_view").is_some() { super::HF_INDEX.to_owned() }
            else { super::index_name_from_config(&config)? };
        let aliases = index_aliases(snapshot, &config, mode, &index_name)?;
        let quant_names: Vec<String> = if mode == AttentionMode::Compact {
            for name in [COMPACT_INDEX, COMPACT_QUANT] {
                ensure!(snapshot.join(name).is_file(), "incomplete compact attention bundle: missing {name}");
            }
            vec![COMPACT_QUANT.into()]
        } else {
            ["quantize_config.json", "quantization_config.json", "hf_quant_config.json"].into_iter()
                .filter(|name| snapshot.join(name).is_file()).map(str::to_owned).collect()
        };
        let allowed_quant = if mode == AttentionMode::Compact { quant_names.clone() } else {
            ["quantize_config.json", "quantization_config.json", "hf_quant_config.json"].map(str::to_owned).to_vec()
        };
        check_references(&config, &allowed_quant, &index_name, &aliases)?;
        let mut hash = Sha256::new();
        hash.update(b"cuteafd-attention-view-v1\0");
        hash.update(match mode { AttentionMode::Hqa => b"hqa".as_slice(),
            AttentionMode::Compact => b"compact", AttentionMode::Fp8Compact => b"fp8-compact" });
        if mode == AttentionMode::Fp8Compact { hash.update(recipe.as_bytes()); }
        let mut metadata_digests = BTreeMap::new();
        let mut quantize_config = None;
        for name in std::iter::once(config_name).chain(quant_names.iter().map(String::as_str)) {
            let path = snapshot.join(name);
            // Storage maps are large; parse their compact fields with the shared
            // streaming visitor, and bind the full original bytes by digest.
            let bytes = if name == config_name { config_bytes.clone() } else { metadata_bytes(&path)? };
            let value = if name == config_name { config.clone() } else {
                crate::formats::exl3_storage::parse_storage_metadata(&path, &bytes, None)?
            };
            check_references(&value, &allowed_quant, &index_name, &aliases)?;
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            metadata_digests.insert(name.to_owned(), Sha256::digest(&bytes).into());
            hash.update(bytes);
            if name != config_name && quantize_config.is_none() { quantize_config = Some(Arc::new(value)); }
        }
        let weight_map = if snapshot.join(&index_name).is_file() {
            #[derive(Deserialize)]
            struct Index { weight_map: BTreeMap<String, String> }
            let bytes = metadata_bytes(&snapshot.join(&index_name))?;
            let index: Index = serde_json::from_slice(&bytes).context("decoding selected weight_map")?;
            hash.update(index_name.as_bytes());
            metadata_digests.insert(index_name.clone(), Sha256::digest(&bytes).into());
            hash.update(bytes);
            Some(Arc::new(index.weight_map))
        } else {
            ensure!(mode != AttentionMode::Compact, "compact attention requires {COMPACT_INDEX}");
            None
        };
        let attention = crate::formats::exl3_contract::AttentionContract::parse(
            config.get("quantization_config").unwrap_or(&Value::Null),
            quantize_config.as_deref().unwrap_or(&Value::Null))?.map(Arc::new);
        if let Some(contract) = &attention {
            if mode == AttentionMode::Fp8Compact { contract.check_fp8_conversion()?; }
            contract.check_selection(config_name, &index_name, &quant_names, weight_map.as_deref())?;
        }
        Ok(Self { snapshot: snapshot.into(), mode, config_name: config_name.into(), index_name,
            quant_names, fingerprint: format!("{:x}", hash.finalize()), config: Arc::new(config),
            quantize_config, weight_map, attention, metadata_digests })
    }

    pub fn snapshot(&self) -> &Path { &self.snapshot }
    pub fn mode(&self) -> AttentionMode { self.mode }
    pub fn config_name(&self) -> &str { &self.config_name }
    pub fn index_name(&self) -> &str { &self.index_name }
    pub fn quant_names(&self) -> &[String] { &self.quant_names }
    pub fn fingerprint(&self) -> &str { &self.fingerprint }
    pub fn metadata_sha256(&self, name: &str) -> Result<String> {
        let digest = self.metadata_digests.get(name).context("metadata was not selected")?;
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }
    pub fn config(&self) -> Result<Value> { Ok(self.config.as_ref().clone()) }
    pub(super) fn quantize_config(&self) -> Option<Value> { self.quantize_config.as_deref().cloned() }
    pub(super) fn weight_map(&self) -> Option<&BTreeMap<String, String>> { self.weight_map.as_deref() }
    pub(super) fn native_attention_names(&self) -> impl Iterator<Item = &str> {
        self.attention.iter().flat_map(|contract| contract.native_tensor_names())
    }
    pub(crate) fn check_attention_headers<'a>(&self,
        headers: impl IntoIterator<Item = &'a crate::SafetensorsTensorMetadata>) -> Result<()> {
        if let Some(contract) = &self.attention { contract.check_headers(headers)?; }
        Ok(())
    }
    pub(crate) fn check_attention_manifest(&self, manifest: &Value) -> Result<()> {
        let parsed = crate::formats::exl3_contract::AttentionContract::parse(
            self.config.get("quantization_config").unwrap_or(&Value::Null), manifest)?;
        ensure!(parsed.as_ref() == self.attention.as_deref(), "attention metadata changed after bundle selection");
        Ok(())
    }
    pub(crate) fn check_metadata_bytes(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let expected = self.metadata_digests.get(name).context("metadata was not selected")?;
        let actual: [u8; 32] = Sha256::digest(bytes).into();
        ensure!(*expected == actual, "selected metadata {name} changed after bundle selection");
        Ok(())
    }
    pub fn index_path(&self) -> PathBuf { self.snapshot.join(&self.index_name) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bundle(root: &Path, compact: bool) {
        let (config, index, quant) = if compact { (COMPACT_CONFIG, COMPACT_INDEX, COMPACT_QUANT) }
            else { ("config.json", HF_INDEX, "quantize_config.json") };
        std::fs::write(root.join(config), json!({"quantization_config": {
            "per_module_metadata": format!("{quant}:tensor_storage")}}).to_string()).unwrap();
        std::fs::write(root.join(index), json!({"weight_map": {"a.weight": "a.safetensors"}}).to_string()).unwrap();
        std::fs::write(root.join(quant), "{}").unwrap();
    }

    #[test]
    fn compact_aliases_require_declared_identical_maps_and_keep_canonical_identity() -> Result<()> {
        let root = tempfile::tempdir()?;
        bundle(root.path(), true);
        let alias = "consolidated.compact-attention.safetensors.index.json";
        let mut config: Value = serde_json::from_slice(&std::fs::read(root.path().join(COMPACT_CONFIG))?)?;
        config["transformers_weights"] = alias.into();
        std::fs::write(root.path().join(COMPACT_CONFIG), config.to_string())?;
        std::fs::write(root.path().join("config.json"), json!({"quantization_config":{"attention_view":{
            "index_aliases":{"compact-attention":[alias]}}}}).to_string())?;
        std::fs::copy(root.path().join(COMPACT_INDEX), root.path().join(alias))?;
        let view = CheckpointView::select(root.path(), AttentionMode::Compact)?;
        assert_eq!(view.index_name(), COMPACT_INDEX);
        // Formatting and non-map metadata of an alias are not a second identity.
        std::fs::write(root.path().join(alias), json!({"metadata":{},"weight_map":{"a.weight":"a.safetensors"}}).to_string())?;
        assert_eq!(CheckpointView::select(root.path(), AttentionMode::Compact)?.fingerprint(), view.fingerprint());
        std::fs::write(root.path().join(alias), r#"{"weight_map":{"a.weight":"wrong.safetensors"}}"#)?;
        assert!(CheckpointView::select(root.path(), AttentionMode::Compact).unwrap_err().downcast_ref::<InvalidIndexAlias>().is_some());
        std::fs::write(root.path().join("config.json"), "{}")?;
        assert!(CheckpointView::select(root.path(), AttentionMode::Compact).is_err());
        Ok(())
    }

    #[test]
    fn bundle_is_complete_and_never_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        bundle(dir.path(), false);
        assert!(CheckpointView::select(dir.path(), AttentionMode::Compact).is_err());
        bundle(dir.path(), true);
        let view = CheckpointView::select(dir.path(), AttentionMode::Compact).unwrap();
        assert_eq!(view.index_name(), COMPACT_INDEX);
        assert_eq!(view.quant_names(), &[COMPACT_QUANT]);
        std::fs::remove_file(dir.path().join(COMPACT_QUANT)).unwrap();
        assert!(CheckpointView::select(dir.path(), AttentionMode::Compact).is_err());
        bundle(dir.path(), true);
        std::fs::copy(dir.path().join("config.json"), dir.path().join(COMPACT_CONFIG)).unwrap();
        assert!(CheckpointView::select(dir.path(), AttentionMode::Compact).unwrap_err().to_string()
            .contains("unselected quantization"));
    }

    #[test]
    fn compact_opens_only_indexed_tensors_without_native_shards() {
        use crate::plan::{Checkpoint, testing::{t, write_safetensors}};
        let root = tempfile::tempdir().unwrap();
        bundle(root.path(), false);
        bundle(root.path(), true);
        std::fs::write(root.path().join(HF_INDEX), json!({"weight_map": {
            "a.weight": "native-absent.safetensors"}}).to_string()).unwrap();
        std::fs::write(root.path().join(COMPACT_INDEX), json!({"weight_map": {
            "a.trellis": "parent.safetensors", "a.suh": "parent.safetensors"}}).to_string()).unwrap();
        write_safetensors(&root.path().join("parent.safetensors"), &[
            t("a.trellis", "I16", &[1, 1, 64]), t("a.suh", "F16", &[16]),
            t("a.weight", "BF16", &[16, 16]), t("unused.weight", "BF16", &[16, 16])]);
        let compact = CheckpointView::select(root.path(), AttentionMode::Compact).unwrap();
        let checkpoint = Checkpoint::open_view_for_role(&compact, "coordinator", |_| true).unwrap();
        assert_eq!(checkpoint.tensors.iter().map(|t| t.meta.name.as_str()).collect::<Vec<_>>(), ["a.suh", "a.trellis"]);
        assert_eq!(checkpoint.projection("a").unwrap().storage, super::super::ProjectionStorage::Exl3);
        let hqa = CheckpointView::select(root.path(), AttentionMode::Hqa).unwrap();
        assert!(Checkpoint::open_view_for_role(&hqa, "coordinator", |_| true).is_err());
        assert!(Checkpoint::open_view_for_role(&CheckpointView::select(root.path(), AttentionMode::Fp8Compact).unwrap(),
            "coordinator", |_| true).is_err());
    }

    #[test]
    fn selected_metadata_is_immutable_after_selection() {
        let root = tempfile::tempdir().unwrap();
        bundle(root.path(), true);
        let view = CheckpointView::select(root.path(), AttentionMode::Compact).unwrap();
        let config = view.config().unwrap();
        std::fs::write(root.path().join(COMPACT_CONFIG), "{}").unwrap();
        std::fs::write(root.path().join(COMPACT_INDEX), "{}").unwrap();
        assert_eq!(view.config().unwrap(), config);
        assert_eq!(view.weight_map().unwrap()["a.weight"], "a.safetensors");
        assert!(CheckpointView::select(root.path(), AttentionMode::Compact).is_err());
    }

    #[test]
    fn selected_identity_and_manifest_digest_do_not_follow_mutation() -> Result<()> {
        let root = tempfile::tempdir()?;
        bundle(root.path(), true);
        let view = CheckpointView::select(root.path(), AttentionMode::Compact)?;
        let identity = super::super::identity::CheckpointIdentity::from_view(&view, None)?;
        view.check_metadata_bytes(COMPACT_QUANT, b"{}")?;
        assert!(view.check_metadata_bytes(COMPACT_QUANT, b"{}\n").is_err());
        std::fs::remove_file(root.path().join(COMPACT_CONFIG))?;
        std::fs::remove_file(root.path().join(COMPACT_INDEX))?;
        assert_eq!(identity, super::super::identity::CheckpointIdentity::from_view(&view, None)?);
        Ok(())
    }

    #[test]
    fn selected_index_requires_every_named_tensor_in_readable_shard() -> Result<()> {
        let root = tempfile::tempdir()?;
        bundle(root.path(), true);
        crate::plan::testing::write_safetensors(&root.path().join("a.safetensors"),
            &[crate::plan::testing::t("other.weight", "BF16", &[2])]);
        let view = CheckpointView::select(root.path(), AttentionMode::Compact)?;
        let error = crate::plan::Checkpoint::open_view(&view).unwrap_err().to_string();
        assert!(error.contains("indexed tensor a.weight missing"), "{error}");
        Ok(())
    }

    #[test]
    fn published_hqa_metadata_matches_native_headers_and_selected_catalogs() -> Result<()> {
        let Some(snapshot) = std::env::var_os("CUTEAFD_HQA_TEST_SNAPSHOT") else { return Ok(()); };
        let snapshot = PathBuf::from(snapshot);
        ensure!(snapshot.is_dir(), "CUTEAFD_HQA_TEST_SNAPSHOT is not a directory");
        let hqa = CheckpointView::select(&snapshot, AttentionMode::Hqa)?;
        let fp8 = CheckpointView::select(&snapshot, AttentionMode::Fp8Compact)?;
        let compact = CheckpointView::select(&snapshot, AttentionMode::Compact)?;
        ensure!(hqa.fingerprint() != fp8.fingerprint() && hqa.fingerprint() != compact.fingerprint(),
            "published attention identities collide");
        ensure!(hqa.weight_map() == fp8.weight_map(), "FP8 must select the same native source tensors");
        let contract = hqa.attention.as_ref().context("published HQA has no attention contract")?;
        let quant = hqa.quantize_config().context("published HQA has no quantization metadata")?;
        let native = quant["native_attention"].as_object().context("published native_attention")?;
        ensure!(native.len() == 58, "GLM Flash publication must declare 12 q_b and 46 o projections");
        let names: std::collections::BTreeSet<_> = native.values().flat_map(|entry|
            entry["tensors"].as_object().unwrap().keys().cloned()).collect();
        let checkpoint = super::super::Checkpoint::open_view_for_role(&hqa, "HQA metadata gate", |name| names.contains(name))?;
        ensure!(checkpoint.tensors.len() == 58, "published native header count");
        contract.check_headers(checkpoint.tensors.iter().map(|tensor| &tensor.meta))?;
        ensure!(compact.weight_map().context("compact index")?.values().all(|shard| !shard.starts_with("hqa-attention-")),
            "compact index references native attention shard");
        for view in [&hqa, &fp8, &compact] {
            crate::read_expert_catalog_view(view).with_context(|| format!("published {:?} EXL3 contract", view.mode()))?;
        }
        eprintln!("published HQA metadata gate: 58 native headers; all three selected catalogs validated at {}", snapshot.display());
        Ok(())
    }

    #[test]
    fn identities_separate_modes_and_only_fp8_depends_on_recipe() {
        let dir = tempfile::tempdir().unwrap();
        bundle(dir.path(), false);
        bundle(dir.path(), true);
        let identities = [AttentionMode::Hqa, AttentionMode::Compact, AttentionMode::Fp8Compact]
            .map(|mode| CheckpointView::select(dir.path(), mode).unwrap().fingerprint);
        assert_ne!(identities[0], identities[1]);
        assert_ne!(identities[0], identities[2]);
        assert_ne!(identities[1], identities[2]);
        for mode in [AttentionMode::Hqa, AttentionMode::Compact, AttentionMode::Fp8Compact] {
            let old = CheckpointView::select_recipe(dir.path(), mode, "recipe-a").unwrap();
            let new = CheckpointView::select_recipe(dir.path(), mode, "recipe-b").unwrap();
            assert_eq!(old.fingerprint == new.fingerprint, mode != AttentionMode::Fp8Compact);
        }
    }
}
