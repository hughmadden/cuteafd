//! Family-independent view of a Hugging Face checkpoint: configuration JSON,
//! the safetensors index and every shard header. Nothing here reads tensor data.
use crate::{read_safetensors_metadata, SafetensorsTensorMetadata};
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;

/// One tensor as the checkpoint stores it.
#[derive(Debug, Clone)]
pub struct CheckpointTensor {
    pub shard: String,
    pub meta: SafetensorsTensorMetadata,
}

#[derive(Debug)]
pub struct Checkpoint {
    pub snapshot: PathBuf,
    /// `config.json`, verbatim.
    pub config: Value,
    /// A separate `quantize_config.json` / `quantization_config.json`, when present.
    pub quantize_config: Option<Value>,
    /// Tensors in index order (sorted by name).
    pub tensors: Vec<CheckpointTensor>,
    /// Shards named by the index that are not readable (absent or incomplete).
    pub missing_shards: Vec<String>,
    pub shard_bytes: u64,
}

pub fn read_json(path: &Path) -> Result<Value> {
    let mut bytes = Vec::new();
    File::open(path)
        .with_context(|| format!("opening {}", path.display()))?
        .take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_JSON_BYTES,
        "{} exceeds {MAX_JSON_BYTES} bytes",
        path.display()
    );
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

impl Checkpoint {
    /// Reads the configuration, the index (or a single `model.safetensors`) and
    /// every shard header. Missing shards are recorded rather than fatal so a
    /// partially downloaded checkpoint can still be planned.
    pub fn open(snapshot: &Path) -> Result<Self> {
        let config = read_json(&snapshot.join("config.json"))?;
        let quantize_config = ["quantize_config.json", "quantization_config.json"]
            .iter()
            .map(|name| snapshot.join(name))
            .find(|path| path.is_file())
            .map(|path| read_json(&path))
            .transpose()?;
        let index_path = snapshot.join("model.safetensors.index.json");
        let weight_map: BTreeMap<String, String> = if index_path.is_file() {
            let index = read_json(&index_path)?;
            serde_json::from_value(
                index
                    .get("weight_map")
                    .cloned()
                    .context("index has no weight_map")?,
            )
            .context("decoding weight_map")?
        } else {
            let single = snapshot.join("model.safetensors");
            ensure!(
                single.is_file(),
                "{} has neither model.safetensors.index.json nor model.safetensors",
                snapshot.display()
            );
            read_safetensors_metadata(&single)?
                .into_iter()
                .map(|meta| (meta.name, "model.safetensors".to_owned()))
                .collect()
        };
        let shards: BTreeSet<&String> = weight_map.values().collect();
        let mut tensors = Vec::with_capacity(weight_map.len());
        let mut missing_shards = Vec::new();
        let mut shard_bytes = 0u64;
        for shard in shards {
            let path = snapshot.join(shard);
            let metadata = match path.metadata() {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => {
                    missing_shards.push(shard.clone());
                    continue;
                }
            };
            let headers = match read_safetensors_metadata(&path) {
                Ok(headers) => headers,
                Err(_) => {
                    missing_shards.push(shard.clone());
                    continue;
                }
            };
            shard_bytes += metadata.len();
            for meta in headers {
                if weight_map.get(&meta.name) == Some(shard) {
                    tensors.push(CheckpointTensor {
                        shard: shard.clone(),
                        meta,
                    });
                }
            }
        }
        tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        Ok(Self {
            snapshot: snapshot.to_path_buf(),
            config,
            quantize_config,
            tensors,
            missing_shards,
            shard_bytes,
        })
    }

    /// The text model configuration: `text_config` when present, else the root.
    pub fn text_config(&self) -> &Value {
        self.config
            .get("text_config")
            .filter(|value| value.is_object())
            .unwrap_or(&self.config)
    }

    /// `quantization_config` from config.json, else the external file.
    pub fn quantization(&self) -> Option<&Value> {
        self.config
            .get("quantization_config")
            .or_else(|| self.text_config().get("quantization_config"))
            .or(self.quantize_config.as_ref())
    }

    pub fn architectures(&self) -> Vec<String> {
        self.config
            .get("architectures")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn model_type(&self) -> Option<&str> {
        self.config.get("model_type").and_then(Value::as_str)
    }

    /// Host placement cannot free a tensor also used by the vocabulary head.
    /// Require a separate head in addition to the config's tying declaration.
    pub fn require_untied_embedding(&self, name: &str) -> Result<()> {
        ensure!(self.text_config().get("tie_word_embeddings").and_then(Value::as_bool) != Some(true)
            && self.config.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true),
            "{name}: host embedding is ineligible: tie_word_embeddings=true");
        let embedding = self.tensors.iter().find(|t| t.meta.name == name)
            .with_context(|| format!("host embedding tensor {name} is missing"))?;
        let heads: Vec<_> = self.tensors.iter().filter(|t|
            t.meta.name == "head.weight" || t.meta.name == "lm_head.weight"
                || t.meta.name.ends_with(".lm_head.weight")).collect();
        ensure!(!heads.is_empty(), "{name}: host embedding is ineligible: no separate LM head tensor");
        for head in heads {
            let same_file = head.shard == embedding.shard
                || std::fs::canonicalize(self.snapshot.join(&head.shard)).ok().zip(
                    std::fs::canonicalize(self.snapshot.join(&embedding.shard)).ok())
                    .is_some_and(|(a, b)| a == b);
            let overlaps = head.meta.byte_offset < embedding.meta.byte_offset.saturating_add(embedding.meta.byte_length)
                && embedding.meta.byte_offset < head.meta.byte_offset.saturating_add(head.meta.byte_length);
            ensure!(!(same_file && overlaps),
                "{name}: host embedding is ineligible: LM head {} aliases its storage", head.meta.name);
        }
        Ok(())
    }
}

/// Reads an integer config field, checking `text_config` first.
pub fn usize_field(config: &Value, key: &str) -> Result<usize> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("config field {key} is missing or not an unsigned integer"))
}

pub fn opt_usize_field(config: &Value, key: &str) -> Option<usize> {
    config.get(key).and_then(Value::as_u64).map(|value| value as usize)
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    use crate::SafetensorsTensorMetadata;
    use cuteafd_core::DType;

    fn fixture() -> Checkpoint {
        let t = |name: &str, offset| CheckpointTensor { shard: "model.safetensors".into(),
            meta: SafetensorsTensorMetadata { name: name.into(), dtype: DType::Bf16,
                shape: vec![64, 128], byte_offset: offset, byte_length: 16384 } };
        Checkpoint { snapshot: Default::default(), config: serde_json::json!({"tie_word_embeddings":false}),
            quantize_config: None, missing_shards: vec![], shard_bytes: 0,
            tensors: vec![t("model.embed_tokens.weight", 0), t("lm_head.weight", 16384)] }
    }

    #[test]
    fn host_embedding_requires_untied_distinct_storage() {
        let mut c = fixture();
        assert!(c.require_untied_embedding("model.embed_tokens.weight").is_ok());
        c.config["tie_word_embeddings"] = Value::Bool(true);
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("tie_word_embeddings"));
        c.config = serde_json::json!({"text_config":{"tie_word_embeddings":true}});
        assert!(c.require_untied_embedding("model.embed_tokens.weight").is_err());
        c.config = serde_json::json!({});
        c.tensors[1].meta.byte_offset = 8192;
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("aliases"));
        c.tensors.pop();
        assert!(c.require_untied_embedding("model.embed_tokens.weight").unwrap_err().to_string().contains("no separate LM head"));
    }
}
