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
