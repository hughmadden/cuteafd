//! Header-only, inference-only admission plan for the officially bundled MiMo tower.
use super::EncoderId;
use crate::{read_safetensors_metadata, SafetensorsTensorMetadata};
use cuteafd_core::DType;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
};

const CODEBOOKS: [usize; 20] = [
    1024, 1024, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
    128, 128,
];

#[derive(Debug, thiserror::Error)]
pub enum AudioTowerError {
    #[error("audio tower unsupported: {0}; add the matching audio exporter/kernel")]
    Unsupported(String),
    #[error("audio tower I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("audio tower config: {0}")]
    Json(#[from] serde_json::Error),
    #[error("audio weight admission needs {required} bytes, got {admitted}")]
    Admission { required: u64, admitted: u64 },
}
type Result<T> = std::result::Result<T, AudioTowerError>;
fn unsupported(reason: impl Into<String>) -> AudioTowerError {
    AudioTowerError::Unsupported(reason.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioStorage {
    /// Checkpoint BF16 matrices, FP32 normalization/bias vectors and RVQ tables.
    MixedBf16,
    /// FP32 resident inference weights, for qualifying the discontinuous RVQ path.
    Fp32,
}
#[derive(Clone, Debug)]
pub struct AudioTensorRead {
    pub metadata: SafetensorsTensorMetadata,
    pub path: PathBuf,
    pub destination: u64,
    pub resident_dtype: DType,
}
#[derive(Clone, Debug)]
pub struct AudioTowerPlan {
    output_width: usize,
    storage: AudioStorage,
    weight_bytes: u64,
    reads: BTreeMap<String, AudioTensorRead>,
}
#[derive(Deserialize)]
struct RootConfig {
    model_type: String,
    hidden_size: usize,
    audio_config: serde_json::Value,
}
fn number(config: &serde_json::Value, key: &str, want: u64) -> Result<()> {
    if config.get(key).and_then(serde_json::Value::as_u64) != Some(want) {
        return Err(unsupported(format!(
            "{key}={:?}, need {want}",
            config.get(key)
        )));
    }
    Ok(())
}
fn boolean(config: &serde_json::Value, key: &str, want: bool) -> Result<()> {
    if config.get(key).and_then(serde_json::Value::as_bool) != Some(want) {
        return Err(unsupported(format!(
            "{key}={:?}, need {want}",
            config.get(key)
        )));
    }
    Ok(())
}
fn validate_config(root: &RootConfig, tokenizer: &serde_json::Value) -> Result<()> {
    if root.model_type != "mimo_v2" || ![4096, 6144].contains(&root.hidden_size) {
        return Err(unsupported(
            "only bundled MiMo V2.6 Flash/Pro audio geometry is implemented",
        ));
    }
    let a = &root.audio_config;
    for (key, want) in [
        ("audio_channels", 20),
        ("audio_segment_size", 6000),
        ("group_size", 4),
        ("input_local_attn_heads", 16),
        ("input_local_dim", 1024),
        ("input_local_head_dim", 64),
        ("input_local_intermediate_size", 4096),
        ("input_local_layers", 6),
        ("projection_layers", 2),
        ("rope_theta", 640000),
        ("out_hidden_size", root.hidden_size as u64),
    ] {
        number(a, key, want)?;
    }
    for key in ["add_post_norm", "input_full_attention"] {
        boolean(a, key, true)?;
    }
    for (key, want) in [
        ("speech_vocab_size", "1280"),
        ("speech_zeroemb_idx", "1024"),
    ] {
        if a.get(key).and_then(serde_json::Value::as_str) != Some(want) {
            return Err(unsupported(format!("audio_config.{key} must be {want}")));
        }
    }
    if a.get("partial_rotary_factor")
        .and_then(serde_json::Value::as_f64)
        != Some(1.0)
        || a.get("input_local_hidden_dropout")
            .and_then(serde_json::Value::as_f64)
            != Some(0.0)
    {
        return Err(unsupported("audio patch rotary/dropout geometry"));
    }
    for (key, want) in [
        ("max_audio_seconds", 300),
        ("stride_size", 2),
        ("avg_pooler", 2),
        ("d_model", 1024),
        ("kernel_size", 3),
        ("encoder_layers", 24),
        ("encoder_skip_layer_id", 3),
        ("encoder_attention_heads", 16),
        ("encoder_ffn_dim", 4096),
        ("nfft", 960),
        ("n_mels", 128),
        ("sampling_rate", 24000),
        ("hop_length", 240),
        ("window_size", 960),
        ("fmin", 0),
        ("num_quantizers", 20),
        ("rope_theta", 10000),
    ] {
        number(tokenizer, key, want)?;
    }
    boolean(tokenizer, "encoder_causal", true)?;
    boolean(tokenizer, "hybrid_attention", true)?;
    number(tokenizer, "swa_per_block", 2)?;
    boolean(tokenizer, "scale_embedding", false)?;
    for (key, want) in [
        ("activation_function", "gelu"),
        ("position_embedding_type", "rope"),
        ("rope_type", "default"),
        ("ln_type", "LayerNorm"),
    ] {
        if tokenizer.get(key).and_then(serde_json::Value::as_str) != Some(want) {
            return Err(unsupported(format!("audio_tokenizer.{key} must be {want}")));
        }
    }
    if tokenizer.get("encoder_attn_window_size") != Some(&serde_json::json!([128, 0]))
        || tokenizer.get("codebook_size") != Some(&serde_json::json!(CODEBOOKS))
        || !tokenizer
            .get("fmax")
            .is_some_and(serde_json::Value::is_null)
    {
        return Err(unsupported("audio tokenizer window/codebook/mel geometry"));
    }
    Ok(())
}

type Catalog = BTreeMap<String, (PathBuf, SafetensorsTensorMetadata)>;
fn catalog(snapshot: &Path) -> Result<Catalog> {
    let mut files = BTreeSet::new();
    let tokenizer = snapshot.join("audio_tokenizer/model.safetensors");
    if !tokenizer.is_file() {
        return Err(unsupported(
            "missing bundled audio_tokenizer/model.safetensors",
        ));
    }
    files.insert(tokenizer);
    let index = snapshot.join("model.safetensors.index.json");
    if index.exists() {
        let map: serde_json::Value = serde_json::from_reader(File::open(index)?)?;
        for (name, file) in map["weight_map"]
            .as_object()
            .ok_or_else(|| unsupported("missing safetensors weight_map"))?
        {
            if name.starts_with("audio_encoder.") || name.starts_with("speech_embeddings.") {
                let name = file
                    .as_str()
                    .ok_or_else(|| unsupported("invalid audio shard filename"))?;
                let path = Path::new(name);
                if path.is_absolute()
                    || path
                        .components()
                        .any(|c| !matches!(c, Component::Normal(_)))
                {
                    return Err(unsupported(format!("invalid audio shard path {name}")));
                }
                files.insert(snapshot.join(path));
            }
        }
    } else {
        files.insert(snapshot.join("model.safetensors"));
    }
    let mut result = Catalog::new();
    for path in files {
        for metadata in read_safetensors_metadata(&path).map_err(|e| unsupported(e.to_string()))? {
            if metadata.name.starts_with("encoder.")
                || metadata.name.starts_with("audio_encoder.")
                || metadata.name.starts_with("speech_embeddings.")
            {
                if result
                    .insert(metadata.name.clone(), (path.clone(), metadata))
                    .is_some()
                {
                    return Err(unsupported("duplicate audio tensor"));
                }
            }
        }
    }
    Ok(result)
}
impl AudioTowerPlan {
    pub fn output_width(&self) -> usize {
        self.output_width
    }
    pub fn storage(&self) -> AudioStorage {
        self.storage
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn reads(&self) -> &BTreeMap<String, AudioTensorRead> {
        &self.reads
    }
    pub fn from_snapshot(snapshot: &Path, storage: AudioStorage) -> Result<Self> {
        let root: RootConfig = serde_json::from_reader(File::open(snapshot.join("config.json"))?)?;
        let tokenizer =
            serde_json::from_reader(File::open(snapshot.join("audio_tokenizer/config.json"))?)?;
        validate_config(&root, &tokenizer)?;
        Self::from_catalog(root.hidden_size, storage, &catalog(snapshot)?)
    }
    fn from_catalog(output_width: usize, storage: AudioStorage, catalog: &Catalog) -> Result<Self> {
        let mut result = Self {
            output_width,
            storage,
            weight_bytes: 0,
            reads: BTreeMap::new(),
        };
        let mut put = |name: String, shape: &[usize], source: DType, vector: bool| -> Result<()> {
            let (path, m) = catalog
                .get(&name)
                .ok_or_else(|| unsupported(format!("missing {name}")))?;
            let elements = shape
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d as u64))
                .ok_or_else(|| unsupported("audio tensor shape overflow"))?;
            let source_bytes = match source {
                DType::Bf16 => 2,
                DType::F32 => 4,
                _ => return Err(unsupported("audio source dtype")),
            };
            if m.dtype != source || m.shape != shape || m.byte_length != elements * source_bytes {
                return Err(unsupported(format!(
                    "{name}: {:?} {:?}, need {source:?} {shape:?}",
                    m.dtype, m.shape
                )));
            }
            let resident_dtype = if vector || storage == AudioStorage::Fp32 {
                DType::F32
            } else {
                source
            };
            let bytes = elements * if resident_dtype == DType::F32 { 4 } else { 2 };
            let destination = (result.weight_bytes + 255) & !255;
            result.weight_bytes = destination
                .checked_add(bytes)
                .ok_or_else(|| unsupported("audio weight arena overflow"))?;
            result.reads.insert(
                name,
                AudioTensorRead {
                    metadata: m.clone(),
                    path: path.clone(),
                    destination,
                    resident_dtype,
                },
            );
            Ok(())
        };
        for (name, shape, vector) in [
            ("conv1.weight", vec![1024, 128, 3], false),
            ("conv1.bias", vec![1024], true),
            ("conv2.weight", vec![1024, 1024, 3], false),
            ("conv2.bias", vec![1024], true),
            ("down_sample_layer.0.weight", vec![1024, 1024, 2], false),
            ("layer_norm.weight", vec![1024], true),
            ("layer_norm.bias", vec![1024], true),
            ("down_sample_norm.weight", vec![1024], true),
            ("down_sample_norm.bias", vec![1024], true),
        ] {
            put(format!("encoder.{name}"), &shape, DType::Bf16, vector)?;
        }
        for i in 0..24 {
            for (name, shape, vector) in [
                ("self_attn.q_proj.weight", vec![1024, 1024], false),
                ("self_attn.q_proj.bias", vec![1024], true),
                ("self_attn.k_proj.weight", vec![1024, 1024], false),
                ("self_attn.v_proj.weight", vec![1024, 1024], false),
                ("self_attn.v_proj.bias", vec![1024], true),
                ("self_attn.out_proj.weight", vec![1024, 1024], false),
                ("self_attn.out_proj.bias", vec![1024], true),
                ("self_attn_layer_norm.weight", vec![1024], true),
                ("self_attn_layer_norm.bias", vec![1024], true),
                ("final_layer_norm.weight", vec![1024], true),
                ("final_layer_norm.bias", vec![1024], true),
                ("fc1.weight", vec![4096, 1024], false),
                ("fc1.bias", vec![4096], true),
                ("fc2.weight", vec![1024, 4096], false),
                ("fc2.bias", vec![1024], true),
            ] {
                put(
                    format!("encoder.layers.{i}.{name}"),
                    &shape,
                    DType::Bf16,
                    vector,
                )?;
            }
        }
        for (i, size) in CODEBOOKS.iter().enumerate() {
            put(
                format!("encoder.quantizer.vq.layers.{i}._codebook.embed"),
                &[*size, 1024],
                DType::F32,
                false,
            )?;
            put(
                format!("speech_embeddings.{i}.weight"),
                &[1280, 1024],
                DType::Bf16,
                false,
            )?;
        }
        for i in 0..6 {
            for (name, shape, vector) in [
                ("input_layernorm.weight", vec![1024], true),
                ("post_attention_layernorm.weight", vec![1024], true),
                ("self_attn.q_proj.weight", vec![1024, 1024], false),
                ("self_attn.q_proj.bias", vec![1024], true),
                ("self_attn.k_proj.weight", vec![1024, 1024], false),
                ("self_attn.k_proj.bias", vec![1024], true),
                ("self_attn.v_proj.weight", vec![1024, 1024], false),
                ("self_attn.v_proj.bias", vec![1024], true),
                ("self_attn.o_proj.weight", vec![1024, 1024], false),
                ("mlp.gate_proj.weight", vec![4096, 1024], false),
                ("mlp.up_proj.weight", vec![4096, 1024], false),
                ("mlp.down_proj.weight", vec![1024, 4096], false),
            ] {
                put(
                    format!("audio_encoder.input_local_transformer.layers.{i}.{name}"),
                    &shape,
                    DType::Bf16,
                    vector,
                )?;
            }
        }
        put(
            "audio_encoder.input_local_transformer.norm.weight".into(),
            &[1024],
            DType::Bf16,
            true,
        )?;
        put(
            "audio_encoder.projection.mlp.0.weight".into(),
            &[16384, 4096],
            DType::Bf16,
            false,
        )?;
        put(
            "audio_encoder.projection.mlp.2.weight".into(),
            &[output_width, 16384],
            DType::Bf16,
            false,
        )?;
        drop(put);
        result.weight_bytes = (result.weight_bytes + 255) & !255;
        // Only the explicitly unused RVQ training buffers may be ignored. Unknown
        // inference tensors must not silently change the bundled tower architecture.
        for name in catalog.keys() {
            if !result.reads.contains_key(name)
                && ![".cluster_size", ".embed_avg", ".inited"]
                    .iter()
                    .any(|suffix| {
                        name.starts_with("encoder.quantizer.vq.layers.") && name.ends_with(suffix)
                    })
            {
                return Err(unsupported(format!("unexpected inference tensor {name}")));
            }
        }
        Ok(result)
    }
    /// Backend/toolchain identity is supplied by the resident native runtime.
    pub fn encoder_id(&self, revision: &str, sm: u32, backend: &str) -> EncoderId {
        let headers = self.reads.iter().map(|(name,read)| {
            let m=&read.metadata;
            (name.clone(),serde_json::json!({"dtype":format!("{:?}",m.dtype),"shape":m.shape,
                "offset":m.byte_offset,"length":m.byte_length,"resident_dtype":format!("{:?}",read.resident_dtype)}))
        }).chain([( "$backend".into(),serde_json::json!({"backend":backend,"storage":format!("{:?}",self.storage)}))]).collect();
        EncoderId::derive("mimo_v26_audio", revision, &headers, 1, sm)
    }
    /// The caller admits the complete native weights/scratch ledger first. This
    /// additional weight-only check prevents accidental cold-path payload reads.
    pub fn load_weights(&self, admitted_weight_bytes: u64) -> Result<Vec<u8>> {
        if admitted_weight_bytes < self.weight_bytes {
            return Err(AudioTowerError::Admission {
                required: self.weight_bytes,
                admitted: admitted_weight_bytes,
            });
        }
        let mut arena = Vec::new();
        arena
            .try_reserve_exact(self.weight_bytes as usize)
            .map_err(|e| unsupported(e.to_string()))?;
        arena.resize(self.weight_bytes as usize, 0);
        for read in self.reads.values() {
            let m = &read.metadata;
            let mut file = File::open(&read.path)?;
            file.seek(SeekFrom::Start(m.byte_offset))?;
            let start = read.destination as usize;
            if m.dtype == read.resident_dtype {
                file.read_exact(&mut arena[start..start + m.byte_length as usize])?;
            } else {
                // Bounded conversion staging, independent of tensor size.
                let mut staging = [0u8; 16384];
                let mut remaining = m.byte_length as usize;
                let mut destination = start;
                while remaining > 0 {
                    let count = remaining.min(staging.len());
                    file.read_exact(&mut staging[..count])?;
                    for bytes in staging[..count].chunks_exact(2) {
                        let bits = u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16;
                        arena[destination..destination + 4].copy_from_slice(&bits.to_le_bytes());
                        destination += 4;
                    }
                    remaining -= count;
                }
            }
        }
        Ok(arena)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_precedes_payload_reads_and_identity_includes_backend() {
        let mut plan = AudioTowerPlan {
            output_width: 4096,
            storage: AudioStorage::Fp32,
            weight_bytes: 256,
            reads: BTreeMap::new(),
        };
        plan.reads.insert(
            "encoder.conv1.bias".into(),
            AudioTensorRead {
                path: PathBuf::from("/nonexistent/audio-tower.safetensors"),
                destination: 0,
                resident_dtype: DType::F32,
                metadata: SafetensorsTensorMetadata {
                    name: "encoder.conv1.bias".into(),
                    dtype: DType::Bf16,
                    shape: vec![1024],
                    byte_offset: 0,
                    byte_length: 2048,
                },
            },
        );
        assert!(matches!(
            plan.load_weights(255),
            Err(AudioTowerError::Admission { .. })
        ));
        let key = plan.encoder_id("snapshot", 120, "cufft13.2/cublas13.2");
        assert_ne!(
            key,
            plan.encoder_id("snapshot", 121, "cufft13.2/cublas13.2")
        );
        assert_ne!(
            key,
            plan.encoder_id("snapshot", 120, "cufft13.3/cublas13.2")
        );
        assert_ne!(key, plan.encoder_id("other", 120, "cufft13.2/cublas13.2"));
    }
    #[test]
    fn missing_inference_tensor_is_named() {
        let error = AudioTowerPlan::from_catalog(4096, AudioStorage::MixedBf16, &Catalog::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("encoder.conv1.weight"));
    }
}
