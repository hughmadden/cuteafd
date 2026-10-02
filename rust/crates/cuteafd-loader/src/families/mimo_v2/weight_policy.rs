//! Measured single-copy defaults for the qualified MiMo V2.6 Pro checkpoint.
//! Signatures identify config/header metadata, not the tensor payload. Local
//! copies keep the policy regardless of directory names; explicit selections
//! still take precedence. Other checkpoints retain their source formats.
use crate::{plan::checkpoint::Checkpoint, SafetensorsTensorMetadata};
use cuteafd_core::DType;
use serde_json::Value;
use sha2::{Digest, Sha256};
use super::MimoV2Config;
use super::projection::{MimoProjectionLayout, MimoProjectionLayoutError, MimoProjectionRepresentation};

const TARGET_CONFIG: &str = "e274f898974f0b13a170e33c5328a82e15e5ce2829ae95ebfc7b23c8ffc8a013";
const TARGET_HEADERS: &str = "d8bcb52b6c216dca96330d0e36614922659b50a701174ddcdd94b2fbc07cf4fc";
const DRAFT_CONFIG: &str = "c5327fcee7e0b697f0a2eae787041a3f31ad2ce53ab946a90a0ffe38cf040abb";
const DRAFT_HEADERS: &str = "0531c3f3ee7db347013884d58270f7065d400613598b1e6ab4915f7aaa2886c7";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoDefaultPolicy { Checkpoint, QualifiedProFp8 }

/// Aggregate physical head/O storage across coordinator ranks; this does not
/// include other target weights, optional drafters, workspaces or caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualifiedProjectionMemory {
    pub source_bytes: u64,
    pub resident_bytes: u64,
    /// One drained source block, not the sum of every converted matrix.
    pub max_load_staging: u64,
}

pub fn qualified_projection_memory(cfg: &MimoV2Config) -> Result<QualifiedProjectionMemory, MimoProjectionLayoutError> {
    let head = |format| MimoProjectionLayout::new(cfg.vocab_size as u64, cfg.hidden as u64, format, 1);
    let output = |format| MimoProjectionLayout::new(cfg.hidden as u64,
        (cfg.heads as u64).checked_mul(cfg.v_head_dim as u64).ok_or(MimoProjectionLayoutError)?, format, 2);
    let source_head = head(MimoProjectionRepresentation::Bf16)?;
    let source_output = output(MimoProjectionRepresentation::Bf16)?;
    let resident_head = head(MimoProjectionRepresentation::Fp8)?;
    let resident_output = output(MimoProjectionRepresentation::Fp8)?;
    let aggregate = |head: MimoProjectionLayout, output: MimoProjectionLayout| {
        let head_bytes = head.resident_bytes()?;
        output.resident_bytes()?.checked_mul(cfg.layers as u64)
            .and_then(|bytes| bytes.checked_add(head_bytes))
            .ok_or(MimoProjectionLayoutError)
    };
    Ok(QualifiedProjectionMemory {
        source_bytes: aggregate(source_head, source_output)?,
        resident_bytes: aggregate(resident_head, resident_output)?,
        max_load_staging: resident_head.max_load_staging.max(resident_output.max_load_staging),
    })
}

fn string(hash: &mut Sha256, text: &str) {
    hash.update((text.len() as u64).to_le_bytes());
    hash.update(text.as_bytes());
}

// Numeric values use their IEEE representation, independent of JSON spelling;
// object keys are sorted so serializer formatting/order never changes identity.
fn canonical(hash: &mut Sha256, value: &Value) {
    match value {
        Value::Null => hash.update(b"0"),
        Value::Bool(value) => hash.update(if *value { b"t" } else { b"f" }),
        Value::Number(value) => {
            hash.update(b"n");
            hash.update(value.as_f64().expect("JSON number has an f64 representation").to_le_bytes());
        }
        Value::String(value) => { hash.update(b"s"); string(hash, value); }
        Value::Array(values) => {
            hash.update(b"a"); hash.update((values.len() as u64).to_le_bytes());
            for value in values { canonical(hash, value); }
        }
        Value::Object(values) => {
            hash.update(b"o"); hash.update((values.len() as u64).to_le_bytes());
            let mut keys: Vec<_> = values.keys().collect(); keys.sort();
            for key in keys { string(hash, key); canonical(hash, &values[key]); }
        }
    }
}

fn config_signature(value: &Value) -> String {
    let mut hash = Sha256::new(); hash.update(b"mimo-qualified-config-v1");
    canonical(&mut hash, value); format!("{:x}", hash.finalize())
}

fn header_signature<'a>(headers: impl IntoIterator<Item = &'a SafetensorsTensorMetadata>) -> String {
    let mut headers: Vec<_> = headers.into_iter().collect();
    headers.sort_by(|a, b| a.name.cmp(&b.name));
    let mut hash = Sha256::new(); hash.update(b"mimo-qualified-headers-v1");
    for header in headers {
        string(&mut hash, &header.name); string(&mut hash, &format!("{:?}", header.dtype));
        hash.update((header.shape.len() as u64).to_le_bytes());
        for &dimension in &header.shape { hash.update((dimension as u64).to_le_bytes()); }
    }
    format!("{:x}", hash.finalize())
}

pub fn default_policy(checkpoint: &Checkpoint, cfg: &MimoV2Config) -> MimoDefaultPolicy {
    if (cfg.layers, cfg.hidden, cfg.heads, cfg.full_kv_heads, cfg.swa_kv_heads,
        cfg.head_dim, cfg.v_head_dim, cfg.experts, cfg.topk, cfg.moe_intermediate, cfg.vocab_size)
        != (70, 6144, 128, 8, 8, 192, 128, 384, 8, 2048, 152576)
        || config_signature(&checkpoint.config) != TARGET_CONFIG {
        return MimoDefaultPolicy::Checkpoint;
    }
    let mut selected = Vec::with_capacity(71);
    for name in std::iter::once("lm_head.weight".to_string())
        .chain((0..70).map(|layer| format!("model.layers.{layer}.self_attn.o_proj.weight"))) {
        let Ok(index) = checkpoint.tensors.binary_search_by(|tensor| tensor.meta.name.cmp(&name)) else {
            return MimoDefaultPolicy::Checkpoint;
        };
        let tensor = &checkpoint.tensors[index].meta;
        if tensor.dtype != DType::Bf16 { return MimoDefaultPolicy::Checkpoint; }
        selected.push(tensor);
    }
    if header_signature(selected) == TARGET_HEADERS {
        MimoDefaultPolicy::QualifiedProFp8
    } else { MimoDefaultPolicy::Checkpoint }
}

/// Only the embedded drafter's exact qualified config and source headers get
/// the automatic FP8 representation. An external drafter remains independent.
pub fn qualified_draft(config: &Value, headers: &[SafetensorsTensorMetadata]) -> bool {
    headers.len() == 63 && headers.iter().all(|tensor| tensor.dtype == DType::Bf16)
        && config_signature(config) == DRAFT_CONFIG && header_signature(headers) == DRAFT_HEADERS
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::plan::checkpoint::CheckpointTensor;

    pub fn fixture(snapshot: &str) -> (Checkpoint, MimoV2Config, Value, Vec<SafetensorsTensorMetadata>) {
        let value: Value = serde_json::from_str(include_str!("../../../tests/fixtures/mimo-qualified-pro.json")).unwrap();
        let metadata = |key: &str| -> Vec<SafetensorsTensorMetadata> {
            value[key].as_object().unwrap().iter().map(|(name, tensor)| SafetensorsTensorMetadata {
                name: name.clone(), dtype: DType::from_safetensors(tensor["dtype"].as_str().unwrap()),
                shape: tensor["shape"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as usize).collect(),
                byte_offset: 0, byte_length: 0,
            }).collect()
        };
        let mut tensors: Vec<_> = metadata("target_headers").into_iter().map(|meta| CheckpointTensor { shard: "test.safetensors".into(), meta }).collect();
        tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
        let cfg = MimoV2Config::from_hf(&value["target_config"]).unwrap();
        let checkpoint = Checkpoint { snapshot: snapshot.into(), config: value["target_config"].clone(),
            quantize_config: None, tensors, missing_shards: Vec::new(), shard_bytes: 0 };
        (checkpoint, cfg, value["draft_config"].clone(), metadata("draft_headers"))
    }

    #[test]
    fn qualified_local_copy_keeps_default_without_hf_path_identity() {
        for snapshot in ["/arbitrarily-renamed-local-copy", "/hf/models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/rev"] {
            let (checkpoint, cfg, draft, headers) = fixture(snapshot);
            assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::QualifiedProFp8);
            assert!(qualified_draft(&draft, &headers));
            let spec = crate::plan::families::mimo::spec_from(&cfg, &checkpoint);
            assert!(spec.notes.iter().any(|note| note.contains("qualified MiMo V2.6 Pro default: single-copy FP8")));
            assert!(spec.notes.iter().any(|note| note.contains("checkpoint source storage") && note.contains("not complete resident-memory admission")));
        }
    }

    #[test]
    fn selected_projection_report_uses_physical_layout_and_one_loading_stage() {
        let (_, cfg, _, _) = fixture("/local");
        let memory = qualified_projection_memory(&cfg).unwrap();
        assert_eq!(memory.source_bytes, 15_967_715_328);
        assert_eq!(memory.resident_bytes, 8_453_554_176);
        assert_eq!(memory.max_load_staging, 33_554_432);
    }

    #[test]
    fn same_geometry_with_changed_config_or_source_headers_keeps_checkpoint_formats() {
        let (mut checkpoint, mut cfg, _, _) = fixture("/local");
        checkpoint.config["attention_value_scale"] = serde_json::json!(0.5);
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
        let (mut checkpoint, _, _, _) = fixture("/local");
        checkpoint.tensors[0].meta.dtype = DType::F8E4M3;
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
        let (checkpoint, _, _, _) = fixture("/local");
        cfg.hidden = 4096;
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
    }

    #[test]
    fn changed_draft_config_or_header_does_not_inherit_target_policy() {
        let (_, _, mut draft, mut headers) = fixture("/local");
        draft["block_size"] = serde_json::json!(16);
        assert!(!qualified_draft(&draft, &headers));
        let (_, _, draft, _) = fixture("/local");
        headers[0].shape[0] += 1;
        assert!(!qualified_draft(&draft, &headers));
    }
}
