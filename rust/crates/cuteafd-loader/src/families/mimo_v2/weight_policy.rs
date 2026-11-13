//! Target weights preserve checkpoint precision; head/O FP8 conversion is opt-in.
//! Drafter policy is independent because target verification determines output.
use crate::plan::checkpoint::Checkpoint;
use super::MimoV2Config;
use super::projection::{MimoProjectionLayout, MimoProjectionLayoutError, MimoProjectionRepresentation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoDefaultPolicy { Checkpoint, Fp8 }

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

/// Automatic target loading never changes checkpoint precision.
pub fn default_policy(_checkpoint: &Checkpoint, _cfg: &MimoV2Config) -> MimoDefaultPolicy {
    MimoDefaultPolicy::Checkpoint
}

/// Preserve the independently qualified FP8 drafter default.
pub fn default_draft_fp8(cfg: &MimoV2Config) -> bool {
    !matches!(cfg.program_family().ok(), Some("mimof" | "mimof2"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::plan::checkpoint::CheckpointTensor;
    use crate::SafetensorsTensorMetadata;
    use cuteafd_core::DType;
    use serde_json::Value;

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
        let checkpoint = Checkpoint { view: None, snapshot: snapshot.into(), config: value["target_config"].clone(),
            quantize_config: None, weight_map: Default::default(), tensors, missing_shards: Vec::new(), shard_bytes: 0 };
        (checkpoint, cfg, value["draft_config"].clone(), metadata("draft_headers"))
    }

    #[test]
    fn mimo_checkpoints_default_to_source_head_and_output() {
        for snapshot in ["/arbitrarily-renamed-local-copy", "/hf/models--XiaomiMiMo--MiMo-V2.6-Pro-RL/snapshots/rev"] {
            let (checkpoint, cfg, _, _) = fixture(snapshot);
            assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
            let spec = crate::plan::families::mimo::spec_from(&cfg, &checkpoint);
            assert!(spec.notes.iter().any(|note| note.contains("checkpoint precision")));
            assert!(default_draft_fp8(&cfg));
        }
    }

    #[test]
    fn flash_mopd_does_not_inherit_qualified_lossy_target_defaults() {
        let dir = tempfile::tempdir().unwrap();
        crate::plan::testing::write_snapshot(dir.path(), &crate::plan::testing::mimo_flash_mopd_config(),
            &crate::plan::testing::mimo_flash_mopd_tensors(), Some(4));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let cfg = MimoV2Config::from_hf(&checkpoint.config).unwrap();
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
        assert_eq!(default_policy(&checkpoint, &cfg.head_split(2).unwrap()), MimoDefaultPolicy::Checkpoint);
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
    fn missing_or_unaligned_projection_keeps_checkpoint_formats() {
        let (mut checkpoint, cfg, _, _) = fixture("/local");
        let at = checkpoint.tensors.iter().position(|t| t.meta.name == "lm_head.weight").unwrap();
        checkpoint.tensors[at].meta.shape[1] = 6100;
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
        checkpoint.tensors.remove(at);
        assert_eq!(default_policy(&checkpoint, &cfg), MimoDefaultPolicy::Checkpoint);
    }
}
